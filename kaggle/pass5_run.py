"""ArtaListen pass5 / v5 — forced-Russian, junk-filtered, multi-variant recovery.

Modes (env ARTALISTEN_MODE):
  voice       — one Voice *.m4a (+ optional anna.json profile for cosine match)
  cafe_pair   — anna+brother train fit + frozen test (same as cafe kernel)

Always: language=ru, condition_on_previous_text=False, strong junk filter,
short independent windows + one full-context pass, consensus across variants.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np


def _package_src() -> Path:
    local = Path(__file__).resolve().parent.parent / "src" / "artalisten" / "cli.py"
    if local.is_file():
        return local.parents[1]
    found = [p for p in Path("/kaggle/input").rglob("artalisten/cli.py") if p.is_file()]
    if len(found) == 1:
        return found[0].parents[1]
    dest = Path("/kaggle/working/srcpkg")
    dest.mkdir(parents=True, exist_ok=True)
    for archive in Path("/kaggle/input").rglob("*.zip"):
        with zipfile.ZipFile(archive) as handle:
            handle.extractall(dest)
    found = [p for p in dest.rglob("artalisten/cli.py") if p.is_file()]
    if len(found) != 1:
        raise SystemExit(f"expected one artalisten package, found {found}")
    return found[0].parents[1]


sys.path.insert(0, str(_package_src()))

MODE = os.environ.get("ARTALISTEN_MODE", "voice").strip().lower()


def _pip(constraints: Path, args: list[str]) -> int:
    cmd = [sys.executable, "-m", "pip", "install", "--upgrade-strategy", "only-if-needed", "-c", str(constraints), *args]
    print("PIP", " ".join(cmd), flush=True)
    return subprocess.call(cmd)


def _install() -> None:
    import torch

    constraints = Path("/tmp/artalisten-constraints.txt")
    constraints.write_text(f"torch=={torch.__version__}\n", encoding="utf-8")
    _pip(
        constraints,
        [
            "demucs",
            "faster-whisper",
            "speechbrain",
            "scikit-learn",
            "silero-vad",
            "soundfile",
            "scipy",
        ],
    )
    pkg_root = _package_src()  # .../src containing artalisten/
    project = pkg_root.parent if (pkg_root / "artalisten").is_dir() else pkg_root
    # Prefer editable project install when pyproject.toml exists; else keep sys.path.
    if (project / "pyproject.toml").is_file():
        _pip(constraints, ["-e", str(project)])
    print(f"SRC {pkg_root} project {project}", flush=True)


def _find_audio() -> Path:
    root = Path("/kaggle/input")
    names = [
        "Voice 261005_102742.m4a",
        "Voice 261005_104834.m4a",
        "Voice 261005_111011.m4a",
        "Voice 261005_111604.m4a",
        "anna+brother.m4a",
        "anna+brother_test.m4a",
        "annabrother.m4a",
        "annabrother_test.m4a",
    ]
    marker = os.environ.get("ARTALISTEN_DATASET_MARKER", "")
    matches = []
    for name in names:
        for path in root.rglob(name):
            if path.is_file():
                matches.append(path)
    if marker:
        matches = [p for p in matches if marker in str(p)]
    # Prefer the file named in ARTALISTEN_FILE if set
    want = os.environ.get("ARTALISTEN_FILE", "")
    if want:
        matches = [p for p in matches if p.name == want]
    if len(matches) != 1:
        listing = [str(p) for p in root.rglob("*")][:80] if root.exists() else []
        raise SystemExit(f"expected one audio, found {matches}. tree={listing}")
    return matches[0]


def _find_anna_profile() -> dict | None:
    root = Path("/kaggle/input")
    paths = [p for p in root.rglob("anna.json") if p.is_file()]
    # also working
    paths += [p for p in Path("/kaggle/working").rglob("anna.json") if p.is_file()] if Path("/kaggle/working").exists() else []
    if not paths:
        return None
    return json.loads(paths[0].read_text(encoding="utf-8"))


def _band(audio: np.ndarray, sr: int, low: float, high: float) -> np.ndarray:
    from artalisten.enhance import anna_bandpass

    return anna_bandpass(audio, sr, low, high)


def _dynnorm(audio: np.ndarray, sr: int) -> np.ndarray:
    from artalisten.enhance import dynamic_normalize

    return dynamic_normalize(audio, sr)


def _demucs_vocals(src: Path, out_dir: Path) -> tuple[np.ndarray, int]:
    from artalisten.audioio import decode_wav, read_mono, resample, write_wav
    from artalisten.device import release_memory
    from artalisten.separate import separate_speech

    out_dir.mkdir(parents=True, exist_ok=True)
    decoded = out_dir / "decoded.wav"
    vocals = out_dir / "vocals.wav"
    decode_wav(src, decoded, 44100, 2, None)
    separate_speech(decoded, vocals, out_dir / "envelopes.npz", device="cuda", shifts=2)
    audio, sr = read_mono(vocals)
    audio = resample(audio, sr, 16000)
    decoded.unlink(missing_ok=True)
    release_memory()
    return np.asarray(audio, dtype=np.float32), 16000


def _sepformer(audio: np.ndarray, sr: int) -> np.ndarray:
    """Optional SpeechBrain sepformer; on failure return input."""
    try:
        import torch
        from speechbrain.inference.separation import SepformerSeparation

        model = SepformerSeparation.from_hparams(
            source="speechbrain/sepformer-wham16k",
            savedir="/kaggle/working/sb-sepformer",
            run_opts={"device": "cuda"},
        )
        wav = torch.from_numpy(audio).float().unsqueeze(0)
        est = model.separate_batch(wav)
        if isinstance(est, (list, tuple)):
            est = est[0]
        arr = est.squeeze().detach().cpu().numpy()
        if arr.ndim > 1:
            # pick loudest stream
            rms = [float(np.sqrt(np.mean(ch * ch) + 1e-12)) for ch in arr]
            arr = arr[int(np.argmax(rms))]
        peak = float(np.max(np.abs(arr))) + 1e-9
        return np.ascontiguousarray((arr / peak * 0.9).astype(np.float32))
    except Exception as exc:
        print(f"SEPFORMER skip: {exc}", flush=True)
        return audio


def _load_raw_16k(src: Path) -> np.ndarray:
    from artalisten.audioio import decode_wav, read_mono, resample

    tmp = Path("/kaggle/working/_raw_decode.wav")
    decode_wav(src, tmp, 16000, 1, None)
    audio, sr = read_mono(tmp)
    if sr != 16000:
        audio = resample(audio, sr, 16000)
    tmp.unlink(missing_ok=True)
    return np.asarray(audio, dtype=np.float32)


def _variants(src: Path, out_dir: Path) -> dict[str, np.ndarray]:
    demucs, sr = _demucs_vocals(src, out_dir / "demucs")
    write = {}
    from artalisten.audioio import write_wav

    A = demucs
    B = _sepformer(demucs, sr)
    E = _band(demucs, sr, 140.0, 6500.0)
    # level norm for E
    peak = float(np.max(np.abs(E))) + 1e-9
    E = (E / peak * 0.9).astype(np.float32)
    raw = _load_raw_16k(src)
    F = _band(raw, sr, 140.0, 6500.0)
    peak = float(np.max(np.abs(F))) + 1e-9
    F = (F / peak * 0.9).astype(np.float32)
    G = _dynnorm(_band(demucs, sr, 140.0, 6500.0), sr)
    variants = {
        "A_demucs_only": A,
        "B_demucs_sepformer": B,
        "E_demucs_anna_band": E,
        "F_no_separation_band": F,
        "G_demucs_dynnorm_anna_band": G,
    }
    for name, audio in variants.items():
        write_wav(out_dir / f"variant_{name}.wav", audio, 16000)
    return variants


def _two_speaker_and_anna(audio: np.ndarray, anna_profile: dict | None) -> dict:
    from artalisten.speakers import _embed, cluster_global, speech_windows

    # Silero VAD
    from silero_vad import get_speech_timestamps, load_silero_vad
    import torch

    model = load_silero_vad()
    ts = get_speech_timestamps(
        torch.from_numpy(audio),
        model,
        sampling_rate=16000,
        threshold=0.3,
        min_speech_duration_ms=80,
        min_silence_duration_ms=200,
    )
    windows = [(t["start"] / 16000, t["end"] / 16000) for t in ts]
    windows = speech_windows([(float(a), float(b)) for a, b in windows]) if windows else []
    if len(windows) < 2:
        return {
            "windows": [],
            "speakers_present": 0,
            "two_speaker_test": None,
            "turns": [],
            "anna_cluster": None,
        }
    clips = [np.ascontiguousarray(audio[int(s * 16000) : int(e * 16000)], dtype=np.float32) for s, e in windows]
    emb, _ = _embed(clips, "cuda", Path("/kaggle/working/spk-cache"))
    labels = cluster_global(emb, 2)
    # F0 rough via numpy autocorrelation-ish skip — use energy + cosine to Anna
    sizes = [int(np.sum(labels == 0)), int(np.sum(labels == 1))]
    cos_to_anna = {0: None, 1: None}
    if anna_profile and anna_profile.get("embedding"):
        ref = np.asarray(anna_profile["embedding"], dtype=np.float64)
        ref = ref / (np.linalg.norm(ref) + 1e-9)
        for lab in (0, 1):
            idx = np.where(labels == lab)[0]
            if len(idx) == 0:
                continue
            mean = emb[idx].mean(axis=0)
            mean = mean / (np.linalg.norm(mean) + 1e-9)
            cos_to_anna[lab] = float(np.dot(mean, ref))
    # Prefer higher cosine-to-Anna; else higher-F0 proxy = quieter? Use cosine then size.
    if cos_to_anna[0] is not None and cos_to_anna[1] is not None:
        anna_cluster = 0 if cos_to_anna[0] >= cos_to_anna[1] else 1
    else:
        anna_cluster = int(np.argmax(sizes))  # majority when solo
    # Collapse if tiny second cluster
    speakers_present = 2 if min(sizes) >= 3 and max(sizes) / max(min(sizes), 1) < 20 else 1
    if speakers_present == 1:
        anna_cluster = int(np.argmax(sizes))
    turns = [{"start": float(s), "end": float(e), "speaker": "Anna" if int(l) == anna_cluster else "Other", "cluster": int(l)} for (s, e), l in zip(windows, labels)]
    return {
        "windows": [[float(s), float(e), "Anna" if int(l) == anna_cluster else "Other", None] for (s, e), l in zip(windows, labels)],
        "speakers_present": speakers_present,
        "two_speaker_test": {
            "sizes": sizes,
            "cos_to_anna": cos_to_anna,
            "anna_cluster": anna_cluster,
        },
        "turns": turns,
        "anna_cluster": anna_cluster,
        "vad": [[float(s), float(e)] for s, e in windows],
    }


def _anna_stem(audio: np.ndarray, turns: list[dict], anna_cluster: int | None) -> np.ndarray:
    from artalisten.enhance import apply_speaker_mask, anna_bandpass

    if anna_cluster is None or not turns:
        return anna_bandpass(audio, 16000)
    keep = {"Anna", anna_cluster, str(anna_cluster)}
    masked = apply_speaker_mask(audio, 16000, turns, keep, keep_gain=1.35, other_gain=0.12)
    return anna_bandpass(masked, 16000)


def _decode_variant(model, audio: np.ndarray, turns: list[dict], tag: str) -> dict:
    from artalisten.asr import transcribe_independent_windows
    from artalisten.align import group_utterances, iter_chunks
    from artalisten.junk import filter_segments, good_word_count, is_junk
    from artalisten.speakers import speaker_at

    # Full-context (still no previous-text conditioning), forced Russian
    words_full = []
    duration = float(len(audio) / 16000)
    for index, (start, end) in enumerate(iter_chunks(duration, 300.0, 3.0)):
        piece = np.ascontiguousarray(audio[int(start * 16000) : int(end * 16000)], dtype=np.float32)
        segments, _ = model.transcribe(
            piece,
            language="ru",
            task="transcribe",
            word_timestamps=True,
            beam_size=5,
            temperature=0.0,
            vad_filter=False,
            condition_on_previous_text=False,
        )
        for segment in segments:
            no_speech = getattr(segment, "no_speech_prob", None)
            avg_lp = getattr(segment, "avg_logprob", None)
            cr = getattr(segment, "compression_ratio", None)
            text = (segment.text or "").strip()
            segs_words = []
            for word in segment.words or []:
                if word.start is None or word.end is None:
                    continue
                segs_words.append(
                    [
                        float(word.start) + start,
                        float(word.end) + start,
                        word.word,
                        None if word.probability is None else float(word.probability),
                    ]
                )
            abs_start = float(segment.start) + start
            abs_end = float(segment.end) + start
            speaker, _sid = speaker_at(turns, 0.5 * (abs_start + abs_end))
            words_full.append(
                {
                    "start": abs_start,
                    "end": abs_end,
                    "text": text,
                    "avg_logprob": None if avg_lp is None else float(avg_lp),
                    "no_speech_prob": None if no_speech is None else float(no_speech),
                    "compression_ratio": None if cr is None else float(cr),
                    "words": segs_words,
                    "speaker": speaker,
                }
            )

    kept_full, rej_full = filter_segments(words_full, text_key="text", min_word_prob=0.4)

    # Short independent windows
    win_words = transcribe_independent_windows(
        model, audio, 16000, turns, task="translate" if False else "transcribe", language="ru",
        window_seconds=10.0, hop_seconds=8.0, vad_filter=False,
    )
    # group into segment-like dicts
    win_segs = []
    for utt in group_utterances(win_words):
        text = (utt.get("text") or "").strip()
        win_segs.append(
            {
                "start": float(utt["start"]),
                "end": float(utt["end"]),
                "text": text,
                "words": [
                    [float(w["start"]), float(w["end"]), w.get("word"), w.get("probability")]
                    for w in (utt.get("words") or [])
                ],
                "speaker": utt.get("speaker"),
                "avg_logprob": None,
                "no_speech_prob": None,
            }
        )
    kept_win, rej_win = filter_segments(win_segs, text_key="text", min_word_prob=0.4)

    score = float(good_word_count(kept_full, 0.4) + 0.5 * good_word_count(kept_win, 0.4))
    print(f"CFG {tag} score={score:.1f} full_kept={len(kept_full)} win_kept={len(kept_win)}", flush=True)

    # Translate only on full-context kept Russian (audio same)
    en_kept = []
    en_rej = []
    if kept_full:
        segments, _ = model.transcribe(
            audio,
            language="ru",
            task="translate",
            word_timestamps=True,
            beam_size=5,
            temperature=0.0,
            vad_filter=False,
            condition_on_previous_text=False,
        )
        en_segs = []
        for segment in segments:
            text = (segment.text or "").strip()
            en_segs.append(
                {
                    "start": float(segment.start),
                    "end": float(segment.end),
                    "text": text,
                    "avg_logprob": None if segment.avg_logprob is None else float(segment.avg_logprob),
                    "no_speech_prob": None if segment.no_speech_prob is None else float(segment.no_speech_prob),
                    "compression_ratio": None if segment.compression_ratio is None else float(segment.compression_ratio),
                    "words": [
                        [float(w.start), float(w.end), w.word, None if w.probability is None else float(w.probability)]
                        for w in (segment.words or [])
                        if w.start is not None
                    ],
                    "speaker": speaker_at(turns, 0.5 * (float(segment.start) + float(segment.end)))[0],
                }
            )
        en_kept, en_rej = filter_segments(en_segs, text_key="text", min_word_prob=None)

    return {
        "source": tag,
        "score": score,
        "good_words": good_word_count(kept_full, 0.4),
        "kept": kept_full,
        "rejected": rej_full,
        "windows_kept": kept_win,
        "windows_rejected": rej_win,
        "translate_kept": en_kept,
        "translate_rejected": en_rej,
    }


def run_voice() -> None:
    from artalisten.asr import load_model
    from artalisten.audioio import write_wav
    from artalisten.consensus import consensus_phrases, final_only
    from artalisten.device import release_memory

    src = _find_audio()
    out = Path("/kaggle/working")
    print(f"INPUT {src} MODE voice", flush=True)
    _install()
    anna = _find_anna_profile()
    variants = _variants(src, out / "clean")
    # Speaker analysis on E (Anna band)
    spk = _two_speaker_and_anna(variants["E_demucs_anna_band"], anna)
    turns = spk["turns"]
    stem = _anna_stem(variants["E_demucs_anna_band"], turns, spk.get("anna_cluster"))
    write_wav(out / f"anna_{src.stem.replace(' ', '_')}.wav", stem, 16000)
    variants["stem_Anna"] = stem

    model = load_model("large-v3", "float16", device="cuda")
    decodes = []
    try:
        for name, audio in variants.items():
            decodes.append(_decode_variant(model, audio, turns, name))
            # Also store a translate-only entry shape compatible with pass4
            d = decodes[-1]
            if d.get("translate_kept") is not None:
                decodes.append(
                    {
                        "source": name,
                        "task": "translate",
                        "score": None,
                        "good_words": None,
                        "kept": d["translate_kept"],
                        "rejected": d["translate_rejected"],
                    }
                )
            decodes[-2 if d.get("translate_kept") is not None else -1]["task"] = "transcribe"
    finally:
        del model
        release_memory()

    # Consensus across transcribe variants only
    by_var = {}
    for d in decodes:
        if d.get("task") != "transcribe":
            continue
        by_var[d["source"]] = d.get("kept") or []
    groups = consensus_phrases(by_var, min_variants=2)
    strong = final_only(groups, min_variants=2)

    result = {
        "id": src.stem.replace("Voice ", "").replace(" ", "_"),
        "source": src.name,
        "seconds": float(len(variants["E_demucs_anna_band"]) / 16000),
        "pass": "pass5",
        "speakers_present": spk["speakers_present"],
        "two_speaker_test": spk["two_speaker_test"],
        "windows": spk["windows"],
        "vad_b_0.3": spk.get("vad"),
        "turns": [[t["start"], t["end"], t["speaker"]] for t in turns],
        "decodes": [
            {
                "source": d["source"],
                "task": d.get("task", "transcribe"),
                "score": d.get("score"),
                "good_words": d.get("good_words"),
                "kept": d.get("kept") or [],
                "rejected": d.get("rejected") or [],
                **(
                    {"windows_kept": d.get("windows_kept"), "windows_rejected": d.get("windows_rejected")}
                    if d.get("task") == "transcribe"
                    else {}
                ),
            }
            for d in decodes
        ],
        "consensus": groups,
        "consensus_final": strong,
    }
    out_json = out / f"pass5_{result['id']}.json"
    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    # Drop huge intermediate wavs except anna stem + E
    for wav in (out / "clean").rglob("*.wav"):
        if wav.name.startswith("variant_E") or wav.name.startswith("variant_G"):
            continue
        if "vocals" in wav.name or "decoded" in wav.name:
            wav.unlink(missing_ok=True)
    print(f"DONE {result['id']} speakers={result['speakers_present']} consensus={len(strong)}", flush=True)


def run_cafe_pair() -> None:
    # Delegate to the improved cafe script in the same package tree if present.
    # Inline import of kaggle.run main after setting paths.
    cafe = Path(__file__).resolve().parent / "run.py"
    if not cafe.is_file():
        # on Kaggle, look in input
        found = list(Path("/kaggle/input").rglob("kaggle/run.py")) + list(Path("/kaggle/input").rglob("run.py"))
        cafe = found[0] if found else None
    if cafe is None or not Path(cafe).is_file():
        raise SystemExit("cafe run.py not found for cafe_pair mode")
    print(f"CAFE_PAIR delegating to {cafe}", flush=True)
    # Execute as __main__
    code = Path(cafe).read_text(encoding="utf-8")
    ns = {"__name__": "__main__", "__file__": str(cafe)}
    exec(compile(code, str(cafe), "exec"), ns)


def main() -> None:
    print(f"PASS5 mode={MODE}", flush=True)
    if MODE in {"voice", "anna_voice"}:
        run_voice()
    elif MODE in {"cafe_pair", "cafe"}:
        run_cafe_pair()
    else:
        raise SystemExit(f"unknown ARTALISTEN_MODE={MODE}")


if __name__ == "__main__":
    main()
