"""Kaggle recovery: enhance, keep the quiet speaker, fit only on the train file.

The test recording is never passed to speaker fitting.
DeepFilterNet has no CPython 3.12 wheel. This script uses a SpeechBrain
enhancer that installs on 3.12, and falls back to DeepFilterNet under Python 3.11.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np


def _package_src() -> Path:
    local = Path(__file__).resolve().parent / "src" / "artalisten" / "cli.py"
    if local.is_file():
        return local.parents[1]
    found = [path for path in Path("/kaggle/input").rglob("artalisten/cli.py") if path.is_file()]
    if len(found) == 1:
        return found[0].parents[1]
    import zipfile

    dest = Path("/kaggle/working/srcpkg")
    dest.mkdir(parents=True, exist_ok=True)
    zips = [path for path in Path("/kaggle/input").rglob("*.zip") if path.is_file()]
    for archive in zips:
        with zipfile.ZipFile(archive) as handle:
            handle.extractall(dest)
    found = [path for path in dest.rglob("artalisten/cli.py") if path.is_file()]
    if len(found) != 1:
        raise SystemExit(f"expected one artalisten package, found {found} from {zips}")
    return found[0].parents[1]


sys.path.insert(0, str(_package_src()))

TRAIN_NAMES = ("anna+brother.m4a", "annabrother.m4a")
TEST_NAMES = ("anna+brother_test.m4a", "annabrother_test.m4a")

JUNK = (
    r"продолжение\s+следует",
    r"субтитр",
    r"dima\s*torzok",
    r"симон",
    r"подогнал",
    r"играет\s+музыка",
    r"звучит\s+музыка",
    r"звучить\s+музика",
    r"music\s+playing",
    r"девушки\s+отдыхают",
    r"дякую\s+за\s+перегляд",
    r"thanks\s+for\s+watching",
    r"говорит\s+на\s+русском",
    r"на\s+русском\s+языке",
    r"what is it",
    r"i can'?t hear you",
    r"tomorrow i will go to sleep",
    r"this is a story for the people",
    r"(?:yes[, ]+)?it is possible",
    r"i don'?t know what to (?:say|do)",
    r"i'?m going to start with a simple one",
    r"we'?re done",
    r"i want to hug you",
    r"i love you(?:\s+so\s+much)?",
    r"i can'?t take it anymore",
    r"what are you doing\?",
    r"i'?m telling you",
    r"now i will\b",
    r"let'?s do it",
    r"\boleg\b",
    r"stsq\d*",
)


def _one(names: tuple[str, ...], dataset_marker: str) -> Path:
    root = Path("/kaggle/input")
    matches = [path for name in names for path in root.rglob(name) if path.is_file()]
    scoped = [path for path in matches if dataset_marker in str(path)]
    if len(scoped) != 1:
        listing = [str(path) for path in root.rglob("*")] if root.exists() else ["<no /kaggle/input>"]
        raise SystemExit(
            f"expected one {names} inside {dataset_marker}, found {scoped} (all matches {matches}). "
            f"input tree: {listing[:80]}"
        )
    return scoped[0]


def _pip(constraints: Path, args: list[str]) -> int:
    cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--upgrade-strategy",
        "only-if-needed",
        "-c",
        str(constraints),
        *args,
    ]
    print("PIP", " ".join(cmd), flush=True)
    return subprocess.call(cmd)


def _install() -> None:
    import torch

    constraints = Path("/tmp/artalisten-constraints.txt")
    constraints.write_text(f"torch=={torch.__version__}\n", encoding="utf-8")
    code = _pip(
        constraints,
        [
            "demucs",
            "faster-whisper",
            "speechbrain",
            "silero-vad",
            "soundfile",
            "scikit-learn",
            "scipy",
            "numpy",
        ],
    )
    if code != 0:
        raise SystemExit(f"pip install failed with exit {code}")


def _match_len(audio: np.ndarray, size: int) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if audio.size >= size:
        return audio[:size]
    out = np.zeros(size, dtype=np.float32)
    out[: audio.size] = audio
    return out


def _overlap_add(audio: np.ndarray, sample_rate: int, fn, chunk_s: float = 10.0, overlap_s: float = 1.0) -> np.ndarray:
    chunk = int(chunk_s * sample_rate)
    overlap = int(overlap_s * sample_rate)
    hop = max(1, chunk - overlap)
    out = np.zeros(len(audio), dtype=np.float32)
    weight = np.zeros(len(audio), dtype=np.float32)
    pos = 0
    while pos < len(audio):
        end = min(len(audio), pos + chunk)
        den = _match_len(fn(audio[pos:end]), end - pos)
        window = np.ones(len(den), dtype=np.float32)
        if overlap and pos > 0:
            fade = min(overlap, len(den))
            window[:fade] = np.linspace(0.0, 1.0, fade, dtype=np.float32)
        if overlap and end < len(audio):
            fade = min(overlap, len(den))
            window[-fade:] = np.linspace(1.0, 0.0, fade, dtype=np.float32)
        out[pos : pos + len(den)] += den * window
        weight[pos : pos + len(den)] += window
        if end >= len(audio):
            break
        pos += hop
    return out / np.maximum(weight, 1e-6)


def _tensor_to_audio(est) -> np.ndarray:
    audio = est.detach().cpu().numpy() if hasattr(est, "detach") else np.asarray(est)
    audio = np.squeeze(audio)
    if audio.ndim == 2:
        if audio.shape[0] <= 4 and audio.shape[0] < audio.shape[1]:
            audio = audio[0]
        else:
            audio = audio[:, 0]
    return np.asarray(audio, dtype=np.float32).reshape(-1)


def _load_speechbrain(device: str):
    import torch

    errors: list[str] = []
    saved = Path("/kaggle/working/enhancer-weights")
    saved.mkdir(parents=True, exist_ok=True)
    try:
        from speechbrain.inference.separation import SepformerSeparation

        model = SepformerSeparation.from_hparams(
            source="speechbrain/sepformer-wham16k-enhancement",
            savedir=str(saved / "sepformer-wham16k-enhancement"),
            run_opts={"device": device},
        )
        name = "speechbrain/sepformer-wham16k-enhancement"

        def apply(piece: np.ndarray) -> np.ndarray:
            tensor = torch.from_numpy(np.ascontiguousarray(piece)).float().unsqueeze(0).to(device)
            with torch.no_grad():
                est = model.separate_batch(tensor)
            return _tensor_to_audio(est)

        print("ENHANCER", name, flush=True)
        return name, apply, model
    except Exception as exc:
        errors.append(f"sepformer: {exc}")
        print("ENHANCER_FAIL sepformer", exc, flush=True)
    try:
        from speechbrain.inference.enhancement import SpectralMaskEnhancement

        model = SpectralMaskEnhancement.from_hparams(
            source="speechbrain/metricgan-plus-voicebank",
            savedir=str(saved / "metricgan-plus-voicebank"),
            run_opts={"device": device},
        )
        name = "speechbrain/metricgan-plus-voicebank"

        def apply(piece: np.ndarray) -> np.ndarray:
            tensor = torch.from_numpy(np.ascontiguousarray(piece)).float().unsqueeze(0).to(device)
            lengths = torch.ones(1, device=device)
            with torch.no_grad():
                est = model.enhance_batch(tensor, lengths=lengths)
            return _tensor_to_audio(est)

        print("ENHANCER", name, flush=True)
        return name, apply, model
    except Exception as exc:
        errors.append(f"metricgan: {exc}")
        print("ENHANCER_FAIL metricgan", exc, flush=True)
    raise RuntimeError("; ".join(errors))


def _load_enhancer(device: str):
    try:
        return _load_speechbrain(device)
    except Exception as exc:
        print("ENHANCER_FAIL speechbrain", exc, flush=True)
    name, apply = _load_deepfilter_311()
    print("ENHANCER", name, flush=True)
    return name, apply, None


DF311 = r'''
import sys
from pathlib import Path
import numpy as np
import soundfile as sf
import torch

src, dst, device = sys.argv[1], sys.argv[2], sys.argv[3]
audio, sr = sf.read(src, always_2d=False, dtype="float32")
audio = np.asarray(audio, dtype=np.float32).reshape(-1)
if sr != 48000:
    import torchaudio
    audio = torchaudio.functional.resample(torch.from_numpy(audio).unsqueeze(0), sr, 48000).squeeze(0).numpy()
    sr = 48000
from df.enhance import enhance, init_df
model, state, _ = init_df(default_model="DeepFilterNet3", post_filter=True)
model.eval()
chunk = 20 * sr
hop = 19 * sr
out = np.zeros(len(audio), dtype=np.float32)
weight = np.zeros(len(audio), dtype=np.float32)
pos = 0
overlap = sr
while pos < len(audio):
    end = min(len(audio), pos + chunk)
    piece = torch.from_numpy(np.ascontiguousarray(audio[pos:end])).float().unsqueeze(0)
    with torch.no_grad():
        den = enhance(model, state, piece, pad=True)
    den = den.detach().cpu().numpy().reshape(-1)[: end - pos]
    window = np.ones(len(den), dtype=np.float32)
    if pos > 0:
        fade = min(overlap, len(den))
        window[:fade] = np.linspace(0, 1, fade, dtype=np.float32)
    if end < len(audio):
        fade = min(overlap, len(den))
        window[-fade:] = np.linspace(1, 0, fade, dtype=np.float32)
    out[pos:pos + len(den)] += den * window
    weight[pos:pos + len(den)] += window
    if end >= len(audio):
        break
    pos += hop
out = out / np.maximum(weight, 1e-6)
sf.write(dst, out, sr, subtype="PCM_16")
print("DF311_OK", device)
'''


def _load_deepfilter_311():
    py = _ensure_py311()
    script = Path("/kaggle/working/df311.py")
    script.write_text(DF311, encoding="utf-8")

    def apply_file(src_wav: Path, dst_wav: Path) -> None:
        subprocess.check_call([str(py), str(script), str(src_wav), str(dst_wav), "cpu"])

    return "DeepFilterNet3-py311", apply_file


def _ensure_py311() -> Path:
    py = Path("/kaggle/working/py311/bin/python")
    if py.is_file():
        return py
    subprocess.check_call([sys.executable, "-m", "pip", "install", "uv"])
    subprocess.check_call(["uv", "python", "install", "3.11"])
    subprocess.check_call(["uv", "venv", "/kaggle/working/py311", "--python", "3.11"])
    subprocess.check_call(
        [
            str(py),
            "-m",
            "pip",
            "install",
            "torch",
            "torchaudio",
            "deepfilternet",
            "soundfile",
            "numpy",
        ]
    )
    return py


def _sensitive_vad(audio: np.ndarray, sample_rate: int) -> tuple[list[dict], dict]:
    import torch
    from silero_vad import get_speech_timestamps, load_silero_vad

    if sample_rate != 16000:
        raise RuntimeError("Silero VAD expects 16 kHz audio")
    duration = float(len(audio) / sample_rate) if sample_rate else 0.0
    model = load_silero_vad()
    wav = torch.from_numpy(np.ascontiguousarray(audio)).float()
    tried = []
    chosen = None
    for threshold in (0.30, 0.18, 0.10, 0.05):
        timestamps = get_speech_timestamps(
            wav,
            model,
            sampling_rate=16000,
            threshold=threshold,
            min_speech_duration_ms=80,
            min_silence_duration_ms=180,
            speech_pad_ms=400,
            return_seconds=True,
        )
        speech = sum(float(item["end"]) - float(item["start"]) for item in timestamps)
        row = {"threshold": threshold, "speech_seconds": round(speech, 1), "regions": len(timestamps)}
        tried.append(row)
        print(
            f"VAD threshold {threshold:.2f} speech {speech:.1f}s regions {len(timestamps)}",
            flush=True,
        )
        if speech > 0.80 * duration:
            print("VAD stop: this threshold marks most of the file", flush=True)
            break
        chosen = (threshold, timestamps, speech)
        if speech >= 120.0 or speech >= 0.12 * duration:
            break
    del model
    if chosen is None:
        raise SystemExit(f"VAD found no usable speech mask: {tried}")
    info = {
        "tried": tried,
        "chosen_threshold": chosen[0],
        "speech_seconds": round(chosen[2], 1),
        "regions": len(chosen[1]),
    }
    return chosen[1], info


def _rms_by_cluster(audio: np.ndarray, sample_rate: int, turns: list[dict]) -> dict[int, float]:
    buckets: dict[int, list[tuple[int, float]]] = {}
    for turn in turns:
        i0 = max(0, int(float(turn["start"]) * sample_rate))
        i1 = min(len(audio), int(float(turn["end"]) * sample_rate))
        piece = audio[i0:i1]
        if piece.size == 0:
            continue
        rms = float(np.sqrt(np.mean(np.square(piece))))
        buckets.setdefault(int(turn["cluster"]), []).append((piece.size, rms))
    levels = {}
    for cluster, parts in buckets.items():
        weight = sum(size for size, _rms in parts)
        levels[cluster] = sum(size * rms for size, rms in parts) / max(weight, 1)
    return levels


def _apply_gain(audio: np.ndarray, sample_rate: int, turns: list[dict], cluster: int, gain: float) -> np.ndarray:
    out = np.array(audio, dtype=np.float32, copy=True)
    for turn in turns:
        if int(turn["cluster"]) != cluster:
            continue
        i0 = max(0, int(float(turn["start"]) * sample_rate))
        i1 = min(len(out), int(float(turn["end"]) * sample_rate))
        out[i0:i1] *= gain
    peak = float(np.max(np.abs(out))) if out.size else 0.0
    if peak > 0.99:
        out *= 0.99 / peak
    return out


def _cyrillic_count(text: str) -> int:
    return sum(1 for ch in text if "\u0400" <= ch <= "\u04FF")


def _is_junk(text: str) -> bool:
    try:
        from artalisten.junk import is_junk as _lib_junk
        return _lib_junk(text)
    except Exception:
        folded = (text or "").casefold().strip()
        if not folded:
            return False
        if folded in {"говор", "на", "русском", "языке", "языке.", "дякую", "дякую."}:
            return True
        return any(re.search(pattern, folded) for pattern in JUNK)


def _mean_prob(utt: dict) -> float:
    probs = [
        float(word["probability"])
        for word in (utt.get("words_ru") or [])
        if word.get("probability") is not None
    ]
    if not probs:
        return 0.0
    return sum(probs) / len(probs)


def _better(new: dict, old: dict) -> bool:
    new_cyr = _cyrillic_count(new.get("text_ru") or "")
    old_cyr = _cyrillic_count(old.get("text_ru") or "")
    if new_cyr != old_cyr:
        return new_cyr > old_cyr
    return _mean_prob(new) > _mean_prob(old)


def _merge_utterances(main_utts: list[dict], quiet_utts: list[dict]) -> tuple[list[dict], list[dict]]:
    from artalisten.align import group_utterances

    def split(words: list[dict]) -> tuple[list[dict], list[dict]]:
        kept, rejected = [], []
        for utt in group_utterances(words):
            if _is_junk(utt.get("text_ru") or ""):
                rejected.append(utt)
            else:
                kept.append(utt)
        return kept, rejected

    main_kept, main_rejected = split(main_utts)
    quiet_kept, quiet_rejected = split(quiet_utts)
    ordered = sorted(main_kept + quiet_kept, key=lambda utt: (float(utt["start"]), float(utt["end"])))
    chosen: list[dict] = []
    for utt in ordered:
        if not chosen:
            chosen.append(utt)
            continue
        prev = chosen[-1]
        overlap = min(float(prev["end"]), float(utt["end"])) - max(float(prev["start"]), float(utt["start"]))
        shorter = min(float(prev["end"]) - float(prev["start"]), float(utt["end"]) - float(utt["start"]))
        if shorter > 0 and overlap > 0.5 * shorter:
            if _better(utt, prev):
                chosen[-1] = utt
            continue
        chosen.append(utt)
    return chosen, main_rejected + quiet_rejected


def _decode(model, audio: np.ndarray, sample_rate: int, turns: list[dict], task: str, vad: bool) -> list[dict]:
    from artalisten.align import drop_overlap_words, iter_chunks
    from artalisten.speakers import speaker_at

    words: list[dict] = []
    duration = float(len(audio) / sample_rate)
    spans = list(iter_chunks(duration, 300.0, 3.0))
    for index, (start, end) in enumerate(spans):
        piece = np.ascontiguousarray(audio[int(start * sample_rate) : int(end * sample_rate)], dtype=np.float32)
        kwargs = dict(
            language="ru",
            task=task,
            word_timestamps=True,
            beam_size=5,
            temperature=0.0,
            vad_filter=vad,
            condition_on_previous_text=False,
            hallucination_silence_threshold=2.0,
        )
        if vad:
            kwargs["vad_parameters"] = {
                "threshold": 0.1,
                "min_speech_duration_ms": 80,
                "min_silence_duration_ms": 200,
                "speech_pad_ms": 350,
            }
        try:
            segments, _info = model.transcribe(piece, **kwargs)
        except TypeError:
            kwargs.pop("hallucination_silence_threshold", None)
            segments, _info = model.transcribe(piece, **kwargs)
        chunk_words = []
        for segment in segments:
            no_speech = getattr(segment, "no_speech_prob", None)
            segment_words = list(segment.words or [])
            pieces = segment_words or []
            if pieces:
                for word in pieces:
                    if word.start is None or word.end is None:
                        continue
                    abs_start = float(word.start) + start
                    abs_end = float(word.end) + start
                    speaker, speaker_id = speaker_at(turns, 0.5 * (abs_start + abs_end))
                    chunk_words.append(
                        {
                            "start": abs_start,
                            "end": abs_end,
                            "word": word.word,
                            "probability": None if word.probability is None else float(word.probability),
                            "speaker": speaker,
                            "speaker_id": speaker_id,
                            "no_speech_prob": None if no_speech is None else float(no_speech),
                        }
                    )
            elif getattr(segment, "text", "").strip():
                abs_start = float(segment.start) + start
                abs_end = float(segment.end) + start
                speaker, speaker_id = speaker_at(turns, 0.5 * (abs_start + abs_end))
                chunk_words.append(
                    {
                        "start": abs_start,
                        "end": abs_end,
                        "word": segment.text,
                        "probability": None,
                        "speaker": speaker,
                        "speaker_id": speaker_id,
                        "no_speech_prob": None if no_speech is None else float(no_speech),
                    }
                )
        words.extend(drop_overlap_words(chunk_words, start, 3.0, is_first=index == 0))
        print(f"ASR {task} chunk {index + 1}/{len(spans)} words {len(chunk_words)} vad {vad}", flush=True)
    return words


def _clock(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    return f"{total // 3600:02d}:{(total % 3600) // 60:02d}:{total % 60:02d}"


def _write_transcript(path: Path, title: str, header: list[str], utterances: list[dict], rejected: list[dict]) -> None:
    lines = [f"# {title}", "", *header, "", "## Conversation", ""]
    if not utterances:
        lines.append("No Russian words were kept. Empty minutes are not filled in.")
        lines.append("")
    for utt in utterances:
        ru = (utt.get("text_ru") or "").strip() or "—"
        en = (utt.get("text_en") or "").strip() or "—"
        lines.append(
            f"### {_clock(float(utt['start']))} – {_clock(float(utt['end']))} · {utt.get('speaker')}"
        )
        lines.append("")
        lines.append(f"**Russian:** {ru}")
        lines.append("")
        lines.append(f"**English:** {en}")
        lines.append("")
    lines.append("## Rejected loops")
    lines.append("")
    lines.append("These model lines are not dialogue. They are the known Whisper loops and credits.")
    lines.append("")
    if not rejected:
        lines.append("None.")
    for utt in rejected:
        text = (utt.get("text_ru") or utt.get("text_en") or "").strip()
        lines.append(f"- {_clock(float(utt['start']))}–{_clock(float(utt['end']))}: {text}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _prepare(src: Path, out_dir: Path, enhancer_name: str, enhance_fn, file_mode: bool) -> np.ndarray:
    from artalisten.audioio import decode_wav, highpass, peak_normalize, read_mono, resample, write_wav
    from artalisten.device import release_memory
    from artalisten.separate import separate_speech

    out_dir.mkdir(parents=True, exist_ok=True)
    decoded = out_dir / "decoded.wav"
    vocals = out_dir / "vocals.wav"
    decode_wav(src, decoded, 44100, 2, None)
    separate_speech(decoded, vocals, out_dir / "envelopes.npz", device="cuda", shifts=0)
    if decoded.exists():
        decoded.unlink()
    audio, sample_rate = read_mono(vocals)
    audio = resample(audio, sample_rate, 16000)
    audio = highpass(audio, 16000, 70.0)
    audio = peak_normalize(audio)
    if vocals.exists():
        vocals.unlink()
    release_memory()
    if file_mode:
        rough = out_dir / "pre_enhance.wav"
        done = out_dir / "enhanced.wav"
        write_wav(rough, audio, 16000)
        enhance_fn(rough, done)
        audio, sample_rate = read_mono(done)
        if sample_rate != 16000:
            audio = resample(audio, sample_rate, 16000)
        rough.unlink(missing_ok=True)
    else:
        print(f"ENHANCE start {src.name} {len(audio) / 16000:.1f}s", flush=True)
        audio = _overlap_add(audio, 16000, enhance_fn)
        write_wav(out_dir / "enhanced.wav", audio, 16000)
    print(f"ENHANCE done {src.name} via {enhancer_name}", flush=True)
    return np.asarray(audio, dtype=np.float32)


def _speakers(audio: np.ndarray, fit: bool, profile: dict | None) -> tuple[list[dict], dict]:
    from artalisten.speakers import (
        _centroids,
        _embed,
        cluster_global,
        nearest_centroids,
        speech_windows,
        turns_from_windows,
    )

    timestamps, vad_info = _sensitive_vad(audio, 16000)
    windows = speech_windows(timestamps)
    duration = float(len(audio) / 16000)
    cache = Path("/kaggle/working/spk-cache")
    print(f"WINDOWS {len(windows)} fit {fit}", flush=True)
    if len(windows) < 2:
        raise SystemExit(f"not enough speech windows after sensitive VAD: {vad_info}")
    clips = []
    sample_rate = 16000
    for start, end in windows:
        i0 = max(0, int(start * sample_rate))
        i1 = min(len(audio), int(end * sample_rate))
        clips.append(np.ascontiguousarray(audio[i0:i1], dtype=np.float32))
    embeddings, embed_device = _embed(clips, "cuda", cache)
    if fit:
        labels = cluster_global(embeddings, 2)
        method = "global-ecapa-agglomerative"
        learning = "global"
    else:
        if profile is None:
            raise SystemExit("test inference requires a frozen profile")
        labels = nearest_centroids(embeddings, np.asarray(profile["centroids"], dtype=np.float64))
        method = "frozen-centroid-assignment"
        learning = "none"
    labeled = [(start, end, int(label)) for (start, end), label in zip(windows, labels)]
    turns = turns_from_windows(labeled, duration)
    info = {
        "vad": vad_info,
        "windows": len(windows),
        "method": method,
        "speaker_learning": learning,
        "embed_device": embed_device,
        "embeddings": embeddings,
        "labels": labels,
    }
    return turns, info


def _name_turns(turns: list[dict], quiet_cluster: int) -> None:
    for turn in turns:
        cluster = int(turn["cluster"])
        if cluster == quiet_cluster:
            turn["speaker"] = "Anna (quiet cluster)"
            turn["speaker_id"] = f"S{cluster}"
        else:
            turn["speaker"] = "brother (louder cluster)"
            turn["speaker_id"] = f"S{cluster}"


def _recover_file(
    src: Path,
    out_dir: Path,
    audio: np.ndarray,
    fit: bool,
    profile: dict | None,
    model,
) -> dict:
    from artalisten.align import fill_translations

    turns, info = _speakers(audio, fit=fit, profile=profile)
    if fit:
        levels = _rms_by_cluster(audio, 16000, turns)
        if len(levels) < 2:
            quiet = min(levels) if levels else 0
        else:
            quiet = min(levels, key=levels.get)
        loud = max(levels, key=levels.get) if levels else quiet
        quiet_rms = levels.get(quiet, 0.0)
        loud_rms = levels.get(loud, quiet_rms)
        gain = 1.5 if quiet_rms <= 1e-5 else float(np.clip(loud_rms / quiet_rms, 1.5, 6.0))
        centroids = _centroid_lists(info["embeddings"], info["labels"])
        profile = {
            "fit_on": src.name,
            "centroids": centroids,
            "quiet_cluster": int(quiet),
            "quiet_gain": gain,
            "quiet_rms": quiet_rms,
            "loud_rms": loud_rms,
            "speaker_learning": "global",
            "method": "global-ecapa-agglomerative",
            "fit_speakers": True,
        }
    else:
        quiet = int(profile["quiet_cluster"])
        gain = float(profile["quiet_gain"])
    _name_turns(turns, quiet)
    boosted = _apply_gain(audio, 16000, turns, quiet, gain)
    print(
        f"QUIET cluster {quiet} gain {gain:.2f} fit {fit} file {src.name}",
        flush=True,
    )
    ru_main = _decode(model, audio, 16000, turns, "transcribe", vad=True)
    ru_quiet = _decode(model, boosted, 16000, turns, "transcribe", vad=True)
    kept, rejected = _merge_utterances(ru_main, ru_quiet)
    real_words = sum(len((utt.get("text_ru") or "").split()) for utt in kept)
    if real_words < 40:
        print(f"RETRY quieter stem without whisper VAD, kept words {real_words}", flush=True)
        ru_open = _decode(model, boosted, 16000, turns, "transcribe", vad=False)
        kept, rejected = _merge_utterances(
            [word for utt in kept for word in utt.get("words_ru") or []],
            ru_open,
        )
        real_words = sum(len((utt.get("text_ru") or "").split()) for utt in kept)
    if real_words < 40:
        print(f"RETRY independent 10s windows on quiet stem, kept words {real_words}", flush=True)
        from artalisten.asr import transcribe_independent_windows
        from artalisten.align import group_utterances

        win_words = transcribe_independent_windows(
            model,
            boosted,
            16000,
            turns,
            task="transcribe",
            language="ru",
            window_seconds=10.0,
            hop_seconds=8.0,
            vad_filter=False,
        )
        kept, rejected = _merge_utterances(
            [word for utt in kept for word in utt.get("words_ru") or []],
            win_words,
        )
    en_words = _decode(model, boosted, 16000, turns, "translate", vad=True)
    en_words = [word for word in en_words if not _is_junk(str(word.get("word") or ""))]
    kept = fill_translations(kept, en_words)
    kept = [utt for utt in kept if (utt.get("text_ru") or utt.get("text_en"))]
    rejected_en = [utt for utt in kept if _is_junk(utt.get("text_en") or "") and not (utt.get("text_ru") or "").strip()]
    kept = [utt for utt in kept if utt not in rejected_en]
    for utt in kept:
        if _is_junk(utt.get("text_en") or ""):
            utt["text_en"] = ""
            utt["words_en"] = []
    header = [
        f"- File: {src.name}",
        f"- Processed: {_clock(len(audio) / 16000)}",
        f"- Speaker fit: {'yes, this file only' if fit else 'no, frozen centroids from ' + str(profile.get('fit_on'))}",
        f"- Quiet cluster: S{quiet} Anna, gain {gain:.2f}",
        f"- VAD chosen threshold: {info['vad']['chosen_threshold']}",
        f"- VAD speech seconds: {info['vad']['speech_seconds']}",
        f"- Kept utterances: {len(kept)}",
        f"- Rejected loop lines: {len(rejected) + len(rejected_en)}",
        "- Empty stretches are omitted. They were not filled.",
    ]
    _write_transcript(out_dir / "transcript.md", src.name, header, kept, rejected + rejected_en)
    manifest = {
        "source_name": src.name,
        "source_seconds": round(len(audio) / 16000, 3),
        "fit_speakers": bool(fit),
        "fit_on": profile.get("fit_on"),
        "speaker_learning": "global" if fit else "none",
        "speaker_method": info["method"],
        "quiet_cluster": quiet,
        "quiet_gain": gain,
        "vad": info["vad"],
        "kept_utterances": len(kept),
        "rejected_loops": len(rejected) + len(rejected_en),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (out_dir / "utterances.json").write_text(
        json.dumps(kept, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return profile


def _centroid_lists(embeddings: np.ndarray, labels: np.ndarray) -> list[list[float]]:
    from artalisten.speakers import _centroids

    return [[float(value) for value in center] for center in _centroids(embeddings, labels, 2)]


def main() -> None:
    from artalisten.asr import load_model
    from artalisten.audioio import probe_duration
    from artalisten.device import release_memory

    train = _one(TRAIN_NAMES, "anna-brother-train")
    test = _one(TEST_NAMES, "anna-brother-test")
    if train.resolve() == test.resolve():
        raise SystemExit("train and test resolved to the same file")
    print(
        f"FILES train {train.name} {probe_duration(train):.1f}s test {test.name} {probe_duration(test):.1f}s",
        flush=True,
    )
    _install()
    enhancer_name, enhance_fn, enhancer_model = _load_enhancer("cuda:0")
    file_mode = enhancer_model is None
    train_audio = _prepare(train, Path("/kaggle/working/train"), enhancer_name, enhance_fn, file_mode)
    test_audio = _prepare(test, Path("/kaggle/working/test"), enhancer_name, enhance_fn, file_mode)
    del enhancer_model
    release_memory()
    model = load_model("large-v3", "float16", device="cuda")
    try:
        profile = _recover_file(train, Path("/kaggle/working/train"), train_audio, True, None, model)
        Path("/kaggle/working/speaker_profile.json").write_text(
            json.dumps(profile, indent=2), encoding="utf-8"
        )
        if profile.get("fit_on") != train.name or not profile.get("centroids"):
            raise SystemExit("train profile was not fit on the train file")
        _recover_file(test, Path("/kaggle/working/test"), test_audio, False, profile, model)
    finally:
        del model
        release_memory()
    test_manifest = json.loads(Path("/kaggle/working/test/manifest.json").read_text(encoding="utf-8"))
    if test_manifest.get("fit_speakers") is not False:
        raise SystemExit("test manifest says speakers were fit")
    if test_manifest.get("fit_on") != train.name:
        raise SystemExit(f"test fit_on is {test_manifest.get('fit_on')!r}")
    proof = {
        "train_file": train.name,
        "test_file": test.name,
        "train_path": str(train),
        "test_path": str(test),
        "train_fit_speakers": True,
        "test_fit_speakers": False,
        "profile_fit_on": profile.get("fit_on"),
        "quiet_cluster": profile.get("quiet_cluster"),
        "quiet_gain": profile.get("quiet_gain"),
        "enhancer": enhancer_name,
        "test_speaker_learning": test_manifest.get("speaker_learning"),
        "test_speaker_method": test_manifest.get("speaker_method"),
        "train_vad": json.loads(Path("/kaggle/working/train/manifest.json").read_text(encoding="utf-8")).get("vad"),
        "test_vad": test_manifest.get("vad"),
    }
    Path("/kaggle/working/split_proof.json").write_text(json.dumps(proof, indent=2), encoding="utf-8")
    for wav in Path("/kaggle/working").rglob("*.wav"):
        wav.unlink()
    print("DONE", json.dumps({k: proof[k] for k in ("enhancer", "train_fit_speakers", "test_fit_speakers", "quiet_cluster", "quiet_gain")}), flush=True)


if __name__ == "__main__":
    main()
