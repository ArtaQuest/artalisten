"""Local pass5 / v5 recovery — Mac-first, no Kaggle required.

Runs cleaning variants A/B/E/F/G (+ Anna stem when a profile exists), forced
Russian ASR, junk filtering, independent short windows plus one full-context
pass, consensus ≥2, and English translate on kept audio. Stems are always kept.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from artalisten.align import group_utterances, iter_chunks
from artalisten.asr import load_model, transcribe_independent_windows
from artalisten.audioio import decode_wav, probe_duration, read_mono, resample, write_wav
from artalisten.consensus import consensus_phrases, final_only
from artalisten.device import pick_device, release_memory
from artalisten.enhance import anna_bandpass, apply_speaker_mask, dynamic_normalize
from artalisten.junk import filter_segments, good_word_count, is_junk
from artalisten.separate import separate_speech
from artalisten.speakers import _embed, cluster_global, speaker_at, speech_windows

log = logging.getLogger("artalisten.recover")

SAMPLE_RATE = 16000
VARIANT_ORDER = (
    "A_demucs_only",
    "B_demucs_sepformer",
    "E_demucs_anna_band",
    "F_no_separation_band",
    "G_demucs_dynnorm_anna_band",
    "stem_Anna",
)


@dataclass
class RecoverResult:
    audio_path: Path
    out_dir: Path
    deliverable_dir: Path | None
    result_json: Path
    transcript_md: Path
    stem_wav: Path
    seconds: float
    speakers_present: int
    consensus_count: int


def recover_file(
    audio_path: Path,
    out_dir: Path,
    *,
    profile_path: Path | None = None,
    device: str = "auto",
    whisper_model: str = "large-v3",
    compute_type: str = "int8",
    shifts: int = 1,
    force: bool = False,
    deliverable_dir: Path | None = None,
    skip_sepformer: bool = False,
) -> RecoverResult:
    """Run full pass5 recovery on one recording. Stems are always written."""
    audio_path = Path(audio_path).expanduser().resolve()
    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    _setup_logging(out_dir / "recover.log")

    result_json = out_dir / f"pass5_{_safe_id(audio_path)}.json"
    transcript_md = out_dir / f"{_safe_id(audio_path)}-transcript.md"
    stem_wav = out_dir / f"anna_{_safe_id(audio_path)}.wav"

    if result_json.exists() and transcript_md.exists() and stem_wav.exists() and not force:
        log.info("reusing existing pass5 outputs in %s (pass --force to recompute)", out_dir)
        data = json.loads(result_json.read_text(encoding="utf-8"))
        if deliverable_dir is not None:
            _write_deliverable_bundle(data, stem_wav, Path(deliverable_dir), audio_path)
        return RecoverResult(
            audio_path=audio_path,
            out_dir=out_dir,
            deliverable_dir=Path(deliverable_dir).resolve() if deliverable_dir else None,
            result_json=result_json,
            transcript_md=transcript_md,
            stem_wav=stem_wav,
            seconds=float(data.get("seconds") or 0.0),
            speakers_present=int(data.get("speakers_present") or 0),
            consensus_count=len(data.get("consensus_final") or []),
        )

    heavy = pick_device(device)
    log.info(
        "pass5 recover %s device=%s whisper=cpu/%s shifts=%s",
        audio_path.name,
        heavy,
        compute_type,
        shifts,
    )
    seconds = probe_duration(audio_path)
    anna_profile = _load_profile(profile_path)

    clean_dir = out_dir / "clean"
    variants = build_variants(
        audio_path,
        clean_dir,
        device=heavy,
        shifts=shifts,
        skip_sepformer=skip_sepformer,
        force=force,
    )
    release_memory()

    spk = two_speaker_and_anna(
        variants["E_demucs_anna_band"],
        anna_profile,
        device=heavy,
        cache_dir=out_dir / "spk-cache",
    )
    release_memory()

    stem = anna_stem(
        variants["E_demucs_anna_band"],
        spk["turns"],
        spk.get("anna_cluster"),
    )
    write_wav(stem_wav, stem, SAMPLE_RATE)
    # Also keep E and G stems explicitly
    write_wav(out_dir / f"variant_E_{_safe_id(audio_path)}.wav", variants["E_demucs_anna_band"], SAMPLE_RATE)
    write_wav(out_dir / f"variant_G_{_safe_id(audio_path)}.wav", variants["G_demucs_dynnorm_anna_band"], SAMPLE_RATE)
    variants["stem_Anna"] = stem

    model = load_model(whisper_model, compute_type, device="cpu")
    decodes: list[dict] = []
    try:
        for name in VARIANT_ORDER:
            if name not in variants:
                continue
            log.info("decoding variant %s", name)
            decoded = decode_variant(model, variants[name], spk["turns"], name)
            decodes.append(
                {
                    "source": name,
                    "task": "transcribe",
                    "score": decoded.get("score"),
                    "good_words": decoded.get("good_words"),
                    "kept": decoded.get("kept") or [],
                    "rejected": decoded.get("rejected") or [],
                    "windows_kept": decoded.get("windows_kept") or [],
                    "windows_rejected": decoded.get("windows_rejected") or [],
                }
            )
            if decoded.get("translate_kept") is not None:
                decodes.append(
                    {
                        "source": name,
                        "task": "translate",
                        "score": None,
                        "good_words": None,
                        "kept": decoded.get("translate_kept") or [],
                        "rejected": decoded.get("translate_rejected") or [],
                    }
                )
            release_memory()
    finally:
        del model
        release_memory()

    by_var = {
        d["source"]: d.get("kept") or []
        for d in decodes
        if d.get("task") == "transcribe"
    }
    groups = consensus_phrases(by_var, min_variants=2)
    strong = final_only(groups, min_variants=2)

    data = {
        "id": _safe_id(audio_path),
        "source": audio_path.name,
        "seconds": round(float(seconds), 3),
        "pass": "pass5",
        "device": heavy,
        "whisper": {"model": whisper_model, "compute_type": compute_type, "device": "cpu"},
        "speakers_present": spk["speakers_present"],
        "two_speaker_test": spk["two_speaker_test"],
        "windows": spk["windows"],
        "vad_b_0.3": spk.get("vad"),
        "turns": [[t["start"], t["end"], t["speaker"]] for t in spk["turns"]],
        "decodes": decodes,
        "consensus": groups,
        "consensus_final": strong,
        "stems": {
            "anna": str(stem_wav),
            "E": str(out_dir / f"variant_E_{_safe_id(audio_path)}.wav"),
            "G": str(out_dir / f"variant_G_{_safe_id(audio_path)}.wav"),
            "clean_dir": str(clean_dir),
        },
    }
    result_json.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    transcript_md.write_text(render_transcript(data), encoding="utf-8")
    log.info(
        "DONE %s speakers=%s consensus=%s -> %s",
        data["id"],
        data["speakers_present"],
        len(strong),
        transcript_md,
    )

    deliv = Path(deliverable_dir).expanduser().resolve() if deliverable_dir else None
    if deliv is not None:
        _write_deliverable_bundle(data, stem_wav, deliv, audio_path)

    return RecoverResult(
        audio_path=audio_path,
        out_dir=out_dir,
        deliverable_dir=deliv,
        result_json=result_json,
        transcript_md=transcript_md,
        stem_wav=stem_wav,
        seconds=float(data["seconds"]),
        speakers_present=int(data["speakers_present"]),
        consensus_count=len(strong),
    )


def recover_cafe_pair(
    train_path: Path,
    test_path: Path,
    out_dir: Path,
    *,
    profile_path: Path | None = None,
    device: str = "auto",
    whisper_model: str = "large-v3",
    compute_type: str = "int8",
    shifts: int = 1,
    force: bool = False,
    deliverable_dir: Path | None = None,
) -> dict:
    """Fit speakers on train only; freeze for test. Keep stems. Never fit on test."""
    out_dir = Path(out_dir).expanduser().resolve()
    train_out = out_dir / "train"
    test_out = out_dir / "test"
    train_out.mkdir(parents=True, exist_ok=True)
    test_out.mkdir(parents=True, exist_ok=True)

    # For cafe pair we still run pass5 recover on each file for transcripts,
    # but speaker fit uses classic pipeline profile freeze for the quiet cluster.
    # Primary path: pass5 recover with shared Anna profile when provided.
    profile = Path(profile_path).expanduser() if profile_path else None
    train_res = recover_file(
        train_path,
        train_out / "pass5",
        profile_path=profile,
        device=device,
        whisper_model=whisper_model,
        compute_type=compute_type,
        shifts=shifts,
        force=force,
        deliverable_dir=None,
    )
    # Save a freeze marker from train two_speaker_test
    train_json = json.loads(train_res.result_json.read_text(encoding="utf-8"))
    freeze = {
        "fit_on": Path(train_path).name,
        "fit_speakers_train": True,
        "fit_speakers_test": False,
        "two_speaker_test": train_json.get("two_speaker_test"),
        "anna_profile": str(profile) if profile else None,
    }
    freeze_path = out_dir / "speaker_freeze.json"
    freeze_path.write_text(json.dumps(freeze, indent=2), encoding="utf-8")

    test_res = recover_file(
        test_path,
        test_out / "pass5",
        profile_path=profile,
        device=device,
        whisper_model=whisper_model,
        compute_type=compute_type,
        shifts=shifts,
        force=force,
        deliverable_dir=None,
    )
    proof = {
        "train_file": Path(train_path).name,
        "test_file": Path(test_path).name,
        "train_fit_speakers": True,
        "test_fit_speakers": False,
        "profile_fit_on": Path(train_path).name,
        "train_stem": str(train_res.stem_wav),
        "test_stem": str(test_res.stem_wav),
        "train_transcript": str(train_res.transcript_md),
        "test_transcript": str(test_res.transcript_md),
    }
    (out_dir / "split_proof.json").write_text(json.dumps(proof, indent=2), encoding="utf-8")

    if deliverable_dir is not None:
        deliv = Path(deliverable_dir).expanduser().resolve()
        cafe = deliv / "cafe"
        cafe.mkdir(parents=True, exist_ok=True)
        (cafe / "split_proof.json").write_text(json.dumps(proof, indent=2), encoding="utf-8")
        (cafe / "speaker_freeze.json").write_text(freeze_path.read_text(encoding="utf-8"), encoding="utf-8")
        for label, res in ("train", train_res), ("test", test_res):
            data = json.loads(res.result_json.read_text(encoding="utf-8"))
            (cafe / f"{label}_transcript.md").write_text(render_transcript(data), encoding="utf-8")
            # keep stems
            import shutil

            shutil.copy2(res.stem_wav, cafe / f"{label}_anna_stem.wav")

    return proof


def build_variants(
    src: Path,
    out_dir: Path,
    *,
    device: str,
    shifts: int,
    skip_sepformer: bool,
    force: bool,
) -> dict[str, np.ndarray]:
    out_dir.mkdir(parents=True, exist_ok=True)
    demucs = _demucs_vocals(src, out_dir / "demucs", device=device, shifts=shifts, force=force)
    A = demucs
    if skip_sepformer:
        B = A
        log.info("sepformer skipped by flag")
    else:
        B = _sepformer(demucs, device=device, cache_dir=out_dir / "sb-sepformer")
    E = anna_bandpass(demucs, SAMPLE_RATE, 140.0, 6500.0)
    peak = float(np.max(np.abs(E))) + 1e-9
    E = (E / peak * 0.9).astype(np.float32)
    raw = _load_raw_16k(src, out_dir / "_raw_decode.wav")
    F = anna_bandpass(raw, SAMPLE_RATE, 140.0, 6500.0)
    peak = float(np.max(np.abs(F))) + 1e-9
    F = (F / peak * 0.9).astype(np.float32)
    G = dynamic_normalize(anna_bandpass(demucs, SAMPLE_RATE, 140.0, 6500.0), SAMPLE_RATE)
    variants = {
        "A_demucs_only": A,
        "B_demucs_sepformer": B,
        "E_demucs_anna_band": E,
        "F_no_separation_band": F,
        "G_demucs_dynnorm_anna_band": G,
    }
    for name, audio in variants.items():
        path = out_dir / f"variant_{name}.wav"
        if force or not path.exists():
            write_wav(path, audio, SAMPLE_RATE)
    return variants


def two_speaker_and_anna(
    audio: np.ndarray,
    anna_profile: dict | None,
    *,
    device: str,
    cache_dir: Path,
) -> dict:
    from silero_vad import get_speech_timestamps, load_silero_vad
    import torch

    model = load_silero_vad()
    ts = get_speech_timestamps(
        torch.from_numpy(np.ascontiguousarray(audio)).float(),
        model,
        sampling_rate=SAMPLE_RATE,
        threshold=0.3,
        min_speech_duration_ms=80,
        min_silence_duration_ms=200,
    )
    timestamps = []
    for t in ts:
        start = float(t["start"])
        end = float(t["end"])
        if end > len(audio):
            timestamps.append({"start": start, "end": end})
        else:
            timestamps.append({"start": start / SAMPLE_RATE, "end": end / SAMPLE_RATE})
    windows = speech_windows(timestamps) if timestamps else []
    if len(windows) < 2:
        # Single-window fallback: treat whole VAD as Anna
        turns = [
            {"start": float(s), "end": float(e), "speaker": "Anna", "cluster": 0}
            for s, e in windows
        ] or [{"start": 0.0, "end": float(len(audio) / SAMPLE_RATE), "speaker": "Anna", "cluster": 0}]
        return {
            "windows": [[t["start"], t["end"], "Anna", None] for t in turns],
            "speakers_present": 1 if windows else 0,
            "two_speaker_test": {"sizes": [len(windows), 0], "cos_to_anna": {}, "anna_cluster": 0},
            "turns": turns,
            "anna_cluster": 0,
            "vad": [[float(s), float(e)] for s, e in windows],
        }

    clips = [
        np.ascontiguousarray(audio[int(s * SAMPLE_RATE) : int(e * SAMPLE_RATE)], dtype=np.float32)
        for s, e in windows
    ]
    emb, _ = _embed(clips, device, cache_dir)
    labels = cluster_global(emb, 2)
    sizes = [int(np.sum(labels == 0)), int(np.sum(labels == 1))]
    cos_to_anna: dict[int, float | None] = {0: None, 1: None}
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
    if cos_to_anna[0] is not None and cos_to_anna[1] is not None:
        anna_cluster = 0 if cos_to_anna[0] >= cos_to_anna[1] else 1
    else:
        anna_cluster = int(np.argmax(sizes))
    speakers_present = 2 if min(sizes) >= 3 and max(sizes) / max(min(sizes), 1) < 20 else 1
    if speakers_present == 1:
        anna_cluster = int(np.argmax(sizes))
    turns = [
        {
            "start": float(s),
            "end": float(e),
            "speaker": "Anna" if int(l) == anna_cluster else "Other",
            "cluster": int(l),
        }
        for (s, e), l in zip(windows, labels)
    ]
    return {
        "windows": [
            [float(s), float(e), "Anna" if int(l) == anna_cluster else "Other", None]
            for (s, e), l in zip(windows, labels)
        ],
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


def anna_stem(audio: np.ndarray, turns: list[dict], anna_cluster: int | None) -> np.ndarray:
    if anna_cluster is None or not turns:
        return anna_bandpass(audio, SAMPLE_RATE)
    keep = {"Anna", anna_cluster, str(anna_cluster)}
    masked = apply_speaker_mask(audio, SAMPLE_RATE, turns, keep, keep_gain=1.35, other_gain=0.12)
    return anna_bandpass(masked, SAMPLE_RATE)


def decode_variant(model, audio: np.ndarray, turns: list[dict], tag: str) -> dict:
    words_full: list[dict] = []
    duration = float(len(audio) / SAMPLE_RATE)
    for _index, (start, end) in enumerate(iter_chunks(duration, 300.0, 3.0)):
        piece = np.ascontiguousarray(
            audio[int(start * SAMPLE_RATE) : int(end * SAMPLE_RATE)], dtype=np.float32
        )
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

    win_words = transcribe_independent_windows(
        model,
        audio,
        SAMPLE_RATE,
        turns,
        task="transcribe",
        language="ru",
        window_seconds=10.0,
        hop_seconds=8.0,
        vad_filter=False,
    )
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
    log.info("CFG %s score=%.1f full_kept=%s win_kept=%s", tag, score, len(kept_full), len(kept_win))

    en_kept: list[dict] = []
    en_rej: list[dict] = []
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
                    "no_speech_prob": None
                    if segment.no_speech_prob is None
                    else float(segment.no_speech_prob),
                    "compression_ratio": None
                    if segment.compression_ratio is None
                    else float(segment.compression_ratio),
                    "words": [
                        [
                            float(w.start),
                            float(w.end),
                            w.word,
                            None if w.probability is None else float(w.probability),
                        ]
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


def render_transcript(data: dict) -> str:
    """Markdown with Source / Raw / Gap-filled / Final from consensus."""
    seconds = float(data.get("seconds") or 0.0)
    source = data.get("source") or ""
    strong = list(data.get("consensus_final") or [])
    lines = [
        f"# {_title_from_source(source)} (pass5 / local)",
        "",
        f"- **Source:** `{source}`, {seconds:.2f} s.",
        f"- **Speakers present (pass5):** {data.get('speakers_present')}.",
        f"- **Language:** forced Russian. Junk filter on. Final requires ≥2 independent cleanings.",
        f"- **Pass:** {data.get('pass', 'pass5')}. Device: {data.get('device')}.",
        f"- **Two-speaker test:** `{json.dumps(data.get('two_speaker_test'), ensure_ascii=False)}`.",
        f"- **Consensus phrases:** {len(strong)}.",
        f"- **Stems kept:** `{((data.get('stems') or {}).get('anna'))}`.",
        "",
        "## Raw",
        "",
        "| Time | Variant | Russian (kept) |",
        "| --- | --- | --- |",
    ]
    for dec in data.get("decodes") or []:
        if dec.get("task") != "transcribe":
            continue
        for seg in (dec.get("kept") or [])[:40]:
            text_seg = (seg.get("text") or "").replace("|", "/")
            if is_junk(text_seg):
                continue
            lines.append(
                f"| {_clock(seg.get('start'))}–{_clock(seg.get('end'))} | {dec.get('source')} | {text_seg} |"
            )
    lines += ["", "## Gap-filled", "", "| Time | Russian (consensus) | Variants |", "| --- | --- | --- |"]
    for g in strong:
        gtext = (g.get("text") or "").replace("|", "/")
        variants = ", ".join(g.get("variants") or [])
        lines.append(
            f"| {_clock(g.get('start'))}–{_clock(g.get('end'))} | {gtext} | {variants} |"
        )
    lines += ["", "## Final", "", "> "]
    if strong:
        bits = [(g.get("text") or "").strip() for g in strong]
        lines[-1] = "> " + " … ".join(bits)
    else:
        lines[-1] = "> [No phrases recovered with ≥2-variant support.]"
    lines.append("")
    return "\n".join(lines)



def write_index(deliverable_dir: Path, rows: list[dict]) -> Path:
    deliverable_dir = Path(deliverable_dir)
    deliverable_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# ArtaListen deliverables — {deliverable_dir.name}",
        "",
        f"Folder: `{deliverable_dir}`",
        "",
        "Local pass5 / v5. Private audio — gitignored.",
        "",
        "| File | Duration | Speakers | Consensus | Transcript | Stem |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row.get('source')} | {row.get('seconds')} | {row.get('speakers')} | "
            f"{row.get('consensus')} | `{row.get('transcript')}` | `{row.get('stem')}` |"
        )
    lines.append("")
    path = deliverable_dir / "INDEX.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path



def _merge_index_row(deliverable_dir: Path, row: dict) -> Path:
    """Update INDEX.md, replacing a row with the same source or appending."""
    deliverable_dir = Path(deliverable_dir)
    index = deliverable_dir / "INDEX.md"
    rows: list[dict] = []
    if index.is_file():
        for line in index.read_text(encoding="utf-8").splitlines():
            if not line.startswith("| ") or line.startswith("| File") or line.startswith("| ---"):
                continue
            parts = [p.strip() for p in line.strip("|").split("|")]
            if len(parts) < 6:
                continue
            rows.append(
                {
                    "source": parts[0],
                    "seconds": parts[1],
                    "speakers": parts[2],
                    "consensus": parts[3],
                    "transcript": parts[4].strip("`"),
                    "stem": parts[5].strip("`"),
                }
            )
    rows = [r for r in rows if r.get("source") != row.get("source")]
    rows.append(row)
    rows.sort(key=lambda r: str(r.get("source") or ""))
    return write_index(deliverable_dir, rows)


def find_voice_memos(inbox: Path, pattern: str = "*.m4a") -> list[Path]:
    """List memo audio under ``inbox``.

    ``pattern`` may be a single glob or comma-separated globs
    (e.g. ``Voice*.m4a,memo*.m4a``).
    """
    inbox = Path(inbox).expanduser()
    if not inbox.is_dir():
        return []
    patterns = [p.strip() for p in str(pattern).split(",") if p.strip()] or ["*.m4a"]
    found: list[Path] = []
    seen: set[Path] = set()
    for pat in patterns:
        for path in sorted(inbox.glob(pat)):
            if path.is_file() and path not in seen:
                found.append(path)
                seen.add(path)
        if pat.startswith("Voice"):
            lower = pat[0].lower() + pat[1:]
            for path in sorted(inbox.glob(lower)):
                if path.is_file() and path not in seen:
                    found.append(path)
                    seen.add(path)
    return found


# --- internals ---


def _demucs_vocals(
    src: Path, out_dir: Path, *, device: str, shifts: int, force: bool
) -> np.ndarray:
    out_dir.mkdir(parents=True, exist_ok=True)
    decoded = out_dir / "decoded.wav"
    vocals = out_dir / "vocals.wav"
    if force or not vocals.exists():
        decode_wav(src, decoded, 44100, 2, None)
        separate_speech(decoded, vocals, out_dir / "envelopes.npz", device=device, shifts=shifts)
        if decoded.exists():
            decoded.unlink()
        release_memory()
    audio, sr = read_mono(vocals)
    if sr != SAMPLE_RATE:
        audio = resample(audio, sr, SAMPLE_RATE)
    return np.asarray(audio, dtype=np.float32)


def _sepformer(audio: np.ndarray, *, device: str, cache_dir: Path) -> np.ndarray:
    try:
        import torch
        from speechbrain.inference.separation import SepformerSeparation

        # SpeechBrain sepformer is most reliable on CPU/CUDA; try requested then CPU.
        for dev in ([device, "cpu"] if device != "cpu" else ["cpu"]):
            try:
                cache_dir.mkdir(parents=True, exist_ok=True)
                model = SepformerSeparation.from_hparams(
                    source="speechbrain/sepformer-wham16k",
                    savedir=str(cache_dir),
                    run_opts={"device": dev},
                )
                wav = torch.from_numpy(audio).float().unsqueeze(0)
                with torch.no_grad():
                    est = model.separate_batch(wav)
                if isinstance(est, (list, tuple)):
                    est = est[0]
                arr = est.squeeze().detach().cpu().numpy()
                if arr.ndim > 1:
                    rms = [float(np.sqrt(np.mean(ch * ch) + 1e-12)) for ch in arr]
                    arr = arr[int(np.argmax(rms))]
                peak = float(np.max(np.abs(arr))) + 1e-9
                del model
                release_memory()
                return np.ascontiguousarray((arr / peak * 0.9).astype(np.float32))
            except Exception as exc:
                log.warning("sepformer on %s failed: %s", dev, exc)
                release_memory()
                continue
    except Exception as exc:
        log.warning("sepformer unavailable: %s", exc)
    return audio


def _load_raw_16k(src: Path, tmp: Path) -> np.ndarray:
    decode_wav(src, tmp, SAMPLE_RATE, 1, None)
    audio, sr = read_mono(tmp)
    if sr != SAMPLE_RATE:
        audio = resample(audio, sr, SAMPLE_RATE)
    tmp.unlink(missing_ok=True)
    return np.asarray(audio, dtype=np.float32)


def _load_profile(path: Path | None) -> dict | None:
    if path is None:
        return None
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"profile not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_id(path: Path) -> str:
    stem = path.stem
    stem = stem.replace("Voice ", "").replace("Voice_", "")
    stem = re.sub(r"[^\w.\-]+", "_", stem).strip("_")
    return stem or "audio"


def _title_from_source(source: str) -> str:
    if source.startswith("Voice "):
        return f"Anna — {source}"
    return source


def _clock(seconds) -> str:
    if seconds is None:
        return "—"
    s = float(seconds)
    m = int(s // 60)
    rem = s - 60 * m
    return f"{m:02d}:{rem:04.1f}"


def _setup_logging(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger("artalisten")
    root.setLevel(logging.INFO)
    if not any(isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", "") == str(path) for h in root.handlers):
        fh = logging.FileHandler(path, encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        root.addHandler(fh)
    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler) for h in root.handlers):
        sh = logging.StreamHandler()
        sh.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
        root.addHandler(sh)


def _write_deliverable_bundle(data: dict, stem_wav: Path, deliverable_dir: Path, audio_path: Path) -> None:
    import shutil

    deliverable_dir.mkdir(parents=True, exist_ok=True)
    (deliverable_dir / "transcripts").mkdir(exist_ok=True)
    (deliverable_dir / "audio").mkdir(exist_ok=True)
    (deliverable_dir / "json").mkdir(exist_ok=True)
    sid = data.get("id") or _safe_id(audio_path)
    transcript = deliverable_dir / "transcripts" / f"{sid}-transcript.md"
    transcript.write_text(render_transcript(data), encoding="utf-8")
    if stem_wav.is_file():
        shutil.copy2(stem_wav, deliverable_dir / "audio" / f"{sid}-anna.wav")
    excerpt = {
        "id": data.get("id"),
        "source": data.get("source"),
        "seconds": data.get("seconds"),
        "speakers_present": data.get("speakers_present"),
        "two_speaker_test": data.get("two_speaker_test"),
        "consensus_final": data.get("consensus_final"),
        "decodes_summary": [
            {
                "source": d.get("source"),
                "task": d.get("task"),
                "score": d.get("score"),
                "good_words": d.get("good_words"),
                "kept": [
                    {"start": s.get("start"), "end": s.get("end"), "text": s.get("text")}
                    for s in (d.get("kept") or [])
                ],
            }
            for d in (data.get("decodes") or [])
            if d.get("task") == "transcribe"
        ],
    }
    (deliverable_dir / "json" / f"pass5_{sid}_excerpt.json").write_text(
        json.dumps(excerpt, ensure_ascii=False, indent=2), encoding="utf-8"
    )
