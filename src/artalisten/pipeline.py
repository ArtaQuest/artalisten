"""Run separation, enhancement, full-file speaker learning, then transcription."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from artalisten.align import fill_translations, group_utterances
from artalisten.asr import load_model, transcribe_chunks
from artalisten.audioio import (
    decode_wav,
    highpass,
    peak_normalize,
    probe_duration,
    read_mono,
    resample,
    write_wav,
)
from artalisten.device import pick_device, release_memory, unified_memory_gb
from artalisten.enhance import MODEL_NAME as ENHANCER_NAME
from artalisten.enhance import enhance_speech
from artalisten.report import render_markdown
from artalisten.separate import MODEL_NAME as SEPARATOR_NAME
from artalisten.separate import separate_speech
from artalisten.speakers import assign_from_profile, hf_token, learn_speakers

log = logging.getLogger("artalisten.pipeline")

DECODE_RATE = 44100


def process(args) -> int:
    audio_path = Path(args.audio).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve() if args.out else _default_out(audio_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    _setup_logging(out_dir / "artalisten.log")
    _check_stamp(out_dir, audio_path, args.max_seconds, args.force)

    source_seconds = probe_duration(audio_path)
    device = pick_device(args.device)
    log.info("source %s duration %.1fs device preference %s", audio_path.name, source_seconds, device)
    if args.max_seconds is not None:
        log.info(
            "processing a %.1fs prefix. Speaker profiles will be learned on that prefix only.",
            args.max_seconds,
        )

    decoded = out_dir / "decoded.wav"
    vocals = out_dir / "vocals.wav"
    envelopes = out_dir / "envelopes.npz"
    enhanced = out_dir / "enhanced.wav"
    speech = out_dir / "speech_16k.wav"

    if args.force or not vocals.exists():
        if args.force or not decoded.exists():
            decode_wav(audio_path, decoded, DECODE_RATE, 2, args.max_seconds)
        separation = separate_speech(
            decoded, vocals, envelopes, device=device, shifts=args.shifts
        )
        _write_json(out_dir / "separation.json", separation)
        if decoded.exists():
            decoded.unlink()
    else:
        log.info("reusing %s", vocals.name)
        separation = _read_json(out_dir / "separation.json")

    if args.force or not enhanced.exists():
        enhancement = enhance_speech(vocals, enhanced, device=device)
        _write_json(out_dir / "enhancement.json", enhancement)
    else:
        log.info("reusing %s", enhanced.name)
        enhancement = _read_json(out_dir / "enhancement.json")

    if args.force or not speech.exists():
        _prepare_speech(enhanced, speech)
    else:
        log.info("reusing %s", speech.name)

    samples, sample_rate = read_mono(speech)
    processed_seconds = float(len(samples) / sample_rate) if sample_rate else 0.0
    diar_path = out_dir / "diarization.json"
    profile_path = getattr(args, "profile", None)
    cache_dir = Path.home() / ".cache" / "artalisten"
    if profile_path:
        if args.force or not diar_path.exists():
            log.info(
                "frozen speakers from %s — not fitting on %s (%.1fs)",
                profile_path,
                audio_path.name,
                processed_seconds,
            )
            profile = json.loads(Path(profile_path).read_text(encoding="utf-8"))
            diarization = assign_from_profile(
                samples, sample_rate, profile, device=device, cache_dir=cache_dir
            )
            _write_json(diar_path, diarization)
            if args.force:
                _clear_asr(out_dir / "asr")
        else:
            log.info("reusing frozen assignment %s", diar_path.name)
            diarization = _read_json(diar_path)
    elif args.force or not diar_path.exists():
        log.info("learning speakers on the full processed audio (%.1fs) before transcription", processed_seconds)
        diarization = learn_speakers(
            samples,
            sample_rate,
            num_speakers=args.speakers,
            device=device,
            cache_dir=cache_dir,
            hf_token=hf_token(),
        )
        diarization["fit_on"] = audio_path.name
        _write_json(diar_path, diarization)
        if getattr(args, "save_profile", None):
            _write_json(Path(args.save_profile), diarization)
            log.info("saved speaker profile %s (fit on %s only)", args.save_profile, audio_path.name)
        if args.force:
            _clear_asr(out_dir / "asr")
    else:
        log.info("reusing speaker profiles %s", diar_path.name)
        diarization = _read_json(diar_path)
    release_memory()

    turns = diarization.get("turns") or []
    asr_dir = out_dir / "asr"
    whisper_device = "cuda" if device == "cuda" else "cpu"
    model = None
    if _asr_incomplete(asr_dir, samples, sample_rate, args.force):
        model = load_model(args.whisper_model, args.compute_type, device=whisper_device)
    try:
        ru_words, ru_meta = transcribe_chunks(
            samples,
            sample_rate,
            turns,
            asr_dir,
            task="transcribe",
            language=args.language,
            model_name=args.whisper_model,
            compute_type=args.compute_type,
            beam_size=args.beam_size,
            force=args.force,
            model=model,
            device=whisper_device,
        )
        en_words, en_meta = transcribe_chunks(
            samples,
            sample_rate,
            turns,
            asr_dir,
            task="translate",
            language=args.language,
            model_name=args.whisper_model,
            compute_type=args.compute_type,
            beam_size=args.beam_size,
            force=args.force,
            model=model,
            device=whisper_device,
        )
    finally:
        if model is not None:
            del model
            release_memory()

    _write_json(out_dir / "words_ru.json", ru_words)
    _write_json(out_dir / "words_en.json", en_words)
    utterances = fill_translations(group_utterances(ru_words), en_words)
    _write_json(out_dir / "utterances.json", utterances)

    ram = unified_memory_gb()
    manifest = {
        "source_name": audio_path.name,
        "source_path": str(audio_path),
        "source_seconds": source_seconds,
        "processed_seconds": processed_seconds,
        "prefix_only": bool(args.max_seconds is not None and processed_seconds + 0.5 < source_seconds),
        "max_seconds": args.max_seconds,
        "device_preference": device,
        "separator": f"HTDemucs ({SEPARATOR_NAME}) on {separation.get('device', device)}",
        "enhancer": f"{ENHANCER_NAME} on {enhancement.get('device', device)}",
        "speaker_method": diarization.get("method"),
        "speaker_model": diarization.get("model"),
        "speaker_learning": diarization.get("speaker_learning"),
        "fit_speakers": profile_path is None,
        "fit_on": diarization.get("fit_on"),
        "speakers": diarization.get("speakers"),
        "asr": (
            f"faster-whisper {ru_meta['model']} {ru_meta['compute_type']} "
            f"on {ru_meta['device']}, language {args.language}"
        ),
        "translation": (
            f"faster-whisper {en_meta['model']} task=translate to {args.translate} "
            f"({en_meta['compute_type']} CPU)"
        ),
        "translation_reason": (
            f"SeamlessM4T is not used. This machine has {ram:.1f} GB unified memory; "
            "SeamlessM4T v2 large does not leave a safe margin beside the OS and large-v3. "
            "The Whisper translate task reuses the same int8 weights."
        ),
        "utterances": len(utterances),
        "lyric_flags": sum(1 for item in utterances if item.get("likely_english_lyric_bleed")),
        "separation": separation,
        "enhancement": enhancement,
        "asr_ru": ru_meta,
        "asr_en": en_meta,
    }
    _write_json(out_dir / "manifest.json", manifest)
    markdown = render_markdown(manifest, utterances)
    transcript = out_dir / "transcript.md"
    transcript.write_text(markdown, encoding="utf-8")
    log.info(
        "wrote %s (%s utterances, %s lyric flags)",
        transcript,
        manifest["utterances"],
        manifest["lyric_flags"],
    )
    print(transcript)
    return 0


def _prepare_speech(enhanced: Path, speech: Path) -> None:
    audio, sample_rate = read_mono(enhanced)
    audio = highpass(audio, sample_rate, 70.0)
    audio = resample(audio, sample_rate, 16000)
    audio = peak_normalize(audio)
    write_wav(speech, audio, 16000)
    log.info("wrote 16 kHz speech %s (%.1fs)", speech.name, len(audio) / 16000.0)


def _default_out(audio_path: Path) -> Path:
    stem = audio_path.stem.replace("+", "-")
    return Path.cwd() / "runs" / stem


def _asr_incomplete(asr_dir: Path, audio: np.ndarray, sample_rate: int, force: bool) -> bool:
    if force:
        return True
    from artalisten.asr import CHUNK_SECONDS, OVERLAP_SECONDS
    from artalisten.align import iter_chunks

    duration = float(len(audio) / sample_rate) if sample_rate else 0.0
    spans = list(iter_chunks(duration, CHUNK_SECONDS, OVERLAP_SECONDS)) if duration else []
    for index in range(len(spans)):
        if not (asr_dir / f"ru_{index:04d}.json").exists():
            return True
        if not (asr_dir / f"en_{index:04d}.json").exists():
            return True
    return False


def _clear_asr(path: Path) -> None:
    if not path.exists():
        return
    for child in path.glob("*.json"):
        child.unlink()


def _check_stamp(out_dir: Path, audio_path: Path, max_seconds: float | None, force: bool) -> None:
    stamp_path = out_dir / "stamp.json"
    stamp = {
        "source": str(audio_path),
        "max_seconds": max_seconds,
        "mtime_ns": audio_path.stat().st_mtime_ns,
    }
    if stamp_path.exists() and not force:
        previous = json.loads(stamp_path.read_text(encoding="utf-8"))
        if previous != stamp:
            raise SystemExit(
                "this output directory was built for a different input or prefix. "
                "Pass --force or choose another --out."
            )
    _write_json(stamp_path, stamp)


def _setup_logging(path: Path) -> None:
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    root.handlers.clear()
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    handle = logging.FileHandler(path, encoding="utf-8")
    handle.setFormatter(formatter)
    root.addHandler(stream)
    root.addHandler(handle)


def _write_json(path: Path, payload) -> None:
    path.write_text(
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _read_json(path: Path):
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    return value
