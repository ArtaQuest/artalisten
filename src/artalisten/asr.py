"""Transcribe and translate with faster-whisper large-v3.

CTranslate2 has no MPS backend. On this Mac the model runs int8 on CPU.
The same weights do both the Russian transcript and the English translation.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import numpy as np

from artalisten.align import drop_overlap_words, iter_chunks
from artalisten.device import release_memory
from artalisten.speakers import speaker_at

log = logging.getLogger("artalisten.asr")

WHISPER_MODEL = "large-v3"
CHUNK_SECONDS = 300.0
OVERLAP_SECONDS = 3.0


def transcribe_chunks(
    audio: np.ndarray,
    sample_rate: int,
    turns: list[dict],
    out_dir: Path,
    task: str,
    language: str,
    model_name: str = WHISPER_MODEL,
    compute_type: str = "int8",
    beam_size: int = 5,
    force: bool = False,
    model=None,
    device: str = "cpu",
) -> tuple[list[dict], dict]:
    """Run one Whisper task over the whole file in resumable chunks.

    ``model`` may be passed in so transcription and translation share one load.
    """
    if sample_rate != 16000:
        raise RuntimeError("faster-whisper expects 16 kHz audio")
    if task not in {"transcribe", "translate"}:
        raise ValueError(f"unsupported whisper task {task}")
    out_dir.mkdir(parents=True, exist_ok=True)
    duration = float(len(audio) / sample_rate)
    spans = list(iter_chunks(duration, CHUNK_SECONDS, OVERLAP_SECONDS))
    owns_model = model is None
    if any(_missing(out_dir, task, index, force) for index in range(len(spans))):
        if model is None:
            model = load_model(model_name, compute_type, device=device)
    words: list[dict] = []
    try:
        for index, (start, end) in enumerate(spans):
            path = _chunk_path(out_dir, task, index)
            if path.exists() and not force:
                chunk_words = json.loads(path.read_text(encoding="utf-8"))
                log.info("reusing %s", path.name)
            else:
                chunk_words = _transcribe_span(
                    model,
                    audio,
                    sample_rate,
                    start,
                    end,
                    turns,
                    task,
                    language,
                    beam_size,
                    is_first=index == 0,
                )
                _atomic_json(path, chunk_words)
                log.info("%s chunk %s/%s saved (%s words)", task, index + 1, len(spans), len(chunk_words))
            words.extend(chunk_words)
    finally:
        if owns_model and model is not None:
            del model
            release_memory()
    words.sort(key=lambda item: (float(item["start"]), float(item["end"])))
    meta = {
        "model": model_name,
        "compute_type": compute_type,
        "device": device,
        "task": task,
        "language": language,
        "beam_size": beam_size,
        "chunk_seconds": CHUNK_SECONDS,
        "overlap_seconds": OVERLAP_SECONDS,
        "chunks": len(spans),
        "words": len(words),
    }
    return words, meta


def load_model(model_name: str, compute_type: str, device: str = "cpu"):
    from faster_whisper import WhisperModel

    threads = min(8, os.cpu_count() or 4)
    log.info("loading faster-whisper %s (%s, %s)", model_name, compute_type, device)
    kwargs = {"device": device, "compute_type": compute_type, "num_workers": 1}
    if device == "cpu":
        kwargs["cpu_threads"] = threads
    return WhisperModel(model_name, **kwargs)


def _transcribe_span(
    model,
    audio: np.ndarray,
    sample_rate: int,
    start: float,
    end: float,
    turns: list[dict],
    task: str,
    language: str,
    beam_size: int,
    is_first: bool,
) -> list[dict]:
    i0 = int(start * sample_rate)
    i1 = int(end * sample_rate)
    piece = np.ascontiguousarray(audio[i0:i1], dtype=np.float32)
    prompt = "Разговор на русском языке." if task == "transcribe" else None
    kwargs = dict(
        language=language,
        task=task,
        word_timestamps=True,
        beam_size=beam_size,
        temperature=0.0,
        vad_filter=True,
        vad_parameters={
            "threshold": 0.3,
            "min_speech_duration_ms": 120,
            "min_silence_duration_ms": 400,
            "speech_pad_ms": 250,
        },
        condition_on_previous_text=False,
        hallucination_silence_threshold=2.0,
    )
    if prompt:
        kwargs["initial_prompt"] = prompt
    try:
        segments, _info = model.transcribe(piece, **kwargs)
    except TypeError:
        kwargs.pop("hallucination_silence_threshold", None)
        segments, _info = model.transcribe(piece, **kwargs)
    words: list[dict] = []
    for segment in segments:
        no_speech = getattr(segment, "no_speech_prob", None)
        segment_words = list(segment.words or [])
        if segment_words:
            for word in segment_words:
                if word.start is None or word.end is None:
                    continue
                abs_start = float(word.start) + start
                abs_end = float(word.end) + start
                speaker, speaker_id = speaker_at(turns, 0.5 * (abs_start + abs_end))
                words.append(
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
        elif segment.text and segment.text.strip():
            abs_start = float(segment.start) + start
            abs_end = float(segment.end) + start
            speaker, speaker_id = speaker_at(turns, 0.5 * (abs_start + abs_end))
            words.append(
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
    return drop_overlap_words(words, start, OVERLAP_SECONDS, is_first=is_first)


def _chunk_path(out_dir: Path, task: str, index: int) -> Path:
    tag = "ru" if task == "transcribe" else "en"
    return out_dir / f"{tag}_{index:04d}.json"


def _missing(out_dir: Path, task: str, index: int, force: bool) -> bool:
    return force or not _chunk_path(out_dir, task, index).exists()


def _atomic_json(path: Path, payload) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)
