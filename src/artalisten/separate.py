"""Separate the background song from speech with HTDemucs."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import soundfile as sf

from artalisten.audioio import write_wav
from artalisten.device import release_memory, torch_device

log = logging.getLogger("artalisten.separate")

MODEL_NAME = "htdemucs"


def separate_speech(
    wav_path: Path,
    vocals_path: Path,
    envelope_path: Path,
    device: str,
    shifts: int = 0,
    overlap: float = 0.25,
) -> dict:
    """Keep the vocals stem. Try MPS, then CPU if MPS cannot run the model."""
    devices = [device] if device == "cpu" else [device, "cpu"]
    last: Exception | None = None
    for name in devices:
        try:
            return _separate(wav_path, vocals_path, envelope_path, name, shifts, overlap)
        except Exception as exc:
            last = exc
            release_memory()
            if name == devices[-1]:
                raise
            log.warning("HTDemucs failed on %s (%s). Retrying on CPU.", name, exc)
    raise RuntimeError(f"separation failed: {last}")


def _separate(
    wav_path: Path,
    vocals_path: Path,
    envelope_path: Path,
    device: str,
    shifts: int,
    overlap: float,
) -> dict:
    import torch
    from demucs.apply import apply_model
    from demucs.audio import convert_audio
    from demucs.pretrained import get_model

    log.info("loading HTDemucs (%s) on %s", MODEL_NAME, device)
    model = get_model(MODEL_NAME)
    model.eval()
    dev = torch_device(device)
    model.to(dev)

    audio, sample_rate = sf.read(str(wav_path), always_2d=True, dtype="float32")
    mixture = torch.from_numpy(np.ascontiguousarray(audio.T))
    mixture = convert_audio(mixture, sample_rate, model.samplerate, model.audio_channels)
    reference = mixture.mean(0)
    mixture = (mixture - reference.mean()) / (reference.std() + 1e-8)

    log.info(
        "separating on %s (shifts=%s, overlap=%s, split=True)",
        device,
        shifts,
        overlap,
    )
    with torch.no_grad():
        sources = apply_model(
            model,
            mixture[None],
            device=dev,
            shifts=shifts,
            split=True,
            overlap=overlap,
            progress=True,
            num_workers=0,
        )[0]
    sources = sources * reference.std() + reference.mean()
    sources = sources.cpu()

    names = list(model.sources)
    if "vocals" not in names:
        raise RuntimeError(f"HTDemucs sources {names} have no vocals stem")
    vocal_index = names.index("vocals")
    vocals = sources[vocal_index].mean(0)
    accompaniment = sources.sum(0) - sources[vocal_index]
    accompaniment = accompaniment.mean(0)

    sample_rate_out = int(model.samplerate)
    write_wav(vocals_path, vocals.numpy(), sample_rate_out)
    envelopes = {
        "hz": np.float32(10.0),
        "sample_rate": np.int32(sample_rate_out),
        "vocals_rms": _rms(vocals.numpy(), sample_rate_out, 10.0),
        "accompaniment_rms": _rms(accompaniment.numpy(), sample_rate_out, 10.0),
    }
    np.savez(envelope_path, **envelopes)
    log.info("wrote speech stem %s", vocals_path.name)
    del model, sources, mixture
    release_memory()
    return {
        "model": MODEL_NAME,
        "device": device,
        "sample_rate": sample_rate_out,
        "sources": names,
        "shifts": shifts,
        "overlap": overlap,
    }


def _rms(mono: np.ndarray, sample_rate: int, hz: float) -> np.ndarray:
    hop = max(1, int(round(sample_rate / hz)))
    count = int(np.ceil(len(mono) / hop))
    out = np.zeros(count, dtype=np.float32)
    for index in range(count):
        piece = mono[index * hop : (index + 1) * hop]
        if piece.size:
            out[index] = float(np.sqrt(np.mean(np.square(piece, dtype=np.float64))))
    return out
