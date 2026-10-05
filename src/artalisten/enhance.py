"""Enhance very low-SNR speech with DeepFilterNet3."""

from __future__ import annotations

import logging
import os
import sys
import types
from pathlib import Path

import numpy as np

from artalisten.audioio import read_mono, resample, write_wav
from artalisten.device import release_memory

log = logging.getLogger("artalisten.enhance")

MODEL_NAME = "DeepFilterNet3"


def enhance_speech(
    vocals_path: Path,
    enhanced_path: Path,
    device: str,
    chunk_seconds: float = 20.0,
    overlap_seconds: float = 1.0,
) -> dict:
    """Denoise the speech stem. Fall back to CPU when MPS cannot run the model."""
    devices = [device] if device == "cpu" else [device, "cpu"]
    last: Exception | None = None
    for name in devices:
        try:
            return _enhance(
                vocals_path, enhanced_path, name, chunk_seconds, overlap_seconds
            )
        except Exception as exc:
            last = exc
            release_memory()
            if name == devices[-1]:
                raise
            log.warning("DeepFilterNet3 failed on %s (%s). Retrying on CPU.", name, exc)
    raise RuntimeError(f"enhancement failed: {last}")


def _enhance(
    vocals_path: Path,
    enhanced_path: Path,
    device: str,
    chunk_seconds: float,
    overlap_seconds: float,
) -> dict:
    import torch

    _shim_torchaudio_backend()
    # DeepFilterNet reads DEVICE from the environment and ignores MPS on its own.
    previous_device = os.environ.get("DEVICE")
    os.environ["DEVICE"] = device
    from df.enhance import enhance, init_df

    log.info("loading %s on %s", MODEL_NAME, device)
    model = None
    try:
        try:
            model, state, _ = init_df(default_model=MODEL_NAME, post_filter=True)
            post_filter = True
        except TypeError:
            model, state, _ = init_df()
            post_filter = False
        model.eval()

        audio, sample_rate = read_mono(vocals_path)
        target_rate = int(state.sr()) if hasattr(state, "sr") else 48000
        audio = resample(audio, sample_rate, target_rate)
        chunk = int(chunk_seconds * target_rate)
        overlap = int(overlap_seconds * target_rate)

        def _one(piece: np.ndarray) -> np.ndarray:
            tensor = torch.from_numpy(np.ascontiguousarray(piece)).float().unsqueeze(0)
            with torch.no_grad():
                denoised = enhance(model, state, tensor, pad=True)
            if isinstance(denoised, torch.Tensor):
                denoised = denoised.detach().cpu().numpy()
            denoised = np.asarray(denoised, dtype=np.float32).reshape(-1)
            if denoised.size >= piece.size:
                return denoised[: piece.size]
            out = np.zeros(piece.size, dtype=np.float32)
            out[: denoised.size] = denoised
            return out

        # Fail fast if this device cannot run the network, before the long file.
        _one(np.zeros(target_rate, dtype=np.float32))
        log.info(
            "enhancing %.1f min in %.0fs chunks on %s",
            len(audio) / target_rate / 60.0,
            chunk_seconds,
            device,
        )
        denoised = _overlap_add(audio, chunk, overlap, _one, target_rate)
        write_wav(enhanced_path, denoised, target_rate)
        log.info("wrote enhanced speech %s", enhanced_path.name)
        return {
            "model": MODEL_NAME,
            "device": device,
            "sample_rate": target_rate,
            "post_filter": post_filter,
            "chunk_seconds": chunk_seconds,
        }
    finally:
        if previous_device is None:
            os.environ.pop("DEVICE", None)
        else:
            os.environ["DEVICE"] = previous_device
        del model
        release_memory()


def _shim_torchaudio_backend() -> None:
    """DeepFilterNet 0.5.6 imports a torchaudio module removed in current releases."""
    if "torchaudio.backend.common" in sys.modules:
        return
    try:
        import torchaudio.backend.common  # noqa: F401
        return
    except ModuleNotFoundError:
        pass
    backend = types.ModuleType("torchaudio.backend")
    common = types.ModuleType("torchaudio.backend.common")

    class AudioMetaData:
        def __init__(
            self,
            sample_rate: int = 0,
            num_frames: int = 0,
            num_channels: int = 0,
            bits_per_sample: int = 0,
            encoding: str = "",
        ):
            self.sample_rate = sample_rate
            self.num_frames = num_frames
            self.num_channels = num_channels
            self.bits_per_sample = bits_per_sample
            self.encoding = encoding

    common.AudioMetaData = AudioMetaData
    backend.common = common
    sys.modules["torchaudio.backend"] = backend
    sys.modules["torchaudio.backend.common"] = common


def _overlap_add(
    audio: np.ndarray,
    chunk: int,
    overlap: int,
    enhance_fn,
    sample_rate: int,
) -> np.ndarray:
    total = len(audio)
    if total == 0:
        return audio.astype(np.float32)
    overlap = min(overlap, max(0, chunk // 4))
    hop = max(1, chunk - overlap)
    out = np.zeros(total, dtype=np.float32)
    weight = np.zeros(total, dtype=np.float32)
    pos = 0
    index = 0
    while pos < total:
        end = min(total, pos + chunk)
        piece = audio[pos:end]
        denoised = enhance_fn(piece)
        size = min(len(denoised), end - pos)
        window = np.ones(size, dtype=np.float32)
        if overlap > 0 and pos > 0:
            fade = min(overlap, size)
            window[:fade] = np.linspace(0.0, 1.0, fade, dtype=np.float32)
        if overlap > 0 and end < total:
            fade = min(overlap, size)
            window[-fade:] = np.linspace(1.0, 0.0, fade, dtype=np.float32)
        out[pos : pos + size] += denoised[:size] * window
        weight[pos : pos + size] += window
        index += 1
        if index % 10 == 0:
            log.info(
                "enhanced %.1f / %.1f min",
                pos / sample_rate / 60.0,
                total / sample_rate / 60.0,
            )
        if end >= total:
            break
        pos += hop
    return out / np.maximum(weight, 1e-6)


def anna_bandpass(
    audio,
    sample_rate: int,
    low_hz: float = 140.0,
    high_hz: float = 6500.0,
):
    """Keep the higher female speech band; attenuate rumble and hiss."""
    from artalisten.audioio import highpass
    import numpy as np
    from scipy.signal import butter, sosfiltfilt

    x = np.asarray(audio, dtype=np.float32)
    if x.size == 0:
        return x
    # high-pass then low-pass via second-order sections
    x = highpass(x, sample_rate, low_hz) if low_hz > 0 else x
    if high_hz and high_hz < 0.45 * sample_rate:
        sos = butter(4, high_hz / (0.5 * sample_rate), btype="low", output="sos")
        x = sosfiltfilt(sos, x).astype(np.float32)
    return np.ascontiguousarray(x, dtype=np.float32)


def dynamic_normalize(audio, sample_rate: int, target_rms: float = 0.08, ceiling: float = 0.95):
    """Gentle RMS lift for ultra-quiet speech (variant G).

    Raises quiet stretches toward ``target_rms`` without hard compression that
    would pump the cafe noise floor too aggressively.
    """
    import numpy as np

    x = np.asarray(audio, dtype=np.float64)
    if x.size == 0:
        return x.astype(np.float32)
    frame = max(1, int(0.05 * sample_rate))
    hop = max(1, frame // 2)
    out = np.copy(x)
    for i0 in range(0, len(x), hop):
        i1 = min(len(x), i0 + frame)
        piece = x[i0:i1]
        rms = float(np.sqrt(np.mean(piece * piece) + 1e-12))
        if rms < 1e-6:
            continue
        gain = min(ceiling / (np.max(np.abs(piece)) + 1e-9), target_rms / rms)
        gain = float(np.clip(gain, 1.0, 8.0))
        out[i0:i1] *= gain
    peak = float(np.max(np.abs(out))) if out.size else 0.0
    if peak > ceiling:
        out *= ceiling / peak
    return np.ascontiguousarray(out, dtype=np.float32)


def apply_speaker_mask(
    audio,
    sample_rate: int,
    turns: list[dict],
    keep_speakers: set[str] | set[int],
    *,
    keep_gain: float = 1.0,
    other_gain: float = 0.15,
):
    """Lift Anna (or quiet) turns and attenuate everyone else / gaps."""
    import numpy as np

    x = np.asarray(audio, dtype=np.float32)
    out = x * float(other_gain)
    keep = {str(s) for s in keep_speakers} | {s for s in keep_speakers}
    for turn in turns:
        sp = turn.get("speaker")
        cid = turn.get("cluster")
        label = turn.get("speaker_id")
        keys = {sp, cid, label, str(sp), str(cid), str(label)}
        if keys & {str(k) for k in keep} or (cid in keep_speakers) or (sp in keep_speakers):
            i0 = max(0, int(float(turn["start"]) * sample_rate))
            i1 = min(len(out), int(float(turn["end"]) * sample_rate))
            out[i0:i1] = x[i0:i1] * float(keep_gain)
    peak = float(np.max(np.abs(out))) if out.size else 0.0
    if peak > 0.99:
        out *= np.float32(0.99 / peak)
    return np.ascontiguousarray(out, dtype=np.float32)
