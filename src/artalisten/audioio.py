"""Decode recordings with ffmpeg and move audio between sample rates."""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

log = logging.getLogger("artalisten.audio")


def require_ffmpeg() -> str:
    path = shutil.which("ffmpeg")
    if not path:
        raise RuntimeError("ffmpeg is required to decode recordings")
    return path


def probe_duration(path: Path) -> float:
    ffprobe = shutil.which("ffprobe")
    if ffprobe:
        try:
            out = subprocess.check_output(
                [
                    ffprobe,
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "default=noprint_wrappers=1:nokey=1",
                    str(path),
                ],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
            if out:
                return float(out)
        except (subprocess.CalledProcessError, ValueError):
            pass
    info = sf.info(str(path))
    return float(info.duration)


def decode_wav(
    src: Path,
    dst: Path,
    sample_rate: int,
    channels: int,
    max_seconds: float | None,
) -> None:
    """Decode any ffmpeg-readable file to 16-bit PCM WAV."""
    ffmpeg = require_ffmpeg()
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [ffmpeg, "-y", "-i", str(src)]
    if max_seconds is not None:
        cmd += ["-t", f"{max_seconds:.3f}"]
    cmd += ["-ac", str(channels), "-ar", str(sample_rate), "-c:a", "pcm_s16le", str(dst)]
    log.info("decoding %s", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        tail = (proc.stderr or "")[-2000:]
        raise RuntimeError(f"ffmpeg failed to decode {src.name}: {tail}")


def read_mono(path: Path) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(str(path), always_2d=True, dtype="float32")
    mono = audio.mean(axis=1).astype(np.float32, copy=False)
    return mono, int(sr)


def write_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    y = np.asarray(audio, dtype=np.float32).reshape(-1)
    y = np.clip(y, -1.0, 1.0)
    sf.write(str(path), y, sample_rate, subtype="PCM_16")


def resample(audio: np.ndarray, sample_rate: int, target_rate: int) -> np.ndarray:
    if sample_rate == target_rate:
        return np.asarray(audio, dtype=np.float32)
    import torch
    import torchaudio

    tensor = torch.from_numpy(np.asarray(audio, dtype=np.float32)).unsqueeze(0)
    out = torchaudio.functional.resample(tensor, sample_rate, target_rate)
    return out.squeeze(0).cpu().numpy().astype(np.float32)


def highpass(audio: np.ndarray, sample_rate: int, cutoff_hz: float = 70.0) -> np.ndarray:
    """Drop cafe rumble below a male fundamental."""
    from scipy.signal import butter, sosfiltfilt

    if len(audio) < sample_rate:
        return np.asarray(audio, dtype=np.float32)
    sos = butter(4, cutoff_hz, btype="highpass", fs=sample_rate, output="sos")
    return sosfiltfilt(sos, np.asarray(audio, dtype=np.float64)).astype(np.float32)


def peak_normalize(audio: np.ndarray, target: float = 0.9, quiet_floor: float = 0.08) -> np.ndarray:
    """Lift a distant mic. Leave a healthy recording alone."""
    y = np.asarray(audio, dtype=np.float32)
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak < 1e-6:
        return y
    if peak < quiet_floor or peak > 1.0:
        y = y * (target / peak)
    return np.clip(y, -1.0, 1.0).astype(np.float32)
