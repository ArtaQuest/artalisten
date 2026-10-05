"""Learn speaker profiles from the whole recording, then label time with those profiles.

Chunk-by-chunk clustering gives each chunk its own arbitrary speaker ids, so the
same voice changes name every few minutes. This module embeds every speech
window in the file and clusters those embeddings once.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

log = logging.getLogger("artalisten.speakers")

ECAPA_SOURCE = "speechbrain/spkrec-ecapa-voxceleb"
WINDOW_SECONDS = 2.0
HOP_SECONDS = 1.0
MIN_WINDOW_SECONDS = 0.8
VAD_MODEL = "silero-vad"


def nearest_centroids(embeddings: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Assign each embedding to a frozen centroid. This does not fit a new clustering."""
    points = np.asarray(embeddings, dtype=np.float64)
    centers = np.asarray(centroids, dtype=np.float64)
    if points.ndim != 2 or centers.ndim != 2 or points.shape[1] != centers.shape[1]:
        raise ValueError("embeddings and centroids must share a feature dimension")
    points = points / np.clip(np.linalg.norm(points, axis=1, keepdims=True), 1e-8, None)
    centers = centers / np.clip(np.linalg.norm(centers, axis=1, keepdims=True), 1e-8, None)
    return np.argmax(points @ centers.T, axis=1).astype(np.int32)


def cluster_global(embeddings: np.ndarray, n_speakers: int) -> np.ndarray:
    """Cluster every embedding from the recording in a single fit."""
    matrix = np.asarray(embeddings, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("embeddings must have shape (windows, dim)")
    if len(matrix) < n_speakers:
        raise RuntimeError(
            f"only {len(matrix)} speech windows; need at least {n_speakers} to learn speakers"
        )
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix = matrix / np.clip(norms, 1e-8, None)
    from sklearn.cluster import AgglomerativeClustering

    try:
        model = AgglomerativeClustering(
            n_clusters=n_speakers,
            metric="cosine",
            linkage="average",
        )
    except TypeError:
        model = AgglomerativeClustering(
            n_clusters=n_speakers,
            affinity="cosine",
            linkage="average",
        )
    return model.fit_predict(matrix).astype(np.int32)


def map_sex(cluster_f0: dict[int, float | None], min_gap_hz: float = 15.0) -> dict[int, str]:
    """Map cluster ids to man/woman using median F0. The label is always estimated."""
    measured = [(cluster, hz) for cluster, hz in cluster_f0.items() if hz is not None]
    labels = {
        cluster: f"speaker_{cluster} (estimated, sex uncertain)" for cluster in cluster_f0
    }
    if len(measured) < 2:
        return labels
    measured.sort(key=lambda item: item[1])
    low_cluster, low_hz = measured[0]
    high_cluster, high_hz = measured[-1]
    if high_hz - low_hz < min_gap_hz or low_cluster == high_cluster:
        return labels
    labels[low_cluster] = "man (estimated)"
    labels[high_cluster] = "woman (estimated)"
    return labels


def median_f0(audio: np.ndarray, sample_rate: int, fmin: float = 75.0, fmax: float = 350.0) -> float | None:
    """Weak autocorrelation pitch estimate. Used only to name the two clusters."""
    y = np.asarray(audio, dtype=np.float64)
    if y.size < int(0.04 * sample_rate):
        return None
    y = y - np.mean(y)
    if float(np.max(np.abs(y))) < 1e-4:
        return None
    frame = int(0.04 * sample_rate)
    hop = int(0.02 * sample_rate)
    min_lag = max(1, int(sample_rate / fmax))
    max_lag = min(frame - 1, int(sample_rate / fmin))
    if max_lag <= min_lag:
        return None
    window = np.hanning(frame)
    pitches: list[float] = []
    for start in range(0, len(y) - frame, hop):
        segment = y[start : start + frame] * window
        energy = float(np.sqrt(np.mean(segment**2)))
        if energy < 0.01:
            continue
        size = 1 << int(np.ceil(np.log2(frame * 2)))
        spectrum = np.fft.rfft(segment, n=size)
        corr = np.fft.irfft(spectrum * np.conj(spectrum))[: max_lag + 1]
        if corr[0] <= 0:
            continue
        corr = corr / corr[0]
        region = corr[min_lag : max_lag + 1]
        lag = int(np.argmax(region)) + min_lag
        if corr[lag] < 0.35:
            continue
        pitches.append(sample_rate / lag)
    if len(pitches) < 3:
        return None
    return float(np.median(pitches))


def speech_windows(timestamps: list[dict]) -> list[tuple[float, float]]:
    windows: list[tuple[float, float]] = []
    for stamp in timestamps:
        start = float(stamp["start"])
        end = float(stamp["end"])
        if end - start < MIN_WINDOW_SECONDS:
            if end - start >= 0.5:
                windows.append((start, end))
            continue
        cursor = start
        while cursor + MIN_WINDOW_SECONDS <= end:
            stop = min(cursor + WINDOW_SECONDS, end)
            if stop - cursor >= MIN_WINDOW_SECONDS:
                windows.append((cursor, stop))
            cursor += HOP_SECONDS
    return windows


def turns_from_windows(
    windows: list[tuple[float, float, int]],
    duration: float,
    bin_s: float = 0.2,
    min_turn: float = 0.6,
) -> list[dict]:
    """Majority-vote overlapping windows, then absorb blips shorter than ``min_turn``."""
    if duration <= 0 or not windows:
        return []
    n_bins = int(np.ceil(duration / bin_s))
    n_speakers = max(cluster for _, _, cluster in windows) + 1
    votes = np.zeros((n_bins, n_speakers), dtype=np.float32)
    for start, end, cluster in windows:
        i0 = max(0, int(np.floor(start / bin_s)))
        i1 = min(n_bins, int(np.ceil(end / bin_s)))
        if i1 > i0 and 0 <= cluster < n_speakers:
            votes[i0:i1, cluster] += 1.0
    covered = votes.sum(axis=1) > 0
    winners = votes.argmax(axis=1)
    winners = np.where(covered, winners, -1)
    winners = _absorb_short(winners, bin_s, min_turn)
    return _runs(winners, bin_s, duration)


def _absorb_short(labels: np.ndarray, bin_s: float, min_turn: float) -> np.ndarray:
    out = labels.copy()
    min_bins = max(1, int(round(min_turn / bin_s)))
    start = 0
    while start < len(out):
        end = start + 1
        while end < len(out) and out[end] == out[start]:
            end += 1
        length = end - start
        if out[start] >= 0 and length < min_bins:
            left = out[start - 1] if start > 0 else -1
            right = out[end] if end < len(out) else -1
            if left >= 0:
                out[start:end] = left
            elif right >= 0:
                out[start:end] = right
        start = end
    return out


def _runs(labels: np.ndarray, bin_s: float, duration: float) -> list[dict]:
    turns: list[dict] = []
    start = 0
    while start < len(labels):
        end = start + 1
        while end < len(labels) and labels[end] == labels[start]:
            end += 1
        if int(labels[start]) >= 0:
            t0 = start * bin_s
            t1 = min(duration, end * bin_s)
            if t1 > t0:
                turns.append(
                    {
                        "start": float(t0),
                        "end": float(t1),
                        "speaker_id": f"S{int(labels[start])}",
                        "cluster": int(labels[start]),
                    }
                )
        start = end
    return turns


def speaker_at(turns: list[dict], moment: float) -> tuple[str, str | None]:
    for turn in turns:
        if float(turn["start"]) <= moment < float(turn["end"]):
            return str(turn["speaker"]), turn.get("speaker_id")
    nearest = None
    best = 1e9
    for turn in turns:
        gap = min(abs(moment - float(turn["start"])), abs(moment - float(turn["end"])))
        if gap < best:
            best = gap
            nearest = turn
    if nearest is not None and best <= 0.3:
        return str(nearest["speaker"]), nearest.get("speaker_id")
    return "unknown", None


def learn_speakers(
    audio: np.ndarray,
    sample_rate: int,
    num_speakers: int,
    device: str,
    cache_dir: Path,
    hf_token: str | None = None,
) -> dict:
    """Build global speaker profiles, then a timeline. Pyannote only if a token exists."""
    duration = float(len(audio) / sample_rate) if sample_rate else 0.0
    if hf_token:
        pyannote = _try_pyannote(audio, sample_rate, num_speakers, device, hf_token, cache_dir)
        if pyannote is not None:
            return _name_turns(pyannote, audio, sample_rate, duration)
    log.info(
        "no usable pyannote pipeline; learning speakers with %s on the full file",
        ECAPA_SOURCE,
    )
    return _learn_open(audio, sample_rate, num_speakers, device, cache_dir, duration)


def _name_turns(result: dict, audio: np.ndarray, sample_rate: int, duration: float) -> dict:
    clusters = sorted({int(turn["cluster"]) for turn in result["turns"]})
    f0 = {cluster: _cluster_f0_from_turns(result["turns"], cluster, audio, sample_rate) for cluster in clusters}
    names = map_sex(f0)
    for turn in result["turns"]:
        turn["speaker"] = names[int(turn["cluster"])]
        turn["median_f0_hz"] = f0.get(int(turn["cluster"]))
    result["speakers"] = _summaries(result["turns"], names, f0)
    result["duration_seconds"] = duration
    result["f0_basis"] = "estimated"
    return result


def _learn_open(
    audio: np.ndarray,
    sample_rate: int,
    num_speakers: int,
    device: str,
    cache_dir: Path,
    duration: float,
) -> dict:
    timestamps = _vad(audio, sample_rate)
    windows = speech_windows(timestamps)
    log.info("VAD speech regions: %s; embedding windows: %s", len(timestamps), len(windows))
    if len(windows) < num_speakers:
        raise RuntimeError(
            "not enough speech to learn "
            f"{num_speakers} speakers (windows={len(windows)}). "
            "The enhancer may have removed the voices along with the noise."
        )
    clips = [_slice(audio, sample_rate, start, end) for start, end in windows]
    embeddings, embed_device = _embed(clips, device, cache_dir)
    labels = cluster_global(embeddings, num_speakers)
    labeled = [(start, end, int(label)) for (start, end), label in zip(windows, labels)]
    turns = turns_from_windows(labeled, duration)
    f0 = {}
    for cluster in range(num_speakers):
        f0[cluster] = _cluster_f0(labeled, cluster, audio, sample_rate)
    names = map_sex(f0)
    for turn in turns:
        turn["speaker"] = names[int(turn["cluster"])]
    centroids = _centroids(embeddings, labels, num_speakers)
    return {
        "method": "global-ecapa-agglomerative",
        "speaker_learning": "global",
        "model": ECAPA_SOURCE,
        "vad": VAD_MODEL,
        "device": embed_device,
        "num_speakers_prior": num_speakers,
        "windows": len(windows),
        "duration_seconds": duration,
        "f0_basis": "estimated",
        "note": (
            "Embeddings were taken across the whole processed recording and clustered once. "
            "Sex labels use median F0 only as a weak hint and are marked estimated."
        ),
        "centroid_cosine_distance": _centroid_distance(centroids),
        "centroids": [_plain_vector(center) for center in centroids],
        "speakers": _summaries(turns, names, f0),
        "turns": turns,
    }


def assign_from_profile(
    audio: np.ndarray,
    sample_rate: int,
    profile: dict,
    device: str,
    cache_dir: Path,
) -> dict:
    """Label speech with a profile learned earlier. Does not fit or update that profile."""
    centroids = np.asarray(profile.get("centroids") or [], dtype=np.float64)
    if centroids.ndim != 2 or len(centroids) < 1:
        raise RuntimeError("speaker profile has no centroids; refusing to fit a new clustering")
    duration = float(len(audio) / sample_rate) if sample_rate else 0.0
    names = {
        int(speaker["cluster"]): str(speaker["label"])
        for speaker in profile.get("speakers") or []
        if "cluster" in speaker
    }
    timestamps = _vad(audio, sample_rate)
    windows = speech_windows(timestamps)
    log.info(
        "inference only: %s speech regions, %s windows, %s frozen centroids (no clustering fit)",
        len(timestamps),
        len(windows),
        len(centroids),
    )
    if not windows:
        turns: list[dict] = []
        embed_device = device
    else:
        clips = [_slice(audio, sample_rate, start, end) for start, end in windows]
        embeddings, embed_device = _embed(clips, device, cache_dir)
        labels = nearest_centroids(embeddings, centroids)
        labeled = [(start, end, int(label)) for (start, end), label in zip(windows, labels)]
        turns = turns_from_windows(labeled, duration)
        for turn in turns:
            cluster = int(turn["cluster"])
            turn["speaker"] = names.get(cluster, f"speaker_{cluster} (estimated, sex uncertain)")
    return {
        "method": "frozen-centroid-assignment",
        "speaker_learning": "none",
        "model": profile.get("model") or ECAPA_SOURCE,
        "vad": VAD_MODEL,
        "device": embed_device,
        "num_speakers_prior": int(profile.get("num_speakers_prior") or len(centroids)),
        "windows": len(windows),
        "duration_seconds": duration,
        "f0_basis": "estimated",
        "fit_on": profile.get("fit_on"),
        "note": (
            "Speakers were assigned to frozen centroids from the training recording. "
            "This file was not used to fit, adapt, or rename the profiles."
        ),
        "speakers": _summaries(
            turns,
            names,
            {int(speaker["cluster"]): speaker.get("median_f0_hz") for speaker in profile.get("speakers") or []},
        ),
        "turns": turns,
    }


def _plain_vector(center: np.ndarray) -> list[float]:
    return [float(value) for value in np.asarray(center).reshape(-1)]


def _summaries(turns: list[dict], names: dict[int, str], f0: dict[int, float | None]) -> list[dict]:
    totals: dict[int, float] = {}
    for turn in turns:
        cluster = int(turn["cluster"])
        totals[cluster] = totals.get(cluster, 0.0) + float(turn["end"]) - float(turn["start"])
    summaries = []
    for cluster, label in sorted(names.items(), key=lambda item: item[0]):
        summaries.append(
            {
                "speaker_id": f"S{cluster}",
                "cluster": cluster,
                "label": label,
                "median_f0_hz": f0.get(cluster),
                "f0_basis": "estimated",
                "speech_seconds": round(totals.get(cluster, 0.0), 3),
            }
        )
    return summaries


def _centroids(embeddings: np.ndarray, labels: np.ndarray, n_speakers: int) -> list[np.ndarray]:
    centers = []
    for cluster in range(n_speakers):
        rows = embeddings[labels == cluster]
        if len(rows) == 0:
            centers.append(np.zeros(embeddings.shape[1], dtype=np.float64))
            continue
        center = rows.mean(axis=0)
        norm = np.linalg.norm(center) + 1e-8
        centers.append(center / norm)
    return centers


def _centroid_distance(centroids: list[np.ndarray]) -> float | None:
    if len(centroids) < 2:
        return None
    a, b = centroids[0], centroids[1]
    return float(1.0 - np.dot(a, b))


def _slice(audio: np.ndarray, sample_rate: int, start: float, end: float) -> np.ndarray:
    i0 = max(0, int(start * sample_rate))
    i1 = min(len(audio), int(end * sample_rate))
    return np.ascontiguousarray(audio[i0:i1], dtype=np.float32)


def _cluster_f0(
    labeled: list[tuple[float, float, int]],
    cluster: int,
    audio: np.ndarray,
    sample_rate: int,
) -> float | None:
    chosen = [(end - start, start, end) for start, end, label in labeled if label == cluster]
    chosen.sort(reverse=True)
    pitches = []
    for _, start, end in chosen[:8]:
        pitch = median_f0(_slice(audio, sample_rate, start, end), sample_rate)
        if pitch is not None:
            pitches.append(pitch)
    if not pitches:
        return None
    return float(np.median(pitches))


def _cluster_f0_from_turns(turns, cluster, audio, sample_rate) -> float | None:
    labeled = [
        (float(turn["start"]), float(turn["end"]), int(turn["cluster"]))
        for turn in turns
        if int(turn["cluster"]) == cluster
    ]
    return _cluster_f0(labeled, cluster, audio, sample_rate)


def _vad(audio: np.ndarray, sample_rate: int) -> list[dict]:
    import torch
    from silero_vad import get_speech_timestamps, load_silero_vad

    if sample_rate != 16000:
        raise RuntimeError("Silero VAD expects 16 kHz audio")
    model = load_silero_vad()
    wav = torch.from_numpy(np.ascontiguousarray(audio)).float()
    timestamps: list[dict] = []
    for threshold in (0.4, 0.25, 0.15):
        timestamps = get_speech_timestamps(
            wav,
            model,
            sampling_rate=16000,
            threshold=threshold,
            min_speech_duration_ms=200,
            min_silence_duration_ms=300,
            speech_pad_ms=200,
            return_seconds=True,
        )
        speech = sum(float(item["end"]) - float(item["start"]) for item in timestamps)
        log.info("silero threshold %.2f found %.1fs of speech", threshold, speech)
        if speech >= 5.0 and len(timestamps) >= 2:
            break
    del model
    return timestamps


def _embed(clips: list[np.ndarray], device: str, cache_dir: Path) -> tuple[np.ndarray, str]:
    devices = [device] if device == "cpu" else [device, "cpu"]
    last: Exception | None = None
    for name in devices:
        try:
            return _embed_on(clips, name, cache_dir), name
        except Exception as exc:
            last = exc
            if name == devices[-1]:
                raise
            log.warning("speaker embedding failed on %s (%s). Retrying on CPU.", name, exc)
    raise RuntimeError(f"embedding failed: {last}")


def _embed_on(clips: list[np.ndarray], device: str, cache_dir: Path) -> np.ndarray:
    import torch
    from speechbrain.inference.speaker import EncoderClassifier

    cache_dir.mkdir(parents=True, exist_ok=True)
    log.info("loading %s on %s", ECAPA_SOURCE, device)
    classifier = EncoderClassifier.from_hparams(
        source=ECAPA_SOURCE,
        savedir=str(cache_dir / "spkrec-ecapa-voxceleb"),
        run_opts={"device": device},
    )
    embeddings = []
    batch_size = 8
    for start in range(0, len(clips), batch_size):
        batch = clips[start : start + batch_size]
        lengths = [len(clip) for clip in batch]
        width = max(lengths)
        array = np.zeros((len(batch), width), dtype=np.float32)
        for index, clip in enumerate(batch):
            array[index, : len(clip)] = clip
        tensor = torch.from_numpy(array)
        ratios = torch.tensor([length / width for length in lengths], dtype=torch.float32)
        with torch.no_grad():
            encoded = classifier.encode_batch(tensor, ratios)
        encoded = encoded.detach().cpu().numpy()
        encoded = np.squeeze(encoded, axis=1) if encoded.ndim == 3 else np.squeeze(encoded)
        if encoded.ndim == 1:
            encoded = encoded[None, :]
        embeddings.append(encoded.astype(np.float32))
        done = min(len(clips), start + batch_size)
        if done % 80 == 0 or done == len(clips):
            log.info("embedded %s / %s windows", done, len(clips))
    del classifier
    return np.concatenate(embeddings, axis=0)


def _try_pyannote(audio, sample_rate, num_speakers, device, token: str, cache_dir: Path):
    try:
        from pyannote.audio import Pipeline
    except ImportError:
        log.info("Hugging Face token is set, but pyannote.audio is not installed. Using ECAPA.")
        return None
    import soundfile as sf
    import torch

    wav_path = cache_dir / "_pyannote_input.wav"
    cache_dir.mkdir(parents=True, exist_ok=True)
    sf.write(str(wav_path), audio, sample_rate, subtype="PCM_16")
    log.info("running pyannote/speaker-diarization-3.1 on the full file")
    try:
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=token,
        )
    except TypeError:
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", token=token)
    if pipeline is None:
        log.warning("pyannote pipeline did not load. Using ECAPA.")
        return None
    try:
        pipeline.to(torch.device(device if device != "mps" else "cpu"))
    except Exception:
        log.info("pyannote staying on CPU")
    annotation = pipeline(str(wav_path), num_speakers=num_speakers)
    raw_turns = []
    ids: dict[str, int] = {}
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        if speaker not in ids:
            ids[speaker] = len(ids)
        raw_turns.append(
            {
                "start": float(turn.start),
                "end": float(turn.end),
                "cluster": ids[speaker],
                "speaker_id": f"S{ids[speaker]}",
            }
        )
    try:
        wav_path.unlink()
    except OSError:
        pass
    return {
        "method": "pyannote-speaker-diarization-3.1",
        "speaker_learning": "global",
        "model": "pyannote/speaker-diarization-3.1",
        "vad": "pyannote",
        "device": "cpu" if device == "mps" else device,
        "num_speakers_prior": num_speakers,
        "note": (
            "pyannote 3.1 diarizes the full recording. "
            "Sex labels use median F0 only as a weak hint and are marked estimated."
        ),
        "turns": raw_turns,
    }


def hf_token() -> str | None:
    import os

    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        value = os.environ.get(key)
        if value:
            return value.strip()
    for path in (
        Path.home() / ".cache" / "huggingface" / "token",
        Path.home() / ".huggingface" / "token",
    ):
        if path.is_file():
            text = path.read_text(encoding="utf-8").strip()
            if text:
                return text
    return None
