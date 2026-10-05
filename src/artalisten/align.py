"""Group word timestamps into speaker turns and attach the English translation."""

from __future__ import annotations

from artalisten.bleed import likely_english_lyric


def join_words(words: list[dict]) -> str:
    parts: list[str] = []
    for word in words:
        token = str(word.get("word", "")).strip()
        if token:
            parts.append(token)
    return " ".join(parts)


def group_utterances(words: list[dict], gap: float = 0.8) -> list[dict]:
    """Merge consecutive words that share a speaker and sit close in time."""
    utterances: list[dict] = []
    current: dict | None = None
    for word in words:
        start = float(word["start"])
        end = float(word["end"])
        speaker = word.get("speaker") or "unknown"
        speaker_id = word.get("speaker_id")
        if current is None:
            current = _new_utterance(word, start, end, speaker, speaker_id)
            continue
        same = speaker == current["speaker"] and speaker_id == current["speaker_id"]
        if same and start - float(current["end"]) <= gap:
            current["end"] = end
            current["words_ru"].append(word)
            if word.get("no_speech_prob") is not None:
                prev = current.get("no_speech_prob")
                prob = float(word["no_speech_prob"])
                current["no_speech_prob"] = prob if prev is None else max(float(prev), prob)
            continue
        utterances.append(_finish(current))
        current = _new_utterance(word, start, end, speaker, speaker_id)
    if current is not None:
        utterances.append(_finish(current))
    return utterances


def _new_utterance(word: dict, start: float, end: float, speaker: str, speaker_id) -> dict:
    prob = word.get("no_speech_prob")
    return {
        "start": start,
        "end": end,
        "speaker": speaker,
        "speaker_id": speaker_id,
        "words_ru": [word],
        "words_en": [],
        "no_speech_prob": None if prob is None else float(prob),
    }


def _finish(current: dict) -> dict:
    text = join_words(current["words_ru"])
    current["text_ru"] = text
    current["text_en"] = ""
    current["likely_english_lyric_bleed"] = likely_english_lyric(
        text, current.get("no_speech_prob")
    )
    return current


def fill_translations(utterances: list[dict], en_words: list[dict], pad: float = 0.05) -> list[dict]:
    """Attach English words whose midpoints fall inside each Russian turn."""
    used: set[int] = set()
    for utt in utterances:
        in_span: list[tuple[int, dict]] = []
        for index, word in enumerate(en_words):
            mid = 0.5 * (float(word["start"]) + float(word["end"]))
            if float(utt["start"]) - pad <= mid <= float(utt["end"]) + pad:
                in_span.append((index, word))
        # Keep the translation on the same global speaker. A neighbouring voice
        # that only overlaps in time stays in its own turn.
        picked_pairs = [
            pair for pair in in_span if pair[1].get("speaker") == utt.get("speaker")
        ]
        for index, _word in picked_pairs:
            used.add(index)
        picked = [word for _, word in picked_pairs]
        utt["words_en"] = picked
        utt["text_en"] = join_words(picked)
    leftovers = [word for index, word in enumerate(en_words) if index not in used]
    utterances.extend(_leftover_utterances(leftovers))
    utterances.sort(key=lambda item: (float(item["start"]), float(item["end"])))
    return utterances


def _leftover_utterances(words: list[dict], gap: float = 0.8) -> list[dict]:
    """English words with no Russian partner. Keep them and flag lyric-like lines."""
    if not words:
        return []
    grouped = group_utterances(words, gap=gap)
    extras = []
    for utt in grouped:
        text = utt["text_ru"]
        extras.append(
            {
                "start": utt["start"],
                "end": utt["end"],
                "speaker": utt["speaker"],
                "speaker_id": utt["speaker_id"],
                "words_ru": [],
                "words_en": utt["words_ru"],
                "text_ru": "",
                "text_en": text,
                "no_speech_prob": utt.get("no_speech_prob"),
                "likely_english_lyric_bleed": likely_english_lyric(
                    text, utt.get("no_speech_prob")
                ),
            }
        )
    return extras


def drop_overlap_words(
    words: list[dict],
    chunk_start: float,
    overlap: float,
    is_first: bool,
) -> list[dict]:
    """Drop the left edge of an overlapping ASR chunk so words are not doubled."""
    if is_first or overlap <= 0:
        return words
    cutoff = chunk_start + overlap / 2.0
    return [word for word in words if float(word["start"]) >= cutoff - 1e-3]


def iter_chunks(duration: float, chunk: float, overlap: float):
    if duration <= 0:
        return
    if chunk <= overlap:
        raise ValueError("chunk length must be longer than the overlap")
    start = 0.0
    while start < duration - 1e-3:
        end = min(duration, start + chunk)
        yield start, end
        if end >= duration - 1e-3:
            break
        start = end - overlap
