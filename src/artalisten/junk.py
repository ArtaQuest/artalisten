"""Reject Whisper hallucination patterns common on extreme low-SNR cafe audio.

Forced-Russian decodes still invent subtitle credits, music captions, Ukrainian
viewer-thanks, and English lyric/translate loops. Those lines must never enter
a Final transcript.
"""

from __future__ import annotations

import re

# Case-folded regexes. Keep this list aggressive: a false reject of a real word
# is recoverable from another variant; a false keep poisons consensus.
JUNK_PATTERNS: tuple[str, ...] = (
    # Russian / Ukrainian subtitle & continuation credits
    r"продолжение\s+следует",
    r"субтитр",
    r"dima\s*torzok",
    r"симон",
    r"подогнал",
    r"создавал\s+dima",
    r"сделал\s+dima",
    # Music / non-speech captions
    r"играет\s+музыка",
    r"звучит\s+музыка",
    r"звучить\s+музика",
    r"music\s+playing",
    r"девушки\s+отдыхают",
    # Ukrainian viewer fluff
    r"дякую\s+за\s+перегляд",
    r"thanks\s+for\s+watching",
    # Whisper language-probe fluff
    r"говорит\s+на\s+русском",
    r"на\s+русском\s+языке",
    r"^говор$",
    # English translate / lyric loops seen on these files
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
    # StSq-style burned-in tags
    r"stsq\d*",
    r"\bst\s*sq\b",
)

_COMPILED = tuple(re.compile(p, re.IGNORECASE) for p in JUNK_PATTERNS)

# Exact short tokens that are never content on these recordings.
_EXACT_JUNK = frozenset(
    {
        "говор",
        "на",
        "русском",
        "языке",
        "языке.",
        "дякую.",
        "дякую",
    }
)


def is_junk(text: str | None) -> bool:
    """True when ``text`` matches a known hallucination / caption / credit."""
    folded = (text or "").casefold().strip()
    if not folded:
        return False
    if folded in _EXACT_JUNK:
        return True
    # Strip common punctuation for exact-ish checks
    tight = re.sub(r"[\s.!?…,«»\"']+", " ", folded).strip()
    if tight in _EXACT_JUNK:
        return True
    return any(pat.search(folded) for pat in _COMPILED)


def filter_segments(
    segments: list[dict],
    text_key: str = "text",
    *,
    min_word_prob: float | None = 0.4,
    word_list_key: str = "words",
) -> tuple[list[dict], list[dict]]:
    """Split segments into (kept, rejected).

    A segment is rejected when its text matches junk patterns, or when
    ``min_word_prob`` is set and no word reaches that probability.
    Rejected items gain a ``reject`` reason string.
    """
    kept: list[dict] = []
    rejected: list[dict] = []
    for seg in segments:
        item = dict(seg)
        text = str(item.get(text_key) or item.get("text_ru") or item.get("word") or "")
        if is_junk(text):
            item["reject"] = "known loop/credit"
            rejected.append(item)
            continue
        if min_word_prob is not None:
            words = item.get(word_list_key) or item.get("words_ru") or []
            probs = []
            for w in words:
                if isinstance(w, dict):
                    p = w.get("probability")
                    if p is None and len(w) >= 4 and isinstance(w, (list, tuple)):
                        p = w[3]
                    if p is not None:
                        probs.append(float(p))
                elif isinstance(w, (list, tuple)) and len(w) >= 4:
                    probs.append(float(w[3]))
            if probs and max(probs) < float(min_word_prob) and not any(
                isinstance(w, dict) and w.get("probability") is None for w in words
            ):
                # Only enforce when we actually have probabilities.
                if all(p < float(min_word_prob) for p in probs):
                    item["reject"] = f"all word p < {min_word_prob}"
                    rejected.append(item)
                    continue
        kept.append(item)
    return kept, rejected


def good_word_count(segments: list[dict], min_prob: float = 0.4) -> int:
    """Count non-junk words with probability ≥ ``min_prob`` (or unknown)."""
    n = 0
    for seg in segments:
        text = str(seg.get("text") or seg.get("text_ru") or "")
        if is_junk(text):
            continue
        words = seg.get("words") or seg.get("words_ru") or []
        if not words:
            # Fall back to whitespace tokens when Whisper omitted word timestamps.
            for tok in text.split():
                if tok.strip() and not is_junk(tok):
                    n += 1
            continue
        for w in words:
            if isinstance(w, dict):
                token = str(w.get("word") or w.get("text") or "")
                prob = w.get("probability")
            elif isinstance(w, (list, tuple)) and len(w) >= 3:
                token = str(w[2])
                prob = w[3] if len(w) > 3 else None
            else:
                continue
            if is_junk(token):
                continue
            if prob is None or float(prob) >= min_prob:
                n += 1
    return n
