"""Flag transcript spans that are probably the English song, not the conversation."""

from __future__ import annotations

import re

# Function words that show up constantly in English pop lyrics and almost never
# as a run of Latin tokens inside Russian speech.
_ENGLISH = {
    "a", "an", "and", "are", "baby", "be", "but", "can", "come", "don",
    "dont", "don't", "for", "get", "go", "gonna", "heart", "i", "i'm", "im",
    "in", "is", "it", "just", "know", "let", "like", "love", "me", "my",
    "never", "now", "of", "oh", "on", "our", "she", "so", "that", "the",
    "this", "to", "up", "wanna", "we", "what", "when", "with", "yeah", "you",
    "your",
}

_TOKEN = re.compile(r"[A-Za-z']+")


def script_counts(text: str) -> tuple[int, int]:
    latin = 0
    cyrillic = 0
    for ch in text:
        if "A" <= ch <= "Z" or "a" <= ch <= "z":
            latin += 1
        elif "\u0400" <= ch <= "\u04FF" or "\u0500" <= ch <= "\u052F":
            cyrillic += 1
    return latin, cyrillic


def likely_english_lyric(text: str, no_speech_prob: float | None = None) -> bool:
    """True when a span looks like English singing that survived separation.

    Russian conversation is Cyrillic. A distant cafe song often remains in the
    vocal stem and Whisper will write it in Latin letters. That is a flag, not
    a deletion: the line stays in the transcript.
    """
    latin, cyrillic = script_counts(text)
    letters = latin + cyrillic
    if letters == 0:
        return False
    latin_ratio = latin / letters
    tokens = [m.group(0).lower().replace("'", "") for m in _TOKEN.finditer(text)]
    english_hits = sum(1 for tok in tokens if tok in _ENGLISH or tok.replace("'", "") in _ENGLISH)

    if latin >= 8 and latin_ratio >= 0.55 and latin > cyrillic:
        return True
    if english_hits >= 3 and latin_ratio >= 0.40:
        return True
    if (
        no_speech_prob is not None
        and no_speech_prob >= 0.6
        and latin >= 4
        and latin_ratio >= 0.5
    ):
        return True
    return False
