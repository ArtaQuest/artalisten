"""Cross-variant consensus for low-SNR transcripts.

A content phrase enters Final only when ≥2 independent cleaning variants agree
on it. Single-variant hits stay marked weak.
"""

from __future__ import annotations

import re
from collections import defaultdict

from artalisten.junk import is_junk

_WORD = re.compile(r"[0-9A-Za-zА-Яа-яЁёЇїІіЄєҐґ']+", re.UNICODE)


def normalize_token(token: str) -> str:
    t = (token or "").casefold().strip().strip(".,!?;:«»\"'…—–-")
    return t


def tokenize(text: str) -> list[str]:
    return [normalize_token(m.group(0)) for m in _WORD.finditer(text or "") if normalize_token(m.group(0))]


def soft_overlap(a: str, b: str, *, min_shared: int = 2) -> bool:
    """True when two phrases share enough content tokens (order-free)."""
    ta = {t for t in tokenize(a) if len(t) >= 3 and not is_junk(t)}
    tb = {t for t in tokenize(b) if len(t) >= 3 and not is_junk(t)}
    if not ta or not tb:
        return False
    return len(ta & tb) >= min_shared


def consensus_phrases(
    variant_segments: dict[str, list[dict]],
    *,
    text_key: str = "text",
    time_slack: float = 2.5,
    min_variants: int = 2,
) -> list[dict]:
    """Find phrases supported by ≥ ``min_variants`` independent sources.

    ``variant_segments`` maps variant name → list of segment dicts with
    ``start``, ``end``, and ``text`` (or ``text_key``).

    Returns consensus rows::

        {
          "start", "end", "text", "variants": [...],
          "support": int, "weak": bool, "texts_by_variant": {...}
        }
    """
    # Flatten to timed hits.
    hits: list[dict] = []
    for variant, segs in variant_segments.items():
        for seg in segs:
            text = str(seg.get(text_key) or seg.get("text_ru") or "").strip()
            if not text or is_junk(text):
                continue
            tokens = [t for t in tokenize(text) if len(t) >= 2]
            if len(tokens) < 1:
                continue
            hits.append(
                {
                    "variant": variant,
                    "start": float(seg.get("start") or 0.0),
                    "end": float(seg.get("end") or seg.get("start") or 0.0),
                    "text": text,
                    "tokens": tokens,
                }
            )
    hits.sort(key=lambda h: (h["start"], h["end"]))

    used = [False] * len(hits)
    groups: list[dict] = []
    for i, hit in enumerate(hits):
        if used[i]:
            continue
        members = [hit]
        used[i] = True
        for j in range(i + 1, len(hits)):
            if used[j]:
                continue
            other = hits[j]
            if other["start"] - hit["end"] > time_slack and other["start"] - members[-1]["end"] > time_slack:
                # Far in the future; later hits may still match if we keep scanning near window.
                if other["start"] - hit["start"] > time_slack * 3:
                    break
            mid_i = 0.5 * (hit["start"] + hit["end"])
            mid_j = 0.5 * (other["start"] + other["end"])
            if abs(mid_i - mid_j) > time_slack:
                continue
            if soft_overlap(hit["text"], other["text"]) or hit["variant"] == other["variant"]:
                if hit["variant"] == other["variant"] and not soft_overlap(hit["text"], other["text"]):
                    continue
                # Require either overlap or same rough wording via shared tokens
                if soft_overlap(hit["text"], other["text"], min_shared=1) or (
                    len(set(hit["tokens"]) & set(other["tokens"])) >= 1 and abs(mid_i - mid_j) <= 1.5
                ):
                    members.append(other)
                    used[j] = True
        by_variant: dict[str, str] = {}
        for m in members:
            # Prefer longer text per variant
            prev = by_variant.get(m["variant"], "")
            if len(m["text"]) >= len(prev):
                by_variant[m["variant"]] = m["text"]
        support = len(by_variant)
        starts = [m["start"] for m in members]
        ends = [m["end"] for m in members]
        # Represent text by the most-common-ish longest among agreeing variants
        rep = max(by_variant.values(), key=len) if by_variant else hit["text"]
        groups.append(
            {
                "start": min(starts),
                "end": max(ends),
                "text": rep,
                "variants": sorted(by_variant),
                "support": support,
                "weak": support < min_variants,
                "texts_by_variant": by_variant,
            }
        )
    groups.sort(key=lambda g: (g["start"], g["end"]))
    return groups


def final_only(
    groups: list[dict],
    *,
    min_variants: int = 2,
) -> list[dict]:
    """Filter to consensus rows with enough independent support."""
    return [g for g in groups if int(g.get("support") or 0) >= min_variants and not g.get("weak")]
