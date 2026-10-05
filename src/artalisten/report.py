"""Render a speaker-attributed bilingual transcript."""

from __future__ import annotations


def format_timestamp(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    total = int(round(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def format_duration(seconds: float) -> str:
    return format_timestamp(seconds)


def render_markdown(manifest: dict, utterances: list[dict]) -> str:
    source = manifest.get("source_name") or "recording"
    processed = float(manifest.get("processed_seconds") or 0.0)
    full = float(manifest.get("source_seconds") or processed)
    prefix = processed + 0.5 < full
    lines = [
        f"# {source}",
        "",
        "Speaker-attributed transcript. Sex labels are estimated from pitch and can be wrong.",
        "",
        f"- Processed: {format_duration(processed)} of {format_duration(full)}",
    ]
    if prefix:
        lines.append(
            "- This is a prefix of the recording. Speaker profiles were learned on that prefix only."
        )
    else:
        lines.append(
            "- Speaker profiles were learned on the full processed recording before the transcript was written."
        )
    lines.append(f"- Separation: {manifest.get('separator', 'HTDemucs')}")
    lines.append(f"- Enhancement: {manifest.get('enhancer', 'DeepFilterNet3')}")
    lines.append(f"- Speakers: {manifest.get('speaker_method', 'global clustering')}")
    lines.append(f"- Transcript: {manifest.get('asr', 'faster-whisper large-v3')}")
    lines.append(f"- Translation: {manifest.get('translation', 'Whisper translate task')}")
    lines.append("")
    lines.append("## Speakers")
    lines.append("")
    speakers = manifest.get("speakers") or []
    if not speakers:
        lines.append("No speaker profile was saved.")
    else:
        lines.append("| Label | Median F0 (Hz) | Speech |")
        lines.append("| --- | --- | --- |")
        for speaker in speakers:
            f0 = speaker.get("median_f0_hz")
            f0_text = "n/a" if f0 is None else f"{float(f0):.0f}"
            lines.append(
                f"| {speaker.get('label')} | {f0_text} | {format_duration(float(speaker.get('speech_seconds') or 0))} |"
            )
    lines.append("")
    lines.append("## Conversation")
    lines.append("")
    if not utterances:
        lines.append("No speech was transcribed. Nothing was invented.")
        lines.append("")
        return "\n".join(lines)

    flagged = 0
    for utt in utterances:
        start = format_timestamp(float(utt["start"]))
        end = format_timestamp(float(utt["end"]))
        speaker = utt.get("speaker") or "unknown"
        lines.append(f"### {start} – {end} · {speaker}")
        lines.append("")
        ru = (utt.get("text_ru") or "").strip()
        en = (utt.get("text_en") or "").strip()
        lines.append(f"**Russian:** {ru or '—'}")
        lines.append("")
        lines.append(f"**English:** {en or '—'}")
        lines.append("")
        if utt.get("likely_english_lyric_bleed"):
            flagged += 1
            lines.append("**Flag:** likely English lyric bleed")
            lines.append("")
    lines.append(f"Flagged spans: {flagged}")
    lines.append("")
    return "\n".join(lines)
