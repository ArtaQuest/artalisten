"""Command line for long-form noisy speech."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="artalisten",
        description=(
            "Listen to a long, noisy recording: separate the background, enhance the "
            "speech, learn speakers from the whole file, then transcribe and translate."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)
    process = sub.add_parser(
        "process",
        help="Separate, enhance, learn speakers, transcribe, and translate",
        description=(
            "Stages run in order on one recording. Speaker embeddings are clustered "
            "once across the whole file before any transcript is written. "
            "Weights, audio, and transcripts are local and are not part of the install."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "example:\n"
            "  artalisten process ~/Downloads/anna+brother.m4a --translate en\n"
        ),
    )
    process.add_argument(
        "audio",
        type=Path,
        help="Recording (wav, mp3, m4a, flac, ogg, aac, mp4, or anything ffmpeg can read)",
    )
    process.add_argument(
        "--translate",
        default="en",
        help="Translation language. Only en is supported (Whisper translate task). Default: en",
    )
    process.add_argument(
        "--language",
        default="ru",
        help="Spoken language passed to Whisper. Default: ru",
    )
    process.add_argument(
        "--speakers",
        type=int,
        default=2,
        help="Speaker count prior for global clustering. Default: 2",
    )
    process.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "mps", "cuda"),
        help="Device for separation, enhancement, and embeddings. Whisper uses CUDA when --device cuda, else CPU. Default: auto",
    )
    process.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory. Default: ./runs/<recording name>",
    )
    process.add_argument(
        "--max-seconds",
        type=float,
        default=None,
        help="Process only a prefix of this many seconds. Speaker learning uses that prefix.",
    )
    process.add_argument(
        "--whisper-model",
        default="large-v3",
        help="faster-whisper model name. Default: large-v3",
    )
    process.add_argument(
        "--compute-type",
        default="int8",
        help="CTranslate2 compute type. Default: int8",
    )
    process.add_argument(
        "--beam-size",
        type=int,
        default=5,
        help="Whisper beam size. Default: 5",
    )
    process.add_argument(
        "--shifts",
        type=int,
        default=0,
        help="HTDemucs random-shift passes. 0 is the memory-safe default.",
    )
    process.add_argument(
        "--force",
        action="store_true",
        help="Recompute every stage even if outputs already exist",
    )
    process.add_argument(
        "--save-profile",
        type=Path,
        default=None,
        help="Write the fitted speaker profile here. Only the file being processed is fit.",
    )
    process.add_argument(
        "--profile",
        type=Path,
        default=None,
        help="Assign speakers from this frozen profile. Do not fit or adapt on this recording.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command != "process":
        parser.print_help()
        return 2
    if args.translate.lower() != "en":
        print("only --translate en is supported", file=sys.stderr)
        return 2
    if args.speakers < 1:
        print("--speakers must be at least 1", file=sys.stderr)
        return 2
    audio = Path(args.audio).expanduser()
    if not audio.is_file():
        print(f"audio file not found: {audio}", file=sys.stderr)
        return 2
    if args.profile is not None and not Path(args.profile).expanduser().is_file():
        print(f"speaker profile not found: {args.profile}", file=sys.stderr)
        return 2
    from artalisten.pipeline import process

    return process(args)


if __name__ == "__main__":
    raise SystemExit(main())
