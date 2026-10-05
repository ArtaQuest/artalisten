"""Command line for long-form noisy speech and local pass5 recovery."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="artalisten",
        description=(
            "Local transcription for extreme low-SNR cafe / Voice Memo recordings. "
            "Mac-first: pass5 multi-variant recovery needs no Kaggle."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    process = sub.add_parser(
        "process",
        help="Legacy single-path: Demucs → DeepFilterNet → ECAPA → Whisper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="example:\n  artalisten process ~/Downloads/anna+brother.m4a --translate en\n",
    )
    _add_process_args(process)

    recover = sub.add_parser(
        "recover",
        help="Pass5 / v5 multi-variant recovery on one file (Mac-local)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "example:\n"
            "  caffeinate -i artalisten recover ~/Downloads/'Voice 261005_104834.m4a' \\\n"
            "    --profile profiles/anna.json --out runs/pass5-local/104834 \\\n"
            "    --deliverables deliverables/memo-local/\n"
        ),
    )
    recover.add_argument("audio", type=Path, help="Recording (m4a/wav/…)")
    _add_recover_shared(recover)
    recover.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Run directory. Default: ./runs/pass5-<name>/",
    )
    recover.add_argument(
        "--deliverables",
        type=Path,
        default=None,
        help="Optional deliverables folder (transcripts/audio/json + INDEX update)",
    )
    recover.add_argument(
        "--skip-sepformer",
        action="store_true",
        help="Skip SpeechBrain sepformer (variant B = A). Saves memory on 16 GB.",
    )

    memo = sub.add_parser(
        "memo",
        help="Batch pass5 on Voice Memo m4a files in an inbox folder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "example:\n"
            "  caffeinate -i artalisten memo --inbox ~/Downloads --glob 'Voice*.m4a' \\\n"
            "    --profile profiles/anna.json --deliverables deliverables/memo-local/\n"
        ),
    )
    memo.add_argument(
        "--inbox",
        type=Path,
        default=Path.home() / "Downloads",
        help="Folder of Voice Memo exports. Default: ~/Downloads",
    )
    memo.add_argument(
        "--glob",
        dest="glob_pattern",
        default="*.m4a",
        help="Glob under inbox. Default: *.m4a (Voice Memos + memo-*.m4a in project inbox)",
    )
    memo.add_argument(
        "--cafe-train",
        type=Path,
        default=None,
        help="Optional cafe train file (fit speakers only here)",
    )
    memo.add_argument(
        "--cafe-test",
        type=Path,
        default=None,
        help="Optional cafe held-out test (frozen; never fit)",
    )
    memo.add_argument(
        "--out",
        type=Path,
        default=Path("runs/pass5-memo"),
        help="Parent run directory. Default: ./runs/pass5-memo",
    )
    memo.add_argument(
        "--deliverables",
        type=Path,
        default=Path("deliverables/memo-local"),
        help="Deliverables root. Default: ./deliverables/memo-local",
    )
    _add_recover_shared(memo)
    memo.add_argument(
        "--skip-sepformer",
        action="store_true",
        help="Skip SpeechBrain sepformer for all Voice files",
    )
    memo.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most N Voice memos (after sorting)",
    )

    return parser


def _add_process_args(process: argparse.ArgumentParser) -> None:
    process.add_argument(
        "audio",
        type=Path,
        help="Recording (wav, mp3, m4a, flac, ogg, aac, mp4, or anything ffmpeg can read)",
    )
    process.add_argument("--translate", default="en", help="Only en supported. Default: en")
    process.add_argument("--language", default="ru", help="Spoken language. Default: ru")
    process.add_argument("--speakers", type=int, default=2, help="Speaker prior. Default: 2")
    process.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "mps", "cuda"),
        help="Device for separation/enhancement/embeddings. Default: auto",
    )
    process.add_argument("--out", type=Path, default=None, help="Output directory")
    process.add_argument("--max-seconds", type=float, default=None, help="Process a prefix only")
    process.add_argument("--whisper-model", default="large-v3")
    process.add_argument("--compute-type", default="int8")
    process.add_argument("--beam-size", type=int, default=5)
    process.add_argument("--shifts", type=int, default=0, help="HTDemucs shifts. Default: 0")
    process.add_argument("--force", action="store_true")
    process.add_argument("--save-profile", type=Path, default=None)
    process.add_argument("--profile", type=Path, default=None)


def _add_recover_shared(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--profile",
        type=Path,
        default=None,
        help="Anna ECAPA profile JSON (cosine match). Example: profiles/anna.json",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "mps", "cuda"),
        help="Device for Demucs/embeddings (Whisper stays CPU int8). Default: auto",
    )
    parser.add_argument("--whisper-model", default="large-v3")
    parser.add_argument(
        "--compute-type",
        default="int8",
        help="faster-whisper compute type on CPU. Default: int8",
    )
    parser.add_argument(
        "--shifts",
        type=int,
        default=1,
        help="HTDemucs shifts for pass5. Default: 1 (memory-safer than 2)",
    )
    parser.add_argument("--force", action="store_true", help="Recompute even if outputs exist")


def main(argv: list[str] | None = None) -> int:
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "process":
        return _cmd_process(args)
    if args.command == "recover":
        return _cmd_recover(args)
    if args.command == "memo":
        return _cmd_memo(args)
    parser.print_help()
    return 2


def _cmd_process(args) -> int:
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


def _cmd_recover(args) -> int:
    audio = Path(args.audio).expanduser()
    if not audio.is_file():
        print(f"audio file not found: {audio}", file=sys.stderr)
        return 2
    if args.profile is not None and not Path(args.profile).expanduser().is_file():
        print(f"profile not found: {args.profile}", file=sys.stderr)
        return 2
    from artalisten.recover import _safe_id, recover_file, write_index

    out = Path(args.out).expanduser() if args.out else Path("runs") / f"pass5-{_safe_id(audio)}"
    deliv = Path(args.deliverables).expanduser() if args.deliverables else None
    result = recover_file(
        audio,
        out,
        profile_path=Path(args.profile).expanduser() if args.profile else None,
        device=args.device,
        whisper_model=args.whisper_model,
        compute_type=args.compute_type,
        shifts=args.shifts,
        force=args.force,
        deliverable_dir=deliv,
        skip_sepformer=bool(args.skip_sepformer),
    )
    print(f"transcript: {result.transcript_md}")
    print(f"stem: {result.stem_wav}")
    print(f"json: {result.result_json}")
    if deliv is not None:
        from artalisten.recover import _merge_index_row

        row = {
            "source": audio.name,
            "seconds": f"{result.seconds:.1f}s",
            "speakers": result.speakers_present,
            "consensus": result.consensus_count,
            "transcript": f"transcripts/{_safe_id(audio)}-transcript.md",
            "stem": f"audio/{_safe_id(audio)}-anna.wav",
        }
        _merge_index_row(deliv, row)
        print(f"deliverables: {deliv}")
    return 0


def _cmd_memo(args) -> int:
    from artalisten.recover import (
        _safe_id,
        find_voice_memos,
        recover_cafe_pair,
        recover_file,
        write_index,
    )

    inbox = Path(args.inbox).expanduser()
    files = find_voice_memos(inbox, args.glob_pattern)
    if args.limit is not None:
        files = files[: max(0, int(args.limit))]
    if not files and not args.cafe_train:
        print(f"no files matching {args.glob_pattern!r} in {inbox}", file=sys.stderr)
        return 2
    if args.profile is not None and not Path(args.profile).expanduser().is_file():
        print(f"profile not found: {args.profile}", file=sys.stderr)
        return 2

    out_root = Path(args.out).expanduser()
    deliv = Path(args.deliverables).expanduser()
    deliv.mkdir(parents=True, exist_ok=True)
    profile = Path(args.profile).expanduser() if args.profile else None
    rows: list[dict] = []

    for audio in files:
        print(f"== recover {audio.name} ==", flush=True)
        run_dir = out_root / _safe_id(audio)
        result = recover_file(
            audio,
            run_dir,
            profile_path=profile,
            device=args.device,
            whisper_model=args.whisper_model,
            compute_type=args.compute_type,
            shifts=args.shifts,
            force=args.force,
            deliverable_dir=deliv,
            skip_sepformer=bool(args.skip_sepformer),
        )
        rows.append(
            {
                "source": audio.name,
                "seconds": f"{result.seconds:.1f}s",
                "speakers": result.speakers_present,
                "consensus": result.consensus_count,
                "transcript": f"transcripts/{_safe_id(audio)}-transcript.md",
                "stem": f"audio/{_safe_id(audio)}-anna.wav",
            }
        )
        print(
            f"done {audio.name}: consensus={result.consensus_count} -> {result.transcript_md}",
            flush=True,
        )

    if args.cafe_train is not None:
        train = Path(args.cafe_train).expanduser()
        test = Path(args.cafe_test).expanduser() if args.cafe_test else None
        if not train.is_file():
            print(f"cafe train not found: {train}", file=sys.stderr)
            return 2
        if test is None or not test.is_file():
            print("cafe mode needs --cafe-test as well (frozen; never fit)", file=sys.stderr)
            return 2
        print(f"== cafe pair {train.name} + {test.name} ==", flush=True)
        proof = recover_cafe_pair(
            train,
            test,
            out_root / "cafe",
            profile_path=profile,
            device=args.device,
            whisper_model=args.whisper_model,
            compute_type=args.compute_type,
            shifts=args.shifts,
            force=args.force,
            deliverable_dir=deliv,
        )
        rows.append(
            {
                "source": f"{train.name} + {test.name}",
                "seconds": "cafe pair",
                "speakers": "fit train / freeze test",
                "consensus": "see cafe/",
                "transcript": "cafe/train_transcript.md",
                "stem": "cafe/train_anna_stem.wav",
            }
        )
        print(f"cafe proof: {proof}", flush=True)

    index = write_index(deliv, rows)
    print(f"INDEX: {index}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
