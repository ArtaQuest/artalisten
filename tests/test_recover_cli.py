from pathlib import Path

from artalisten.cli import build_parser
from artalisten.recover import _safe_id, find_voice_memos, render_transcript, write_index


def test_parser_has_recover_and_memo():
    parser = build_parser()
    recover = parser.parse_args(
        ["recover", "x.m4a", "--profile", "profiles/anna.json", "--skip-sepformer"]
    )
    assert recover.command == "recover"
    assert recover.skip_sepformer is True
    memo = parser.parse_args(["memo", "--inbox", "/tmp", "--glob", "Voice*.m4a", "--limit", "1"])
    assert memo.command == "memo"
    assert memo.limit == 1
    assert memo.glob_pattern == "Voice*.m4a"


def test_safe_id_strips_voice_prefix():
    assert _safe_id(Path("Voice 261005_104834.m4a")) == "261005_104834"
    assert _safe_id(Path("anna+brother.m4a")) == "anna_brother"


def test_find_voice_memos(tmp_path: Path):
    (tmp_path / "Voice 261005_104834.m4a").write_bytes(b"abc")
    (tmp_path / "notes.txt").write_text("no")
    found = find_voice_memos(tmp_path, "Voice*.m4a")
    assert len(found) == 1
    assert found[0].name.startswith("Voice")


def test_render_transcript_and_index(tmp_path: Path):
    data = {
        "source": "Voice 261005_104834.m4a",
        "seconds": 13.4,
        "speakers_present": 1,
        "pass": "pass5",
        "device": "mps",
        "two_speaker_test": {"sizes": [3, 1]},
        "stems": {"anna": "anna.wav"},
        "decodes": [
            {
                "source": "E_demucs_anna_band",
                "task": "transcribe",
                "kept": [{"start": 1.0, "end": 2.0, "text": "Спасибо"}],
            }
        ],
        "consensus_final": [
            {"start": 1.0, "end": 2.0, "text": "Спасибо", "variants": ["E", "F"], "support": 2}
        ],
    }
    md = render_transcript(data)
    assert "forced Russian" in md
    assert "Спасибо" in md
    assert "## Final" in md
    idx = write_index(
        tmp_path,
        [
            {
                "source": "Voice 261005_104834.m4a",
                "seconds": "13.4s",
                "speakers": 1,
                "consensus": 1,
                "transcript": "transcripts/x.md",
                "stem": "audio/x.wav",
            }
        ],
    )
    assert idx.is_file()
    assert "Voice 261005_104834.m4a" in idx.read_text()
