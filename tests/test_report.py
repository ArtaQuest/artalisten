import subprocess
import sys

from artalisten.report import render_markdown


def test_transcript_has_both_languages_and_the_flag():
    manifest = {
        "source_name": "anna+brother.m4a",
        "processed_seconds": 30.0,
        "source_seconds": 30.0,
        "separator": "HTDemucs (htdemucs) on cpu",
        "enhancer": "DeepFilterNet3 on cpu",
        "speaker_method": "global-ecapa-agglomerative",
        "asr": "faster-whisper large-v3 int8 on cpu, language ru",
        "translation": "faster-whisper large-v3 task=translate to en",
        "speakers": [
            {
                "label": "man (estimated)",
                "median_f0_hz": 112.0,
                "speech_seconds": 12.0,
            }
        ],
    }
    utterances = [
        {
            "start": 1.2,
            "end": 3.4,
            "speaker": "man (estimated)",
            "text_ru": "Привет",
            "text_en": "Hello",
            "likely_english_lyric_bleed": False,
        },
        {
            "start": 4.0,
            "end": 6.0,
            "speaker": "woman (estimated)",
            "text_ru": "I love you baby",
            "text_en": "I love you baby",
            "likely_english_lyric_bleed": True,
        },
    ]
    text = render_markdown(manifest, utterances)
    assert "Привет" in text
    assert "Hello" in text
    assert "man (estimated)" in text
    assert "likely English lyric bleed" in text
    assert "full processed recording" in text


def test_prefix_is_called_a_prefix():
    text = render_markdown(
        {
            "source_name": "clip.m4a",
            "processed_seconds": 60.0,
            "source_seconds": 120.0,
            "speakers": [],
        },
        [],
    )
    assert "prefix" in text
    assert "Nothing was invented" in text


def test_cli_help_lists_process_and_translate():
    module = subprocess.run(
        [sys.executable, "-m", "artalisten", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert module.returncode == 0
    assert "process" in module.stdout
    process = subprocess.run(
        [sys.executable, "-m", "artalisten", "process", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert process.returncode == 0
    assert "--translate" in process.stdout
    assert "anna+brother" in process.stdout
