from artalisten.align import (
    drop_overlap_words,
    fill_translations,
    group_utterances,
    iter_chunks,
)


def test_groups_words_by_speaker_and_gap():
    words = [
        {"start": 0.0, "end": 0.4, "word": " Привет", "speaker": "man (estimated)", "speaker_id": "S0"},
        {"start": 0.5, "end": 0.9, "word": " брат", "speaker": "man (estimated)", "speaker_id": "S0"},
        {"start": 2.5, "end": 3.0, "word": " Да", "speaker": "woman (estimated)", "speaker_id": "S1"},
    ]
    utterances = group_utterances(words, gap=0.8)
    assert len(utterances) == 2
    assert utterances[0]["text_ru"] == "Привет брат"
    assert utterances[0]["speaker"] == "man (estimated)"
    assert utterances[1]["text_ru"] == "Да"


def test_translation_stays_with_the_same_speaker():
    russian = group_utterances(
        [
            {
                "start": 1.0,
                "end": 2.0,
                "word": " Привет",
                "speaker": "man (estimated)",
                "speaker_id": "S0",
            }
        ]
    )
    english = [
        {"start": 1.1, "end": 1.6, "word": " Hello", "speaker": "man (estimated)", "speaker_id": "S0"},
        {"start": 1.4, "end": 1.8, "word": " yeah", "speaker": "woman (estimated)", "speaker_id": "S1"},
    ]
    filled = fill_translations(russian, english)
    man = next(item for item in filled if item["speaker"].startswith("man"))
    assert man["text_en"] == "Hello"
    woman = next(item for item in filled if item["speaker"].startswith("woman"))
    assert woman["text_en"] == "yeah"
    assert woman["text_ru"] == ""


def test_chunks_cover_the_file():
    spans = list(iter_chunks(650, 300, 3))
    assert spans[0] == (0.0, 300.0)
    assert spans[1][0] == 297.0
    assert spans[-1][1] == 650.0


def test_overlap_drops_the_left_edge():
    words = [
        {"start": 10.0, "end": 10.2, "word": " a"},
        {"start": 12.0, "end": 12.2, "word": " b"},
    ]
    kept = drop_overlap_words(words, chunk_start=10.0, overlap=3.0, is_first=False)
    assert [word["word"] for word in kept] == [" b"]
    assert drop_overlap_words(words, chunk_start=10.0, overlap=3.0, is_first=True) == words
