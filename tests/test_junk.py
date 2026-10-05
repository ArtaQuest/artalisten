from artalisten.junk import filter_segments, good_word_count, is_junk


def test_rejects_subtitle_credits():
    assert is_junk("Субтитры сделал DimaTorzok")
    assert is_junk("Субтитры подогнал „Симон\"")
    assert is_junk("Продолжение следует...")


def test_rejects_music_captions_and_uk_thanks():
    assert is_junk("Играет музыка.")
    assert is_junk("Девушки отдыхают...")
    assert is_junk("Дякую за перегляд!")
    assert is_junk("Звучить музика")
    assert is_junk("Thanks for watching")


def test_rejects_english_loops_and_stsq():
    assert is_junk("I don't know what to say.")
    assert is_junk("I'm going to start with a simple one")
    assert is_junk("I love you so much")
    assert is_junk("StSq3 3.30")
    assert is_junk("Let's do it")


def test_keeps_real_russian():
    assert not is_junk("Проверим себя, может, вернулись.")
    assert not is_junk("Я пока отдыхаю.")
    assert not is_junk("фамилии разные")


def test_filter_segments_and_good_words():
    segs = [
        {"text": "Проверим себя", "words": [{"word": "Проверим", "probability": 0.8}, {"word": "себя", "probability": 0.7}]},
        {"text": "Субтитры сделал DimaTorzok", "words": [{"word": "Субтитры", "probability": 0.9}]},
        {"text": "шум", "words": [{"word": "шум", "probability": 0.1}]},
    ]
    kept, rejected = filter_segments(segs, min_word_prob=0.4)
    assert len(kept) == 1
    assert kept[0]["text"].startswith("Проверим")
    assert any("DimaTorzok" in (r["text"]) for r in rejected)
    assert good_word_count(kept, 0.4) == 2


def test_rejects_split_continuation_fragments():
    assert is_junk("Продолжение")
    assert is_junk("следует...")
    assert is_junk("следует")
