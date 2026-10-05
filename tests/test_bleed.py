from artalisten.bleed import likely_english_lyric


def test_russian_conversation_is_not_flagged():
    text = "Привет, как дела сегодня вечером в этом кафе"
    assert likely_english_lyric(text) is False


def test_english_lyric_is_flagged():
    assert likely_english_lyric("I love you baby oh yeah") is True


def test_short_latin_noise_is_not_enough():
    assert likely_english_lyric("ok") is False
