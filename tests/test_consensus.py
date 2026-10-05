from artalisten.consensus import consensus_phrases, final_only, soft_overlap


def test_soft_overlap():
    assert soft_overlap("Проверим себя, может, вернулись", "Проверим себя может вернулись")
    assert not soft_overlap("Играет музыка", "Я пока отдыхаю")


def test_consensus_requires_two_variants():
    variants = {
        "A": [{"start": 10.0, "end": 12.0, "text": "Проверим себя, может, вернулись"}],
        "E": [{"start": 10.1, "end": 12.1, "text": "Проверим себя, может, вернулись."}],
        "F": [{"start": 30.0, "end": 31.0, "text": "одинокий вариант только здесь"}],
    }
    groups = consensus_phrases(variants, min_variants=2)
    strong = final_only(groups, min_variants=2)
    assert any("Проверим" in g["text"] for g in strong)
    assert all(g["support"] >= 2 for g in strong)
    # The single-variant phrase must be weak / excluded from final
    assert not any("одинокий" in g["text"] for g in strong)


def test_short_russian_phrase_consensus():
    variants = {
        "E": [{"start": 90.8, "end": 92.1, "text": "А я не отдыхаю."}],
        "F": [{"start": 90.4, "end": 92.1, "text": "А я не отдыхаю."}],
        "G": [{"start": 90.0, "end": 92.1, "text": "А я не отдыхаю."}],
    }
    strong = final_only(consensus_phrases(variants, min_variants=2), min_variants=2)
    assert any("отдыхаю" in g["text"] for g in strong)
