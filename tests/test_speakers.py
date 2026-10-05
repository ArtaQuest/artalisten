import numpy as np

from artalisten.speakers import cluster_global, map_sex, median_f0, nearest_centroids, turns_from_windows


def test_global_clustering_keeps_a_voice_the_same_across_the_file():
    rng = np.random.default_rng(1)

    def blob(axis: int, count: int) -> np.ndarray:
        values = rng.normal(scale=0.02, size=(count, 8))
        values[:, axis] += 1.0
        return values

    # First half is voice A then voice B. Second half swaps the order.
    # Clustering each half alone can give voice A a different id in each half.
    embeddings = np.vstack(
        [
            blob(0, 12),
            blob(1, 12),
            blob(1, 12),
            blob(0, 12),
        ]
    )
    labels = cluster_global(embeddings, 2)
    first_voice = set(labels[0:12]).union(set(labels[36:48]))
    second_voice = set(labels[12:36])
    assert first_voice != second_voice
    assert len(first_voice) == 1
    assert len(second_voice) == 1


def test_lower_pitch_is_estimated_man():
    names = map_sex({0: 110.0, 1: 210.0})
    assert names[0] == "man (estimated)"
    assert names[1] == "woman (estimated)"


def test_close_pitch_stays_uncertain():
    names = map_sex({0: 150.0, 1: 158.0})
    assert "uncertain" in names[0]
    assert "uncertain" in names[1]


def test_pitch_tracker_orders_two_sines():
    rate = 16000
    low = median_f0(_sine(120.0, rate), rate)
    high = median_f0(_sine(210.0, rate), rate)
    assert low is not None and high is not None
    assert 100 < low < 140
    assert 180 < high < 240


def test_nearest_centroid_keeps_the_training_index():
    centroids = np.eye(2, 4)
    points = np.vstack([centroids[1], centroids[0], centroids[1]])
    labels = nearest_centroids(points, centroids)
    assert list(labels) == [1, 0, 1]


def test_short_turn_is_absorbed():
    turns = turns_from_windows(
        [(0.0, 3.0, 0), (3.0, 3.3, 1), (3.3, 6.0, 0)],
        duration=6.0,
        bin_s=0.1,
        min_turn=0.6,
    )
    assert [turn["cluster"] for turn in turns] == [0]


def _sine(freq: float, sample_rate: int, seconds: float = 1.0) -> np.ndarray:
    time = np.arange(int(sample_rate * seconds)) / sample_rate
    return (0.2 * np.sin(2 * np.pi * freq * time)).astype(np.float32)
