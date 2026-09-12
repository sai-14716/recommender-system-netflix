import numpy as np
import pandas as pd

from src.data import chronological_split
from src.metrics import ranking_metrics


def sample_ratings():
    rows = []
    for user in range(3):
        for index in range(5):
            rows.append({"userId": user, "movieId": index, "rating": float((index % 5) + 1), "timestamp": user * 100 + index})
    return pd.DataFrame(rows)


def test_chronological_split_has_no_future_training_rows():
    split = chronological_split(sample_ratings(), validation_count=1, test_count=1)
    for user in split.train.userId.unique():
        train_max = split.train.loc[split.train.userId == user, "timestamp"].max()
        validation_min = split.validation.loc[split.validation.userId == user, "timestamp"].min()
        validation_max = split.validation.loc[split.validation.userId == user, "timestamp"].max()
        test_min = split.test.loc[split.test.userId == user, "timestamp"].min()
        assert train_max < validation_min < test_min
    assert set(split.train.userId) == set(split.test.userId)


def test_ranking_metrics_counts_hits():
    recommended = {1: [4, 2, 3], 2: [7, 8]}
    relevant = {1: {2, 9}, 2: {7}}
    metrics = ranking_metrics(recommended, relevant, ks=(2,))
    assert metrics["recall@2"] == 0.75
    assert 0 < metrics["ndcg@2"] <= 1
