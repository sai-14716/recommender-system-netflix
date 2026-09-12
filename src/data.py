from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED_RATING_COLUMNS = {"userId", "movieId", "rating", "timestamp"}


@dataclass
class ChronologicalSplit:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def load_ratings(path: str | Path, max_rows: int | None = None) -> pd.DataFrame:
    frame = pd.read_csv(path, nrows=max_rows, usecols=sorted(EXPECTED_RATING_COLUMNS))
    missing = EXPECTED_RATING_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"ratings file is missing columns: {sorted(missing)}")
    if frame[["userId", "movieId", "timestamp"]].isna().any().any():
        raise ValueError("userId, movieId, and timestamp cannot be null")
    if not frame["rating"].between(0.5, 5.0).all():
        raise ValueError("ratings must be between 0.5 and 5.0")
    return frame.astype({"userId": "int64", "movieId": "int64", "rating": "float32", "timestamp": "int64"})


def chronological_split(ratings: pd.DataFrame, validation_count: int = 1, test_count: int = 1) -> ChronologicalSplit:
    if validation_count < 1 or test_count < 1:
        raise ValueError("validation_count and test_count must be positive")
    ordered = ratings.sort_values(["userId", "timestamp", "movieId"], kind="mergesort")
    group_sizes = ordered.groupby("userId", sort=False).size()
    minimum = validation_count + test_count + 1
    eligible = set(group_sizes[group_sizes >= minimum].index)
    eligible_frame = ordered[ordered["userId"].isin(eligible)]
    train_parts, validation_parts, test_parts = [], [], []
    for _, group in eligible_frame.groupby("userId", sort=False):
        train_parts.append(group.iloc[: -validation_count - test_count])
        validation_parts.append(group.iloc[-validation_count - test_count : -test_count])
        test_parts.append(group.iloc[-test_count:])
    if not train_parts:
        raise ValueError("no users have enough ratings for the requested split")
    return ChronologicalSplit(
        train=pd.concat(train_parts, ignore_index=True),
        validation=pd.concat(validation_parts, ignore_index=True),
        test=pd.concat(test_parts, ignore_index=True),
    )


def make_id_maps(train: pd.DataFrame) -> tuple[dict[int, int], dict[int, int]]:
    users = {value: index for index, value in enumerate(train["userId"].unique())}
    items = {value: index for index, value in enumerate(train["movieId"].unique())}
    return users, items


def encode_frame(frame: pd.DataFrame, user_map: dict[int, int], item_map: dict[int, int]) -> pd.DataFrame:
    encoded = frame.copy()
    encoded["userIndex"] = encoded["userId"].map(user_map)
    encoded["itemIndex"] = encoded["movieId"].map(item_map)
    return encoded.dropna(subset=["userIndex", "itemIndex"]).astype({"userIndex": "int32", "itemIndex": "int32"})