from dataclasses import dataclass

import pandas as pd


REQUIRED_RATING_COLUMNS = {"userId", "movieId", "rating", "timestamp"}


@dataclass(frozen=True)
class TemporalSplit:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    validation_cutoff: int
    test_cutoff: int
    evaluation_users: tuple[int, ...]


def load_ratings(path: str) -> pd.DataFrame:
    ratings = pd.read_csv(path)
    missing_columns = REQUIRED_RATING_COLUMNS.difference(ratings.columns)
    if missing_columns:
        raise ValueError(f"ratings file is missing columns: {sorted(missing_columns)}")
    return ratings.sort_values("timestamp", kind="stable").reset_index(drop=True)


def build_temporal_split(
    ratings: pd.DataFrame,
    *,
    validation_quantile: float = 0.8,
    test_quantile: float = 0.9,
    positive_threshold: float = 4.0,
    min_history: int = 5,
) -> TemporalSplit:
    missing_columns = REQUIRED_RATING_COLUMNS.difference(ratings.columns)
    if missing_columns:
        raise ValueError(f"ratings are missing columns: {sorted(missing_columns)}")
    if not 0 < validation_quantile < test_quantile < 1:
        raise ValueError("quantiles must satisfy 0 < validation < test < 1")

    ordered = ratings.sort_values("timestamp", kind="stable").reset_index(drop=True)
    validation_cutoff = int(ordered["timestamp"].quantile(validation_quantile))
    test_cutoff = int(ordered["timestamp"].quantile(test_quantile))
    if validation_cutoff >= test_cutoff:
        raise ValueError("temporal cutoffs collapse to the same timestamp")

    train = ordered[ordered["timestamp"] < validation_cutoff].copy()
    validation = ordered[
        (ordered["timestamp"] >= validation_cutoff)
        & (ordered["timestamp"] < test_cutoff)
    ].copy()
    test = ordered[ordered["timestamp"] >= test_cutoff].copy()

    train_positive_counts = (
        train[train["rating"] >= positive_threshold].groupby("userId").size()
    )
    test_positive_users = set(
        test.loc[test["rating"] >= positive_threshold, "userId"].astype(int)
    )
    evaluation_users = tuple(
        sorted(
            int(user_id)
            for user_id, count in train_positive_counts.items()
            if count >= min_history and int(user_id) in test_positive_users
        )
    )

    assert_temporal_integrity(train, validation, test, validation_cutoff, test_cutoff)
    return TemporalSplit(
        train=train,
        validation=validation,
        test=test,
        validation_cutoff=validation_cutoff,
        test_cutoff=test_cutoff,
        evaluation_users=evaluation_users,
    )


def assert_temporal_integrity(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    validation_cutoff: int,
    test_cutoff: int,
) -> None:
    if not train.empty and int(train["timestamp"].max()) >= validation_cutoff:
        raise AssertionError("training data contains validation-period events")
    if not validation.empty:
        if int(validation["timestamp"].min()) < validation_cutoff:
            raise AssertionError("validation data contains training-period events")
        if int(validation["timestamp"].max()) >= test_cutoff:
            raise AssertionError("validation data contains test-period events")
    if not test.empty and int(test["timestamp"].min()) < test_cutoff:
        raise AssertionError("test data contains pre-test events")


def positive_items_by_user(
    ratings: pd.DataFrame, positive_threshold: float = 4.0
) -> dict[int, set[int]]:
    positives = ratings[ratings["rating"] >= positive_threshold]
    return {
        int(user_id): set(group["movieId"].astype(int))
        for user_id, group in positives.groupby("userId")
    }
