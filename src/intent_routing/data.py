from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass
class DataSplits:
    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame


def load_banking77_csv(train_path: str | Path, test_path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(train_path, header=None, names=["text", "label"])
    test = pd.read_csv(test_path, header=None, names=["text", "label"])
    for name, df in {"train": train, "test": test}.items():
        if df.isna().any().any():
            raise ValueError(f"{name} contains missing values")
        if not df["text"].map(lambda x: isinstance(x, str) and len(x.strip()) > 0).all():
            raise ValueError(f"{name} contains empty text")
    return train, test


def make_train_valid(train: pd.DataFrame, valid_size: float = 0.15, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    tr, va = train_test_split(
        train,
        test_size=valid_size,
        random_state=seed,
        stratify=train["label"],
    )
    return tr.reset_index(drop=True), va.reset_index(drop=True)


def prepare_splits(train_path: str | Path, test_path: str | Path, valid_size: float = 0.15, seed: int = 42) -> DataSplits:
    train, test = load_banking77_csv(train_path, test_path)
    tr, va = make_train_valid(train, valid_size, seed)
    return DataSplits(tr, va, test.reset_index(drop=True))
