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


def _read_banking77(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    expected = {"text", "category"}
    if set(df.columns) != expected:
        raise ValueError(f"Expected BANKING77 columns {sorted(expected)}, got {list(df.columns)}")
    df = df.rename(columns={"category": "label"})[["text", "label"]]
    if df.isna().any().any():
        raise ValueError(f"{path} contains missing values")
    if not df["text"].map(lambda x: isinstance(x, str) and len(x.strip()) > 0).all():
        raise ValueError(f"{path} contains empty text")
    return df


def load_banking77_csv(train_path: str | Path, test_path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    return _read_banking77(train_path), _read_banking77(test_path)


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
