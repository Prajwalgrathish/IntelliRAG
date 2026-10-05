"""Data loading, splitting and embedding for the intent classifier."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence, Union

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from src.embeddings.transformer_encoder import TextEncoder
from src.nlp.preprocessing import preprocess_query

REQUIRED_COLUMNS = ("text", "label")


def load_intent_dataframe(path: Union[str, Path]) -> pd.DataFrame:
    """Read a CSV with ``text`` and ``label`` columns, dropping empty rows."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Training data not found: {path}")
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Training CSV is missing required column(s): {missing}")
    df = df[list(REQUIRED_COLUMNS)].dropna()
    df["text"] = df["text"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip()
    df = df[(df["text"] != "") & (df["label"] != "")].reset_index(drop=True)
    if df["label"].nunique() < 2:
        raise ValueError("Training data must contain at least two distinct labels.")
    return df


def build_label_list(df: pd.DataFrame) -> list[str]:
    return sorted(df["label"].unique().tolist())


def split_dataframe(
    df: pd.DataFrame, val_size: float, test_size: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified train/validation/test split."""
    if val_size <= 0 or test_size <= 0 or val_size + test_size >= 1:
        raise ValueError("val_size and test_size must be > 0 and sum to < 1.")
    try:
        train_val, test = train_test_split(df, test_size=test_size, stratify=df["label"], random_state=seed)
        rel_val = val_size / (1 - test_size)
        train, val = train_test_split(train_val, test_size=rel_val, stratify=train_val["label"], random_state=seed)
    except ValueError as exc:
        raise ValueError(
            f"Could not create a stratified split (each label needs enough examples for every split): {exc}"
        ) from exc
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def embed_texts(encoder: TextEncoder, texts: Sequence[str], batch_size: int = 32) -> np.ndarray:
    """Embed preprocessed queries with the (frozen) encoder."""
    return encoder.encode_queries([preprocess_query(t) for t in texts], batch_size=batch_size)


class EmbeddingDataset(Dataset):
    """Pre-computed embeddings + integer targets."""

    def __init__(self, embeddings: np.ndarray, targets: np.ndarray) -> None:
        if len(embeddings) != len(targets):
            raise ValueError("embeddings and targets must have the same length")
        self.x = torch.as_tensor(embeddings, dtype=torch.float32)
        self.y = torch.as_tensor(targets, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]
