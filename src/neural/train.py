"""Training pipeline for the intent classifier head."""
from __future__ import annotations

import copy
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.embeddings.transformer_encoder import TextEncoder
from src.neural.dataset import (
    EmbeddingDataset, build_label_list, embed_texts, load_intent_dataframe, split_dataframe,
)
from src.neural.evaluate import evaluate_embeddings
from src.neural.model import IntentClassifier, save_checkpoint
from src.utils.device import resolve_device
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TrainConfig:
    data_path: str = "data/training/intent_queries.csv"
    encoder_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    output_path: str = "models/intent_classifier.pt"
    hidden_dim: int = 256
    dropout: float = 0.2
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 16
    epochs: int = 80
    patience: int = 12
    val_size: float = 0.2
    test_size: float = 0.2
    seed: int = 42
    device: str = "auto"
    embedding_max_length: int = 64


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _run_epoch(model, loader, criterion, device, optimizer=None) -> float:
    training = optimizer is not None
    model.train(training)
    total, count = 0.0, 0
    with torch.set_grad_enabled(training):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            if training:
                optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            if training:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total += loss.item() * len(yb)
            count += len(yb)
    return total / max(count, 1)


def train_intent_classifier(config: TrainConfig, encoder: Optional[TextEncoder] = None) -> dict[str, Any]:
    """Train the MLP head on frozen transformer embeddings and save the best checkpoint.

    Returns a summary dict (paths, labels, per-epoch history, validation/test metrics). All
    numbers are computed from the data you supply; nothing is pre-filled.
    """
    set_seed(config.seed)
    device = resolve_device(config.device)
    if encoder is None:
        from src.embeddings.transformer_encoder import TransformerEncoder

        encoder = TransformerEncoder(
            config.encoder_model_name, device=config.device, max_length=config.embedding_max_length
        )

    df = load_intent_dataframe(config.data_path)
    labels = build_label_list(df)
    label_to_id = {l: i for i, l in enumerate(labels)}
    train_df, val_df, test_df = split_dataframe(df, config.val_size, config.test_size, config.seed)
    logger.info("intent_data_split", extra={"train": len(train_df), "val": len(val_df), "test": len(test_df)})

    def prepare(frame):
        return embed_texts(encoder, frame["text"].tolist()), frame["label"].map(label_to_id).to_numpy()

    x_train, y_train = prepare(train_df)
    x_val, y_val = prepare(val_df)
    x_test, y_test = prepare(test_df)

    model = IntentClassifier(
        input_dim=int(x_train.shape[1]), num_classes=len(labels),
        hidden_dim=config.hidden_dim, dropout=config.dropout,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    generator = torch.Generator().manual_seed(config.seed)
    train_loader = DataLoader(EmbeddingDataset(x_train, y_train), batch_size=config.batch_size, shuffle=True, generator=generator)
    val_loader = DataLoader(EmbeddingDataset(x_val, y_val), batch_size=256)

    history: list[dict[str, float]] = []
    best_state, best_val, stale = None, float("inf"), 0
    started = time.perf_counter()
    for epoch in range(1, config.epochs + 1):
        train_loss = _run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss = _run_epoch(model, val_loader, criterion, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        if val_loss < best_val - 1e-6:
            best_val, best_state, stale = val_loss, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= config.patience:
                logger.info("early_stopping", extra={"epoch": epoch})
                break
    if best_state is not None:
        model.load_state_dict(best_state)

    validation = evaluate_embeddings(model, x_val, y_val, labels, device)
    test = evaluate_embeddings(model, x_test, y_test, labels, device)
    metrics = {"validation": validation, "test": test, "history": history, "best_val_loss": best_val}
    config_dict = asdict(config)
    out = save_checkpoint(config.output_path, model, labels, encoder.model_name, config_dict, metrics)

    metrics_path = Path(out).with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2, default=float), encoding="utf-8")
    return {
        "model_path": str(out),
        "metrics_path": str(metrics_path),
        "labels": labels,
        "epochs_run": len(history),
        "training_seconds": round(time.perf_counter() - started, 2),
        "splits": {"train": len(train_df), "val": len(val_df), "test": len(test_df)},
        "history": history,
        "validation": validation,
        "test": test,
    }
