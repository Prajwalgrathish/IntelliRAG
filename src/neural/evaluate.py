"""Evaluation utilities for the intent classifier."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support

from src.embeddings.transformer_encoder import TextEncoder
from src.neural.dataset import embed_texts, load_intent_dataframe
from src.neural.model import IntentClassifier, load_checkpoint


@torch.inference_mode()
def predict_logits(model: IntentClassifier, embeddings: np.ndarray, device: str = "cpu", batch_size: int = 256) -> torch.Tensor:
    model.eval()
    x = torch.as_tensor(embeddings, dtype=torch.float32)
    outputs = [model(x[i : i + batch_size].to(device)).cpu() for i in range(0, len(x), batch_size)]
    return torch.cat(outputs) if outputs else torch.zeros((0, model.config["num_classes"]))


def evaluate_embeddings(
    model: IntentClassifier, embeddings: np.ndarray, targets: np.ndarray, labels: Sequence[str], device: str = "cpu"
) -> dict[str, Any]:
    """Compute loss, accuracy, macro/weighted metrics, per-class report and confusion matrix."""
    logits = predict_logits(model, embeddings, device)
    y_true = np.asarray(targets)
    y_pred = logits.argmax(dim=1).numpy()
    ids = list(range(len(labels)))
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=ids, average="macro", zero_division=0
    )
    _, _, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=ids, average="weighted", zero_division=0
    )
    return {
        "num_examples": int(len(y_true)),
        "loss": float(F.cross_entropy(logits, torch.as_tensor(y_true, dtype=torch.long)).item()) if len(y_true) else None,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "per_class": classification_report(
            y_true, y_pred, labels=ids, target_names=list(labels), output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=ids).tolist(),
        "labels": list(labels),
    }


def evaluate_checkpoint(
    checkpoint_path: Union[str, Path], data_path: Union[str, Path], encoder: TextEncoder, device: str = "cpu"
) -> dict[str, Any]:
    """Evaluate a saved model on a labelled CSV (labels unseen at training time are rejected)."""
    model, meta = load_checkpoint(checkpoint_path, device=device)
    labels = meta["labels"]
    df = load_intent_dataframe(data_path)
    unknown = sorted(set(df["label"]) - set(labels))
    if unknown:
        raise ValueError(f"Data contains labels unknown to the model: {unknown}")
    label_to_id = {l: i for i, l in enumerate(labels)}
    embeddings = embed_texts(encoder, df["text"].tolist())
    targets = df["label"].map(label_to_id).to_numpy()
    return evaluate_embeddings(model, embeddings, targets, labels, device)
