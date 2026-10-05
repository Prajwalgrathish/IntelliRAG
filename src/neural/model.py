"""Query-intent classifier (PyTorch).

Design: a pretrained transformer encoder (CLASSIFICATION_MODEL_NAME) is used as a FROZEN feature
extractor that turns a query into a sentence embedding; a small trainable MLP head (this
module) learns to map embeddings to intent labels. Only the MLP head is trained, which is fast
on CPU and works with small labelled sets. The predicted intent adapts retrieval depth
(top_k) and adds answer-style guidance to the RAG prompt.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional, Sequence, Union

import torch
from torch import nn

from src.embeddings.transformer_encoder import TextEncoder
from src.nlp.preprocessing import preprocess_query

CHECKPOINT_FORMAT_VERSION = 1


class IntentClassifier(nn.Module):
    """MLP classification head over sentence embeddings."""

    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 256, dropout: float = 0.2) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be >= 2")
        self.config = {
            "input_dim": input_dim, "num_classes": num_classes, "hidden_dim": hidden_dim, "dropout": dropout,
        }
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, max(hidden_dim // 2, num_classes)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(hidden_dim // 2, num_classes), num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class IntentPrediction:
    label: str
    confidence: float
    probabilities: dict[str, float]


def _to_builtin(obj: Any) -> Any:
    """Convert numpy/other scalars to plain Python so checkpoints load with weights_only=True."""
    return json.loads(json.dumps(obj, default=float))


def save_checkpoint(
    path: Union[str, Path],
    model: IntentClassifier,
    labels: Sequence[str],
    encoder_model_name: str,
    training_config: dict[str, Any],
    metrics: Optional[dict[str, Any]] = None,
) -> Path:
    """Persist weights plus everything needed to rebuild the predictor."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "model_config": _to_builtin(model.config),
        "labels": list(labels),
        "encoder_model_name": encoder_model_name,
        "training_config": _to_builtin(training_config),
        "metrics": _to_builtin(metrics) if metrics else None,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "torch_version": str(torch.__version__),
    }
    torch.save(payload, path)
    return path


def load_checkpoint(path: Union[str, Path], device: str = "cpu") -> tuple[IntentClassifier, dict[str, Any]]:
    """Load a checkpoint; returns (model in eval mode, metadata dict)."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Intent classifier checkpoint not found: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(f"Unsupported checkpoint format version: {payload.get('format_version')}")
    model = IntentClassifier(**payload["model_config"])
    model.load_state_dict(payload["state_dict"])
    model.to(device).eval()
    meta = {k: v for k, v in payload.items() if k != "state_dict"}
    return model, meta


class IntentPredictor:
    """Encoder + trained head -> intent labels for raw query strings."""

    def __init__(self, model: IntentClassifier, labels: Sequence[str], encoder: TextEncoder, device: str = "cpu") -> None:
        self.model = model.to(device).eval()
        self.labels = list(labels)
        self.encoder = encoder
        self.device = device
        self.metadata: dict[str, Any] = {}

    @torch.inference_mode()
    def predict(self, texts: Sequence[str]) -> list[IntentPrediction]:
        queries = [preprocess_query(t) for t in texts]
        embeddings = torch.from_numpy(self.encoder.encode_queries(queries)).to(self.device)
        probs = torch.softmax(self.model(embeddings), dim=-1).cpu()
        predictions = []
        for row in probs:
            idx = int(row.argmax())
            predictions.append(
                IntentPrediction(
                    label=self.labels[idx],
                    confidence=float(row[idx]),
                    probabilities={lab: float(p) for lab, p in zip(self.labels, row)},
                )
            )
        return predictions

    @classmethod
    def from_checkpoint(
        cls, path: Union[str, Path], encoder_provider: Callable[[str], TextEncoder], device: str = "cpu"
    ) -> "IntentPredictor":
        """Rebuild a predictor; ``encoder_provider(model_name)`` supplies the matching encoder."""
        model, meta = load_checkpoint(path, device=device)
        encoder = encoder_provider(meta["encoder_model_name"])
        predictor = cls(model, meta["labels"], encoder, device=device)
        predictor.metadata = meta
        return predictor
