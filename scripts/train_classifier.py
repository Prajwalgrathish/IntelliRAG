"""CLI: train / evaluate the PyTorch query-intent classifier (MLP head on frozen transformer embeddings)."""
from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import json
import sys

from app.config import get_settings
from src.neural.train import TrainConfig, train_intent_classifier
from src.utils.logging import configure_logging


def main(argv=None) -> int:
    settings = get_settings()
    defaults = TrainConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=defaults.data_path, help="CSV with 'text' and 'label' columns.")
    parser.add_argument("--encoder-model", default=settings.classification_model_name)
    parser.add_argument("--output", default=settings.intent_model_path)
    parser.add_argument("--hidden-dim", type=int, default=defaults.hidden_dim)
    parser.add_argument("--dropout", type=float, default=defaults.dropout)
    parser.add_argument("--lr", type=float, default=defaults.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=defaults.weight_decay)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--patience", type=int, default=defaults.patience)
    parser.add_argument("--val-size", type=float, default=defaults.val_size)
    parser.add_argument("--test-size", type=float, default=defaults.test_size)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--device", default=settings.device)
    parser.add_argument("--evaluate-only", action="store_true", help="Evaluate an existing checkpoint on --data instead of training.")
    args = parser.parse_args(argv)
    configure_logging(settings.log_level)

    if args.evaluate_only:
        from src.embeddings.transformer_encoder import TransformerEncoder
        from src.neural.evaluate import evaluate_checkpoint

        encoder = TransformerEncoder(args.encoder_model, device=args.device, max_length=defaults.embedding_max_length)
        report = evaluate_checkpoint(args.output, args.data, encoder, device="cpu")
        print(json.dumps(report, indent=2))
        return 0

    config = TrainConfig(
        data_path=args.data, encoder_model_name=args.encoder_model, output_path=args.output, hidden_dim=args.hidden_dim,
        dropout=args.dropout, learning_rate=args.lr, weight_decay=args.weight_decay, batch_size=args.batch_size,
        epochs=args.epochs, patience=args.patience, val_size=args.val_size, test_size=args.test_size,
        seed=args.seed, device=args.device,
    )
    summary = train_intent_classifier(config)
    compact = {k: v for k, v in summary.items() if k != "history"}
    print(json.dumps(compact, indent=2, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
