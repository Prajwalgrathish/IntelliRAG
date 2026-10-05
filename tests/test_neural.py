import numpy as np
import pandas as pd
import pytest
import torch

from src.neural.dataset import EmbeddingDataset, build_label_list, load_intent_dataframe, split_dataframe
from src.neural.evaluate import evaluate_checkpoint, evaluate_embeddings
from src.neural.model import IntentClassifier, IntentPredictor, load_checkpoint, save_checkpoint
from src.neural.train import TrainConfig, train_intent_classifier

KEYWORDS = {
    "summary": ("summarize", "recap", "highlights"),
    "comparison": ("compare", "contrast", "differences"),
    "procedural": ("install", "configure", "procedure"),
    "definition": ("define", "definition", "glossary"),
}
TOPICS = ["pricing", "security", "backup", "billing", "network", "storage", "reporting", "audit"]


def make_dataset_csv(path):
    rows = []
    for label, words in KEYWORDS.items():
        for i, topic in enumerate(TOPICS):
            rows.append({"text": f"{words[i % 3]} {topic} {words[(i + 1) % 3]}", "label": label})
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_model_forward_shape_and_validation():
    model = IntentClassifier(input_dim=32, num_classes=5, hidden_dim=16)
    assert model(torch.randn(4, 32)).shape == (4, 5)
    with pytest.raises(ValueError):
        IntentClassifier(input_dim=8, num_classes=1)


def test_model_is_trainable_end_to_end():
    torch.manual_seed(0)
    model = IntentClassifier(input_dim=8, num_classes=2, hidden_dim=16, dropout=0.0)
    x = torch.randn(32, 8)
    y = (x[:, 0] > 0).long()
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    losses = []
    for _ in range(60):
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(x), y)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0] * 0.5


def test_dataset_loading_validation_and_splitting(tmp_path):
    csv = make_dataset_csv(tmp_path / "d.csv")
    df = load_intent_dataframe(csv)
    assert len(df) == 32 and build_label_list(df) == sorted(KEYWORDS)
    train, val, test = split_dataframe(df, 0.25, 0.25, seed=1)
    assert len(train) + len(val) + len(test) == len(df)
    assert set(train["label"]) == set(val["label"]) == set(test["label"]) == set(KEYWORDS)
    assert not (set(train["text"]) & set(test["text"]))

    pd.DataFrame({"text": ["a"], "label": ["x"]}).to_csv(tmp_path / "one.csv", index=False)
    with pytest.raises(ValueError, match="two distinct"):
        load_intent_dataframe(tmp_path / "one.csv")
    pd.DataFrame({"query": ["a"]}).to_csv(tmp_path / "cols.csv", index=False)
    with pytest.raises(ValueError, match="missing required"):
        load_intent_dataframe(tmp_path / "cols.csv")
    with pytest.raises(FileNotFoundError):
        load_intent_dataframe(tmp_path / "none.csv")
    assert len(EmbeddingDataset(np.zeros((3, 4)), np.array([0, 1, 0]))) == 3


def test_training_pipeline_saves_loadable_checkpoint_and_reports_metrics(tmp_path, encoder):
    csv = make_dataset_csv(tmp_path / "intents.csv")
    out = tmp_path / "models" / "intent.pt"
    config = TrainConfig(
        data_path=str(csv), encoder_model_name=encoder.model_name, output_path=str(out), hidden_dim=32,
        learning_rate=0.01, batch_size=8, epochs=40, patience=100, val_size=0.25, test_size=0.25, device="cpu",
    )
    summary = train_intent_classifier(config, encoder=encoder)

    assert out.is_file() and (tmp_path / "models" / "intent.metrics.json").is_file()
    assert summary["labels"] == sorted(KEYWORDS) and summary["epochs_run"] == 40
    assert summary["history"][-1]["train_loss"] < summary["history"][0]["train_loss"]
    assert 0.0 <= summary["test"]["accuracy"] <= 1.0
    assert summary["test"]["accuracy"] > 1 / len(KEYWORDS)  # better than chance on separable data
    assert np.array(summary["test"]["confusion_matrix"]).shape == (4, 4)
    assert {"macro_f1", "per_class", "weighted_f1"} <= summary["test"].keys()

    model, meta = load_checkpoint(out)
    assert meta["labels"] == summary["labels"] and meta["encoder_model_name"] == encoder.model_name
    assert meta["training_config"]["hidden_dim"] == 32

    predictor = IntentPredictor.from_checkpoint(out, lambda name: encoder)
    predictions = predictor.predict(["please summarize the highlights", "compare pricing and contrast security"])
    assert len(predictions) == 2
    for p in predictions:
        assert p.label in summary["labels"] and sum(p.probabilities.values()) == pytest.approx(1.0, abs=1e-5)
        assert p.confidence == max(p.probabilities.values())

    report = evaluate_checkpoint(out, csv, encoder)
    assert report["accuracy"] > 0.5 and report["num_examples"] == 32


def test_training_is_reproducible_with_same_seed(tmp_path, encoder):
    csv = make_dataset_csv(tmp_path / "intents.csv")

    def run(name):
        cfg = TrainConfig(data_path=str(csv), encoder_model_name=encoder.model_name, output_path=str(tmp_path / name),
                          hidden_dim=16, epochs=5, patience=10, val_size=0.25, test_size=0.25, device="cpu", seed=7)
        return train_intent_classifier(cfg, encoder=encoder)["history"]

    assert run("a.pt") == run("b.pt")


def test_checkpoint_roundtrip_preserves_logits(tmp_path):
    model = IntentClassifier(16, 3, hidden_dim=8).eval()
    path = save_checkpoint(tmp_path / "m.pt", model, ["a", "b", "c"], "enc", {"lr": 0.1}, metrics={"acc": np.float64(0.5)})
    loaded, meta = load_checkpoint(path)
    x = torch.randn(5, 16)
    assert torch.allclose(model(x), loaded(x))
    assert meta["metrics"] == {"acc": 0.5} and meta["labels"] == ["a", "b", "c"]
    with pytest.raises(FileNotFoundError):
        load_checkpoint(tmp_path / "missing.pt")


def test_evaluate_embeddings_perfect_and_imperfect_predictions():
    model = IntentClassifier(4, 2, hidden_dim=8, dropout=0.0)
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
        model.net[-1].bias.copy_(torch.tensor([5.0, 0.0]))  # always predicts class 0
    emb = np.random.RandomState(0).randn(6, 4).astype(np.float32)
    report = evaluate_embeddings(model, emb, np.array([0, 0, 0, 1, 1, 1]), ["x", "y"])
    assert report["accuracy"] == 0.5
    assert report["confusion_matrix"] == [[3, 0], [3, 0]]
    assert report["per_class"]["x"]["recall"] == 1.0 and report["per_class"]["y"]["recall"] == 0.0
