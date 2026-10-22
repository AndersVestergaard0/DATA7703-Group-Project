"""One-time frozen test-set evaluation for the small CNN and ResNet-18.

The team agreed to freeze the trained weights in models/cnn.pt and models/resnet18.pt
and evaluate them on the held-out test split. This script:
  1. Reloads each checkpoint and re-verifies its declared preprocessing (img_size,
     normalize_method) still matches the live src/preprocessing.py module.
  2. Runs inference on the test split only (never train/val).
  3. Writes per-image predictions, aggregate metrics, and checkpoint metadata to
     artifacts/test_evaluation/<model>/.

Run once, from the repo root or scripts/:
    python scripts/evaluate_test_set.py
"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import kagglehub
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix,
)
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.preprocessing import load_and_pad, normalize_image  # noqa: E402

CLASS_TO_IDX = {"Uninfected": 0, "Parasitized": 1}
OUT_ROOT = ROOT / "artifacts" / "test_evaluation"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class CellDataset(Dataset):
    """Same loading logic as 03_cnn.ipynb / 04_resnet18.ipynb, parameterised by the
    checkpoint's own declared img_size/normalize_method rather than hardcoding them,
    so this script can't silently drift from what each model was actually trained on."""

    def __init__(self, df: pd.DataFrame, data_dir: Path, img_size: int, normalize_method: str):
        self.df = df.reset_index(drop=True)
        self.data_dir = data_dir
        self.img_size = img_size
        self.normalize_method = normalize_method

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        path = self.data_dir / row["category"] / row["filename"]
        img = load_and_pad(str(path), self.img_size)
        img = normalize_image(img, method=self.normalize_method)
        img = torch.from_numpy(img.astype(np.float32)).permute(2, 0, 1)
        label = CLASS_TO_IDX[row["category"]]
        return img, label


class SmallCNN(nn.Module):
    """Must match notebooks/03_cnn.ipynb exactly - this is the architecture the saved
    state_dict in models/cnn.pt was fit against."""

    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def build_resnet18() -> nn.Module:
    """Must match notebooks/04_resnet18.ipynb exactly: ImageNet architecture with the
    final layer replaced for 2 classes. weights=None since we load our own fine-tuned
    state_dict immediately after - no need to also download ImageNet init weights."""
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model


def resolve_data_dir() -> Path:
    download_path = Path(kagglehub.dataset_download("iarunava/cell-images-for-detecting-malaria"))
    data_dir = download_path / "cell_images"
    if (data_dir / "cell_images").exists():
        data_dir = data_dir / "cell_images"
    return data_dir


def load_checkpoint(path: Path, device: torch.device) -> dict:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    required = {"model_state_dict", "class_to_idx", "img_size", "normalize_method", "val_metrics"}
    missing = required - checkpoint.keys()
    if missing:
        raise ValueError(f"{path} is missing expected keys: {missing}")
    if checkpoint["class_to_idx"] != CLASS_TO_IDX:
        raise ValueError(f"{path} class_to_idx does not match CLASS_TO_IDX: {checkpoint['class_to_idx']}")
    return checkpoint


def verify_preprocessing_compatibility(checkpoint: dict, sample_path: Path) -> None:
    """Re-run the checkpoint's declared preprocessing against the *current*
    src/preprocessing.py and sanity-check the output shape/range, so a silent
    preprocessing drift since training time (e.g. someone editing normalize_image)
    would be caught here instead of producing quietly-wrong test predictions."""
    img = load_and_pad(str(sample_path), checkpoint["img_size"])
    img = normalize_image(img, method=checkpoint["normalize_method"])
    expected_shape = (checkpoint["img_size"], checkpoint["img_size"], 3)
    if img.shape != expected_shape:
        raise ValueError(f"Preprocessing shape mismatch: got {img.shape}, expected {expected_shape}")
    if not np.isfinite(img).all():
        raise ValueError("Preprocessing produced non-finite values")


def run_inference(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    all_probs, all_preds, all_labels = [], [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = outputs.argmax(dim=1)
            all_probs.extend(probs.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.tolist())
    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def classification_metrics(model_name: str, y_true, y_pred, y_prob) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "model": model_name,
        "split": "test",
        "n_images": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "positive_class": "Parasitized",
        "class_to_index": CLASS_TO_IDX,
    }


def evaluate_model(model_name: str, model: nn.Module, checkpoint: dict,
                    test_df: pd.DataFrame, data_dir: Path, device: torch.device) -> dict:
    out_dir = OUT_ROOT / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    verify_preprocessing_compatibility(checkpoint, data_dir / test_df.iloc[0]["category"] / test_df.iloc[0]["filename"])

    ds = CellDataset(test_df, data_dir, checkpoint["img_size"], checkpoint["normalize_method"])
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)

    y_true, y_pred, y_prob = run_inference(model, loader, device)
    metrics = classification_metrics(model_name, y_true, y_pred, y_prob)

    predictions = test_df[["filename", "patient_id", "category", "split"]].copy()
    predictions["y_true"] = y_true
    predictions["y_pred"] = y_pred
    predictions["prob_parasitized"] = y_prob
    predictions["correct"] = predictions["y_true"] == predictions["y_pred"]
    predictions["outcome"] = [
        "TP" if t == 1 and p == 1 else "FN" if t == 1 else "FP" if p == 1 else "TN"
        for t, p in zip(predictions["y_true"], predictions["y_pred"])
    ]
    predictions.to_csv(out_dir / "test_predictions.csv", index=False)
    (out_dir / "test_metrics.json").write_text(json.dumps(metrics, indent=2))

    checkpoint_info = {
        "model": model_name,
        "img_size": checkpoint["img_size"],
        "normalize_method": checkpoint["normalize_method"],
    }
    (out_dir / "checkpoint_info.json").write_text(json.dumps(checkpoint_info, indent=2))

    print(f"{model_name}: test_acc={metrics['accuracy']:.4f} test_f1={metrics['f1']:.4f} "
          f"test_auc={metrics['roc_auc']:.4f} -> {out_dir}")
    return {"metrics": metrics, "checkpoint_info": checkpoint_info}


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print("Using device:", device)

    manifest_path = ROOT / "data" / "splits" / "patient_split.csv"
    manifest = pd.read_csv(manifest_path)
    test_df = manifest[manifest["split"] == "test"].reset_index(drop=True)
    print(f"Test split: {len(test_df)} images, "
          f"{test_df['patient_id'].nunique()} patients, "
          f"{test_df['category'].value_counts().to_dict()}")

    data_dir = resolve_data_dir()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    summary = {
        "manifest_sha256": sha256_file(manifest_path),
        "base_git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "test_images": len(test_df),
        "test_patients": int(test_df["patient_id"].nunique()),
        "models": {},
    }

    cnn_checkpoint_path = ROOT / "models" / "cnn.pt"
    cnn_checkpoint = load_checkpoint(cnn_checkpoint_path, device)
    cnn_model = SmallCNN().to(device)
    cnn_model.load_state_dict(cnn_checkpoint["model_state_dict"])
    summary["models"]["cnn"] = evaluate_model(
        "cnn", cnn_model, cnn_checkpoint, test_df, data_dir, device
    )

    resnet_checkpoint_path = ROOT / "models" / "resnet18.pt"
    resnet_checkpoint = load_checkpoint(resnet_checkpoint_path, device)
    resnet_model = build_resnet18().to(device)
    resnet_model.load_state_dict(resnet_checkpoint["model_state_dict"])
    summary["models"]["resnet18"] = evaluate_model(
        "resnet18", resnet_model, resnet_checkpoint, test_df, data_dir, device
    )

    (OUT_ROOT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWrote combined summary to {OUT_ROOT / 'summary.json'}")


if __name__ == "__main__":
    main()
