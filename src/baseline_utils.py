"""I/O, integrity checks and figures for Felix's course baseline.
PCA and SVM fitting remain visible in notebooks/02_baseline_svm.ipynb.
"""
from pathlib import Path
import hashlib
import warnings
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             ConfusionMatrixDisplay, RocCurveDisplay,
                             PrecisionRecallDisplay)
from . import preprocessing as shared

CLASS_TO_IDX = {"Uninfected": 0, "Parasitized": 1}


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_manifest(frame):
    required = {"filename", "patient_id", "category", "split"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Missing manifest columns: {required - set(frame.columns)}")
    if frame[list(required)].isna().any().any():
        raise ValueError("Manifest contains missing identifiers or labels")
    if set(frame["split"]) != {"train", "val", "test"}:
        raise ValueError("Expected the team's train/val/test split names")
    if set(frame["category"]) != set(CLASS_TO_IDX):
        raise ValueError("Unexpected category names; do not guess label encoding")
    if frame.duplicated(["category", "filename"]).any():
        raise ValueError("Duplicate image keys in the frozen manifest")
    if not frame["filename"].map(lambda s: Path(s).name == s).all():
        raise ValueError("Manifest filenames must be basenames")
    canonical = frame["patient_id"].str.lower().str.replace(r"[^a-z0-9]", "", regex=True)
    if (frame.assign(canonical=canonical).groupby("canonical")["split"].nunique() > 1).any():
        raise ValueError("Patient overlap across splits (including spelling aliases)")
    if (frame.groupby("split")["category"].nunique() != 2).any():
        raise ValueError("Every split must contain both classes")
    out = frame.copy()
    out["label"] = out["category"].map(CLASS_TO_IDX).astype(int)
    return out


def load_manifest(path, data_root):
    frame = validate_manifest(pd.read_csv(path))
    paths = [Path(data_root) / row.category / row.filename
             for row in frame.itertuples(index=False)]
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} missing images; first: {missing[0]}")
    counts = frame.groupby(["split", "category"]).size().unstack(fill_value=0)
    report = {"manifest_sha256": sha256_file(path), "rows": len(frame),
              "image_counts": frame["split"].value_counts().to_dict(),
              "patient_counts": frame.groupby("split")["patient_id"].nunique().to_dict(),
              "class_counts": counts.to_dict(), "patient_overlap": False,
              "duplicate_image_keys": 0, "missing_images": 0,
              "positive_class": "Parasitized", "class_to_index": CLASS_TO_IDX}
    return frame, report


def legacy_load_and_pad(path, size):
    """Exact geometry/scaling of Anders's current CNN placeholder, at chosen size."""
    with Image.open(path) as original:
        image = original.convert("RGB")
        width, height = image.size
        factor = size / max(width, height)
        dimensions = (max(1, round(width * factor)), max(1, round(height * factor)))
        image = image.resize(dimensions, Image.Resampling.BILINEAR)
        canvas = Image.new("RGB", (size, size), (0, 0, 0))
        canvas.paste(image, ((size - dimensions[0]) // 2, (size - dimensions[1]) // 2))
        return np.asarray(canvas, dtype=np.float32) / 255.0


def check_image(array, size):
    array = np.asarray(array)
    if array.shape != (size, size, 3):
        raise ValueError(f"Expected RGB HWC {(size, size, 3)}, got {array.shape}")
    if not np.isfinite(array).all() or array.min() < 0 or array.max() > 1.000001:
        raise ValueError("Preprocessor must return finite RGB pixels in [0,1]")
    return array.astype(np.float32, copy=False)


def choose_preprocessor(sample_path, size, allow_legacy=True):
    """Probe the shared API once; never silently claim a stub has been implemented."""
    try:
        probe = np.asarray(shared.load_and_pad(str(sample_path), size))
    except NotImplementedError:
        if not allow_legacy:
            raise
        warnings.warn("Shared preprocessing is still a stub. Using documented Anders-compatible fallback.")
        return (lambda p: check_image(legacy_load_and_pad(p, size), size)), "anders_compatible_fallback"
    needs_scaling = probe.dtype == np.uint8 or probe.max() > 1.000001
    if needs_scaling:
        # A raw-pixel shared loader must have a working shared scaling function.
        check_image(shared.scale_pixels(probe), size)
    else:
        check_image(probe, size)

    def load(path):
        image = np.asarray(shared.load_and_pad(str(path), size))
        if needs_scaling:
            image = shared.scale_pixels(image)
        return check_image(image, size)
    return load, "team_shared_load_and_scale" if needs_scaling else "team_shared_scaled_loader"


def load_features(rows, data_root, load_image):
    features = []
    for number, row in enumerate(rows.itertuples(index=False), 1):
        features.append(load_image(Path(data_root) / row.category / row.filename).reshape(-1))
        if number % 5000 == 0:
            print(f"Loaded {number}/{len(rows)} images", flush=True)
    return np.stack(features).astype(np.float32, copy=False)


def classification_metrics(y_true, y_pred, scores):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {"accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_true, scores)),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def prediction_table(rows, predictions, scores):
    table = rows[["filename", "patient_id", "category", "split", "label"]].copy()
    table = table.rename(columns={"label": "y_true"})
    table["y_pred"] = np.asarray(predictions, dtype=int)
    table["decision_score"] = np.asarray(scores, dtype=float)
    table["correct"] = table["y_true"] == table["y_pred"]
    table["outcome"] = ["TP" if t == 1 and p == 1 else "FN" if t == 1 else
                        "FP" if p == 1 else "TN"
                        for t, p in zip(table["y_true"], table["y_pred"])]
    return table


def export_figures(table, data_root, output_dir, title, seed=42):
    import matplotlib.pyplot as plt
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    y, pred, score = table.y_true, table.y_pred, table.decision_score
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    ConfusionMatrixDisplay.from_predictions(y, pred, labels=[0, 1],
        display_labels=["Uninfected", "Parasitized"], cmap="Blues", colorbar=False, ax=axes[0])
    axes[0].set_title("Confusion matrix")
    RocCurveDisplay.from_predictions(y, score, ax=axes[1], name="PCA + linear SVM")
    PrecisionRecallDisplay.from_predictions(y, score, ax=axes[2], name="PCA + linear SVM")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_dir / "evaluation.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    chosen = []
    for col, outcome in enumerate(["TP", "TN", "FP", "FN"]):
        subset = table[table.outcome == outcome]
        subset = subset.sample(min(2, len(subset)), random_state=seed)
        for row_number, ax in enumerate(axes[:, col]):
            ax.axis("off")
            if row_number < len(subset):
                row = subset.iloc[row_number]
                with Image.open(Path(data_root) / row.category / row.filename) as image:
                    ax.imshow(image.convert("RGB"))
                ax.set_title(f"{outcome}: {row.patient_id}\nmargin={row.decision_score:.2f}", fontsize=9)
                chosen.append(row.to_dict())
            else:
                ax.set_title(f"{outcome}: no further cases")
    fig.suptitle(title + " - seeded case sample (SVM margin is not probability)")
    fig.tight_layout()
    fig.savefig(output_dir / "error_cases.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(chosen).to_csv(output_dir / "displayed_cases.csv", index=False)
