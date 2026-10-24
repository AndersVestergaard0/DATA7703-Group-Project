"""Evaluate the registered PCA/SVM checkpoint without fitting or model selection."""
from pathlib import Path
import argparse
import datetime as dt
import importlib.metadata
import json
import os
import subprocess
import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.baseline_utils import (sha256_file, load_manifest, choose_preprocessor,
    load_features, classification_metrics, prediction_table, export_figures)

def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--confirm-frozen-protocol', action='store_true',
        help='Confirm that model, preprocessing and evaluation protocol are fixed.')
    parser.add_argument('--data-root', type=Path)
    args = parser.parse_args()
    if not args.confirm_frozen_protocol:
        parser.error('--confirm-frozen-protocol is required before held-out evaluation')
    checkpoint = ROOT / 'models/pca_linear_svm.joblib'
    registered = json.loads((ROOT / 'models/pca_linear_svm_metadata.json').read_text())
    manifest_path = ROOT / 'data/splits/patient_split.csv'
    checkpoint_hash = sha256_file(checkpoint)
    if checkpoint_hash != registered['sha256']:
        raise ValueError('Checkpoint differs from the registered frozen model')
    if sha256_file(manifest_path) != registered['manifest_sha256']:
        raise ValueError('Frozen patient manifest has changed')
    if sha256_file(ROOT / 'src/preprocessing.py') != registered['shared_preprocessing_sha256']:
        raise ValueError('Shared preprocessing differs from the verified protocol')
    out = ROOT / 'artifacts/felix_svm/test'
    if (out / 'test_metrics.json').exists():
        info = json.loads((out / 'checkpoint_info.json').read_text())
        if info['checkpoint_sha256'] != checkpoint_hash:
            raise ValueError('Existing test results belong to a different checkpoint')
        print('Existing frozen test result; no predictions recomputed.')
        print((out / 'test_metrics.json').read_text())
        return
    local = ROOT / 'data/local_config.json'
    data_root = args.data_root or os.environ.get('MALARIA_DATA_ROOT')
    if data_root is None and local.exists():
        data_root = json.loads(local.read_text()).get('data_root')
    if not data_root:
        parser.error('Set MALARIA_DATA_ROOT or pass --data-root')
    data_root = Path(data_root).expanduser().resolve()
    saved = joblib.load(checkpoint)
    k = saved['n_components']
    pca, svm = saved['pca'], saved['svm']
    if (k != 20 or svm.C != 0.1 or svm.kernel != 'linear'
            or pca.n_features_in_ != 3072
            or saved['class_to_index'] != {'Uninfected': 0, 'Parasitized': 1}
            or list(svm.classes_) != [0, 1]):
        raise ValueError('Saved model does not match the selected baseline')
    manifest, split_audit = load_manifest(manifest_path, data_root)
    val = manifest[manifest.split.eq('val')].reset_index(drop=True)
    test = manifest[manifest.split.eq('test')].reset_index(drop=True)
    if len(test) != 4110 or test.patient_id.nunique() != 31:
        raise ValueError('Unexpected held-out cohort')
    probe = data_root / val.iloc[0].category / val.iloc[0].filename
    load_image, backend = choose_preprocessor(probe, 32, allow_legacy=False)
    # Check the saved model against already-published validation predictions first.
    old = pd.read_csv(ROOT / 'artifacts/felix_svm/validation/validation_predictions.csv')
    old = old.set_index('filename').loc[val.filename]
    z_val = pca.transform(load_features(val, data_root, load_image))[:, :k]
    val_pred, val_scores = svm.predict(z_val), svm.decision_function(z_val)
    if not np.array_equal(val_pred, old.y_pred.to_numpy()):
        raise ValueError('Validation replay changed class predictions')
    if not np.allclose(val_scores, old.decision_score.to_numpy(), atol=2e-6, rtol=1e-6):
        raise ValueError('Validation replay changed decision scores beyond tolerance')
    out.mkdir(parents=True, exist_ok=True)
    started = dt.datetime.now(dt.timezone(dt.timedelta(hours=10))).isoformat()
    info = {
        'model': 'pca_linear_svm', 'checkpoint': 'models/pca_linear_svm.joblib',
        'checkpoint_sha256': checkpoint_hash, 'manifest_sha256': sha256_file(manifest_path),
        'shared_preprocessing_sha256': sha256_file(ROOT / 'src/preprocessing.py'),
        'evaluation_script_sha256': sha256_file(Path(__file__)),
        'source_git_commit': subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip(),
        'image_size': 32, 'colour': 'RGB', 'normalization': 'zero_one',
        'resize': 'aspect-preserving bilinear, centered black padding',
        'preprocessing_backend': backend, 'n_components': k, 'C': svm.C,
        'class_to_index': saved['class_to_index'], 'positive_class': 'Parasitized',
        'score_type': 'signed SVC decision_function; not calibrated probability',
        'decision_threshold': 0.0, 'fit_partitions': ['train'],
        'selection_partition': 'val', 'evaluation_partition': 'test',
        'model_refitted': False, 'protocol_frozen': True,
        'validation_replay': {'images': len(val), 'changed_labels': 0,
            'maximum_margin_difference': float(np.max(np.abs(val_scores-old.decision_score.to_numpy()))),
            'atol': 2e-6, 'rtol': 1e-6},
        'versions': {p: importlib.metadata.version(p) for p in ['numpy','pandas','scikit-learn','Pillow']},
        'started_at_brisbane': started,
    }
    write_json(out / 'evaluation_status.json', {'status':'running', 'started_at':started})
    write_json(out / 'checkpoint_info.json', info)
    write_json(out / 'split_audit.json', split_audit)
    print('Evaluating frozen model on 4,110 test images', flush=True)
    z_test = pca.transform(load_features(test, data_root, load_image))[:, :k]
    predictions, scores = svm.predict(z_test), svm.decision_function(z_test)
    table = prediction_table(test, predictions, scores)
    metrics = {'model':'pca_linear_svm', 'split':'test', 'n_images':len(test),
        'patient_groups':int(test.patient_id.nunique()), 'positive_class':'Parasitized',
        'class_to_index':saved['class_to_index'],
        **classification_metrics(test.label, predictions, scores),
        'n_components':k, 'C':svm.C, 'test_evaluated':True, 'model_refitted':False}
    table.to_csv(out / 'test_predictions.csv', index=False, float_format='%.17g')
    write_json(out / 'test_metrics.json', metrics)
    patient = table.assign(error=~table.correct, fp=table.outcome.eq('FP'), fn=table.outcome.eq('FN'))
    patient = patient.groupby('patient_id').agg(images=('filename','size'), errors=('error','sum'),
        fp=('fp','sum'), fn=('fn','sum'))
    patient['error_rate'] = patient.errors / patient.images
    patient.to_csv(out / 'patient_test_errors.csv')
    export_figures(table, data_root, out, 'Frozen PCA + linear SVM — TEST', seed=42)
    info['checkpoint_file_unchanged'] = sha256_file(checkpoint) == checkpoint_hash
    if not info['checkpoint_file_unchanged']:
        raise RuntimeError('Checkpoint file changed during evaluation')
    write_json(out / 'checkpoint_info.json', info)
    write_json(out / 'evaluation_status.json', {'status':'completed', 'model_refitted':False,
        'n_images':len(test), 'checkpoint_sha256':checkpoint_hash,
        'completed_at_brisbane':dt.datetime.now(dt.timezone(dt.timedelta(hours=10))).isoformat()})
    print(json.dumps(metrics, indent=2), flush=True)

if __name__ == '__main__':
    main()
