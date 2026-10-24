"""Verify exported predictions and compare the three frozen test pipelines."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/felix_svm/test'
METRICS = ['accuracy','precision','recall','f1','roc_auc']

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    manifest_path = ROOT / 'data/splits/patient_split.csv'
    expected = 'e5e60a3dfa5104ddae084e542baa30323d721e7ad663fc42066dfcf9bb5fa869'
    assert sha(manifest_path) == expected, 'Manifest changed'
    manifest = pd.read_csv(manifest_path)
    cohort = manifest[manifest.split.eq('test')].set_index('filename')
    assert len(cohort) == 4110 and cohort.patient_id.nunique() == 31
    info = json.loads((OUT / 'checkpoint_info.json').read_text())
    assert info['model_refitted'] is False and info['checkpoint_file_unchanged']
    assert info['checkpoint_sha256'] == sha(ROOT / 'models/pca_linear_svm.joblib')
    assert info['validation_replay']['changed_labels'] == 0
    assert info['shared_preprocessing_sha256'] == sha(ROOT / 'src/preprocessing.py')
    assert info['evaluation_script_sha256'] == sha(ROOT / 'scripts/evaluate_frozen_svm.py')
    status = json.loads((OUT / 'evaluation_status.json').read_text())
    assert status['status'] == 'completed'
    configs = [
        ('PCA + linear SVM', OUT, 'decision_score',32,'zero_one'),
        ('Small CNN',ROOT/'artifacts/test_evaluation/cnn','prob_parasitized',128,'zero_one'),
        ('ResNet-18',ROOT/'artifacts/test_evaluation/resnet18','prob_parasitized',128,'imagenet'),
    ]
    records, checks = [], []
    for model, folder, score_name, size, norm in configs:
        csv_path, metrics_path = folder/'test_predictions.csv', folder/'test_metrics.json'
        d = pd.read_csv(csv_path)
        saved = json.loads(metrics_path.read_text())
        assert len(d) == len(cohort) and not d.filename.duplicated().any()
        assert set(d.filename) == set(cohort.index)
        source = cohort.loc[d.filename]
        assert np.array_equal(d.patient_id.to_numpy(),source.patient_id.to_numpy())
        assert np.array_equal(d.y_true.to_numpy(),source.category.eq('Parasitized').astype(int).to_numpy())
        assert d.split.eq('test').all() and np.isfinite(d[score_name]).all()
        assert set(d.y_pred.unique()).issubset({0,1})
        recomputed = dict(zip(METRICS,[accuracy_score(d.y_true,d.y_pred),
            precision_score(d.y_true,d.y_pred),recall_score(d.y_true,d.y_pred),
            f1_score(d.y_true,d.y_pred),roc_auc_score(d.y_true,d[score_name])]))
        assert all(abs(saved[k]-recomputed[k]) < 1e-12 for k in METRICS)
        counts = dict(zip(['tn','fp','fn','tp'],[int(x) for x in confusion_matrix(d.y_true,d.y_pred,labels=[0,1]).ravel()]))
        assert all(saved[k] == counts[k] for k in counts)
        if model == 'PCA + linear SVM':
            assert np.array_equal(d.y_pred.to_numpy(),(d.decision_score >= 0).astype(int).to_numpy())
        errors = d[d.y_true.ne(d.y_pred)].groupby('patient_id').size().sort_values(ascending=False)
        record = {'model':model,'split':'test','n_images':len(d),'patient_groups':31,
            'image_size':size,'normalization':norm,**recomputed,**counts,
            'top10_error_count':int(errors.head(10).sum()),'total_errors':int(errors.sum()),
            'top10_error_fraction':float(errors.head(10).sum()/errors.sum())}
        records.append(record)
        checks.append({'model':model,'prediction_csv_sha256':sha(csv_path),
            'metrics_json_sha256':sha(metrics_path),'manifest_filename_patient_label_match':True,
            'recomputed_metrics_and_confusion_matrix_match':True})
    comparison = pd.DataFrame(records)
    comparison.to_csv(OUT/'model_comparison.csv',index=False,float_format='%.17g')
    svm = records[0]
    fig, ax = plt.subplots(figsize=(5.4,4.6))
    cm=np.array([[svm['tn'],svm['fp']],[svm['fn'],svm['tp']]])
    ax.imshow(cm,cmap='BuPu',vmin=0,vmax=cm.max())
    for (r,c),value in np.ndenumerate(cm):
        ax.text(c,r,str(value),ha='center',va='center',fontsize=21,
            color='white' if value>cm.max()*0.55 else '#202c34')
    ax.set(xticks=[0,1],yticks=[0,1],xticklabels=['Uninfected','Parasitized'],
        yticklabels=['Uninfected','Parasitized'],xlabel='Predicted label',ylabel='True label',
        title='PCA + linear SVM | held-out test')
    fig.tight_layout();fig.savefig(OUT/'confusion_matrix.png',dpi=200);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10.7,2.8));ax.axis('off')
    cells=[[r['model'],f"{r['accuracy']:.2%}",f"{r['recall']:.2%}",
        f"{r['f1']:.4f}",f"{r['roc_auc']:.4f}",str(r['fn']),str(r['fp'])] for r in records]
    tab=ax.table(cellText=cells,colLabels=['Pipeline','Accuracy','Recall','F1','ROC-AUC','FN','FP'],
        colWidths=[.27,.14,.14,.12,.13,.10,.10],loc='center',cellLoc='center')
    tab.auto_set_font_size(False);tab.set_fontsize(11);tab.scale(1,1.65)
    for (r,c),cell in tab.get_celld().items():
        cell.set_edgecolor('#e5ded6')
        if r==0:cell.set_facecolor('#7b1e3a');cell.set_text_props(color='white',weight='bold')
        elif r==1:cell.set_facecolor('#f6e8ed')
    ax.set_title('Same held-out cohort: 4,110 images, 31 patient ID groups',fontsize=14,pad=12)
    fig.text(.5,.03,'Pipeline comparison: SVM 32px; CNN/ResNet 128px. ResNet uses ImageNet preprocessing and pretrained initialization.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.07,1,1]);fig.savefig(OUT/'model_comparison.png',dpi=200);plt.close(fig)
    report={'status':'passed','manifest_sha256':expected,'n_images':4110,'patient_groups':31,
        'checkpoint_sha256':info['checkpoint_sha256'],'model_refitted':False,
        'validation_labels_unchanged':True,'checks':checks,
        'interpretation':'Descriptive performance on this split; no significance test or clinical validation claimed.'}
    (OUT/'verification.json').write_text(json.dumps(report,indent=2)+chr(10))
    print(comparison[['model','accuracy','recall','f1','roc_auc','fn','fp']].to_string(index=False))
    print('VERIFIED: all three CSVs match the same frozen test cohort; metrics agree.')

if __name__ == '__main__':
    main()
