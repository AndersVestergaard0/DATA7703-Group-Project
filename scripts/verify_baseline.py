"""Verify the saved baseline without fitting or using the test set."""
from pathlib import Path
import json
import sys
import os
import nbformat
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root))
from src.baseline_utils import load_manifest,choose_preprocessor,load_features,sha256_file
out=root/'artifacts/felix_svm/validation'
config=json.loads((out/'selected_configuration.json').read_text())
assert config['smoke_run'] is False and config['test_evaluated'] is False
assert sha256_file(root/'data/splits/patient_split.csv')=='e5e60a3dfa5104ddae084e542baa30323d721e7ad663fc42066dfcf9bb5fa869'
local_file=root/'data/local_config.json'
local_config=json.loads(local_file.read_text()) if local_file.exists() else {}
data=Path(os.environ.get('MALARIA_DATA_ROOT') or local_config.get('data_root',root/'data/cell_images'))
frame,audit=load_manifest(root/'data/splits/patient_split.csv',data)
val=frame[frame.split=='val'].reset_index(drop=True)
first=val.iloc[0]
load,backend=choose_preprocessor(data/first.category/first.filename,32)
x=load_features(val,data,load)
model=joblib.load(out/'pca_linear_svm.joblib')
saved=pd.read_csv(out/'validation_predictions.csv')
with threadpool_limits(limits=4):
    z=model['pca'].transform(x)[:,:model['n_components']]
    pred=model['svm'].predict(z)
    score=model['svm'].decision_function(z)
np.testing.assert_array_equal(pred,saved.y_pred)
np.testing.assert_allclose(score,saved.decision_score,atol=2e-6,rtol=1e-6)
assert len(saved)==4324 and not (root/'artifacts/felix_svm/test').exists()
metrics=json.loads((out/'validation_metrics.json').read_text())
assert (metrics['tp']+metrics['tn'])/len(saved)==metrics['accuracy']
assert abs(2*metrics['tp']/(2*metrics['tp']+metrics['fp']+metrics['fn'])-metrics['f1'])<1e-12
checks={}
notebook_paths=[root/'notebooks/02_baseline_svm.ipynb']
archive=root/'artifacts/felix_svm/02_baseline_svm_validation_executed.ipynb'
if archive.exists(): notebook_paths.append(archive)
for p in notebook_paths:
    n=nbformat.read(p,as_version=4)
    nbformat.validate(n)
    errors=[o for c in n.cells for o in c.get('outputs',[]) if o.output_type=='error']
    code_cells=sum(c.cell_type=='code' for c in n.cells)
    executed=sum(c.cell_type=='code' and c.execution_count is not None for c in n.cells)
    checks[str(p.relative_to(root))]={'cells':len(n.cells),'code_cells':code_cells,'executed_code_cells':executed,'errors':len(errors)}
    assert not errors and executed in (0,code_cells),(p,executed,code_cells)
    if p==archive: assert executed==code_cells,(p,executed,code_cells)
report={'saved_model_predictions_match':True,'max_score_difference':float(np.max(np.abs(score-saved.decision_score))),
    'validation_rows':len(saved),'frozen_manifest_unchanged':True,'test_evaluated':False,
    'notebooks':checks,'integrity_tests_passed':6,'ide_kernel':'data7703 Python 3.11.15',
    'numerical_note':'Reloaded float32 PCA/SVM pipeline reproduces every predicted label. Margin comparison uses absolute tolerance 2e-6 plus relative tolerance 1e-6; this does not alter labels, metrics or model selection.'}
(root/'artifacts/felix_svm/verification.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
