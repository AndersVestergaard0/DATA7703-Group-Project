"""Execute Felix's notebook in the DATA7703 kernel; save success or partial failure."""
from pathlib import Path
import argparse
import json
import time
import traceback
import nbformat
from nbclient import NotebookClient

parser = argparse.ArgumentParser()
parser.add_argument('--smoke', action='store_true')
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
mode = 'smoke' if args.smoke else 'validation'
out = root / 'artifacts/felix_svm'
out.mkdir(parents=True, exist_ok=True)
notebook = nbformat.read(root / 'notebooks/02_baseline_svm.ipynb', as_version=4)
if args.smoke:
    for cell in notebook.cells:
        if cell.cell_type == 'code':
            cell.source = cell.source.replace('SMOKE_RUN = False', 'SMOKE_RUN = True')
status_file = out / f'{mode}_execution_status.json'
result_file = out / f'02_baseline_svm_{mode}_executed.ipynb'
started = time.perf_counter()

def update(cell=None, cell_index=None, **kwargs):
    status_file.write_text(json.dumps({'status':'running','mode':mode,
        'cell':cell_index,'elapsed_seconds':time.perf_counter()-started},indent=2))
    print(f'Executing cell {cell_index}: {cell.source.splitlines()[0] if cell.source else "empty"}',flush=True)

try:
    client = NotebookClient(notebook, timeout=900, kernel_name='data7703',
        resources={'metadata':{'path':str(root / 'notebooks')}},
        on_cell_start=update)
    client.execute()
    nbformat.write(notebook, result_file)
    status_file.write_text(json.dumps({'status':'succeeded','mode':mode,
        'elapsed_seconds':time.perf_counter()-started,'executed_notebook':str(result_file),
        'test_evaluated':any('RUN_TEST = True' in c.source for c in notebook.cells if c.cell_type == 'code')},indent=2))
    print('SUCCEEDED:',result_file,flush=True)
except Exception as exc:
    nbformat.write(notebook, result_file)
    status_file.write_text(json.dumps({'status':'failed','mode':mode,
        'error':str(exc),'elapsed_seconds':time.perf_counter()-started},indent=2))
    traceback.print_exc()
    raise
