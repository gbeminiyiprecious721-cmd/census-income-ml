from pathlib import Path
import os, json, time
import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter

root = Path(__file__).resolve().parent
os.environ['JUPYTER_PATH'] = str(root / '.jupyter' / 'share' / 'jupyter')
os.environ['IPYTHONDIR'] = str(root / '.ipython')
os.environ['JUPYTER_RUNTIME_DIR'] = str(root / '.jupyter' / 'runtime')
os.environ['MPLCONFIGDIR'] = str(root / '.matplotlib')
path = root / 'Adult_Income_Random_Forest_End_to_End.ipynb'
n = nbformat.read(path, as_version=4)
started = time.time()
def progress(cell_index, **kwargs):
    print('Executing cell', cell_index, flush=True)
client = NotebookClient(n, timeout=3600, kernel_name='adult-income',
    resources={'metadata':{'path':str(root)}}, on_cell_start=progress)
try:
    client.execute()
finally:
    nbformat.write(n, path)
nbformat.validate(n)
errors = [o for c in n.cells if c.cell_type == 'code' for o in c.get('outputs',[]) if o.output_type=='error']
codes = [c for c in n.cells if c.cell_type=='code']
assert not errors and all(c.execution_count for c in codes)
html, _ = HTMLExporter().from_notebook_node(n)
(root/'Adult_Income_Random_Forest_End_to_End.html').write_text(html, encoding='utf-8')
report = {'code_cells':len(codes),'error_outputs':len(errors),'sequential_counts':[c.execution_count for c in codes],
          'seconds':round(time.time()-started,2),'clean_kernel':True}
(root/'artifacts'/'execution_validation.json').write_text(json.dumps(report,indent=2))
print(report,flush=True)
