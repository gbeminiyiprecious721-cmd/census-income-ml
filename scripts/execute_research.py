"""Execute every research cell in a new kernel and retain verified outputs."""
from pathlib import Path
import os,sys,json,time,hashlib
import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter

root=Path(__file__).resolve().parents[1]
for key,relative in [('JUPYTER_PATH','.jupyter/share/jupyter'),('IPYTHONDIR','.ipython'),
                     ('JUPYTER_RUNTIME_DIR','.jupyter/runtime'),('MPLCONFIGDIR','.matplotlib')]:
    os.environ[key]=str(root/relative)
os.environ['MPLBACKEND']='module://matplotlib_inline.backend_inline'
kernel=root/'.jupyter/share/jupyter/kernels/census-income'; kernel.mkdir(parents=True,exist_ok=True)
(kernel/'kernel.json').write_text(json.dumps({'argv':[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],
    'display_name':'Python (CensusIncome ML)','language':'python'}),encoding='utf-8')
path=root/'notebooks/01_income_classification_research.ipynb'
n=nbformat.read(path,4); start=time.time()
def progress(cell_index,**kwargs): print('Executing cell',cell_index,flush=True)
def checkpoint(**kwargs): nbformat.write(n,path)
client=NotebookClient(n,timeout=3600,kernel_name='census-income',resources={'metadata':{'path':str(root)}},
    on_cell_start=progress,on_cell_executed=checkpoint)
try: client.execute()
finally: nbformat.write(n,path)
nbformat.validate(n)
codes=[c for c in n.cells if c.cell_type=='code']
assert [c.execution_count for c in codes]==list(range(1,len(codes)+1))
assert not [o for c in codes for o in c.outputs if o.output_type=='error']
report={'clean_kernel':True,'code_cells':len(codes),'sequential_counts':[c.execution_count for c in codes],
    'error_outputs':0,'seconds':round(time.time()-start,2),
    'embedded_figures':sum('image/png' in o.get('data',{}) for c in codes for o in c.outputs),
    'code_sha256':hashlib.sha256('\n'.join(c.source for c in codes).encode()).hexdigest()}
(root/'reports/execution_validation.json').write_text(json.dumps(report,indent=2))
summary_path=root/'reports/summary.json'
summary=json.loads(summary_path.read_text()); summary['status']='executed and validated'
summary['notebook_execution']=report
summary_path.write_text(json.dumps(summary,indent=2),encoding='utf-8')
html,_=HTMLExporter().from_notebook_node(n)
(root/'reports/research_notebook.html').write_text(html,encoding='utf-8')
print(json.dumps(report,indent=2),flush=True)
