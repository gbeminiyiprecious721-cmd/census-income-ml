"""Execute full research in a fresh isolated copy; never promote its fitted model.

Default strict mode refuses drift from archived results. --record-differences is
explicit and retains differences with a separately labelled notebook and reports.
"""
from pathlib import Path
import argparse,hashlib,json,os,shutil,subprocess,sys,tempfile,time


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--record-differences',action='store_true')
    parser.add_argument('--output',type=Path,default=Path('docs/portfolio_audit/reproduction'))
    args=parser.parse_args();root=Path(__file__).resolve().parents[1]
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    work=Path(tempfile.mkdtemp(prefix='census-research-verification-'))
    for name in ['src','data','notebooks']:
        shutil.copytree(root/name,work/name,ignore=shutil.ignore_patterns('__pycache__','*.egg-info'))
    shutil.copytree(root/'artifacts/historical',work/'artifacts/historical')
    (work/'reports').mkdir();(work/'scripts').mkdir()
    shutil.copy2(root/'scripts/execute_research.py',work/'scripts/execute_research.py')
    nbpath=work/'notebooks/01_income_classification_research.ipynb'
    import nbformat
    nb=nbformat.read(nbpath,4)
    nb.cells[0].source+='\n\n**Isolated verification run:** fresh outputs below are from this run. Its fitted model is not promoted to the dashboard. Archived historical results remain separate.'
    if args.record_differences:
        last=next(c for c in reversed(nb.cells) if c.cell_type=='code')
        last.source=last.source.replace('test_metrics,audit,independence)',"test_metrics,audit,independence,reproduction_policy='record')")
        last.source=last.source.replace('Fresh computations match archived metrics at reported precision.',"Reconciliation matches archived precision: **{summary['reproduction']['matches_reported_precision']}**. Differences, if present, are recorded in `reproduction_difference.json`; this isolated refit does not replace the validated dashboard model.")
        last.source=last.source.replace("'model_changed_from_historical_selection':False", "'model_changed_from_historical_selection':search.best_params_!=json.loads((ARTIFACTS_DIR/'historical/results.json').read_text())['best_parameters']")
    nbformat.write(nb,nbpath)
    source_hashes={str(p.relative_to(work)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (work/'src/census_income').glob('*.py')}
    start=time.time()
    with (output/'execution.log').open('w') as log:
        result=subprocess.run([sys.executable,str(work/'scripts/execute_research.py')],cwd=work,stdout=log,stderr=subprocess.STDOUT)
    shutil.copytree(work/'reports',output/'reports',dirs_exist_ok=True)
    shutil.copy2(nbpath,output/'notebook.ipynb')
    record={'isolated_project':str(work),'authoritative_artifacts_modified':False,
            'policy':'record' if args.record_differences else 'strict','exit_code':result.returncode,
            'seconds':round(time.time()-start,2),'source_hashes':source_hashes,
            'python':sys.version,'platform':sys.platform}
    (output/'verification.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2));raise SystemExit(result.returncode)


if __name__=='__main__':main()
