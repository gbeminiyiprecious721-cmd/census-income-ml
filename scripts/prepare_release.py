"""Prepare a small self-contained source checkout, without Git actions or uploads."""
from pathlib import Path
import argparse,hashlib,json,shutil,zipfile
ROOT=Path(__file__).resolve().parents[1]
FILES={'README.md','LICENSE','NOTICE','.gitignore','.python-version','pyproject.toml',
       'requirements.txt','requirements-dev.txt','requirements-lock.txt','requirements-qa.txt'}
DIRECTORIES={'app','src','notebooks','reports','artifacts','data','archive','tests','scripts','.github','.streamlit'}
DOCS={'data_card.md','model_card.md','deployment_preparation.md','RESEARCH_REPORT.md','INTERVIEW_DEFENSE.md','GITHUB_RELEASE.md'}
EXCLUDED={'.venv','venv','__pycache__','.pytest_cache','.git','.jupyter','.ipython','.matplotlib','.ipynb_checkpoints'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination',type=Path,default=ROOT/'dist/CensusIncomeML')
    parser.add_argument('--replace',action='store_true')
    args=parser.parse_args();destination=args.destination.resolve()
    if destination==ROOT or destination in ROOT.parents:
        parser.error('Destination must not replace the source project or its parents')
    marker=destination/'.release-stage.json'
    if destination.exists():
        if not args.replace or not marker.exists():parser.error('Use a new destination, or --replace for a previously generated stage')
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    files={}
    for source in sorted(ROOT.rglob('*')):
        if not source.is_file():continue
        relative=source.relative_to(ROOT);parts=relative.parts
        if any(p in EXCLUDED or p.endswith('.egg-info') for p in parts):continue
        if source.is_symlink() or source.name.startswith('.env') or source.name in {'secrets.toml','release_stage.log'}:continue
        if parts[:2]==('artifacts','research_store'):continue
        include=(len(parts)==1 and source.name in FILES) or parts[0] in DIRECTORIES
        if parts[0]=='docs' and len(parts)>1:
            include=(len(parts)==2 and parts[1] in DOCS) or parts[1] in {'images','portfolio_audit'}
            if len(parts)>2 and parts[1]=='premium_dashboard':
                include=parts[2] in {'browser_qa.json','layout_qa.json','refresh_qa.json','chart_export_qa.json','after','responsive'}
        if not include:continue
        if source.stat().st_size>50*1024*1024:parser.error(f'File too large for this release: {relative}')
        target=destination/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
        files[str(relative)]=hashlib.sha256(target.read_bytes()).hexdigest()
    record={'description':'Generated source checkout, not a Git commit or deployment',
            'file_count':len(files),'bytes':sum((destination/p).stat().st_size for p in files),'files':files}
    marker.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({k:v for k,v in record.items() if k!='files'},indent=2));print(destination)


if __name__=='__main__':main()
