from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / 'data' / 'raw'
REPORTS_DIR = PROJECT_ROOT / 'reports'
ARTIFACTS_DIR = PROJECT_ROOT / 'artifacts'
MODEL_PATH = ARTIFACTS_DIR / 'model.joblib'
MANIFEST_PATH = ARTIFACTS_DIR / 'model_manifest.json'
SEED = 42
JOBS = min(4, os.cpu_count() or 1)
COLUMNS = ['age','workclass','fnlwgt','education','education-num','marital-status','occupation',
           'relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country','income']
NUMERIC_COLUMNS = ['age','fnlwgt','education-num','capital-gain','capital-loss','hours-per-week']
MODEL_INPUTS = ['age','capital-gain','capital-loss','hours-per-week','education-num','workclass',
                'occupation','marital-status','relationship','native-country']
NUMERIC_RANGES = {'age':(17,90),'capital-gain':(0,99999),'capital-loss':(0,4356),
                  'hours-per-week':(1,99),'education-num':(1,16)}
ORIGINAL = ['age','capital-gain','capital-loss','hours-per-week','sex','race']
EXPANDED = ['age','capital-gain','capital-loss','hours-per-week','education_bin','workclass',
            'occupation','marital-status','relationship','native-country']
FEATURE_SETS = {'Original six':ORIGINAL,'Plus education':ORIGINAL+['education_bin'],
                'Expanded census':EXPANDED,'Expanded + interaction':EXPANDED+['hours_age_interaction']}
NUMERIC_FEATURES = ['age','capital-gain','capital-loss','hours-per-week','hours_age_interaction']
SCORING = {'accuracy':'accuracy','precision':'precision','recall':'recall','f1':'f1','roc_auc':'roc_auc'}
DATA_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/'
SOURCE_HASHES = {'adult.data':'5b00264637dbfec36bdeaab5676b0b309ff9eb788d63554ca0a249491c86603d',
                 'adult.test':'a2a9044bc167a35b2361efbabec64e89d69ce82d9790d2980119aac5fd7e9c05'}
