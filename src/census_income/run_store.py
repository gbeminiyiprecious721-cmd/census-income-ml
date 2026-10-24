"""Validated, immutable local research bundles; no training or network access.

Checksums detect accidental corruption, not malicious replacement of a trusted
bundle and its manifest. Only publish maintainer-controlled project outputs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import importlib.metadata as metadata
import io
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from types import MappingProxyType
from typing import Mapping
import uuid

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold

from . import __version__
from .config import COLUMNS, MODEL_INPUTS, SOURCE_HASHES
from .data import clean_data, split_data, split_independence
from .evaluate import wilson_interval

JSON_REPORTS = ('summary.json', 'data_quality.json', 'eda_summary.json',
                'execution_validation.json', 'calibration_metrics.json')
CSV_COLUMNS = {
    'model_comparison.csv': {'model', 'evaluation', 'accuracy', 'precision', 'recall', 'f1', 'roc_auc'},
    'feature_comparison.csv': {'model', 'accuracy', 'precision', 'recall', 'f1', 'roc_auc'},
    'feature_comparison_folds.csv': {'model', 'fold', 'accuracy', 'precision', 'recall', 'f1', 'roc_auc'},
    'confusion_matrix.csv': {'actual', 'predicted', 'count'},
    'roc_curve.csv': {'fpr', 'tpr', 'threshold'},
    'precision_recall_curve.csv': {'precision', 'recall', 'threshold'},
    'calibration.csv': {'mean_score', 'observed_fraction', 'count'},
    'threshold_diagnostics.csv': {'threshold', 'accuracy', 'precision', 'recall', 'f1'},
    'subgroup_metrics.csv': {'attribute', 'group', 'n', 'positives', 'negatives', 'tn', 'fp', 'fn', 'tp', 'accuracy', 'precision', 'recall_tpr', 'fpr', 'fnr', 'f1', 'prevalence', 'selection_rate',
                             'recall_tpr_ci_low', 'recall_tpr_ci_high', 'fpr_ci_low', 'fpr_ci_high', 'precision_ci_low', 'precision_ci_high'},
    'importance_comparison.csv': {'feature', 'impurity_importance', 'f1_decrease_mean'},
    'coherent_source_permutation.csv': {'source', 'f1_decrease', 'roc_auc_decrease'},
}
REQUIRED_FILES = frozenset(
    [f'reports/{n}' for n in (*JSON_REPORTS, *CSV_COLUMNS)]
    + ['artifacts/model.joblib', 'artifacts/model_manifest.json']
    + [f'data/raw/{n}' for n in SOURCE_HASHES]
)
DEPENDENCIES = ('scikit-learn', 'numpy', 'pandas', 'scipy', 'joblib')


class BundleError(ValueError):
    """A candidate bundle is incomplete, incompatible or inconsistent."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise BundleError(message)


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _json(payload: bytes) -> dict:
    def reject(value):
        raise BundleError(f'Invalid JSON numeric value: {value}')
    value = json.loads(payload, parse_constant=reject)
    _require(isinstance(value, dict), 'Expected JSON object')
    return value


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _frame(files: Mapping[str, bytes]) -> pd.DataFrame:
    parts = []
    for name, skip in [('adult.data', 0), ('adult.test', 1)]:
        payload = files[f'data/raw/{name}']
        _require(_digest(payload) == SOURCE_HASHES[name], f'Source checksum mismatch: {name}')
        parts.append(pd.read_csv(io.BytesIO(payload), names=COLUMNS, header=None, skiprows=skip))
    raw = pd.concat(parts, ignore_index=True)
    _require(raw.shape == (48842, 15), 'Unexpected original source shape')
    return clean_data(raw)


@dataclass(frozen=True)
class Snapshot:
    """Own frozen bytes. Return fresh parsed values so callers cannot mutate state."""
    signature: str
    _manifest: bytes = field(repr=False)
    _files: Mapping[str, bytes] = field(repr=False)

    @property
    def manifest(self) -> dict:
        return _json(self._manifest)

    @property
    def source_counts(self) -> dict:
        return self.manifest['source_counts']

    def bytes(self, name: str) -> bytes:
        return self._files[name]

    def report(self, name: str):
        payload = self._files[f'reports/{name}']
        return _json(payload) if name.endswith('.json') else pd.read_csv(io.BytesIO(payload))

    def source_data(self) -> tuple[pd.DataFrame, dict]:
        return _frame(self._files), self.report('data_quality.json')


def _near(actual, expected, message: str) -> None:
    _require(bool(np.isfinite(actual)) and bool(np.isclose(actual, expected, rtol=0, atol=1e-10)), message)


def _counts(frame: pd.DataFrame, columns: list[str]) -> None:
    values = frame[columns].to_numpy(dtype=float)
    _require(bool(np.isfinite(values).all() and (values >= 0).all() and (values == np.floor(values)).all()), 'Invalid record counts')


def _metrics(tn: int, fp: int, fn: int, tp: int) -> dict:
    divide = lambda a, b: a / b if b else 0.0
    return dict(accuracy=divide(tp+tn, tn+fp+fn+tp), precision=divide(tp, tp+fp),
                recall=divide(tp, tp+fn), f1=divide(2*tp, 2*tp+fp+fn))


def _validate_semantics(snapshot: Snapshot) -> None:
    m = snapshot.manifest
    s = snapshot.report('summary.json')
    model = _json(snapshot.bytes('artifacts/model_manifest.json'))
    for key in ['run_id', 'model_version', 'selected_feature_group']:
        _require(isinstance(s[key], str) and bool(s[key].strip()), f'Invalid summary identity: {key}')
    _require(isinstance(s['limitations'], list) and bool(s['limitations']) and
             all(isinstance(item, str) and bool(item.strip()) for item in s['limitations']), 'Missing or invalid scientific limitations')
    _require(isinstance(s['selected_parameters'], dict) and bool(s['selected_parameters']), 'Missing selected model parameters')
    _require(s['status'] == 'executed and validated', 'Research execution is not validated')
    _require(s['run_id'] == m['run_id'] == s['model_version'] == model['model_version'] == model['metadata']['run_id'], 'Run/model identity mismatch')
    _require(model['artifact_format'] == 'census-income-pipeline-v1' and model['package_version'] == __version__, 'Unsupported model format or package version')
    _require(model['model_inputs'] == MODEL_INPUTS, 'Model input schema mismatch')
    for dependency in DEPENDENCIES:
        _require(model['dependencies'][dependency] == s['dependencies'][dependency] == metadata.version(dependency), f'Model dependency version mismatch: {dependency}')
    payload = snapshot.bytes('artifacts/model.joblib')
    _require(_digest(payload) == model['sha256'] == s['model_sha256'], 'Model checksum mismatch')
    _require(len(payload) == model['bytes'], 'Model byte count mismatch')
    _require(model['metadata']['parameters'] == s['selected_parameters'] and model['metadata']['selected_feature_group'] == s['selected_feature_group'], 'Model selection metadata mismatch')
    _require(model['calibrated'] is False and model['source_year'] == 1994 and model['metadata']['seed'] == s['seed'] == 42, 'Frozen research assumptions mismatch')
    _require(model['decision_rule'] == 'RandomForest argmax; class 0 on exact probability ties', 'Frozen decision rule mismatch')
    execution = snapshot.report('execution_validation.json')
    _require(execution == s['notebook_execution'], 'Execution evidence mismatch')
    _require(execution['clean_kernel'] is True and execution['error_outputs'] == 0 and execution['code_cells'] > 0, 'Notebook execution failed')
    _require(execution['sequential_counts'] == list(range(1, execution['code_cells']+1)), 'Nonsequential notebook execution')
    _require(m['research_execution']['timestamp_basis'] == 'model creation timestamp; execution completion time unavailable', 'Research timestamp basis mismatch')
    _require(m['research_execution']['timestamp_utc'] == model['created_utc'], 'Research timestamp mismatch')
    for value in [m['published_utc'], m['research_execution']['timestamp_utc']]:
        _require(datetime.fromisoformat(value).utcoffset() is not None, 'Timestamp must include timezone')

    source = _frame(snapshot._files)
    counts = {str(k): int(v) for k, v in source.income.value_counts().items()}
    quality, eda = snapshot.report('data_quality.json'), snapshot.report('eda_summary.json')
    _require(m['source_counts'] == eda['income_counts'] == quality['target_distribution'] == s['dataset']['target_distribution'] == counts, 'Source category distribution mismatch')
    _require(eda['rows'] == quality['clean_rows'] == s['dataset']['clean_rows'] == len(source), 'Clean source row count mismatch')
    _require(s['dataset']['raw_rows'] == quality['raw_rows'] == 48842 and s['dataset']['duplicates_removed'] == quality['duplicates_removed'] == 52, 'Raw source population mismatch')
    for name in SOURCE_HASHES:
        _require(s['dataset']['provenance'][name]['sha256'] == quality['provenance'][name]['sha256'] == model['metadata']['source_provenance'][name]['sha256'] == SOURCE_HASHES[name], 'Source provenance mismatch')
    for column, key in [('sex', 'sex_counts'), ('race', 'race_counts')]:
        _require(eda[key] == {str(k): int(v) for k, v in source[column].value_counts().items()}, f'{column} category counts mismatch')
    Xdev, Xtest, ydev, ytest = split_data(source)
    split = m['split']
    _require(split['development_rows'] == s['dataset']['development_rows'] == len(Xdev) and split['test_rows'] == s['dataset']['test_rows'] == len(Xtest), 'Split sizes mismatch')
    for name, frame in [('development', Xdev), ('test', Xtest)]:
        _require(split[f'{name}_indices_sha256'] == _digest(np.asarray(frame.index, dtype='<i8').tobytes()), 'Split membership mismatch')
    independence = split_independence(source, Xdev, Xtest)
    for key, value in independence.items():
        if key != 'interpretation':
            _require(s['dataset']['independence'][key] == value, 'Split independence evidence mismatch')

    frames = {}
    for name, required in CSV_COLUMNS.items():
        frame = snapshot.report(name)
        _require(not frame.empty and required <= set(frame.columns), f'Report schema mismatch: {name}')
        frames[name] = frame
    confusion = frames['confusion_matrix.csv']
    _counts(confusion, ['actual', 'predicted', 'count'])
    _require(len(confusion) == 4 and set(zip(confusion.actual, confusion.predicted)) == {(0,0),(0,1),(1,0),(1,1)}, 'Invalid confusion matrix classes')
    cells = {(int(row.actual), int(row.predicted)): int(row.count) for row in confusion.itertuples()}
    tn, fp, fn, tp = (cells[key] for key in [(0,0),(0,1),(1,0),(1,1)])
    _require(tn+fp+fn+tp == len(Xtest) and tp+fn == int(ytest.sum()), 'Confusion matrix population mismatch')
    for key, value in _metrics(tn,fp,fn,tp).items():
        _near(s['test_metrics'][key], value, f'Test {key} arithmetic mismatch')
    for key, value in s['test_metrics'].items():
        _near(model['metadata']['test_metrics'][key], value, 'Model/test metrics mismatch')
        _require(0 <= value <= 1, 'Invalid test metric')
    comparison = frames['model_comparison.csv']
    fixed_selected = comparison[(comparison.model == s['selected_feature_group']) & (comparison.evaluation == 'fixed forest: five-fold CV')]
    _require(len(fixed_selected) == 1, 'Selected feature group has no unique fixed-model comparison')
    for evaluation, metrics in [('five-fold CV', s['cv_metrics']), ('held-out test reproduction', s['test_metrics'])]:
        selected = comparison[(comparison.model == 'F1-selected forest') & (comparison.evaluation == evaluation)]
        _require(len(selected) == 1, 'Selected comparison row missing or duplicated')
        for key, value in metrics.items():
            _near(selected.iloc[0][key], value, 'Comparison/summary mismatch')
    # Cross-check development summaries against fold-level evidence, never test rows.
    feature = frames['feature_comparison.csv']
    folds = frames['feature_comparison_folds.csv']
    metric_names = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    _require(not feature.model.duplicated().any() and set(feature.model) == set(folds.model), 'Feature comparison identities mismatch')
    for row in feature.itertuples():
        group = folds[folds.model == row.model]
        _require(len(group) == 5 and set(group.fold) == {1,2,3,4,5}, 'Expected five distinct development folds')
        fixed = comparison[(comparison.model == row.model) & (comparison.evaluation == 'fixed forest: five-fold CV')]
        _require(len(fixed) == 1, 'Missing fixed-model comparison')
        for key in metric_names:
            _near(getattr(row, key), group[key].mean(), 'Feature fold mean mismatch')
            _near(fixed.iloc[0][key], getattr(row, key), 'Fixed comparison mismatch')
    baseline = comparison[comparison.model == 'Majority-class baseline']
    _require(len(baseline) == 1 and baseline.iloc[0].evaluation == 'five-fold CV', 'Majority baseline missing')
    # Counts are stratified across five folds, so use saved fold-weighting precision.
    baseline_accuracy = np.mean([(ydev.iloc[test] == 0).mean() for _, test in StratifiedKFold(5, shuffle=True, random_state=42).split(Xdev, ydev)])
    _near(baseline.iloc[0].accuracy, baseline_accuracy, 'Majority baseline population mismatch')
    for key, value in [('precision', 0), ('recall', 0), ('f1', 0), ('roc_auc', .5)]:
        _near(baseline.iloc[0][key], value, 'Majority baseline constant-prediction mismatch')
    for name, columns in {
        'model_comparison.csv': metric_names,
        'feature_comparison.csv': metric_names,
        'feature_comparison_folds.csv': metric_names,
        'roc_curve.csv': ['fpr','tpr'],
        'precision_recall_curve.csv': ['precision','recall'],
        'calibration.csv': ['mean_score','observed_fraction'],
        'threshold_diagnostics.csv': ['threshold', *metric_names[:-1]],
    }.items():
        values = frames[name][columns].to_numpy(dtype=float)
        _require(bool(np.isfinite(values).all() and (values >= 0).all() and (values <= 1).all()), f'Invalid metric range: {name}')
    calibration = frames['calibration.csv']
    _counts(calibration, ['count'])
    _require(calibration['count'].sum() == len(Xtest), 'Calibration population mismatch')
    ece = np.average(abs(calibration.mean_score-calibration.observed_fraction), weights=calibration['count'])
    _near(snapshot.report('calibration_metrics.json')['ece_10_equal_width_bins'], ece, 'Calibration arithmetic mismatch')
    roc = frames['roc_curve.csv']
    _require(bool((np.diff(roc.fpr) >= 0).all() and (np.diff(roc.tpr) >= 0).all()), 'ROC curve must be monotonic')
    _near(np.trapezoid(roc.tpr, roc.fpr), s['test_metrics']['roc_auc'], 'ROC curve/AUC mismatch')
    subgroup = frames['subgroup_metrics.csv']
    _counts(subgroup, ['n','positives','negatives','tn','fp','fn','tp'])
    expected_groups = {a: set(source[a].unique()) for a in ['sex','race']}
    _require(set(subgroup.attribute) == set(expected_groups), 'Subgroup partitions mismatch')
    for attribute, group in subgroup.groupby('attribute'):
        _require(set(group['group']) == expected_groups[attribute] and not group['group'].duplicated().any(), 'Subgroup labels mismatch')
        _require(group[['tn','fp','fn','tp']].sum().tolist() == [tn,fp,fn,tp], 'Subgroup totals mismatch')
    for row in subgroup.itertuples():
        mask = Xtest[row.attribute].eq(row.group)
        _require(row.n == int(mask.sum()) and row.positives == int(ytest[mask].sum()), 'Subgroup source population mismatch')
        _require(row.n == row.tn+row.fp+row.fn+row.tp and row.positives == row.fn+row.tp and row.negatives == row.tn+row.fp, 'Subgroup count arithmetic mismatch')
        for key, value in _metrics(row.tn,row.fp,row.fn,row.tp).items():
            _near(getattr(row, 'recall_tpr' if key == 'recall' else key), value, 'Subgroup metric arithmetic mismatch')
        _near(row.fpr, row.fp/row.negatives, 'Subgroup FPR mismatch')
        _near(row.fnr, row.fn/row.positives, 'Subgroup FNR mismatch')
        _near(row.prevalence, row.positives/row.n, 'Subgroup prevalence mismatch')
        _near(row.selection_rate, (row.tp+row.fp)/row.n, 'Subgroup selection rate mismatch')
        for key, numerator, denominator in [('recall_tpr', row.tp, row.positives), ('fpr', row.fp, row.negatives), ('precision', row.tp, row.tp+row.fp)]:
            low, high = getattr(row, key+'_ci_low'), getattr(row, key+'_ci_high')
            _require(0 <= low <= getattr(row, key) <= high <= 1, 'Invalid subgroup interval bounds')
            expected_low, expected_high = wilson_interval(numerator, denominator)
            _near(low, expected_low, 'Subgroup Wilson lower bound mismatch')
            _near(high, expected_high, 'Subgroup Wilson upper bound mismatch')
    frozen = frames['threshold_diagnostics.csv']
    frozen = frozen[np.isclose(frozen.threshold, .5)]
    _require(len(frozen) == 1, 'Missing frozen threshold')
    for key in ['accuracy','precision','recall','f1']:
        _near(frozen.iloc[0][key], s['test_metrics'][key], 'Frozen threshold mismatch')


def validate_bundle(bundle: Path, expected_manifest_sha256: str | None = None) -> Snapshot:
    """Read all files once, check integrity and semantics, then freeze their bytes."""
    try:
        bundle = Path(bundle)
        _require(not bundle.is_symlink(), 'Bundle directory cannot be a symlink')
        manifest_path = bundle/'manifest.json'
        _require(not manifest_path.is_symlink(), 'Manifest cannot be a symlink')
        raw = manifest_path.read_bytes()
        if expected_manifest_sha256 is not None:
            _require(_digest(raw) == expected_manifest_sha256, 'Manifest checksum mismatch')
        manifest = _json(raw)
        _require(manifest['schema_version'] == 1, 'Unsupported bundle schema')
        _require(bool(re.fullmatch(r'[a-f0-9]{32}', manifest['bundle_id'])), 'Invalid bundle identity')
        _require(set(manifest['files']) == REQUIRED_FILES, 'Missing or unexpected bundle files')
        files = {}
        for name, entry in manifest['files'].items():
            path = bundle/name
            _require(not any(p.is_symlink() for p in [path, *path.parents] if p != bundle.parent), 'Bundle paths cannot contain symlinks')
            _require(path.resolve().is_relative_to(bundle.resolve()), 'Bundle path escapes store')
            payload = path.read_bytes()
            _require(_digest(payload) == entry['sha256'], f'Checksum mismatch: {name}')
            _require(len(payload) == entry['bytes'], f'Byte count mismatch: {name}')
            files[name] = payload
        snapshot = Snapshot(_digest(raw), raw, MappingProxyType(files))
        _validate_semantics(snapshot)
        return snapshot
    except BundleError:
        raise
    except (OSError, ValueError, TypeError, KeyError, AttributeError, metadata.PackageNotFoundError) as exc:
        raise BundleError(f'Invalid research bundle: {exc}') from exc


def _load_pointer(store: Path, payload: bytes) -> Snapshot:
    try:
        pointer = _json(payload)
        _require(pointer['schema_version'] == 1 and bool(re.fullmatch(r'[a-f0-9]{32}', pointer['bundle_id'])), 'Invalid current pointer')
        _require(bool(re.fullmatch(r'[a-f0-9]{64}', pointer['manifest_sha256'])), 'Invalid manifest checksum')
        bundle = store/'bundles'/pointer['bundle_id']
        _require(bundle.resolve().is_relative_to(store.resolve()) and not (store/'bundles').is_symlink(), 'Pointer escapes publication store')
        snapshot = validate_bundle(bundle, pointer['manifest_sha256'])
        _require(snapshot.manifest['bundle_id'] == pointer['bundle_id'], 'Pointer/bundle identity mismatch')
        return snapshot
    except BundleError:
        raise
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise BundleError(f'Invalid publication pointer: {exc}') from exc


def load_snapshot(store: Path) -> Snapshot:
    """Resolve exactly one atomic pointer; never reread live project files."""
    try:
        return _load_pointer(Path(store), (Path(store)/'current.json').read_bytes())
    except OSError as exc:
        raise BundleError(f'Cannot read publication pointer: {exc}') from exc


def publish_existing(project: Path, store: Path) -> Snapshot:
    """Stage existing trusted outputs, validate, then atomically activate locally."""
    project, store = Path(project), Path(store)
    bundles = store/'bundles'
    bundles.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='.staging-', dir=bundles))
    pointer_tmp = None
    try:
        files = {}
        for name in REQUIRED_FILES:
            payload = (project/name).read_bytes()
            target = stage/name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
            files[name] = {'sha256': _digest(payload), 'bytes': len(payload)}
        s = _json((stage/'reports/summary.json').read_bytes())
        model = _json((stage/'artifacts/model_manifest.json').read_bytes())
        source = _frame({n: (stage/n).read_bytes() for n in REQUIRED_FILES if n.startswith('data/')})
        Xdev, Xtest, _, _ = split_data(source)
        bundle_id = uuid.uuid4().hex
        manifest = dict(schema_version=1, bundle_id=bundle_id, run_id=s['run_id'], published_utc=_now(),
            research_execution={'timestamp_utc': model['created_utc'], 'timestamp_basis': 'model creation timestamp; execution completion time unavailable'},
            source_counts={str(k): int(v) for k,v in source.income.value_counts().items()},
            split={'development_rows': len(Xdev), 'test_rows': len(Xtest),
                   'development_indices_sha256': _digest(np.asarray(Xdev.index,dtype='<i8').tobytes()),
                   'test_indices_sha256': _digest(np.asarray(Xtest.index,dtype='<i8').tobytes())}, files=files)
        (stage/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
        snapshot = validate_bundle(stage)
        os.replace(stage, bundles/bundle_id)
        pointer = dict(schema_version=1, bundle_id=bundle_id, manifest_sha256=snapshot.signature)
        with tempfile.NamedTemporaryFile(mode='w', prefix='.current-', dir=store, delete=False) as stream:
            pointer_tmp = Path(stream.name)
            json.dump(pointer, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(pointer_tmp, store/'current.json')
        return snapshot
    except BundleError:
        raise
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise BundleError(f'Publication rejected: {exc}') from exc
    finally:
        if stage.exists():
            shutil.rmtree(stage)
        if pointer_tmp is not None and pointer_tmp.exists():
            pointer_tmp.unlink()


@dataclass
class SnapshotSession:
    """One session's last-valid snapshot, retained when a new candidate fails."""
    snapshot: Snapshot | None = None
    error: str | None = None
    stale: bool = False
    loaded_utc: str | None = None
    _pointer: bytes | None = field(default=None, repr=False)

    def refresh(self, store: Path, force: bool = False) -> Snapshot | None:
        try:
            payload = (Path(store)/'current.json').read_bytes()
            if not force and payload == self._pointer and self.error is None:
                return self.snapshot
            candidate = _load_pointer(Path(store), payload)
            self.snapshot = candidate
            self._pointer = payload
            self.loaded_utc = _now()
            self.error = None
            self.stale = False
        except (OSError, BundleError) as exc:
            self.error = str(exc)
            self.stale = self.snapshot is not None
        return self.snapshot
