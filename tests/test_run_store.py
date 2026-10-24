"""Contract and failure tests operate only on temporary publication stores."""
import hashlib
import json
from pathlib import Path
import shutil

import pandas as pd
import pytest

from census_income.run_store import BundleError, SnapshotSession, load_snapshot, publish_existing

ROOT = Path(__file__).resolve().parents[1]

@pytest.fixture(scope="module")
def published(tmp_path_factory):
    store = tmp_path_factory.mktemp("verified-store")
    publish_existing(ROOT, store)
    return store

@pytest.fixture
def store(published, tmp_path):
    dest = tmp_path / "store"
    shutil.copytree(published, dest)
    return dest

def rewrite(store, name, mutate):
    """Re-sign an intentionally bad fixture to test semantics beyond checksums."""
    pointer = json.loads((store / "current.json").read_text())
    bundle = store / "bundles" / pointer["bundle_id"]
    manifest = json.loads((bundle / "manifest.json").read_text())
    path = bundle / name
    mutate(path)
    payload = path.read_bytes()
    manifest["files"][name]["sha256"] = hashlib.sha256(payload).hexdigest()
    manifest["files"][name]["bytes"] = len(payload)
    (bundle / "manifest.json").write_text(json.dumps(manifest))
    pointer["manifest_sha256"] = hashlib.sha256((bundle / "manifest.json").read_bytes()).hexdigest()
    (store / "current.json").write_text(json.dumps(pointer))

def test_real_snapshot_and_source_categories(published):
    snapshot = load_snapshot(published)
    assert snapshot.source_counts == {"<=50K":37109, ">50K":11681}
    assert snapshot.report("summary.json")["test_metrics"]["accuracy"] == 0.8322402131584341
    assert snapshot.manifest["split"]["test_rows"] == 9758
    assert snapshot.manifest["research_execution"]["timestamp_basis"] != "publication"

@pytest.mark.parametrize("case", ["corrupt", "missing", "schema", "counts", "subgroups", "run", "compatibility", "category_keys", "category_distribution", "split", "execution"])
def test_invalid_bundles_rejected(store, case):
    pointer = json.loads((store / "current.json").read_text())
    bundle = store / "bundles" / pointer["bundle_id"]
    if case == "corrupt":
        (bundle / "reports/summary.json").write_text("{}")
    elif case == "missing":
        (bundle / "reports/roc_curve.csv").unlink()
    elif case == "schema":
        rewrite(store, "reports/confusion_matrix.csv", lambda p: pd.DataFrame({"incorrect":[1]}).to_csv(p,index=False))
    elif case in {"counts", "subgroups"}:
        name = "confusion_matrix.csv" if case == "counts" else "subgroup_metrics.csv"
        def edit(p):
            frame = pd.read_csv(p)
            frame.loc[0, "count" if case == "counts" else "fp"] += 1
            frame.to_csv(p,index=False)
        rewrite(store, "reports/"+name, edit)
    else:
        name = "artifacts/model_manifest.json" if case == "compatibility" else "reports/summary.json"
        if case in {"category_keys", "category_distribution"}: name = "reports/eda_summary.json"
        if case == "execution": name = "reports/execution_validation.json"
        def edit(p):
            data = json.loads(p.read_text())
            if case == "run": data["run_id"] = "different"
            if case == "compatibility": data["dependencies"]["numpy"] = "incompatible"
            if case == "category_keys": data["income_counts"] = {"bad":37109, ">50K":11681}
            if case == "category_distribution": data["income_counts"] = {"<=50K":37108, ">50K":11682}
            if case == "split": data["dataset"]["test_rows"] -= 1
            if case == "execution": data["error_outputs"] = 1
            p.write_text(json.dumps(data))
        rewrite(store,name,edit)
    with pytest.raises(BundleError): load_snapshot(store)

def test_last_valid_snapshot_never_mixes_files(store):
    session = SnapshotSession()
    original = session.refresh(store)
    expected = original.report("confusion_matrix.csv")
    (store / "current.json").write_text('{"broken":true}')
    assert session.refresh(store) is original
    assert session.error and session.stale
    pd.testing.assert_frame_equal(session.snapshot.report("confusion_matrix.csv"), expected)
    empty = SnapshotSession()
    assert empty.refresh(store) is None and empty.error

def test_valid_transition_and_manual_refresh(store):
    session = SnapshotSession()
    first = session.refresh(store)
    assert session.refresh(store) is first
    assert session.refresh(store, force=True) is not first
    publish_existing(ROOT, store)
    new = session.refresh(store)
    assert new.signature != first.signature and not session.stale
    assert len(list((store / "bundles").iterdir())) == 2

def test_pointer_resolved_once_and_bundle_bytes_frozen(store, monkeypatch):
    original = Path.read_bytes
    reads = []
    def read(path):
        if path == store / "current.json": reads.append(path)
        return original(path)
    monkeypatch.setattr(Path,"read_bytes",read)
    snapshot = load_snapshot(store)
    assert len(reads) == 1
    bundle = store / "bundles" / snapshot.manifest["bundle_id"]
    (bundle / "reports/summary.json").write_text("{}")
    assert snapshot.report("summary.json")["run_id"] == "20261022T103759Z"

def test_failed_publication_leaves_pointer_intact(store, monkeypatch):
    import census_income.run_store as runs
    previous = (store / "current.json").read_bytes()
    def fail(*args, **kwargs): raise BundleError("incomplete staged output")
    monkeypatch.setattr(runs,"validate_bundle",fail)
    with pytest.raises(BundleError): publish_existing(ROOT,store)
    assert (store / "current.json").read_bytes() == previous

def test_pointer_cannot_escape_store(store):
    (store / "current.json").write_text(json.dumps({"schema_version":1,"bundle_id":"../../elsewhere","manifest_sha256":"0"*64}))
    with pytest.raises(BundleError): load_snapshot(store)

def test_snapshot_inference_exact_parity(published, valid_record):
    from census_income.inference import predict_records
    from census_income.snapshot_inference import predict_snapshot_records
    # The forest sums trees on threads. Sequential scheduling removes summation
    # order noise for a bit-exact parity check without changing saved parameters.
    from joblib import parallel_backend
    with parallel_backend('sequential'):
        expected = predict_records(valid_record)
        actual = predict_snapshot_records(load_snapshot(published), valid_record)
    pd.testing.assert_frame_equal(actual,expected,check_exact=True)


def test_recovered_pointer_clears_stale_status(store):
    session = SnapshotSession()
    original = session.refresh(store)
    pointer = (store/'current.json').read_bytes()
    (store/'current.json').write_text('{}')
    assert session.refresh(store) is original and session.stale
    (store/'current.json').write_bytes(pointer)
    assert session.refresh(store).signature == original.signature
    assert not session.stale and session.error is None


def test_snapshot_callers_cannot_mutate_retained_evidence(published):
    snapshot = load_snapshot(published)
    snapshot.manifest['source_counts']['>50K'] = 1
    snapshot.report('summary.json')['test_metrics']['f1'] = 0
    frame = snapshot.report('confusion_matrix.csv')
    frame.loc[:, 'count'] = 0
    source, _ = snapshot.source_data()
    source.loc[:, 'income'] = '>50K'
    assert snapshot.source_counts['>50K'] == 11681
    assert snapshot.report('summary.json')['test_metrics']['f1'] == 0.6990255561684133
    assert snapshot.report('confusion_matrix.csv')['count'].sum() == 9758
    assert snapshot.source_data()[0]['income'].value_counts().to_dict() == {'<=50K': 37109, '>50K': 11681}


@pytest.mark.parametrize('name,column,value', [
    ('roc_curve.csv', 'tpr', 1.5),
    ('precision_recall_curve.csv', 'precision', -0.1),
    ('feature_comparison.csv', 'f1', 0.1),
    ('calibration.csv', 'count', -1),
])
def test_resigned_semantic_errors_rejected(store, name, column, value):
    def edit(path):
        frame = pd.read_csv(path)
        frame.loc[0, column] = value
        frame.to_csv(path, index=False)
    rewrite(store, 'reports/'+name, edit)
    with pytest.raises(BundleError):
        load_snapshot(store)


def test_subgroup_population_cannot_be_shifted_between_groups(store):
    def edit(path):
        frame = pd.read_csv(path)
        # Shift a correctly classified negative between groups and recompute rates.
        # Overall totals and within-row arithmetic still agree.
        for index, sign in [(0, -1), (1, 1)]:
            frame.loc[index, ['tn', 'n', 'negatives']] += sign
            row = frame.loc[index]
            expected = {'accuracy': (row.tn+row.tp)/row.n,
                        'precision': row.tp/(row.tp+row.fp),
                        'recall': row.tp/row.positives,
                        'f1': 2*row.tp/(2*row.tp+row.fp+row.fn)}
            for key, value in expected.items():
                frame.loc[index, 'recall_tpr' if key == 'recall' else key] = value
            frame.loc[index, 'fpr'] = row.fp / row.negatives
            frame.loc[index, 'fnr'] = row.fn / row.positives
        frame.to_csv(path, index=False)
    rewrite(store, 'reports/subgroup_metrics.csv', edit)
    with pytest.raises(BundleError):
        load_snapshot(store)


def test_roc_auc_must_agree_with_curve(store):
    def edit(path):
        frame = pd.read_csv(path)
        frame['tpr'] = frame['tpr'] ** 2
        frame.to_csv(path, index=False)
    rewrite(store, 'reports/roc_curve.csv', edit)
    with pytest.raises(BundleError):
        load_snapshot(store)


@pytest.mark.parametrize('case', ['missing_limits', 'limits_type', 'limits_item', 'missing_group'])
def test_overview_required_metadata_rejected_before_activation(store, case):
    def edit(path):
        data = json.loads(path.read_text())
        if case == 'missing_limits': data.pop('limitations')
        elif case == 'limits_type': data['limitations'] = 'not a list'
        elif case == 'limits_item': data['limitations'] = [1]
        else: data['selected_feature_group'] = 'missing-group'
        path.write_text(json.dumps(data))
    rewrite(store, 'reports/summary.json', edit)
    if case == 'missing_group':
        def model_edit(path):
            data = json.loads(path.read_text())
            data['metadata']['selected_feature_group'] = 'missing-group'
            path.write_text(json.dumps(data))
        rewrite(store, 'artifacts/model_manifest.json', model_edit)
    with pytest.raises(BundleError):
        load_snapshot(store)


@pytest.mark.parametrize('bounds', [(2, 3), (.5, .1), (.04, .05)])
def test_invalid_or_inaccurate_wilson_intervals_rejected(store, bounds):
    def edit(path):
        frame = pd.read_csv(path)
        frame.loc[0, ['fpr_ci_low', 'fpr_ci_high']] = bounds
        frame.to_csv(path, index=False)
    rewrite(store, 'reports/subgroup_metrics.csv', edit)
    with pytest.raises(BundleError):
        load_snapshot(store)


@pytest.mark.parametrize('metric,value', [('precision', .4), ('recall', .2), ('f1', .3), ('roc_auc', .7)])
def test_majority_baseline_must_match_constant_prediction(store, metric, value):
    def edit(path):
        frame = pd.read_csv(path)
        frame.loc[frame.model == 'Majority-class baseline', metric] = value
        frame.to_csv(path, index=False)
    rewrite(store, 'reports/model_comparison.csv', edit)
    with pytest.raises(BundleError):
        load_snapshot(store)
