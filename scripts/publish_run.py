"""Validate and activate existing trusted research locally. Never train or deploy."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from census_income.run_store import BundleError, publish_existing


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--store', type=Path, default=ROOT/'artifacts/research_store')
    args = parser.parse_args()
    try:
        snapshot = publish_existing(ROOT, args.store)
    except BundleError as exc:
        parser.exit(1, f'Publication rejected; previous pointer preserved: {exc}\n')
    print(f"Validated run {snapshot.manifest['run_id']} activated locally in {args.store}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
