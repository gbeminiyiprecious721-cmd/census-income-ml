# Data card

## Source and attribution

Becker, B. & Kohavi, R. (1996). *Adult* [Dataset]. UCI Machine Learning Repository. [doi:10.24432/C5XW20](https://doi.org/10.24432/C5XW20). The [UCI source](https://archive.ics.uci.edu/dataset/2/adult) describes extraction from the 1994 Census database and distributes the dataset under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

The two original files, `adult.data` and `adult.test`, contain 48,842 rows, 14 predictors and an income label. Preserve their source URLs and SHA-256 hashes. Historical hashes appear in `artifacts/historical/provenance.json`; current data-quality reports should record the actual files used by a fresh run.

## Cleaning and analytical population

Whitespace and the test-file target suffix are normalised. Question-mark missing markers become missing values; labels are validated before encoding. Exact full-record duplicates are removed before a seeded, stratified 80/20 split. The historical cleaned population contains 48,790 rows after removing 52 duplicates: 39,032 development and 9,758 test records.

This pools and resplits the original UCI files to preserve the reviewed benchmark. Results are not directly an evaluation on the original UCI test partition. Two predictor profiles occur in both partitions with conflicting labels; full-record deduplication cannot resolve this overlap. Any alternative split should be a separately labelled experiment rather than an undocumented benchmark change.

## Variables and transformation

Numeric predictors include age, education number, capital gain/loss and weekly hours. Categorical fields describe employment, education, household relationships and country. Race and sex support subgroup diagnostics; the final prediction input excludes them. The census weight (`fnlwgt`) and duplicate education representation are not final model inputs.

Fixed education categories and age × hours are computed per record. Imputation and categorical encoding fit inside training folds and are saved with the estimator. Missing values use fitted imputers. Unseen categories are handled by the encoder and disclosed through inference warnings. The app displays aggregates and provides explicit browsing/search/export of the public historical UCI records. It does not expose private uploads or person identifiers.

## Acquisition and reproducibility

Use the package's configured raw-data directory (`data/raw/`) for the two unchanged source files. The data loader supports explicit downloading through its `download` argument; consult the training CLI help for exposed acquisition options. Retain raw files and provenance so offline reruns can use the same bytes. Record cleaning counts, class distribution, missingness, split sizes and cross-split overlap in `reports/data_quality.json`.

## Coverage and limitations

The source is a selected historical US population with documented extraction restrictions, missing values and coarse demographic categories. Labels use a historical nominal $50,000 boundary. Results do not establish current population representativeness, causal relationships or individual income validity. Subgroup and intersectional coverage is incomplete; rate estimates can be unstable when outcome denominators are small. Neither removal of protected attributes nor aggregate accuracy establishes fairness.
