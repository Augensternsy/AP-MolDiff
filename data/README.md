# Research data

Download the release archives linked in `../dataset_manifest.csv` and extract them at the repository root. CSV IDs in the stage-one tables require the deposited categorical mapping. Keep row order when locating per-row protein features. The validation/test tables contain 115 rows each; their historical paper adoption and split/leakage properties have not been certified.

The feature-index gzip contains 1,153,824 relative-path records, including 16,842 recorded missing paths. Its per-cache SHA256 values are historical index metadata, not fresh verification. Protein feature payloads and the original projection parameters are not included. Do not train on the dataset loader's zero-condition fallback and call that a historical reproduction.

See SOURCES.md for attribution, upstream acquisition and distribution scope. No private logs or original all-BindingDB dump are included.
