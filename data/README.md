# Research data

Current checkout data is grouped under `../datasets/stage1/`, `../datasets/e2po/`, `../datasets/targets/`, `../datasets/evaluation_records/` and `../datasets/feature_index/`. Run `git lfs pull` to obtain LFS-backed contents. The existing GitHub Release remains a separate archival download source. CSV IDs in the stage-one tables require the deposited categorical mapping. Keep row order when locating per-row protein features. The validation/test tables contain 115 rows each; their historical paper adoption and split/leakage properties have not been certified.

The feature-index gzip contains 1,153,824 relative-path records, including 16,842 recorded missing paths. Its per-cache SHA256 values are historical index metadata, not fresh verification. Protein feature payloads and the original projection parameters are not included. Do not train on the dataset loader's zero-condition fallback and call that a historical reproduction.

See SOURCES.md for attribution, upstream acquisition and distribution scope. No private logs or original all-BindingDB dump are included.
