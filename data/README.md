# Research data location

The published inputs are documented in [datasets/README.md](../datasets/README.md), with attribution and availability in [datasets/SOURCES.md](../datasets/SOURCES.md).

Training inputs are under `datasets/stage1/` and `datasets/e2po/`; target references are under `datasets/targets/`. The historical feature index is under `datasets/feature_index/` and is not the actual tensor cache. Historical generation and evaluation records are under `results/generation/` and `results/evaluation/`. No scientific data values were changed by this reorganization. Run `git lfs pull` to obtain the LFS-backed inputs.
