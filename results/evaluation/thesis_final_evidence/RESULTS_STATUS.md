# Evaluation data scope

Stage II in paper Tables 3 and 4 means **strict HardClamp**, not NoClamp or SoftClamp.

- `paper_models_per_target_metrics.csv` contains 108 historical target-summary records: 12 targets for each of the eight Table 3 methods, plus 12 Stage I targets. Percent columns are in percentage points, not fractions. Full-precision source numbers are preserved; this is a derived summary, not raw generated or docked molecules. Source dataset IDs, source-file SHA256, source row indices and evidence levels are included.
- Historical master/statistical tables, per-molecule property exports and their protocol metadata were backed up privately before removal from this public checkout. The retained target-summary CSV does not replace those raw and derived records or establish a complete reproducibility package.

## Verified scope and remaining mismatches

A prior audit found that the two Table 4 groups matched all 18 displayed values under the then-documented molecule-evaluation definitions and target-macro aggregation. The supporting property exports and protocol metadata are now archived privately, not included in this checkout. This does not prove historical training/checkpoint lineage or full paper reproduction. Novelty for these two groups means maximum Morgan similarity to target actives below 0.5, not exact absence from the training set.

The available historical target summaries match 66 of 72 displayed Table 3 numbers. Baseline raw experimental lineage remains partially verified even when a displayed value agrees. Six numbers remain unsupported at the supplied paper precision:

| Method | Metric | Paper | Available target-summary mean |
|---|---|---:|---:|
| REINVENT4 | Validity (%) | 97.13 | 100.0 |
| TamGen | Validity (%) | 99.75 | 100.0 |
| PAFlow | Validity (%) | 74.05 | 100.0 |
| DiffSBDD | Mean Vina (kcal/mol) | -6.7210 | -6.721475579628282 |
| MolCRAFT | SA | 4.0370 | 4.036888891594083 |
| MolCRAFT | Mean Vina (kcal/mol) | -7.5130 | -7.512579438711091 |

The three 100% validity entries come from corrected summaries of already-valid inputs; original generation-attempt denominators are not established by those summaries. Do not replace them with paper numbers. Other conflicting mean values also remain unchanged. No training, generation or docking was rerun, and no historical scientific values were edited to force agreement.
