# Evaluation data scope

Stage II in paper Tables 3 and 4 means **strict HardClamp**, not NoClamp or SoftClamp.

- `paper_models_per_target_metrics.csv` contains 108 historical target-summary records: 12 targets for each of the eight Table 3 methods, plus 12 Stage I targets. Percent columns are in percentage points, not fractions. Full-precision source numbers are preserved; this is a derived summary, not raw generated or docked molecules. Source dataset IDs, source-file SHA256, source row indices and evidence levels are included.
- The unchanged original `unified_master_results_table.csv` mixes older experiment versions and different diversity definitions. It is retained because it also contains evidence relevant to the paper; it is **not** a fully verified final Table 3.
- The unchanged 76-row `unified_statistical_tests_with_ci.csv` is retained as unique statistical evidence, including comparisons outside Tables 3 and 4. Several baseline inputs use older evaluation versions. Internal BH arithmetic consistency does not establish paper-batch consistency or reproduce all bootstrap intervals.
- `../unified_molecular_properties_20261104/TG_PathB50k_HistoricalNoClamp_molecules.csv` corresponds row by row to the published Stage I generation batch.
- `../unified_molecular_properties_20261104/E2PO_v3_15k_HardClamp_molecules.csv` corresponds row by row to the published Clamping generation batch. Only `source_csv` metadata has been remapped to repository-relative paths of byte-identical published source files; all molecular, numerical, checkpoint-hash and source-hash fields remain unchanged.
- The StrictNoClamp molecule-property export is outside the scope of Tables 3 and 4 and has been archived outside this Git repository before removal. Its historical evidence and the unmodified statistical test table are preserved. `environment_and_rules.json` is the unchanged historical protocol record and describes both comparisons; it does not claim that both exports remain in this checkout.

## Verified scope and remaining mismatches

The two Table 4 groups match all 18 displayed values under the documented molecule-evaluation definitions and target-macro aggregation. This does not prove historical training/checkpoint lineage or full paper reproduction. Novelty for these two groups means maximum Morgan similarity to target actives below 0.5, not exact absence from the training set.

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
