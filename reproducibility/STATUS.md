# Reproduction status

The release makes selected source and research artifacts available. It does not certify full paper reproduction.

- Reused prior evidence: CPU synthetic ordinary/SC forward/backward and optimizer tests; synthetic E2PO update with independent/frozen reference; strict 337-key PathB/v3 loading and finite synthetic forward; tokenizer contracts; standalone property/statistical regressions; limited real docking evidence.
- Not repeated for this publication: training, checkpoint loading tests, molecule generation, molecular scoring, statistical inference and Vina.
- Historical 640-to-1024 projection parameters remain unverified. A recorded same-sequence cached/online comparison had RMSE about 0.160236. A seed-42 audit projection is not historical recovery.
- The cache payload set is about 1 TB and has 16,842 recorded missing paths. Only a portable historical index is released. The loader's zero-condition fallback is not evidence of complete valid conditioning.
- The latest separate deployment N=1 sampling completed but produced zero RDKit-valid molecules. It is neither a successful valid-molecule reproduction nor proof of overall model failure.
- v3a-to-v3 lineage, the historical reference/initial model identity and the final manuscript's checkpoint/result adoption remain unverified.
- PathB's saved tied embedding/head conflict is retained; ordinary loading takes the later lm_head value. v3 tied keys agree. PathB none still uses rounding; v3 none uses no callback.
- Diversity has differing historical definitions; old SA evaluation can substitute 3.0 on error, while the independent tool records NaN. Validity needs a real attempt denominator, novelty needs a versioned reference, and BH conclusions belong to their specified comparison family.
- Stage-one source corpus/version and split provenance are incomplete. The two released 115-row validation/test tables are the actual existing files, not newly established leakage-free benchmark splits.
- Publication is owner-authorized, but source/data/weight historical identity is not inferred from that authorization. Third-party terms remain applicable.

Compressed member digests and archive SHA256 were recomputed for this release. Internal server path values in evaluation metadata were made project-relative or reduced to package-relative/basename identifiers; scientific scores were not recomputed. The two small statistics input tables reformat existing selected-docking scores. Their success denominator is the selected docking rows, not all generated attempts. They are not a new paper analysis.
