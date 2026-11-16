# PAFlow: local baseline output provenance and scope

These are historical **local reproduction outputs** and clearly marked derived
indexes from AP-MolDiff's baseline experiments. They are not claimed to be official
sample downloads or original implementations of the upstream method.

The repository owner explicitly authorized public hosting of independently
generated local outputs in this release. The existing targeted audit identified
no explicit prohibition on redistribution of these particular output records.
Earlier UNVERIFIED findings are not relabeled as third-party dataset grants:
publication here relies on the owner's authorization for the local outputs, not
on automatically converting a software license into an output-data license.
No blanket MIT or new third-party data license is assigned to these data, and
no endorsement, commercial clearance, or rights in excluded inputs are asserted.

## Method attribution and upstream terms

- Original implementation and authors: [https://github.com/CMACH508/PAFlow](https://github.com/CMACH508/PAFlow).
- Original method/paper: [https://arxiv.org/abs/2509.01486](https://arxiv.org/abs/2509.01486).
- Observed upstream terms: No explicit upstream license was found in the reviewed repository; no upstream source code, model or input-structure bundle is redistributed here.

## Actual data and unverified limitations

There are 8,886 existing single-molecule SDF files from 12 local runs, plus derived SMILES indexes. The original runs had 12,000 numeric predictions: 3,114 predictions have no saved SDF and are not presented as valid molecules. The recovered_numeric_predictions directories contain losslessly extracted final positions and encoded type indices, per-record payload checksums and metadata. They are existing numeric outputs, not new chemical reconstructions or 12,000 decoded/valid molecules. Do not add index rows and SDF records as independent molecules.

The 12 original result.pt trajectory/input-context files remain private because they also contain third-party structural inputs. Metadata source_pt references describe those withheld historical sources, not promised public downloads. Full intermediate trajectories are not supplied by the final-prediction NPZ files.

Only eight complete target Vina CSVs are included (800 rows). ESR1, P2RX3, IDH1 and CCR9 target CSVs are withheld whole because six individual generation/score associations remain unresolved across them; no original rows were filtered or rewritten to manufacture a complete cohort. The included targets have existing complete molecule/selection association evidence.

No molecules were generated, reconstructed or docked again for this upload. Original record bytes and filenames are unchanged. Missing rows, failed scores and unresolved provenance are not filled or silently treated as successful experiments.
