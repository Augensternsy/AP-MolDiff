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

All **12 original target Vina CSVs are now included (1,200 rows, 100 per target)**.
The eight previously public CSVs (800 rows) are unchanged; the four additions are
ESR1, P2RX3, IDH1 and CCR9 (400 original rows). All 1,200 File IDs point to existing
source SDF files and match the source-file/cleaned-SMILES selection snapshots.

**Five stereochemical structure associations remain UNVERIFIED.** There are
1,194 direct canonical-SMILES matches plus one additional chirality-aware graph
isomorphism (P2RX3/433.sdf), giving 1,195 strict structure matches. The other five
records are retained without changing their SMILES, original source SDF, scores
or missing values; including the original files does not resolve their stereo
differences or prove full paper reproduction.

| Target | Original File ID | Unresolved issue |
|---|---|---|
| ESR1 | 624.sdf | Stereo information differs; original Vina/LE remain nan |
| P2RX3 | 975.sdf | Stereo information differs |
| P2RX3 | 395.sdf | Stereo information differs |
| IDH1 | 846.sdf | Stereo information differs |
| CCR9 | 798.sdf | Stereo information differs |

These are the original locally saved evaluation CSVs. Different historical
full/partial-pool backup CSVs are not spliced into them, and no original rows are
filtered or rewritten to manufacture a verified cohort. No explicit output-data
redistribution ban was identified in the reused review; public research hosting
is authorized by the repository owner. This does not redistribute the private
input-context PT files or grant rights in upstream source/models.

| Target | Original Vina file | Rows | SHA256 |
|---|---|---:|---|
| ESR1 | vina_evaluated_100/vina_ESR1.csv | 100 | `020f774b7e38bb386ab6f7ccba9c5055ce41e8ad8eba9d20c3acb2fcd12773b1` |
| P2RX3 | vina_evaluated_100/vina_P2RX3.csv | 100 | `f2a3b0264fec82b2561a4e3ea2f8530538c8b2c5792b731adb61761a52028c04` |
| IDH1 | vina_evaluated_100/vina_IDH1.csv | 100 | `9bb50316e8016312f6b76f73c35ec97f2c2a439fc266690a69646a18eed56ddf` |
| CCR9 | vina_evaluated_100/vina_CCR9.csv | 100 | `ac7678aea742d67bb7743f73e1aee55a3174f8a44c4a41285b66362017267fb3` |
| HCRTR1 | vina_evaluated_100/vina_HCRTR1.csv | 100 | `332928dbf04e28ccedebbd3b02fdc1c078bb3e69b59ee447486b68089e239a18` |
| JAK1 | vina_evaluated_100/vina_JAK1.csv | 100 | `871863ae68d50be49a53debefcb0ee8311d550c42b1c3efb4a5731ca938c02eb` |
| KDM1A | vina_evaluated_100/vina_KDM1A.csv | 100 | `24d1e2d2cb4ff4d8044ab7b9bda300f30b7f08fab0ec21100551053e1c8e6172` |
| RIOK1 | vina_evaluated_100/vina_RIOK1.csv | 100 | `b826bfb2c71f15492ffa4b0eb1ccf4f8b4e909a1a21dd5427a492034557e7a23` |
| NR4A1 | vina_evaluated_100/vina_NR4A1.csv | 100 | `65337027c0299a8d2cba127650f2c5356bc88e31c60625cdbb3e2902af7ac4d0` |
| GRIK1 | vina_evaluated_100/vina_GRIK1.csv | 100 | `5ec0ee39fafc575fb875ccadd2933b0e84f4201fc6a4314e616e139fe5b46d15` |
| FTO | vina_evaluated_100/vina_FTO.csv | 100 | `eb074f8f2c5103f077659978d16c19638bd264e03871b56973ae5bfb7e33ca27` |
| SPIN1 | vina_evaluated_100/vina_SPIN1.csv | 100 | `dabdc0fa0093d2c35fc898f0802f2b673603fb88f1a199d1af9e23f2b8973a03` |

No molecules were generated, reconstructed or docked again for this upload. Original record bytes and filenames are unchanged. Missing rows, failed scores and unresolved provenance are not filled or silently treated as successful experiments.
