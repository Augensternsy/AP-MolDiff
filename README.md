# AP-MolDiff

AP-MolDiff studies protein-sequence-conditioned continuous SMILES diffusion, Self-Conditioning and DLPS-E2PO preference optimization. The research builds on [TGM-DLM](https://github.com/CRIPAC-DIG/tgm-dlm). Public source uses the explicitly licensed Diffusion-LM/OpenAI foundations and author-authorized interfaces described in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

**This is a source and research-resource release, not a claim of full paper reproduction.** Original historical projection parameters, complete feature payloads, v3a-to-v3 lineage and final manuscript adoption remain unresolved. Existing model tests were not repeated for this publication. Weights and data are owner-authorized for distribution; upstream data/model terms remain separate from code MIT.

## Two stages and repository layout

Stage one learns a protein-conditioned continuous SMILES diffusion model; the SC path feeds a prior denoising prediction into the next model call. Stage two compares winner/loser denoising errors relative to an independently loaded, frozen reference, retaining the actual published E2PO objective. ESM-2 provides 640-dimensional protein-token features; the external adapter produces 1024-dimensional conditions, separate from the model's internal adapter. No PC-CFG is introduced.

```text
improved-diffusion/  # diffusion, Transformer, SC, stage-one entry points, tokenizer
scripts/            # preference preparation, E2PO training, stage-two evaluation
datasets/           # Python dataset modules; downloaded tables/features are not in Git
evaluation/         # candidate preparation, Vina, portable existing-score inputs
tools/              # standalone molecular-property and statistical tools
configs/            # BERT config and resource descriptions (not historical run snapshots)
checkpoints/        # download location; no checkpoint binaries in ordinary Git
data/               # data/source instructions; downloaded tables remain untracked
reproducibility/    # evidence boundaries
checkpoint_manifest.csv
dataset_manifest.csv
evaluation_manifest.csv
checksums.sha256
```

## Installation

The prior isolated CPU checks used Python 3.10, torch 2.6.0+cpu, Transformers 4.50.3 and the pinned RDKit/analysis dependencies. Install `uv` separately if unavailable. Do not upgrade a historical environment in place. The following corrected installation recipe is documented, not newly executed in this release:

```bash
python3.10 -m venv --without-pip --copies .venv-model
uv pip install --python .venv-model/bin/python "torch==2.6.0+cpu" --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv-model/bin/python -r requirements-model-sources.txt
. .venv-model/bin/activate
export PYTHONPATH="$PWD:$PWD/improved-diffusion:$PWD/improved-diffusion/scripts${PYTHONPATH:+:$PYTHONPATH}"
export WANDB_MODE=disabled
export SWANLAB_MODE=disabled
```

GPU/CUDA/DDP/FP16 and optimizer-resume compatibility are not certified by the CPU checks. SwanLab/wandb remain real dependencies even with tracking disabled. PyTDC is an additional dependency for rebuilding original preference inputs; its historical version/snapshot is not pinned by the basic model requirements. ESM's external model/tokenizer must be acquired separately from the [official model page](https://huggingface.co/facebook/esm2_t30_150M_UR50D), which lists upstream MIT terms. Do not confuse ESM weights with either diffusion checkpoint.

## Resource downloads and verification

The [Release](https://github.com/Augensternsy/AP-MolDiff/releases/tag/research-resources-20261115) hosts the following freshly hashed assets; none are ordinary Git blobs:

| Asset | Bytes | SHA256 |
|---|---:|---|
| [PLAIN_ema_0.9999_050000.pt](https://github.com/Augensternsy/AP-MolDiff/releases/download/research-resources-20261115/PLAIN_ema_0.9999_050000.pt) | 731593951 | `59d113a8aaaa4c77903b60136061b8476d06ddf382d83eb2ce2b476fef92ab87` |
| [direct_ki_e2po_15k.pt](https://github.com/Augensternsy/AP-MolDiff/releases/download/research-resources-20261115/direct_ki_e2po_15k.pt) | 731576807 | `2bd5e18825308b15a5f07ba31d3ead3ca3f7351e1984f9c6f5e9f2a6476582e8` |
| [ap_moldiff_stage1_data.tar.gz](https://github.com/Augensternsy/AP-MolDiff/releases/download/research-resources-20261115/ap_moldiff_stage1_data.tar.gz) | 31899991 | `3ec5f9e1a40f3c3f17578cc0f7841d01e57185e6ea1cfea6c623bb8dd74ace8a` |
| [ap_moldiff_e2po_data.tar.gz](https://github.com/Augensternsy/AP-MolDiff/releases/download/research-resources-20261115/ap_moldiff_e2po_data.tar.gz) | 48283472 | `34a2b4d9b9f6a52e304e32a5860471740bc7b9b3d8108ccd3cf4daf4648af441` |
| [ap_moldiff_targets.tar.gz](https://github.com/Augensternsy/AP-MolDiff/releases/download/research-resources-20261115/ap_moldiff_targets.tar.gz) | 174329 | `4bc97208cb78a22b3087d5e4a85f62f280379fe812a65730f14b76df99faa015` |
| [ap_moldiff_evaluation_records.tar.gz](https://github.com/Augensternsy/AP-MolDiff/releases/download/research-resources-20261115/ap_moldiff_evaluation_records.tar.gz) | 2924840 | `1a348effd02cc215a1ba2a1bd350a2298468b2f770787fba5761bfa454efe61b` |
| [ap_moldiff_feature_index.csv.gz](https://github.com/Augensternsy/AP-MolDiff/releases/download/research-resources-20261115/ap_moldiff_feature_index.csv.gz) | 54408904 | `3c199837500b9f6bac8f9d8be4f21642ac6481eee3e154ce80fcace219857a31` |

With GitHub CLI available, download and restore at the repository root. These are resource-dependent usage commands, not a new training/generation test:

```bash
mkdir -p downloads checkpoints
gh release download research-resources-20261115 --repo Augensternsy/AP-MolDiff --dir downloads
cp downloads/PLAIN_ema_0.9999_050000.pt checkpoints/
cp downloads/direct_ki_e2po_15k.pt checkpoints/
tar -xzf downloads/ap_moldiff_stage1_data.tar.gz -C .
tar -xzf downloads/ap_moldiff_e2po_data.tar.gz -C .
tar -xzf downloads/ap_moldiff_targets.tar.gz -C .
tar -xzf downloads/ap_moldiff_evaluation_records.tar.gz -C .
sha256sum -c checksums.sha256
```

`checksums.sha256` covers published source/configuration/manifest files, all seven payload assets under downloads/, the restored checkpoints, and dataset/evaluation members. It excludes itself to avoid a circular digest; its exact content is also anchored by the Git commit and Release asset digest. It does not claim current verification of individual cache files merely mentioned in the historical index. Only extract archives from the verified Release. No public download link is claimed for the missing projection or prepared receptors.

## Dataset acquisition and formats

| Deposited CSV | Actual data rows |
|---|---:|
| `datasets/train.csv` | 1153824 |
| `datasets/valid.csv` | 115 |
| `datasets/test.csv` | 115 |
| `data/bindingdb_ki/le_affinity_preserving_preference_pairs_delta1.0.csv` | 328157 |
| `data/bindingdb_ki/le_affinity_preserving_clean_ligands.csv` | 55095 |
| `data/bindingdb_ki/test_12_targets_heldout_pairs.csv` | 4490 |
| `data/bindingdb_ki/test_target_active_smiles.csv` | 87 |

The 12-target package contains 12 FASTA files, 12 active-reference CSVs and the 12-row sequence table. These are the deposited input versions, not a guarantee of canonical full-length proteins. Stage-one CSVs contain categorical SMILES/sequence IDs and need categorical_mappings.json; training caches are addressed by CSV row index as `datasets/output/train_desc_states/NNNN/ROW.pt`, NNNN=ROW//10000. Do not reorder rows. The actual validation/test tables each contain 115 records; their historical source/split/leakage provenance is unverified.

Preference rows contain protein_sequence, winner_smiles, loser_smiles, winner_pKi, loser_pKi and delta_pKi, plus stored selection properties. The 328,157 preference pairs are not 328,157 independent proteins; the existing selection summary reports 323 proteins. The held-out file has 4,490 pairs; final paper adoption remains unverified.

Use [BindingDB official versioned downloads](https://www.bindingdb.org/rwd/bind/chemsearch/marvin/Download.jsp) and its linked quarterly archives for original data. TDC [DTI BindingDB_Ki](https://tdcommons.ai/multi_pred_tasks/dti/) lists CC BY 3.0 US; preserve applicable attribution and terms. A current download is not the historical training snapshot. See [data/SOURCES.md](data/SOURCES.md) for source and distribution scope.

Preference rebuilding entry (not executed here; may download data, requires PyTDC and validated exclusion/filter settings):

```bash
python scripts/prepare_bindingdb_ki_preferences_v3.py --out_dir runs/new-preferences --delta_pki 1.0 --fasta_dir datasets/fasta --test_target_seq_file data/test_targets/test_target_sequences.csv
```

This does not establish the exact historical filtering/configuration. The external feature cache payloads are NOT in this release: the gzip index records 1,136,982 historical file entries and 16,842 missing paths, not a complete currently verified cache set. Missing/corrupt features can trigger zero-condition fallback; that is not valid historical conditioning.

## Stage-one TG and SC training

Templates only, not newly executed training. They require complete valid feature caches and separately acquired historical initialization files. The 200k ancestor and TG100k initializer are not included in this two-final-checkpoint release; no fabricated download links are supplied. Supply absolute paths in the environment variables before running. Use new output directories; the historical scripts can auto-resume.

```bash
(
  cd improved-diffusion/scripts
  python train_protein_170.py --csv_path ../../datasets/train.csv --pretrained_model_path "${APM_STAGE1_INITIALIZER:?Supply the verified historical initializer}" --checkpoint_path ../../runs/new-tg
)
(
  cd improved-diffusion/scripts
  python train_protein_self_cond.py --csv_path ../../datasets/train.csv --pretrained_model_path "${APM_TG100K_INITIALIZER:?Supply the verified TG100k initializer}" --checkpoint_path ../../runs/new-sc --self_cond True
)
```

These templates preserve current script defaults; they are not recovered immutable paper run commands. Synthetic CPU ordinary/SC optimizer checks previously passed; full training and historical input recovery remain unverified.

## Stage-two E2PO training

Template only. The policy/reference paths are explicit; the reference is independently loaded and frozen. Obtain and verify the original projection rather than using an arbitrary seed-42 candidate. Set beta from a documented experimental configuration; no default or calibration fallback is promoted to the paper value.

```bash
python scripts/train_direct_ki_e2po.py --csv_path data/bindingdb_ki/le_affinity_preserving_preference_pairs_delta1.0.csv --model_path checkpoints/PLAIN_ema_0.9999_050000.pt --reference_model_path checkpoints/PLAIN_ema_0.9999_050000.pt --esm2_path esm2_t30_150M_UR50D --projection_weights_path "${APM_HISTORICAL_PROJECTION:?Original projection not recovered: stop}" --beta "${APM_BETA:?Set documented experiment beta}" --output_dir runs/new-e2po
```

A prior synthetic update exercised the actual preference loss/update with independent frozen reference; it was not the historical v3 training. v3a-to-v3 save/copy/migration identity remains unverified.

## Protein-conditioned generation and SMILES decoding

PathB template, not executed for this release. Supply a legal local ESM/tokenizer directory and a trustworthy historical projection before treating results as historical reproduction:

```bash
python improved-diffusion/scripts/sample_protein_self_cond.py --model_path checkpoints/PLAIN_ema_0.9999_050000.pt --esm2_path esm2_t30_150M_UR50D --fasta_dir datasets/fasta --target_protein ESR1 --projection_weights_path "${APM_HISTORICAL_PROJECTION:?Original projection not recovered: stop}" --num_samples 2 --batch_size 1 --timestep_spacing 10 --clamp none --output_dir runs/new-pathb
```

Stage-two technical usage template, NOT a history-matched paper run. Its combined CLI has limited projection/FASTA path exposure and uses its existing initialization path; do not invent a `--projection_weights_path` option for it. Its default fixed-seed projection is not the recovered historical layer. `--dock_samples 0` disables actual docking:

```bash
python scripts/eval_ki_e2po_12_targets.py --checkpoint checkpoints/direct_ki_e2po_15k.pt --esm2_path esm2_t30_150M_UR50D --num_samples 2 --batch_size 1 --dock_samples 0 --clamp none --output_dir runs/new-v3-technical
```

It traverses 12 targets; it is not an ESR1-only CLI. Both sampling entries use the existing SMILES tokenizer/decoding interfaces. Stage-one output includes generated_ESR1.csv/SMILES; retain all failed/INVALID attempts. Do not reindex 37 active token IDs to replace the 312-row historical embedding capacity. No SMILES repair or mock output is introduced. PathB clamp=none still reaches the historical rounding callback, whereas v3 none does not. Strict weight loading, finite forward, executed sampling, valid SMILES and reproduced paper metrics are distinct claims.

## RDKit molecular evaluation

The standalone tool handles validity denominators, QED, SA and Ro5; it does not promise Diversity/Novelty/MMD. Use unknown scope if the file does not prove all attempts. Example for an existing downloaded historical generation table; not executed here:

```bash
python tools/ev_molecular_properties.py --gen-file generation_results_self_cond_50k_noclamp_1000/generated_ESR1.csv --input-scope unknown --output-dir runs/new-properties
```

Only use all-attempts when every attempt is recorded; valid-only records need a documented total. Empty/unknown totals give undefined generation validity. SA failure is NaN in this independent tool, unlike the old research helper's possible 3.0 fallback. Historical Diversity definitions differ and Novelty needs a versioned reference set; do not mix same-named columns across protocols.

## AutoDock Vina evaluation

Existing entry and real options, template only. Receptors and reference ligands are not distributed here; supply validated legally acquired files and a separately prepared candidate directory:

```bash
python evaluation/docking_candidate_selection/select_docking_candidates.py --help
python evaluation/evaluate_e2po_1000_docking_chunked.py --receptor-root resources/CrossDocked2020 --selected-root runs/selected-candidates --output-dir runs/new-vina
```

Prior limited two-ligand real docking is not the paper-scale protocol; no Vina run was performed in this publication. The archive includes existing Vina score records, not newly docked poses. Their success fraction uses selected docking attempts, not all generated molecules.

## Existing results, statistics and paper reproduction

Downloaded per-molecule records, raw selected Vina tables and historical aggregate/statistical tables are listed in evaluation_manifest.csv. Only metadata path fields were made portable; numerical values were not recomputed. The two Git-hosted release_inputs tables format the existing 24 model-target Vina tables for the standalone tool. Model labels retain historical naming and do not prove matching NoClamp semantics. Statistical template, not newly run:

```bash
MPLBACKEND=Agg python tools/audit_and_plot_mean_vina_target_loto.py --per-target-csv evaluation/release_inputs/per_target.csv --detailed-csv evaluation/release_inputs/detailed.csv --output-dir runs/new-statistics
```

Specify the comparison family, paired unit, one/two-sided Wilcoxon choice, BH correction and bootstrap design before interpreting results. The fixed-target LOTO analyses are dependent sensitivity checks. Do not turn non-significant results into improvement claims or transplant a q-value into another family. Existing paper tables are candidate archival artifacts until final manuscript adoption is bound to a checkpoint and batch.

## Availability and known limitations

- Two selected checkpoints, existing train/validation/test tables, Ki preference/clean/held-out records, 12-target inputs, portable existing evaluation records and a historical cache index are available in the Release. Other checkpoint candidates, full BindingDB/ESM copies, private logs and audits are excluded.
- The historical 640-to-1024 projection is UNVERIFIED. A recorded same-sequence cached/online comparison had RMSE about 0.160236. Fixed-seed audit candidates must not be presented as the original parameters.
- The approximately 1 TB cache payload set is not provided, and 16,842 expected paths remain unresolved. Only the index is released. Full first-stage historical training reproduction is therefore blocked.
- The PathB EMA has differing saved shared embedding/head keys; ordinary assign=False loading uses the later lm_head value. v3's keys agree. No historical checkpoint was repaired or resaved.
- v3a-to-v3 lineage, true initialization/reference identity and the final manuscript model/result adoption remain unverified. The 200k ancestor's first source is not established.
- A recent separate deployment N=1 completed sampling with zero RDKit-valid/returned molecules. It is not successful valid-molecule validation and not proof that the model fails overall. Earlier technical samples do not establish historical paper reproduction.
- Current resource JSONs are descriptions, not frozen original training configs. GPU/DDP/FP16/resume, exact historical data/split lineage and full numerical paper reproduction are not certified.

See [reproducibility/STATUS.md](reproducibility/STATUS.md). Code MIT applies only within the author-authorized scope in LICENSE/THIRD_PARTY_NOTICES; it does not blanket-relicense datasets, model weights, Diffusion-LM, Transformers, ESM or third-party original data. TGM-DLM citation: Gong, Haisong; Liu, Qiang; Wu, Shu; Wang, Liang. *Text-Guided Molecule Generation with Diffusion Language Model*. AAAI 2024, 38(1), 109-117, DOI:10.1609/aaai.v38i1.27761. No unverified AP-MolDiff paper citation is invented.
