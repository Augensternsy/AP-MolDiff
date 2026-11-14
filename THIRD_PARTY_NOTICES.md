# Source and dependency notices

MIT LICENSE in this repository: Copyright (c) 2026 Wsy. Its scope is the two
standalone tools, twelve author-confirmed research entry/data/evaluation source
files listed below, the explicitly authorized local collate contribution,
and new original compatibility adapter/test/documentation contributions.
This is not a blanket relicensing of inherited third-party source.

The twelve author-confirmed research source files are:

- `improved-diffusion/scripts/train_protein_170.py`
- `improved-diffusion/scripts/train_protein_self_cond.py`
- `improved-diffusion/scripts/sample_protein_self_cond.py`
- `scripts/eval_ki_e2po_12_targets.py`
- `scripts/train_direct_ki_e2po.py`
- `scripts/prepare_bindingdb_ki_preferences_v3.py`
- `datasets/__init__.py`
- `datasets/ki_preference_dataset.py`
- `improved-diffusion/scripts/extract_esm.py`
- `improved-diffusion/scripts/process_train_csv.py`
- `evaluation/docking_candidate_selection/select_docking_candidates.py`
- `evaluation/evaluate_e2po_1000_docking_chunked.py`

## Licensed foundations retained

- Diffusion-LM, https://github.com/XiangLi1999/Diffusion-LM,
  commit 759889d58ef38e2eed41a8c34db8032e072826f4, Apache-2.0.
  Exact mathematical/training foundation bodies are retained in gaussian_diffusion.py
  and train_util.py under clearly named `_Licensed...` classes. Respacing formulas,
  seven script_util helpers and three test_util helpers are reused with their
  Apache-2.0 terms. This is explicit reuse, not an originality claim.
- Existing logger.py, losses.py and unet.py retain the
  previously reviewed Diffusion-LM / Hugging Face source and original comments.
- OpenAI improved-diffusion, commit 1bc7bbbdc414d83d4abf2ad8cc1446dc36c4e4d5,
  https://github.com/openai/improved-diffusion, MIT, Copyright (c) OpenAI.
  Existing __init__.py and nn.py and inherited OpenAI portions retain those terms.
- fp16_util.py and resample.py have byte-identical licensed Diffusion-LM ancestors;
  Wsy authorized their local None-gradient/CPU changes. resample also replaces the
  removed NumPy np.int alias with int. No sampling formula is changed by that edit.
- OpenAI Baselines MIT notices for logger ancestry are retained separately.
- Hugging Face Transformers: https://github.com/huggingface/transformers,
  Apache-2.0; use external Transformers 4.50.3 BertEncoder/AutoConfig/ESM interfaces.
  The small bundled config is generated from BertConfig defaults. No private
  model weights, tokenizer downloads or vendored Transformers distribution is shipped.
- The actual license texts are retained in THIRD_PARTY_LICENSES; retain all
  upstream source comments and those texts when redistributing affected files.

## Newly expressed interfaces and author-authorized research

No TGM-DLM class/function body is used as the source of the new adapters. They
are expressed from frozen input/output, tensor-schema, mathematical and RNG
behavior requirements, using explicitly licensed foundations. They preserve
legacy quirks intentionally; finite tests are not proof of universal equivalence.
This work does not claim a legally certified clean-room development process.

The author confirmed local contributions were their own and authorized MIT
publication, without outstanding copied-source, coauthor or institutional
restrictions. That declaration does not license third-party inherited work.
The datasets/mytokenizers release interfaces exclude the old TGM ChEBI and unused
legacy tokenizer branches; no unlicensed TGM implementation is redistributed.

TGM-DLM research lineage: https://github.com/CRIPAC-DIG/tgm-dlm .
Gong, Haisong; Liu, Qiang; Wu, Shu; Wang, Liang. Text-Guided Molecule Generation
with Diffusion Language Model. AAAI 2024. Attribution is not a grant of code rights.

## External dependencies and scientific assets

NumPy, pandas, PyTorch, RDKit, SciPy, statsmodels, Matplotlib, Transformers,
safetensors, blobfile, SwanLab, wandb, Vina, Meeko, Gemmi and tqdm are installed
separately with their respective licenses; they are not relicensed as Wsy code.
RDKit SA_Score BSD-3-Clause notice is retained in THIRD_PARTY_LICENSES.
No checkpoints, ESM weight copies, BindingDB/private data, internal reports or
server symlinks are included. Research/model/data access permissions remain separate.
