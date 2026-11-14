# Selected checkpoints

Only PathB-50k and Ki-E2PO-v3-15k are distributed. See `../checkpoint_manifest.csv` for exact bytes, freshly computed SHA256 and Release download URLs. Store downloaded weights here; checkpoint binaries are excluded from ordinary Git history.

The JSON files under configs describe the present public model/resource interface; they are not recovered immutable historical training configurations. Use the existing sampling scripts, not a fictional generic `--config` flag. ESM-2 and the original external projection are separate resources. The 200k ancestor and TG100k initialization needed for the historical training progression are not included in this two-checkpoint release.

PathB tied embedding/head values differ in the saved EMA file; the tested ordinary assign=False loading uses the later lm_head copy. v3's two stored keys agree. Do not rewrite or repair these historical files silently. The v3a-to-v3 training/save/migration identity remains unverified.
