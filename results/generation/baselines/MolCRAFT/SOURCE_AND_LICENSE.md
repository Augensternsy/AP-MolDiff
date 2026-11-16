# MolCRAFT: local baseline output provenance and scope

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

- Original implementation and authors: [https://github.com/GenSI-THUAIR/MolCRAFT/tree/master/MolCRAFT](https://github.com/GenSI-THUAIR/MolCRAFT/tree/master/MolCRAFT).
- Original method/paper: [https://arxiv.org/abs/2404.12141](https://arxiv.org/abs/2404.12141).
- Observed upstream terms: CC-BY-NC-SA-4.0 for the upstream project. This notice does not automatically assign that license to independently generated output data.
- The complete upstream text is retained in `UPSTREAM_SOFTWARE_LICENSE.txt` as a software/model notice, **not** a new license imposed on locally generated outputs.

## Actual data and unverified limitations

The historical original SDF files retain 11,787 records. Those 12 source SDF files are withheld because their per-molecule titles embed private server absolute paths; no original SDF content was edited for publication. This public directory instead contains the existing derived SMILES-index CSVs covering those 11,787 original SDF records. Source_File in each index identifies its withheld historical SDF source, not a promised public download. The 213 missing records relative to the requested 12,000 were not recovered or synthesized. Derived SMILES indexes are secondary views of existing SDF records, not additional generated molecules. The 1,200 local selected Vina rows have existing generation membership links. Full-precision macro SA and Vina differ from the paper's four-decimal display; do not replace raw values with paper numbers or claim complete paper reproduction.

No molecules were generated, reconstructed or docked again for this upload. Original record bytes and filenames are unchanged. Missing rows, failed scores and unresolved provenance are not filled or silently treated as successful experiments.
