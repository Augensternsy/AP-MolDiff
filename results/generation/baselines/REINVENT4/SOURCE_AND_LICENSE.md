# REINVENT4: local baseline output provenance and scope

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

- Original implementation and authors: [https://github.com/MolecularAI/REINVENT4](https://github.com/MolecularAI/REINVENT4).
- Original method/paper: [https://doi.org/10.1186/s13321-024-00812-5](https://doi.org/10.1186/s13321-024-00812-5).
- Observed upstream terms: Apache-2.0 for upstream software/official priors; it is not treated as an automatic output-data license.
- The complete upstream text is retained in `UPSTREAM_SOFTWARE_LICENSE.txt` as a software/model notice, **not** a new license imposed on locally generated outputs.

## Actual data and unverified limitations

The requested-1000 local batch retains 11,656 unique valid generated rows. A further 277 explicitly failed-sanitization strings are separate derived log extractions, with no invented attempt indices or reordered full-attempt table. The remaining 67 filtered entries lack a complete state record and must not all be asserted invalid: unique_molecules=true filters duplicates too. Thus 11,656/12,000 is a retained unique-valid yield, not automatically the usual unfiltered Validity.

Six matched-batch full/partial Vina pools contain 5,712 original rows: ESR1 965, HCRTR1 973, JAK1 981, P2RX3 972, KDM1A 971 and IDH1 850. IDH1 has 974 retained generated rows, so this pool is partial; the other six target pools were not recovered. Index and exact SMILES match the generation batch. These are NOT a recovered formal 100-per-target selection. No selected100 files are provided or reconstructed from the pool. The different legacy 1,177-row Vina batch is not included.

No molecules were generated, reconstructed or docked again for this upload. Original record bytes and filenames are unchanged. Missing rows, failed scores and unresolved provenance are not filled or silently treated as successful experiments.
