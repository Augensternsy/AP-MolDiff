# DiffSBDD: local baseline output provenance and scope

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

- Original implementation and authors: [https://github.com/arneschneuing/DiffSBDD](https://github.com/arneschneuing/DiffSBDD).
- Original method/paper: [https://doi.org/10.1038/s43588-024-00737-x](https://doi.org/10.1038/s43588-024-00737-x).
- Observed upstream terms: MIT for upstream software; CC-BY-4.0 for specific official Zenodo models/samples. These local outputs are not byte-identical copies of the checked official sample package.
- The complete upstream text is retained in `UPSTREAM_SOFTWARE_LICENSE.txt` as a software/model notice, **not** a new license imposed on locally generated outputs.

## Actual data and unverified limitations

The 12 original multi-molecule SDF files retain 12,000 saved records, including previously identified invalid records; they must not be described as 12,000 valid molecules. Derived SMILES indexes preserve failed decodes rather than inventing strings. The 1,200 local selected Vina rows have existing generation membership links. This local 12-target run is not the official Zenodo sample archive. Historical model/run-to-paper lineage is not fully sealed.

No molecules were generated, reconstructed or docked again for this upload. Original record bytes and filenames are unchanged. Missing rows, failed scores and unresolved provenance are not filled or silently treated as successful experiments.
