# Data and model sources

The project owner explicitly authorized distribution of the two selected trained models and the project-constructed research data in this release. This authorizes publication of the deposited artifacts; it does not establish a missing historical training/provenance chain or relicense third-party original data.

The source-code MIT notice does not automatically apply to model weights or datasets. Owner-authorized compilations, mappings, target selections and evaluation records are provided as research artifacts. The complete historical source/version mapping of the stage-one corpus remains unverified; no claim that it is a uniquely recovered BindingDB release or a certified paper split is made. No additional blanket data/model license is invented here.

The Ki preference preparation code loads Therapeutics Data Commons `DTI(name='BindingDB_Ki')`. TDC's official DTI page lists CC BY 3.0 US for that dataset: https://tdcommons.ai/multi_pred_tasks/dti/ . Retain applicable upstream attribution and conditions for TDC/BindingDB-derived content; license: https://creativecommons.org/licenses/by/3.0/us/ . The preference and clean-ligand files are derived selections with token, affinity, property and leakage filters, not unmodified upstream distributions. The exact historical TDC snapshot still requires provenance confirmation.

BindingDB: Tiqing Liu et al., BindingDB: a web-accessible database of experimentally determined protein-ligand binding affinities, Nucleic Acids Research 35 (2007), D198-D201. Obtain original data from https://www.bindingdb.org/rwd/bind/chemsearch/marvin/Download.jsp . That page links quarterly versioned archives at https://library.ucsd.edu/dc/collection/bb03870458 . Preserve applicable terms of the chosen version and source subsets; do not assume a current download reproduces the old training input.

ESM-2 is external pretrained work by Meta, not a project-trained diffusion model. Obtain the model/tokenizer from https://huggingface.co/facebook/esm2_t30_150M_UR50D and preserve its upstream MIT notice. Copies are not rehosted in this release.

Prepared CrossDocked receptor/PDBQT and reference SDF files are not included because their preparation/provenance and redistribution scope were not closed. No invented matching-download link is supplied. The Vina code requires appropriately acquired receptor and reference-ligand inputs.
