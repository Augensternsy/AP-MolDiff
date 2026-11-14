import os
import sys
import logging
import pandas as pd
import torch
from torch.utils.data import Dataset

# Add scripts directory to path for importing ChemformerTokenizer
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, 'improved-diffusion', 'scripts')
if SCRIPTS_DIR not in sys.path:
    sys.path.append(SCRIPTS_DIR)

from mytokenizers import ChemformerTokenizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("KiPreferenceDataset")

class KiPreferenceDataset(Dataset):
    """
    Dataset for BindingDB Ki Preference Pairs (Winner-Loser SMILES).
    """
    def __init__(
        self,
        csv_path=None,
        max_len=170,
        max_proteins=None,
        max_pairs_per_protein=None,
        seed=2026
    ):
        super().__init__()
        if csv_path is None:
            csv_path = os.path.join(PROJECT_ROOT, 'data', 'bindingdb_ki', 'preference_pairs_delta1.0_no_test_targets.csv')
        
        self.csv_path = csv_path
        self.max_len = max_len
        self.tokenizer = ChemformerTokenizer(max_len=max_len)
        
        logger.info(f"Loading preference dataset from: {csv_path}")
        df = pd.read_csv(csv_path)
        logger.info(f"Raw pairs loaded: {len(df)}")
        
        # Protein-balanced sampling & debugging filters
        if max_proteins is not None or max_pairs_per_protein is not None:
            grouped = df.groupby('protein_sequence', sort=False)
            unique_proteins = list(grouped.groups.keys())
            
            if max_proteins is not None and len(unique_proteins) > max_proteins:
                unique_proteins = unique_proteins[:max_proteins]
                logger.info(f"Capping unique proteins to max_proteins={max_proteins}")
            
            sampled_dfs = []
            for prot in unique_proteins:
                prot_df = grouped.get_group(prot)
                if max_pairs_per_protein is not None and len(prot_df) > max_pairs_per_protein:
                    prot_df = prot_df.sample(n=max_pairs_per_protein, random_state=seed)
                sampled_dfs.append(prot_df)
            
            df = pd.concat(sampled_dfs, ignore_index=True)
            logger.info(f"Dataset size after balanced sampling & filters: {len(df)} pairs across {len(unique_proteins)} proteins")
        
        # Tokenization pre-filtering
        self.samples = []
        skipped_count = 0
        
        for idx, row in df.iterrows():
            w_smi = str(row['winner_smiles']).strip()
            l_smi = str(row['loser_smiles']).strip()
            
            try:
                w_tok = self.tokenizer(w_smi) # shape [1, max_len]
                l_tok = self.tokenizer(l_smi) # shape [1, max_len]
                
                # Verify non-empty tokens
                if (w_tok != 0).sum() <= 2 or (l_tok != 0).sum() <= 2:
                    skipped_count += 1
                    continue
                
                self.samples.append({
                    'protein_sequence': str(row['protein_sequence']).strip(),
                    'winner_smiles': w_smi,
                    'loser_smiles': l_smi,
                    'winner_pKi': float(row['winner_pKi']),
                    'loser_pKi': float(row['loser_pKi']),
                    'delta_pKi': float(row['delta_pKi']),
                    'winner_ids': w_tok.squeeze(0), # shape [max_len]
                    'winner_mask': (w_tok.squeeze(0) != 0).long(),
                    'loser_ids': l_tok.squeeze(0),  # shape [max_len]
                    'loser_mask': (l_tok.squeeze(0) != 0).long()
                })
            except Exception as e:
                skipped_count += 1
                
        if skipped_count > 0:
            logger.warning(f"Skipped {skipped_count} invalid/untokenizable SMILES pairs.")
            
        logger.info(f"Final valid dataset size: {len(self.samples)} pairs.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

if __name__ == '__main__':
    # Quick self-test
    ds = KiPreferenceDataset(max_proteins=5, max_pairs_per_protein=10)
    print("Dataset self-test success! Sample 0:", ds[0]['winner_smiles'], "delta_pKi:", ds[0]['delta_pKi'])
