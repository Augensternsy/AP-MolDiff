"""
prepare_bindingdb_ki_preferences_v3.py

该脚本是第三版 (v3) affinity-preserving ligand-efficiency-aware Ki-E2PO 偏好对构建脚本；
在保持高 pKi 偏好为主目标的前提下，引入 ligand efficiency、MolWt、HeavyAtomCount 和 Ro5 约束，缓解 v2 中分子量上升和 LE 下降的问题。
"""

import os
import sys
import argparse
import json
import time
import hashlib
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import QED, Descriptors, rdMolDescriptors
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# Try importing RDKit's Synthetic Accessibility (SA) Score module
try:
    from rdkit.Contrib.SA_Score import sascorer
except ImportError:
    print("[WARNING] 'sascorer' module not found. SA Score calculations will fallback to 4.0.")
    sascorer = None

# Global seed
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

def get_md5(string):
    """Generate MD5 ID for protein sequences"""
    return hashlib.md5(string.strip().encode('utf-8')).hexdigest()

def load_fasta_sequences(fasta_dir):
    """Load benchmark protein FASTA sequences"""
    benchmark_seqs = {}
    if not os.path.exists(fasta_dir):
        return benchmark_seqs

    for file_name in os.listdir(fasta_dir):
        if file_name.endswith(('.fasta', '.fa')):
            target_name = os.path.splitext(file_name)[0]
            file_path = os.path.join(fasta_dir, file_name)
            seq = ""
            with open(file_path, 'r') as f:
                for line in f:
                    if not line.startswith('>'):
                        seq += line.strip()
            if seq:
                benchmark_seqs[target_name] = seq.upper()
    return benchmark_seqs

def check_benchmark_exclusion(seq, benchmark_seqs):
    seq_upper = seq.strip().upper()
    for name, b_seq in benchmark_seqs.items():
        if seq_upper == b_seq or seq_upper in b_seq or b_seq in seq_upper:
            return name
    return None

# ================= Tokenizer Loader =================

def load_first_stage_tokenizer(tokenizer_path, max_len):
    """Load ChemformerTokenizer"""
    if tokenizer_path:
        tokenizer_path = os.path.abspath(tokenizer_path)
        if os.path.isfile(tokenizer_path):
            dir_name = os.path.dirname(tokenizer_path)
            module_name = os.path.splitext(os.path.basename(tokenizer_path))[0]
        else:
            dir_name = tokenizer_path
            module_name = "mytokenizers"
            
        if dir_name not in sys.path:
            sys.path.insert(0, dir_name)
            
        try:
            import importlib
            module = importlib.import_module(module_name)
            if hasattr(module, 'ChemformerTokenizer'):
                return getattr(module, 'ChemformerTokenizer')(max_len=max_len)
        except Exception as e:
            print(f"[WARNING] Failed to load tokenizer from specified path {tokenizer_path}: {e}. Trying fallback...")
            
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir) if os.path.basename(script_dir) == 'scripts' else script_dir
        if os.path.basename(project_root) == 'improved-diffusion':
            project_root = os.path.dirname(project_root)
        imp_diff_scripts = os.path.join(project_root, 'improved-diffusion', 'scripts')
        if imp_diff_scripts not in sys.path:
            sys.path.insert(0, imp_diff_scripts)
        from mytokenizers import ChemformerTokenizer
        return ChemformerTokenizer(max_len=max_len)
    except Exception as e:
        print(f"[ERROR] Failed to import ChemformerTokenizer from mytokenizers: {e}")
        sys.exit(1)

# ================= SMILES Validation parallel checks =================

_tokenizer = None

def init_worker(tokenizer_path, max_smiles_len):
    global _tokenizer
    _tokenizer = load_first_stage_tokenizer(tokenizer_path, max_smiles_len)

def check_single_smiles(task_tuple):
    idx, raw_smiles, tokenizer_path, max_smiles_len = task_tuple
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = load_first_stage_tokenizer(tokenizer_path, max_smiles_len)
        
    try:
        mol = Chem.MolFromSmiles(raw_smiles)
        if mol is None:
            return idx, None, 0, False, 'rdkit_invalid'
        clean_smi = Chem.MolToSmiles(mol, canonical=True)
    except:
        return idx, None, 0, False, 'rdkit_invalid'
        
    try:
        tokens = _tokenizer.rg.findall(clean_smi)
        if not tokens:
            return idx, clean_smi, 2, False, 'empty_tokens'
            
        has_unk = False
        out_of_vocab = False
        for tok in tokens:
            if tok in _tokenizer.toktoid:
                tid = _tokenizer.toktoid[tok]
                if tid == 3: # UNK
                    has_unk = True
            else:
                has_unk = True
                out_of_vocab = True
                
        unpadded_len = len(tokens) + 2
        if has_unk or out_of_vocab:
            return idx, clean_smi, unpadded_len, False, 'contains_unk'
            
        if unpadded_len > _tokenizer.max_len:
            return idx, clean_smi, unpadded_len, False, 'exceed_max_len'
            
        return idx, clean_smi, unpadded_len, True, 'valid'
    except Exception as e:
        return idx, clean_smi, 0, False, 'tokenizer_error'

def parallel_check_smiles(smiles_series, tokenizer_path, max_smiles_len, num_cores):
    tasks = [(idx, smi, tokenizer_path, max_smiles_len) for idx, smi in enumerate(smiles_series.tolist())]
    print(f"[INFO] Running RDKit + Tokenizer validation on {len(tasks)} SMILES strings using {num_cores} CPU cores...")
    
    results = [None] * len(tasks)
    with Pool(processes=num_cores, initializer=init_worker, initargs=(tokenizer_path, max_smiles_len)) as pool:
        for res in tqdm(pool.imap_unordered(check_single_smiles, tasks, chunksize=1000), total=len(tasks), desc="SMILES Validation"):
            idx, clean_smi, token_len, tokenizable, reason = res
            results[idx] = (clean_smi, token_len, tokenizable, reason)
            
    return results

def sanitize_single_smiles(args):
    idx, smiles = args
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return idx, None
        return idx, Chem.MolToSmiles(mol, canonical=True)
    except:
        return idx, None

def parallel_sanitize_smiles(smiles_series, num_cores):
    tasks = list(enumerate(smiles_series.tolist()))
    print(f"[INFO] Sanitizing {len(tasks)} SMILES strings using {num_cores} CPU cores...")
    
    cleaned_smiles = [None] * len(tasks)
    with Pool(processes=num_cores) as pool:
        for idx, canon_smi in tqdm(pool.imap_unordered(sanitize_single_smiles, tasks, chunksize=1000), total=len(tasks), desc="RDKit Sanitization"):
            cleaned_smiles[idx] = canon_smi
            
    return cleaned_smiles

# ================= RDKit Molecular Properties parallel computation =================

def compute_properties_single(task_tuple):
    idx, smiles = task_tuple
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return idx, None
            
        qed_val = QED.default(mol)
        
        sa_val = np.nan
        if sascorer:
            try:
                sa_val = sascorer.calculateScore(mol)
            except:
                pass
                
        mw_val = Descriptors.MolWt(mol)
        logp_val = Descriptors.MolLogP(mol)
        
        h_donors = Descriptors.NumHDonors(mol)
        h_acceptors = Descriptors.NumHAcceptors(mol)
        ro5_passed = bool((mw_val <= 500) and (logp_val <= 5) and (h_donors <= 5) and (h_acceptors <= 10))
        
        hac_val = mol.GetNumHeavyAtoms()
        
        return idx, {
            'QED': qed_val,
            'SA': sa_val,
            'MolWt': mw_val,
            'LogP': logp_val,
            'Ro5_pass': ro5_passed,
            'HAC': hac_val
        }
    except Exception as e:
        return idx, None

# ================= v3 Group Pair Construction =================

def construct_pairs_for_protein_v3(args):
    target_seq, group_df, delta_pki, max_pairs = args
    
    # Calculate group-specific thresholds
    pki_thresh = group_df['pKi'].quantile(0.60)
    le_median = group_df['Exp_LE'].median()
    
    group_df = group_df.copy()
    group_df['is_winner'] = (
        (group_df['pKi'] >= pki_thresh) &
        (group_df['QED'] >= 0.45) &
        (group_df['SA'] <= 4.0) &
        (group_df['MolWt'] >= 250.0) &
        (group_df['MolWt'] <= 480.0) &
        (group_df['Ro5_pass'] == True) &
        (group_df['HAC'] <= 40) &
        (group_df['Exp_LE'] >= le_median)
    )
    
    winners_df = group_df[group_df['is_winner']]
    if len(winners_df) == 0:
        return []
        
    protein_id = get_md5(target_seq)
    valid_pairs = []
    
    for idx_w, row_w in winners_df.iterrows():
        w_smiles = row_w['SMILES_clean']
        w_pki = row_w['pKi']
        w_qed = row_w['QED']
        w_sa = row_w['SA']
        w_mw = row_w['MolWt']
        w_logp = row_w['LogP']
        w_ro5 = row_w['Ro5_pass']
        w_hac = row_w['HAC']
        w_le = row_w['Exp_LE']
        w_pki_hac = row_w['pKi_per_HAC']
        
        for idx_l, row_l in group_df.iterrows():
            if row_l['SMILES_clean'] == w_smiles:
                continue
                
            diff_pki = w_pki - row_l['pKi']
            if diff_pki >= delta_pki:
                l_pki = row_l['pKi']
                l_qed = row_l['QED']
                l_sa = row_l['SA'] if not np.isnan(row_l['SA']) else 4.0
                l_mw = row_l['MolWt']
                l_logp = row_l['LogP']
                l_ro5 = row_l['Ro5_pass']
                l_hac = row_l['HAC']
                l_le = row_l['Exp_LE']
                l_pki_hac = row_l['pKi_per_HAC']
                
                # Bonus calculation
                loser_bad_property_bonus = 0.0
                if not l_ro5:
                    loser_bad_property_bonus += 0.2
                if l_mw > 480.0:
                    loser_bad_property_bonus += 0.2
                if l_sa > 4.0:
                    loser_bad_property_bonus += 0.2
                if l_le < le_median:
                    loser_bad_property_bonus += 0.2
                    
                # Priority calculation
                priority_score = (
                    2.0 * diff_pki
                    + 0.8 * (w_le - l_le)
                    + 0.3 * (w_qed - l_qed)
                    - 0.15 * max(0.0, w_sa - l_sa)
                    - 0.10 * max(0.0, w_mw - l_mw) / 100.0
                    - 0.10 * max(0.0, w_hac - l_hac) / 10.0
                    + loser_bad_property_bonus
                )
                
                valid_pairs.append({
                    'protein_id': protein_id,
                    'protein_sequence': target_seq,
                    'winner_smiles': w_smiles,
                    'loser_smiles': row_l['SMILES_clean'],
                    'winner_pKi': w_pki,
                    'loser_pKi': l_pki,
                    'delta_pKi': diff_pki,
                    'winner_QED': w_qed,
                    'loser_QED': l_qed,
                    'winner_SA': w_sa,
                    'loser_SA': row_l['SA'],  # Keep original SA (can be NaN or float)
                    'winner_MolWt': w_mw,
                    'loser_MolWt': l_mw,
                    'winner_LogP': w_logp,
                    'loser_LogP': l_logp,
                    'winner_Ro5': w_ro5,
                    'loser_Ro5': l_ro5,
                    'winner_HAC': w_hac,
                    'loser_HAC': l_hac,
                    'winner_Exp_LE': w_le,
                    'loser_Exp_LE': l_le,
                    'delta_Exp_LE': w_le - l_le,
                    'winner_pKi_per_HAC': w_pki_hac,
                    'loser_pKi_per_HAC': l_pki_hac,
                    'pair_priority': priority_score
                })
                
    if len(valid_pairs) == 0:
        return []
        
    valid_pairs.sort(key=lambda x: x['pair_priority'], reverse=True)
    
    if len(valid_pairs) <= max_pairs:
        return valid_pairs
    else:
        return valid_pairs[:max_pairs]

# ================= Pipeline Main =================

def main():
    parser = argparse.ArgumentParser(description="BindingDB_Ki v3 LE-Aware Preference Pairs Builder")
    parser.add_argument('--out_dir', type=str, default='data/bindingdb_ki', help="Output directory")
    parser.add_argument('--min_ligands_per_protein', type=int, default=20, help="Min ligands per protein sequence")
    parser.add_argument('--delta_pki', type=float, default=1.0, help="Delta pKi threshold for preference pairs")
    parser.add_argument('--test_target_seq_file', type=str, default='datasets/targets/test_target_sequences.csv', help="CSV containing test target sequences")
    parser.add_argument('--exclude_test_target_ligands', type=str, default='true', help="Exclude exact test target rows")
    parser.add_argument('--exclude_exact_test_ligand_smiles', type=str, default='true', help="Cross-exclude test active smiles")
    parser.add_argument('--fasta_dir', type=str, default='datasets/targets/fasta', help="Benchmark protein fasta directory")
    parser.add_argument('--num_cores', type=int, default=max(1, cpu_count() - 2), help="Number of CPU cores for parallelism")
    
    # Tokenizer params
    parser.add_argument('--tokenizer_path', type=str, default=None, help="First stage tokenizer script or vocab path")
    parser.add_argument('--max_smiles_len', type=int, default=170, help="Max SMILES length for model token compatibility")
    parser.add_argument('--strict_token_filter', type=str, default='true', help="Strictly filter SMILES via tokenizer compatibility")
    args = parser.parse_args()

    # Paths conversion to absolute
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir) if os.path.basename(script_dir) == 'scripts' else script_dir
    if os.path.basename(project_root) == 'improved-diffusion':
        project_root = os.path.dirname(project_root)
        
    out_dir_abs = os.path.isabs(args.out_dir) and args.out_dir or os.path.abspath(os.path.join(project_root, args.out_dir))
    os.makedirs(out_dir_abs, exist_ok=True)
    
    fasta_dir_abs = os.path.isabs(args.fasta_dir) and args.fasta_dir or os.path.abspath(os.path.join(project_root, 'datasets/targets/fasta'))
    test_seq_file_abs = os.path.isabs(args.test_target_seq_file) and args.test_target_seq_file or os.path.abspath(os.path.join(project_root, args.test_target_seq_file))
    
    tokenizer_path_abs = args.tokenizer_path
    if tokenizer_path_abs:
        if not os.path.isabs(tokenizer_path_abs):
            tokenizer_path_abs = os.path.abspath(os.path.join(project_root, tokenizer_path_abs))

    start_time = time.time()

    exclude_test_target_ligands = args.exclude_test_target_ligands.lower() == 'true'
    exclude_exact_test_ligand_smiles = args.exclude_exact_test_ligand_smiles.lower() == 'true'
    strict_token_filter = args.strict_token_filter.lower() == 'true'

    # Load Tokenizer
    tokenizer = load_first_stage_tokenizer(tokenizer_path_abs, args.max_smiles_len)
    tokenizer_type = type(tokenizer).__name__
    vocab_size = len(tokenizer)
    
    print(f"\n========================================================")
    print(f"Tokenizer Type: {tokenizer_type}")
    print(f"Tokenizer Vocab Size: {vocab_size}")
    print(f"Max SMILES Length: {args.max_smiles_len}")
    print(f"========================================================\n")
    sys.stdout.flush()

    # 1. Import pytdc
    try:
        from tdc.multi_pred import DTI
    except ImportError:
        print("[ERROR] 'pytdc' is not installed. Please run: pip install pytdc")
        sys.exit(1)

    print("[Step 1/8] Downloading/Loading BindingDB_Ki dataset from Therapeutics Data Commons...")
    sys.stdout.flush()
    try:
        data = DTI(name='BindingDB_Ki')
        raw_df = data.get_data()
        df = raw_df.copy()
        print(f"[INFO] Raw dataset loaded. Initial row count: {len(df)}")
    except Exception as e:
        print(f"[ERROR] Failed to load data from pytdc: {e}")
        sys.exit(1)

    # 2. Convert Ki to pKi
    print("\n[Step 2/8] Adaptive detection of affinity metric (pKi vs Ki nM)...")
    y_clean = df['Y'].dropna()
    mean_y = y_clean.mean()
    
    is_already_pki = 3.0 <= mean_y <= 12.0
    if is_already_pki:
        df['pKi'] = df['Y']
    else:
        def convert_ki_to_pki(ki_val):
            try:
                val = float(ki_val)
                if val <= 0: return np.nan
                return 9.0 - np.log10(val)
            except:
                return np.nan
        df['pKi'] = df['Y'].apply(convert_ki_to_pki)

    df = df.dropna(subset=['pKi'])
    original_samples_count = len(df)
    print(f"[INFO] Dataset size after filtering out invalid affinity scores: {original_samples_count}")

    # 3. RDKit Standardization
    print("\n[Step 3/8] Running RDKit Standardization on all SMILES...")
    df['SMILES_clean'] = parallel_sanitize_smiles(df['Drug'], args.num_cores)
    rdkit_valid_count = df['SMILES_clean'].notna().sum()
    print(f"[INFO] RDKit standardization completed. Valid SMILES: {rdkit_valid_count} / {len(df)}")

    # 4. Leakage Prevention
    print("\n[Step 4/8] Handling test target sequences matching & active ligand extraction...")
    test_seqs = set()
    num_excluded_by_protein_seq = 0
    num_excluded_proteins = 0
    test_active_ligands_count = 0
    num_excluded_by_ligand_leakage = 0
    
    if os.path.exists(test_seq_file_abs):
        test_df = pd.read_csv(test_seq_file_abs)
        if 'protein_sequence' in test_df.columns:
            test_seqs = set(test_df['protein_sequence'].str.strip().str.upper().tolist())
        print(f"[INFO] Loaded {len(test_seqs)} test target sequences from {test_seq_file_abs}")
    else:
        print("[WARNING] No test target protein sequences file found. Loading fuzzy name-based exclusion fallback...")
        sys.stdout.flush()

    if len(test_seqs) > 0:
        df['Target_upper'] = df['Target'].str.strip().str.upper()
        excluded_rows = df[df['Target_upper'].isin(test_seqs)]
        df = df[~df['Target_upper'].isin(test_seqs)]
        
        num_excluded_by_protein_seq = len(excluded_rows)
        num_excluded_proteins = excluded_rows['Target_upper'].nunique()
        df = df.drop(columns=['Target_upper'])
        
        test_drugs = excluded_rows['SMILES_clean'].dropna().unique()
        test_active_ligands_count = len(test_drugs)
        
        if exclude_exact_test_ligand_smiles and test_active_ligands_count > 0:
            test_smiles_set = set(test_drugs)
            rows_before_leakage = len(df)
            df = df[~df['SMILES_clean'].isin(test_smiles_set)]
            num_excluded_by_ligand_leakage = rows_before_leakage - len(df)
            print(f"[INFO] Excluded {num_excluded_by_ligand_leakage} rows due to exact active test ligand SMILES leakage protection.")
    else:
        print("[WARNING] Falling back to FASTA directory fuzzy name-based matching...")
        benchmark_seqs = load_fasta_sequences(fasta_dir_abs)
        if benchmark_seqs:
            df['matched_benchmark'] = df['Target'].apply(lambda s: check_benchmark_exclusion(s, benchmark_seqs))
            excluded_rows_df = df[df['matched_benchmark'].notna()]
            excluded_benchmark_targets = set(excluded_rows_df['matched_benchmark'].unique())
            df = df[df['matched_benchmark'].isna()]
            df = df.drop(columns=['matched_benchmark'])
            num_excluded_by_protein_seq = len(excluded_rows_df)
            num_excluded_proteins = len(excluded_benchmark_targets)
            print(f"[INFO] Excluded {num_excluded_by_protein_seq} rows associated with benchmark targets: {list(excluded_benchmark_targets)}")

    post_leakage_exclusion_count = len(df)
    print(f"[INFO] Dataset size after sequence and leakage exclusion: {post_leakage_exclusion_count}")

    # 5. Tokenizer Filter
    print("\n[Step 5/8] ChemformerTokenizer filtering on remaining rows...")
    sys.stdout.flush()
    
    valid_smiles_mask = df['SMILES_clean'].notna()
    valid_smiles_series = df.loc[valid_smiles_mask, 'SMILES_clean']
    
    check_results = parallel_check_smiles(valid_smiles_series, tokenizer_path_abs, args.max_smiles_len, args.num_cores)
    
    token_lens = []
    tokenizables = []
    reasons = []
    
    result_idx = 0
    for idx, row in df.iterrows():
        smi = row['SMILES_clean']
        if pd.isna(smi) or smi is None:
            token_lens.append(0)
            tokenizables.append(False)
            reasons.append('rdkit_invalid')
        else:
            _, token_len, tokenizable, reason = check_results[result_idx]
            token_lens.append(token_len)
            tokenizables.append(tokenizable)
            reasons.append(reason)
            result_idx += 1
            
    df['token_len'] = token_lens
    df['tokenizable'] = tokenizables
    df['token_filter_reason'] = reasons

    tokenizer_pass_count = df['tokenizable'].sum()
    print(f"[INFO] Tokenizer validation: {tokenizer_pass_count} passed, {len(df) - tokenizer_pass_count} failed.")

    # 6. Properties Calculation
    print("\n[Step 6/8] Computing molecular properties (QED, SA, MolWt, LogP, Ro5, HAC) and Exp_LE...")
    sys.stdout.flush()
    
    df_tokenizable = df[df['tokenizable']].copy()
    
    smiles_to_compute = df_tokenizable['SMILES_clean'].tolist()
    tasks = list(enumerate(smiles_to_compute))
    
    properties_results = [None] * len(smiles_to_compute)
    with Pool(processes=args.num_cores) as pool:
        for idx, props in tqdm(pool.imap_unordered(compute_properties_single, tasks, chunksize=1000), total=len(tasks), desc="Properties calculation"):
            properties_results[idx] = props
            
    qed_list, sa_list, mw_list, logp_list, ro5_list, hac_list = [], [], [], [], [], []
    for res in properties_results:
        if res is None:
            qed_list.append(np.nan)
            sa_list.append(np.nan)
            mw_list.append(np.nan)
            logp_list.append(np.nan)
            ro5_list.append(False)
            hac_list.append(0)
        else:
            qed_list.append(res['QED'])
            sa_list.append(res['SA'])
            mw_list.append(res['MolWt'])
            logp_list.append(res['LogP'])
            ro5_list.append(res['Ro5_pass'])
            hac_list.append(res['HAC'])
            
    df_tokenizable['QED'] = qed_list
    df_tokenizable['SA'] = sa_list
    df_tokenizable['MolWt'] = mw_list
    df_tokenizable['LogP'] = logp_list
    df_tokenizable['Ro5_pass'] = ro5_list
    df_tokenizable['HAC'] = hac_list
    
    # Calculate Exp_LE and pKi_per_HAC
    df_tokenizable['Exp_LE'] = np.where(df_tokenizable['HAC'] > 0, 1.364 * df_tokenizable['pKi'] / df_tokenizable['HAC'], np.nan)
    df_tokenizable['pKi_per_HAC'] = np.where(df_tokenizable['HAC'] > 0, df_tokenizable['pKi'] / df_tokenizable['HAC'], np.nan)
    df_tokenizable = df_tokenizable.dropna(subset=['Exp_LE'])

    # Save clean ligands
    clean_ligands_path = os.path.join(out_dir_abs, "le_affinity_preserving_clean_ligands.csv")
    df_tokenizable.to_csv(clean_ligands_path, index=False)
    print(f"[INFO] Saved clean ligands to: {clean_ligands_path}")

    # Save removed ligands
    df_removed = df[~df.index.isin(df_tokenizable.index)].copy()
    removed_ligands_path = os.path.join(out_dir_abs, "le_affinity_preserving_removed_ligands.csv")
    df_removed.to_csv(removed_ligands_path, index=False)
    print(f"[INFO] Saved removed ligands to: {removed_ligands_path}")

    # 7. min_ligands_per_protein Filter
    print(f"\n[Step 7/8] Filtering proteins by minimum ligand count (>= {args.min_ligands_per_protein})...")
    target_counts = df_tokenizable.groupby('Target').size()
    valid_targets = target_counts[target_counts >= args.min_ligands_per_protein].index
    df_final = df_tokenizable[df_tokenizable['Target'].isin(valid_targets)]
    
    print(f"[INFO] Final dataset rows: {len(df_final)} (Unique proteins: {len(valid_targets)})")

    # 8. Construct v3 Pairs
    print(f"\n[Step 8/8] Constructing ligand-efficiency-aware preference pairs (delta_pKi >= {args.delta_pki})...")
    sys.stdout.flush()
    
    group_tasks = []
    for target_seq, group in df_final.groupby('Target'):
        group_tasks.append((target_seq, group, args.delta_pki, 3000))

    final_pairs = []
    with Pool(processes=args.num_cores) as pool:
        for p_list in tqdm(pool.imap_unordered(construct_pairs_for_protein_v3, group_tasks, chunksize=10), total=len(group_tasks), desc="Constructing Pairs v3"):
            if p_list:
                final_pairs.extend(p_list)

    pairs_df = pd.DataFrame(final_pairs)
    pairs_filename = "le_affinity_preserving_preference_pairs_delta1.0.csv"
    pairs_path = os.path.join(out_dir_abs, pairs_filename)
    
    columns_to_keep = [
        'protein_id', 'protein_sequence', 'winner_smiles', 'loser_smiles',
        'winner_pKi', 'loser_pKi', 'delta_pKi',
        'winner_QED', 'loser_QED',
        'winner_SA', 'loser_SA',
        'winner_MolWt', 'loser_MolWt',
        'winner_LogP', 'loser_LogP',
        'winner_Ro5', 'loser_Ro5',
        'winner_HAC', 'loser_HAC',
        'winner_Exp_LE', 'loser_Exp_LE',
        'delta_Exp_LE',
        'winner_pKi_per_HAC', 'loser_pKi_per_HAC',
        'pair_priority'
    ]
    
    if len(pairs_df) > 0:
        pairs_df = pairs_df[columns_to_keep]
        pairs_df.to_csv(pairs_path, index=False)
    else:
        pd.DataFrame(columns=columns_to_keep).to_csv(pairs_path, index=False)
        
    print(f"[INFO] Saved v3 preference pairs ({len(pairs_df)} pairs) to: {pairs_path}")

    # ================= Save summary.json =================
    # Calculate protein stats and distributions
    if len(pairs_df) > 0:
        pair_counts_per_protein = pairs_df.groupby('protein_sequence').size()
        pair_counts_dist = pair_counts_per_protein.describe().to_dict()
    else:
        pair_counts_dist = {}

    # Calculate winner_failed_property_constraints
    failed_top_40 = 0
    failed_qed = 0
    failed_sa = 0
    failed_mw = 0
    failed_ro5 = 0
    failed_hac = 0
    failed_le = 0
    
    for target_seq, group in df_final.groupby('Target'):
        pki_thresh = group['pKi'].quantile(0.60)
        le_median = group['Exp_LE'].median()
        
        failed_top_40 += (group['pKi'] < pki_thresh).sum()
        failed_qed += (group['QED'] < 0.45).sum()
        failed_sa += (group['SA'] > 4.0).sum() + group['SA'].isna().sum()
        failed_mw += ((group['MolWt'] < 250.0) | (group['MolWt'] > 480.0)).sum()
        failed_ro5 += (group['Ro5_pass'] == False).sum()
        failed_hac += (group['HAC'] > 40).sum()
        failed_le += (group['Exp_LE'] < le_median).sum()

    # Identify total unique winners across all groups
    total_valid_winners_count = 0
    for target_seq, group in df_final.groupby('Target'):
        pki_thresh = group['pKi'].quantile(0.60)
        le_median = group['Exp_LE'].median()
        is_w = (
            (group['pKi'] >= pki_thresh) &
            (group['QED'] >= 0.45) &
            (group['SA'] <= 4.0) &
            (group['MolWt'] >= 250.0) &
            (group['MolWt'] <= 480.0) &
            (group['Ro5_pass'] == True) &
            (group['HAC'] <= 40) &
            (group['Exp_LE'] >= le_median)
        )
        total_valid_winners_count += is_w.sum()

    summary = {
        'original_ligand_count': int(len(raw_df)) if 'raw_df' in locals() else 0,
        'rdkit_valid_count': int(rdkit_valid_count),
        'tokenizer_valid_count': int(tokenizer_pass_count),
        'post_protein_seq_exclusion_count': int(post_leakage_exclusion_count),
        'post_leakage_exclusion_count': int(post_leakage_exclusion_count - num_excluded_by_ligand_leakage),
        'total_valid_winners_count': int(total_valid_winners_count),
        'final_preference_pairs_count': int(len(pairs_df)),
        'final_proteins_for_pairs_count': int(pairs_df['protein_sequence'].nunique()) if len(pairs_df) > 0 else 0,
        'pair_counts_per_protein_distribution': {
            'min': int(pair_counts_dist.get('min', 0)),
            'max': int(pair_counts_dist.get('max', 0)),
            'mean': float(pair_counts_dist.get('mean', 0.0)),
            'median': float(pair_counts_dist.get('50%', 0.0)),
            'std': float(pair_counts_dist.get('std', 0.0))
        },
        'token_filter_reasons': {
            'rdkit_invalid': int((df['token_filter_reason'] == 'rdkit_invalid').sum()),
            'contains_unk': int((df['token_filter_reason'] == 'contains_unk').sum()),
            'exceed_max_len': int((df['token_filter_reason'] == 'exceed_max_len').sum()),
            'tokenizer_error': int((df['token_filter_reason'] == 'tokenizer_error').sum()),
            'empty_tokens': int((df['token_filter_reason'] == 'empty_tokens').sum())
        },
        'winner_failed_property_constraints': {
            'failed_top_40_pKi': int(failed_top_40),
            'failed_QED': int(failed_qed),
            'failed_SA': int(failed_sa),
            'failed_MolWt': int(failed_mw),
            'failed_Ro5': int(failed_ro5),
            'failed_HAC': int(failed_hac),
            'failed_Exp_LE_median': int(failed_le)
        },
        'confirm_no_vina_docking_boltz_test_target_info': True
    }

    if len(pairs_df) > 0:
        summary['Exp_LE_distribution'] = {
            'winner': {
                'mean': float(pairs_df['winner_Exp_LE'].mean()),
                'median': float(pairs_df['winner_Exp_LE'].median()),
                'min': float(pairs_df['winner_Exp_LE'].min()),
                'max': float(pairs_df['winner_Exp_LE'].max()),
                'std': float(pairs_df['winner_Exp_LE'].std())
            },
            'loser': {
                'mean': float(pairs_df['loser_Exp_LE'].mean()),
                'median': float(pairs_df['loser_Exp_LE'].median()),
                'min': float(pairs_df['loser_Exp_LE'].min()),
                'max': float(pairs_df['loser_Exp_LE'].max()),
                'std': float(pairs_df['loser_Exp_LE'].std())
            }
        }
        summary['MolWt_distribution'] = {
            'winner': {
                'mean': float(pairs_df['winner_MolWt'].mean()),
                'median': float(pairs_df['winner_MolWt'].median()),
                'min': float(pairs_df['winner_MolWt'].min()),
                'max': float(pairs_df['winner_MolWt'].max()),
                'std': float(pairs_df['winner_MolWt'].std())
            },
            'loser': {
                'mean': float(pairs_df['loser_MolWt'].mean()),
                'median': float(pairs_df['loser_MolWt'].median()),
                'min': float(pairs_df['loser_MolWt'].min()),
                'max': float(pairs_df['loser_MolWt'].max()),
                'std': float(pairs_df['loser_MolWt'].std())
            }
        }
        summary['HAC_distribution'] = {
            'winner': {
                'mean': float(pairs_df['winner_HAC'].mean()),
                'median': float(pairs_df['winner_HAC'].median()),
                'min': float(pairs_df['winner_HAC'].min()),
                'max': float(pairs_df['winner_HAC'].max()),
                'std': float(pairs_df['winner_HAC'].std())
            },
            'loser': {
                'mean': float(pairs_df['loser_HAC'].mean()),
                'median': float(pairs_df['loser_HAC'].median()),
                'min': float(pairs_df['loser_HAC'].min()),
                'max': float(pairs_df['loser_HAC'].max()),
                'std': float(pairs_df['loser_HAC'].std())
            }
        }

    summary_path = os.path.join(out_dir_abs, "le_affinity_preserving_summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)
    print(f"[INFO] Saved summary to: {summary_path}")

    elapsed_time = time.time() - start_time
    print(f"\n========================================================")
    print(f"DATASET PREPARATION COMPLETED IN {elapsed_time/60:.2f} MINUTES")
    print(f"========================================================")

if __name__ == '__main__':
    main()
