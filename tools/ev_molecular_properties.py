import pandas as pd
import numpy as np
import os
import argparse
import sys
from rdkit import Chem
from rdkit.Chem import QED, Descriptors, rdMolDescriptors

# 尝试导入 RDKit 的合成可及性 (SA) 评分模块
try:
    from rdkit.Contrib.SA_Score import sascorer
except ImportError:
    print("【警告】未检测到 'sascorer' 模块。SA Score 将无法计算。")
    sascorer = None


# ==================== 核心工具函数 ====================

def smiles_to_mols(smiles_list):
    """将 SMILES 转换为 Mol 对象，无效的返回 None"""
    mols = []
    for s in smiles_list:
        try:
            mol = Chem.MolFromSmiles(s.strip()) if isinstance(s, str) and s.strip() else None
            mols.append(mol)
        except:
            mols.append(None)
    return mols




# ==================== 评估指标计算函数 ====================











def calculate_comprehensive_metrics(smiles_list):
    """核心评估函数：包含你要求的 SA Score 修改逻辑"""
    mols_raw = smiles_to_mols(smiles_list)
    valid_mask = [m is not None for m in mols_raw]
    validity = sum(valid_mask) / len(smiles_list) if smiles_list else np.nan
    valid_mols = [m for m in mols_raw if m is not None]

    # Uniqueness
    can_smi = []
    for m in valid_mols:
        try:
            can_smi.append(Chem.MolToSmiles(m, canonical=True))
        except:
            pass
    uniqueness = len(set(can_smi)) / len(can_smi) if can_smi else np.nan

    results = {'QED': [], 'SA_Score': [], 'LogP': [], 'MolWt': [], 'TPSA': [], 'Fsp3': [], 'Ro5_Pass': [], 'Validity': valid_mask}

    for m in mols_raw:
        if m is None:
            for k in ['QED', 'SA_Score', 'LogP', 'MolWt', 'TPSA', 'Fsp3', 'Ro5_Pass']: results[k].append(np.nan)
        else:
            # 1. QED
            try:
                results['QED'].append(QED.default(m))
            except:
                results['QED'].append(np.nan)

            # --- 修改部分：你要求的 SA Score 安全计算逻辑 ---
            sa_value = np.nan
            if sascorer:
                try:
                    sa_value = sascorer.calculateScore(m)
                except:
                    sa_value = np.nan
            results['SA_Score'].append(sa_value)
            # ----------------------------------------------

            # 3. LogP, 4. MolWt
            try:
                results['LogP'].append(Descriptors.MolLogP(m))
                results['MolWt'].append(Descriptors.MolWt(m))
            except:
                results['LogP'].append(np.nan)
                results['MolWt'].append(np.nan)

            # 5. TPSA
            try:
                results['TPSA'].append(Descriptors.TPSA(m))
            except:
                results['TPSA'].append(np.nan)

            # 6. Fsp3
            try:
                results['Fsp3'].append(rdMolDescriptors.CalcFractionCSP3(m))
            except:
                results['Fsp3'].append(np.nan)

            # 7. Ro5
            try:
                if (Descriptors.MolWt(m) <= 500 and Descriptors.MolLogP(m) <= 5 and
                        Descriptors.NumHDonors(m) <= 5 and Descriptors.NumHAcceptors(m) <= 10):
                    results['Ro5_Pass'].append(1.0)
                else:
                    results['Ro5_Pass'].append(0.0)
            except:
                results['Ro5_Pass'].append(np.nan)

    df_details = pd.DataFrame(results)
    summary = {
        'Validity': validity,
        'Uniqueness': uniqueness,
        'QED': df_details['QED'].mean(),
        'SA': df_details['SA_Score'].mean(),
        'LogP': df_details['LogP'].mean(),
        'MolWt': df_details['MolWt'].mean(),
        'TPSA': df_details['TPSA'].mean(),
        'Ro5': df_details['Ro5_Pass'].mean(),
        'Avg_QED': df_details['QED'].mean(),
        'Avg_SA': df_details['SA_Score'].mean(),
        'Avg_LogP': df_details['LogP'].mean(),
        'Avg_MolWt': df_details['MolWt'].mean(),
        'Avg_TPSA': df_details['TPSA'].mean(),
        'Avg_Fsp3': df_details['Fsp3'].mean(),
        'Ro5_Rate': df_details['Ro5_Pass'].mean()
    }
    return summary, df_details


# Candidate-only CLI: explicit provenance of the generation denominator.
from pathlib import Path

def main(argv=None):
    parser = argparse.ArgumentParser(description="CPU molecular properties; unknown generation totals remain unknown")
    parser.add_argument('--gen-file', '--gen_file', required=True, dest='gen_file')
    parser.add_argument('--smiles-column', default='SMILES')
    parser.add_argument('--input-scope', choices=['all-attempts', 'valid-only', 'unknown'], default='unknown',
                        help='all-attempts asserts every attempt is recorded including failures; valid-only asserts every valid molecule is recorded')
    parser.add_argument('--total-generated', type=int, help='Externally documented original total, not an inferred count')
    parser.add_argument('--output-dir', '--output_dir', required=True, dest='output_dir')
    args = parser.parse_args(argv)
    try:
        df = pd.read_csv(args.gen_file)
        if args.smiles_column not in df:
            raise ValueError('Missing SMILES column: ' + args.smiles_column)
        smiles = df[args.smiles_column].tolist()
        summary, details = calculate_comprehensive_metrics(smiles)
        valid = int(details['Validity'].sum())
        rows = len(smiles)
        total = args.total_generated
        if total is not None and total < 0:
            raise ValueError('total-generated must be nonnegative')
        if args.input_scope == 'all-attempts':
            if total is not None and total != rows:
                raise ValueError('all-attempts requires total-generated equal to input rows')
            total = rows
        elif args.input_scope == 'valid-only':
            if valid != rows:
                raise ValueError('valid-only contains invalid or missing SMILES')
            if total is not None and total < rows:
                raise ValueError('total-generated is smaller than the valid records')
        elif total is not None:
            raise ValueError('unknown scope cannot establish numerator completeness; specify all-attempts or valid-only')
        summary['Input_Rows'] = rows
        summary['Valid_Records'] = valid
        summary['Input_Valid_Fraction'] = valid / rows if rows else np.nan
        summary['Total_Generated'] = total if total is not None else np.nan
        summary['Input_Scope'] = args.input_scope
        summary['Validity'] = valid / total if total is not None and total > 0 else np.nan
        summary['Validity_Status'] = ('COMPUTED_FROM_DECLARED_RECORDS' if total is not None and total > 0 else
                                      'UNDEFINED_ZERO_ATTEMPTS' if total == 0 else 'UNKNOWN_GENERATION_TOTAL')
        # Chemistry and averaging definitions above are retained; no fake missing SA or generation totals.
        out = Path(args.output_dir)
        if out.exists() and (not out.is_dir() or any(out.iterdir())):
            raise ValueError('Output must be a fresh or empty directory')
        out.mkdir(parents=True, exist_ok=True)
        pd.concat([df.reset_index(drop=True), details.add_prefix('Computed_')], axis=1).to_csv(out/'molecules.csv', index=False)
        pd.DataFrame([summary]).to_csv(out/'summary.csv', index=False)
        print(pd.DataFrame([summary]).to_string(index=False))
        print('Scope is a caller assertion, not independent proof of input completeness.')
    except (ValueError, OSError, pd.errors.ParserError) as exc:
        parser.error(str(exc))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
