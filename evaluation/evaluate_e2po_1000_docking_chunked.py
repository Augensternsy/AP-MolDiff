import os
import argparse
import uuid
from pathlib import Path
import sys
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Lipinski, Descriptors
from multiprocessing import Pool
import time

try:
    from vina import Vina
except ImportError:
    print("【错误】未检测到 'vina' 模块。")
    exit()

try:
    from meeko import MoleculePreparation
except ImportError:
    print("【错误】未检测到 'meeko' 模块。")
    exit()

# ================= 配置区域 =================

GENE_TO_PDB = {
    'ESR1':   '2r6w',
    'HCRTR1': '4zjc',
    'JAK1':   '3eyg',
    'P2RX3':  '5svl',
    'KDM1A':  '5lhg',
    'IDH1':   '4umx',
    'RIOK1':  '4otp',
    'NR4A1':  '3v3q',
    'GRIK1':  '3fv1',
    'FTO':    '4zs3',
    'SPIN1':  '5jsj',
    'CCR9':   '5lwe'
}

PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
DATA_DIR = os.path.join(PROJECT_ROOT, "datasets/CrossDocked2020")
SELECTED_ROOT = os.path.join(PROJECT_ROOT, "evaluation/docking_candidate_selection/selected_candidates")

# 评估的 1000 分子级模型
MODELS = ['SelfCond_NoClamp_PathB_50k_1000', 'Ki-E2PO-v1-20k-1000', 'Ki-E2PO-v2-15k-1000', 'Ki-E2PO-v2-18k-1000', 'Ki-E2PO-v2-20k-1000', 'Ki-E2PO-v2-30k-1000', 'Ki-E2PO-v3-10k-1000', 'Ki-E2PO-v3-15k-1000', 'Ki-E2PO-v3-20k-1000', 'Ki-E2PO-v3-NoPref-15k-1000', 'Ki-E2PO-v3-Hard-15k-1000']

# 输出目录
OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "runs", "docking-" + uuid.uuid4().hex)

BOX_PADDING = 10.0
EXHAUSTIVENESS = 8
N_POSES = 1
N_CORES = 12  # 使用 12 核满负荷加速运行
CHUNK_SIZE = 10  # 每个子任务处理最多 10 个分子，实现动态负载均衡

# ================= 辅助函数 =================

def get_box_config(ref_ligand_path):
    mol = Chem.MolFromMolFile(ref_ligand_path)
    if not mol: return None, None
    conf = mol.GetConformer()
    coords = conf.GetPositions()
    center = np.mean(coords, axis=0)
    size = (np.max(coords, axis=0) - np.min(coords, axis=0)) + BOX_PADDING
    return center, size


def prepare_ligand_pdbqt(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if not mol: return None
        mol = Chem.AddHs(mol)
        if AllChem.EmbedMolecule(mol, randomSeed=42) != 0:
            if AllChem.EmbedMolecule(mol, useRandomCoords=True) != 0: return None
        try:
            AllChem.UFFOptimizeMolecule(mol)
        except:
            pass
        preparator = MoleculePreparation()
        preparator.prepare(mol)
        return preparator.write_pdbqt_string()
    except:
        return None


def dock_chunk_task(args):
    """
    对接单个分子的子任务（一个靶标的某一部分分子）
    """
    model, target, chunk_idx, to_process_sub, receptor_path, center, size, chunk_save_path = args
    records = []
    
    try:
        v = Vina(sf_name='vina', cpu=1)
        v.set_receptor(receptor_path)
        v.compute_vina_maps(center=center, box_size=size)
    except Exception as e:
        print(f"【错误】靶标 {target} 子任务 {chunk_idx} 初始化 Vina 失败: {e}", flush=True)
        for idx, smiles in to_process_sub:
            records.append({
                'Model': model, 'Target': target, 'Index': idx,
                'SMILES': smiles, 'Vina_Score': np.nan, 'Ligand_Efficiency': np.nan
            })
        df_save = pd.DataFrame(records).sort_values('Index')
        df_save.to_csv(chunk_save_path, index=False)
        return target, 0, len(to_process_sub)

    success_count = 0
    for idx, smiles in to_process_sub:
        print(f"[{model}][{target}] 正在处理第 {idx} 个分子 (子任务 {chunk_idx}): {smiles}...", flush=True)
        res = {
            'Model': model, 'Target': target, 'Index': idx,
            'SMILES': smiles, 'Vina_Score': np.nan, 'Ligand_Efficiency': np.nan
        }
        
        # 检查可旋转键数量与分子量以防卡死
        mol = Chem.MolFromSmiles(smiles)
        n_rot = Lipinski.NumRotatableBonds(mol) if mol else 0
        mw = Descriptors.MolWt(mol) if mol else 0
        if n_rot > 15 or mw > 750:
            print(f"  [跳过] 靶标 {target} 分子 {idx} 可旋转键数 ({n_rot}) > 15 或分子量 ({mw:.1f}) > 750，直接记为 NaN。", flush=True)
            records.append(res)
            continue
            
        pdbqt_str = prepare_ligand_pdbqt(smiles)
        if pdbqt_str:
            try:
                v.set_ligand_from_string(pdbqt_str)
                v.dock(exhaustiveness=EXHAUSTIVENESS, n_poses=N_POSES)
                score = v.score()[0]
                res['Vina_Score'] = score
                n_heavy = mol.GetNumHeavyAtoms() if mol else 1
                if n_heavy > 0:
                    res['Ligand_Efficiency'] = -score / n_heavy
                success_count += 1
            except Exception:
                pass
        records.append(res)
        
    df_save = pd.DataFrame(records).sort_values('Index')
    df_save.to_csv(chunk_save_path, index=False)
    return target, success_count, len(to_process_sub)


# ================= 汇总与报告 =================

def print_model_summary(model, stats_dict):
    print(f"\n{'='*70}")
    print(f"  ✅ 模型 [{model}] 全部 12 靶标对接完成！指标汇总如下：")
    print(f"{'='*70}")
    print(f"{'Target':<10} {'SR':>8} {'Mean Vina':>12} {'Median Vina':>13} {'Top-10%':>10} {'Avg LE':>9}")
    print(f"{'-'*70}")

    targets = list(stats_dict.keys())
    sr_vals, vina_vals, median_vals, top10_vals, le_vals = [], [], [], [], []

    for tgt in targets:
        s = stats_dict[tgt]
        sr = s['success_rate']
        mv = s['mean_vina']
        med = s['median_vina']
        t10 = s['top_10_vina']
        le = s['mean_le']

        sr_str   = f"{sr:.1f}%"    if not np.isnan(sr)  else "N/A"
        mv_str   = f"{mv:.3f}"     if not np.isnan(mv)  else "N/A"
        med_str  = f"{med:.3f}"    if not np.isnan(med) else "N/A"
        t10_str  = f"{t10:.3f}"    if not np.isnan(t10) else "N/A"
        le_str   = f"{le:.4f}"     if not np.isnan(le)  else "N/A"

        print(f"{tgt:<10} {sr_str:>8} {mv_str:>12} {med_str:>13} {t10_str:>10} {le_str:>9}")

        for val, lst in [(sr, sr_vals), (mv, vina_vals), (med, median_vals), (t10, top10_vals), (le, le_vals)]:
            if not np.isnan(val):
                lst.append(val)

    print(f"{'-'*70}")
    avg_sr  = np.mean(sr_vals)   if sr_vals   else float('nan')
    avg_mv  = np.mean(vina_vals) if vina_vals else float('nan')
    avg_med = np.mean(median_vals) if median_vals else float('nan')
    avg_t10 = np.mean(top10_vals) if top10_vals else float('nan')
    avg_le  = np.mean(le_vals)   if le_vals   else float('nan')

    print(f"{'AVG':<10} {avg_sr:.1f}% {avg_mv:>12.3f} {avg_med:>13.3f} {avg_t10:>10.3f} {avg_le:>9.4f}")
    print(f"{'='*70}\n")
    sys.stdout.flush()


def compute_stats_from_records(records):
    df = pd.DataFrame(records)
    total = len(df)
    vina_col = 'Vina_Score'
    le_col = 'Ligand_Efficiency'

    df_valid = df.dropna(subset=[vina_col])
    valid_count = len(df_valid)
    success_rate = (valid_count / total * 100.0) if total > 0 else 0.0

    if valid_count > 0:
        mean_vina   = df_valid[vina_col].mean()
        median_vina = df_valid[vina_col].median()
        sorted_v    = df_valid[vina_col].sort_values()
        top_n       = max(1, int(np.ceil(0.1 * valid_count)))
        top_10_vina = sorted_v.iloc[:top_n].mean()
        mean_le     = df_valid[le_col].mean() if le_col in df_valid.columns else float('nan')
    else:
        mean_vina = median_vina = top_10_vina = mean_le = float('nan')

    return {
        'success_rate': success_rate,
        'mean_vina':    mean_vina,
        'median_vina':  median_vina,
        'top_10_vina':  top_10_vina,
        'mean_le':      mean_le
    }


# ================= 主程序 =================

def configure_paths(argv=None):
    global PROJECT_ROOT, DATA_DIR, SELECTED_ROOT, OUTPUT_ROOT
    parser = argparse.ArgumentParser(description="Chunked Vina evaluation (no parameter changes)")
    parser.add_argument("--project-root", default=PROJECT_ROOT)
    parser.add_argument("--receptor-root", help="CrossDocked root containing PDB-id subdirectories")
    parser.add_argument("--selected-root", help="Root containing model/target candidate tables")
    parser.add_argument("--output-dir", default=OUTPUT_ROOT)
    parser.add_argument("--allow-existing-output", action="store_true",
                        help="Opt into legacy reuse/resume behavior")
    args = parser.parse_args(argv)
    PROJECT_ROOT = str(Path(args.project_root).expanduser().resolve())
    DATA_DIR = str(Path(args.receptor_root or os.path.join(PROJECT_ROOT, "datasets/CrossDocked2020")).expanduser().resolve())
    SELECTED_ROOT = str(Path(args.selected_root or os.path.join(PROJECT_ROOT, "evaluation/docking_candidate_selection/selected_candidates")).expanduser().resolve())
    OUTPUT_ROOT = str(Path(args.output_dir).expanduser().resolve())
    if os.path.isdir(OUTPUT_ROOT) and os.listdir(OUTPUT_ROOT) and not args.allow_existing_output:
        parser.error("Output directory is nonempty; choose a fresh --output-dir")
    return args

def main(argv=None):
    configure_paths(argv)
    start_total = time.time()
    targets = list(GENE_TO_PDB.keys())

    for model in MODELS:
        print(f"\n>>> 开始评估模型 [{model}]（动态负载均衡，分块大小={CHUNK_SIZE}）...")
        sys.stdout.flush()

        model_input_root = os.path.join(SELECTED_ROOT, model)
        model_output_dir = os.path.join(OUTPUT_ROOT, model, "vina_scores")
        os.makedirs(model_output_dir, exist_ok=True)

        all_tasks = []
        model_start = time.time()

        for target in targets:
            input_csv = os.path.join(model_input_root, target, "selected_100_for_docking.csv")
            if not os.path.exists(input_csv):
                print(f"  [警告] 找不到 {model}/{target} 的筛选文件，跳过。")
                continue

            df_in = pd.read_csv(input_csv)
            smi_col = next((c for c in ['SMILES', 'smiles', 'canonical_smiles'] if c in df_in.columns), df_in.columns[0])
            smiles_list = df_in[smi_col].dropna().astype(str).tolist()

            protein_dir   = os.path.join(DATA_DIR, GENE_TO_PDB[target])
            receptor_path = os.path.join(protein_dir, f"{GENE_TO_PDB[target]}_protein_cleaned.pdbqt")
            ref_lig_path  = os.path.join(protein_dir, f"{GENE_TO_PDB[target]}_ligand.sdf")

            if not os.path.exists(receptor_path) or not os.path.exists(ref_lig_path):
                print(f"  [跳过] {model}/{target} 缺少受体/配体文件！")
                continue

            center, size = get_box_config(ref_lig_path)
            if center is None:
                print(f"  [跳过] {model}/{target} 无法计算 Box！")
                continue

            # 检查断点续传
            save_path = os.path.join(model_output_dir, f"vina_{target}.csv")
            processed_indices = set()
            existing_records = []
            
            if os.path.exists(save_path):
                try:
                    df_old = pd.read_csv(save_path)
                    if 'Index' in df_old.columns:
                        # 只有当包含得分，或者由于分子量/可旋转键过大跳过（SMILES在 records 中，而结合能为空，但它确实被运行过了）
                        # 识别已经处理的 Index
                        processed_indices = set(df_old['Index'].dropna().astype(int).tolist())
                        existing_records = df_old.to_dict('records')
                except:
                    pass

            # 找出尚未处理的索引
            to_process = []
            for idx, smiles in enumerate(smiles_list):
                if idx not in processed_indices:
                    to_process.append((idx, smiles))

            if not to_process:
                continue

            # 将 to_process 划分为 CHUNK_SIZE 大小的块
            chunks = [to_process[i:i + CHUNK_SIZE] for i in range(0, len(to_process), CHUNK_SIZE)]
            for chunk_idx, chunk_sub in enumerate(chunks):
                chunk_save_path = os.path.join(model_output_dir, f"vina_{target}_chunk_{chunk_idx}.csv")
                all_tasks.append((model, target, chunk_idx, chunk_sub, receptor_path, center, size, chunk_save_path))

        if not all_tasks:
            print(f"  [{model}] 所有分子均已完成对接，直接汇总结果。")
        else:
            print(f"  [{model}] 共有 {len(all_tasks)} 个子任务等待对接，启动 {N_CORES} 进程并行...")
            sys.stdout.flush()

            with Pool(processes=N_CORES) as pool:
                for tgt, num_success, total_in_chunk in pool.imap_unordered(dock_chunk_task, all_tasks):
                    # 打印单任务完成提示
                    pass

        # 合并所有子任务的结果到最终 vina_{target}.csv 并清除中间文件
        print(f"  [{model}] 正在合并分块数据并更新最终结果...")
        for target in targets:
            save_path = os.path.join(model_output_dir, f"vina_{target}.csv")
            existing_records = []
            if os.path.exists(save_path):
                try:
                    existing_records = pd.read_csv(save_path).to_dict('records')
                except:
                    pass

            merged_dict = {r['Index']: r for r in existing_records}

            # 寻找所有的 chunk 文件并合并
            # 注意：子任务个数取决于 missing 的个数，我们可以直接寻找目录下所有 vina_{target}_chunk_*.csv 匹配文件
            chunk_files = [os.path.join(model_output_dir, f) for f in os.listdir(model_output_dir) if f.startswith(f"vina_{target}_chunk_") and f.endswith(".csv")]
            for cf in chunk_files:
                try:
                    df_cf = pd.read_csv(cf)
                    for r in df_cf.to_dict('records'):
                        merged_dict[r['Index']] = r
                    # 删除临时分块文件
                    os.remove(cf)
                except Exception as e:
                    print(f"  [警告] 合并并删除分块文件 {cf} 失败: {e}")

            if merged_dict:
                df_merged = pd.DataFrame(list(merged_dict.values())).sort_values('Index')
                df_merged.to_csv(save_path, index=False)

        # 计算并打印该模型的汇总指标
        model_stats = {}
        for target in targets:
            save_path = os.path.join(model_output_dir, f"vina_{target}.csv")
            if os.path.exists(save_path):
                df_tmp = pd.read_csv(save_path)
                model_stats[target] = compute_stats_from_records(df_tmp.to_dict('records'))
            else:
                model_stats[target] = {
                    'success_rate': float('nan'), 'mean_vina': float('nan'),
                    'median_vina': float('nan'), 'top_10_vina': float('nan'), 'mean_le': float('nan')
                }

        elapsed = (time.time() - model_start) / 60
        print(f"  [{model}] 耗时 {elapsed:.1f} 分钟")
        print_model_summary(model, model_stats)

    print(f"\n[✔ All Done] DPO Models Vina 对接全部完成！")
    print(f"总耗时: {(time.time() - start_total) / 60:.1f} 分钟")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
