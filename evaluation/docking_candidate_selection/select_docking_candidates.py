import os
import sys
import argparse
import uuid
from pathlib import Path
import logging
import json
import random
import pandas as pd
import numpy as np

# 导入 RDKit 模块
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs
    from rdkit.ML.Cluster import Butina
except ImportError:
    print("【错误】未检测到 RDKit 库，此脚本依赖 RDKit 进行分子有效性检查与相似度计算。")
    sys.exit(1)

# 配置日志
def setup_logging(out_dir):
    log_dir = os.path.join(out_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "select_docking_candidates.log")
    
    logger = logging.getLogger("selection")
    logger.setLevel(logging.INFO)
    
    # 清理已有的 handlers，防止重复输出
    if logger.handlers:
        logger.handlers.clear()
        
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    logger.addHandler(file_handler)
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(console_handler)
    
    return logger

# RDKit 处理相关函数
def is_valid_smiles(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except:
        return False

def to_canonical_smiles(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            return Chem.MolToSmiles(mol, canonical=True)
    except:
        pass
    return None

def get_morgan_fp(mol):
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)

# 寻找 reference active 文件
def find_reference_file(project_root, target):
    target_lower = target.lower()
    
    # 优先搜索 datasets/targets/actives_bindingdb_cl/
    cl_dir = os.path.join(project_root, "datasets/targets/actives_bindingdb_cl")
    if os.path.exists(cl_dir):
        for f in os.listdir(cl_dir):
            if target_lower in f.lower() and f.endswith(('.csv', '.txt', '.smi')):
                return os.path.join(cl_dir, f)
                
    # 备选搜索整个 datasets 目录
    datasets_dir = os.path.join(project_root, "datasets")
    if os.path.exists(datasets_dir):
        for root, dirs, files in os.walk(datasets_dir):
            # 排除隐藏目录
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            for f in files:
                f_lower = f.lower()
                if target_lower in f_lower and any(k in f_lower for k in ['active', 'bindingdb', 'reference', 'known']):
                    if f.endswith(('.csv', '.txt', '.smi')):
                        return os.path.join(root, f)
    return None

# 读取 reference smiles
def load_reference_actives(ref_file, logger):
    ref_smiles_set = set()
    ref_mols = []
    ref_fps = []
    
    if not ref_file or not os.path.exists(ref_file):
        return ref_fps
        
    logger.info(f"正在读取参考活性分子文件: {ref_file}")
    try:
        if ref_file.endswith('.csv'):
            df = pd.read_csv(ref_file)
            smiles_col = None
            for col in ["SMILES", "smiles", "Smiles", "mol", "molecule", "sequence"]:
                if col in df.columns:
                    smiles_col = col
                    break
            if smiles_col:
                smiles_list = df[smiles_col].dropna().tolist()
            else:
                # 如果没有匹配的列，尝试第一列
                smiles_list = df.iloc[:, 0].dropna().tolist()
        else:
            with open(ref_file, 'r', encoding='utf-8') as f:
                smiles_list = [line.strip().split()[0] for line in f if line.strip() and not line.startswith('#')]
                
        for s in smiles_list:
            s_str = str(s).strip()
            mol = Chem.MolFromSmiles(s_str)
            if mol:
                can_s = Chem.MolToSmiles(mol, canonical=True)
                if can_s not in ref_smiles_set:
                    ref_smiles_set.add(can_s)
                    ref_mols.append(mol)
                    ref_fps.append(get_morgan_fp(mol))
                    
        logger.info(f"成功加载 {len(ref_fps)} 个唯一的有效参考活性分子。")
    except Exception as e:
        logger.error(f"读取参考分子文件失败 {ref_file}: {e}")
        
    return ref_fps

# 寻找 generated 候选文件
def find_generated_files(project_root, targets):
    candidate_files = []
    exclude_dirs = ['.git', '__pycache__', '.ipynb_checkpoints', 'wandb', 'swanlog', 'node_modules', 'docking_candidate_selection', 'docking_results', 'docking_results_selected', 'ki_e2po_eval', 'ki_e2po_eval_v2']
    
    # 扫描 project_root 目录
    for root, dirs, files in os.walk(project_root):
        # 排除不需要扫描的目录，防止死循环
        dirs[:] = [d for d in dirs if d not in exclude_dirs and not d.startswith('.')]
        
        for f in files:
            if not f.endswith(('.csv', '.smi', '.txt')):
                continue
                
            f_path = os.path.join(root, f)
            rel_path = os.path.relpath(f_path, project_root)
            path_lower = rel_path.lower()
            
            # grep-like 规则
            keywords = ["generated", "generate", "sample", "smiles", "reinvent", "tamgen", "pocketflow", "self", "cond", "clamp"]
            if not any(kw in path_lower for kw in keywords):
                continue
                
            # 确认属于我们的 12 个 target 之一
            matched_target = None
            for t in targets:
                # 为避免 GRIK1 匹配到 KDM1A 等，可做分词检测或直接 substring 检测
                if f"_{t.lower()}" in path_lower or f"/{t.lower()}" in path_lower or f"{t.lower()}_" in path_lower or f"_{t.lower()}." in path_lower:
                    matched_target = t
                    break
                elif t.lower() in f.lower():
                    matched_target = t
                    break
                    
            if matched_target:
                candidate_files.append((f_path, rel_path, matched_target))
                
    return candidate_files

# 解析模型名称
def infer_model_name(rel_path):
    path_lower = rel_path.lower()
    
    model_keywords = {
        "protobind_local": "ProtoBind-Diff-Local",
        "reinvent": "REINVENT4",
        "tamgen": "TamGen",
        "pocketflow": "PocketFlow",
        "protobind": "ProtoBind-Diff",
        "targetdiff": "TargetDiff",
        "pocket2mol": "Pocket2Mol",
        "ki_e2po_v1_20k_1000": "Ki-E2PO-v1-20k-1000",
        "ki_e2po_v2_15k_1000": "Ki-E2PO-v2-15k-1000",
        "ki_e2po_v2_18k_1000": "Ki-E2PO-v2-18k-1000",
        "ki_e2po_v2_20k_1000": "Ki-E2PO-v2-20k-1000",
        "ki_e2po_v2_30k_1000": "Ki-E2PO-v2-30k-1000",
        "ki_e2po_v3_10k_1000": "Ki-E2PO-v3-10k-1000",
        "ki_e2po_v3_15k_1000": "Ki-E2PO-v3-15k-1000",
        "ki_e2po_v3_20k_1000": "Ki-E2PO-v3-20k-1000",
        "ki_e2po_nopref_15k_1000": "Ki-E2PO-v3-NoPref-15k-1000",
        "ki_e2po_hard_v3_15k_1000": "Ki-E2PO-v3-Hard-15k-1000"
    }

    
    for kw, name in model_keywords.items():
        if kw in path_lower:
            return name, False
            
    # 特别针对 PathB 50k 1000分子无夹持运行，防路径覆盖冲突
    if "50k_noclamp_1000" in path_lower:
        return "SelfCond_NoClamp_PathB_50k_1000", False
            
    # 如果包含 self_cond 或 baseline，解析夹持配置
    if "self_cond" in path_lower or "selfcond" in path_lower:
        if "noclamp" in path_lower:
            return "SelfCond_NoClamp", False
        elif "soft" in path_lower:
            return "SelfCond_SoftClamp", False
        elif "hard" in path_lower:
            return "SelfCond_HardClamp", False
        return "SelfCond", False
        
    if "baseline" in path_lower:
        if "noclamp" in path_lower:
            return "Baseline_NoClamp", False
        elif "soft" in path_lower:
            return "Baseline_SoftClamp", False
        elif "hard" in path_lower:
            return "Baseline_HardClamp", False
        return "Baseline", False
        
    if "noclamp" in path_lower:
        return "NoClamp", False
    elif "soft" in path_lower:
        return "SoftClamp", False
    elif "hard" in path_lower:
        return "HardClamp", False
        
    # 无法识别时，使用父目录名
    parent_dir = os.path.basename(os.path.dirname(rel_path))
    if parent_dir and parent_dir != ".":
        return parent_dir, True
    return "UnknownModel", True

# 识别 SMILES 列名
def identify_smiles_column(df):
    candidates = ["SMILES", "smiles", "Smiles", "mol", "molecule", "generated_smiles", "canonical_smiles"]
    for c in candidates:
        if c in df.columns:
            return c
    for c in df.columns:
        if c.lower() == 'smiles':
            return c
    return None

# 多样性选择实现 (Butina 聚类法)
def perform_diverse_selection(mols, fps, max_selected, logger):
    n_fps = len(fps)
    if n_fps <= max_selected:
        return list(range(n_fps)), "all_retained", None
        
    try:
        # 计算距离矩阵
        dists = []
        for i in range(1, n_fps):
            sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
            dists.extend([1 - x for x in sims])
            
        # 使用 Butina 进行聚类 (cutoff=0.4 表示相似度低于 0.6)
        cutoff = 0.4
        clusters = Butina.ClusterData(dists, n_fps, distThresh=cutoff, isDistData=True)
        
        # 按类大小降序排列
        sorted_clusters = sorted(clusters, key=len, reverse=True)
        
        selected_indices = []
        picked_per_cluster = [0] * len(sorted_clusters)
        
        while len(selected_indices) < max_selected:
            added_any = False
            for c_idx, cluster in enumerate(sorted_clusters):
                if len(selected_indices) >= max_selected:
                    break
                current_pick = picked_per_cluster[c_idx]
                if current_pick < len(cluster):
                    idx = cluster[current_pick]
                    selected_indices.append(idx)
                    picked_per_cluster[c_idx] += 1
                    added_any = True
            if not added_any:
                break
                
        return selected_indices, "diverse (Butina clustering)", None
    except Exception as e:
        fallback_reason = f"Butina clustering error: {e}"
        logger.warning(f"多样性选择失败，退回到随机选择。原因: {fallback_reason}")
        return None, "random (fallback)", fallback_reason

def main():
    parser = argparse.ArgumentParser(description="对接前候选分子选择与过滤工具")
    parser.add_argument("--project_root", type=str, default=str(Path(__file__).resolve().parents[2]), help="项目根目录")
    parser.add_argument("--generated_root", type=str, default="auto", help="生成分子的根目录或 'auto'")
    parser.add_argument("--reference_root", type=str, default="auto", help="参考活性分子的根目录或 'auto'")
    parser.add_argument("--out_dir", type=str, default="runs/docking-selection-" + uuid.uuid4().hex, help="结果输出目录；旧位置可显式传入")
    parser.add_argument("--targets", type=str, default="ESR1,HCRTR1,JAK1,P2RX3,KDM1A,IDH1,RIOK1,NR4A1,GRIK1,CCR9,FTO,SPIN1", help="需要处理的靶标")
    parser.add_argument("--max_selected", type=int, default=100, help="每个模型每个靶标选择的最大分子数")
    parser.add_argument("--novelty_threshold", type=float, default=0.5, help="相似度阈值 (Tanimoto < threshold 判为 novel)")
    parser.add_argument("--selection_mode", type=str, default="diverse", choices=["random", "diverse"], help="选择模式")
    parser.add_argument("--seed", type=int, default=42, help="固定随机数种子")
    parser.add_argument("--dry_run", action="store_true", help="只进行文件发现与匹配检查，不输出候选分子")
    
    parser.add_argument("--allow-existing-output", action="store_true", help="显式允许复用历史输出目录")
    args = parser.parse_args()
    args.project_root = str(Path(args.project_root).expanduser().resolve())
    
    # 随机种子固定
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # 转换 targets 列表
    targets_list = [t.strip() for t in args.targets.split(",") if t.strip()]
    
    # 创建输出目录
    out_dir_abs = os.path.isabs(args.out_dir) and args.out_dir or os.path.join(args.project_root, args.out_dir)
    if os.path.isdir(out_dir_abs) and os.listdir(out_dir_abs) and not args.allow_existing_output:
        parser.error("输出目录非空；指定新目录或显式使用 --allow-existing-output")
    os.makedirs(out_dir_abs, exist_ok=True)
    
    # 初始化日志
    logger = setup_logging(out_dir_abs)
    
    logger.info("==================================================================")
    logger.info("开始执行对接候选分子筛选与过滤任务")
    logger.info("==================================================================")
    logger.info(f"项目根目录: {args.project_root}")
    logger.info(f"输出目录: {out_dir_abs}")
    logger.info(f"选择模式: {args.selection_mode} | 随机种子: {args.seed}")
    logger.info(f"最大筛选数量: {args.max_selected} | Novelty 相似度阈值: {args.novelty_threshold}")
    logger.info(f"Dry Run 模式: {args.dry_run}")
    
    # 1. 扫描生成结果文件
    logger.info("\n>>> 正在扫描候选生成文件...")
    raw_found_files = find_generated_files(args.project_root, targets_list)
    logger.info(f"共发现 {len(raw_found_files)} 个符合条件的生成数据文件。")
    
    # 对 (model_name, target) 进行去重，优先选择包含 evaluation/vina 等更完整评估数据的源文件
    deduped = {}
    for f_path, rel_path, target in raw_found_files:
        model_name, _ = infer_model_name(rel_path)
        key = (model_name, target)
        if key not in deduped:
            deduped[key] = (f_path, rel_path, target)
        else:
            existing_path = deduped[key][1].lower()
            new_path = rel_path.lower()
            
            def get_file_priority(p):
                score = 0
                if 'vina' in p:
                    score += 4
                if 'evaluated' in p:
                    score += 2
                if 'summary' in p:
                    score -= 2 # summary.csv 往往只包含统计指标，不包含具体的分子行数据
                return score
                
            if get_file_priority(new_path) > get_file_priority(existing_path):
                deduped[key] = (f_path, rel_path, target)
                
    all_found_files = list(deduped.values())
    logger.info(f"去重及优先级筛选后，实际处理 {len(all_found_files)} 个文件。")
    
    report_rows = []
    ref_fps_cache = {}
    
    # 逐个文件处理
    for file_path, rel_path, target in all_found_files:
        model_name, inferred_from_parent = infer_model_name(rel_path)
        logger.info(f"\n------------------------------------------------------------------")
        logger.info(f"处理文件: {rel_path}")
        logger.info(f"识别目标: Target={target} | Model={model_name} (推导自父目录: {inferred_from_parent})")
        
        status = "Success"
        notes_list = []
        
        # 2. 读取输入分子文件
        try:
            if file_path.endswith('.csv'):
                df_input = pd.read_csv(file_path)
                smi_col = identify_smiles_column(df_input)
                if smi_col:
                    logger.info(f"识别到 SMILES 列: '{smi_col}'")
                    smiles_series = df_input[smi_col].dropna().astype(str)
                else:
                    logger.warning(f"CSV 文件中未识别到标准的 SMILES 列，尝试第一列...")
                    smiles_series = df_input.iloc[:, 0].dropna().astype(str)
                    notes_list.append("SMILES column not identified, fallback to first column")
            else:
                # .smi 或 .txt 文件
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = [line.strip() for line in f if line.strip() and not line.startswith('#')]
                smiles_list = [line.split()[0] for line in lines]
                df_input = pd.DataFrame({'original_smiles': smiles_list})
                smiles_series = df_input['original_smiles']
                logger.info(f"识别到文本文件，共计 {len(smiles_series)} 行。")
        except Exception as e:
            logger.error(f"读取文件失败 {rel_path}: {e}")
            report_rows.append({
                "Model": model_name, "Target": target, "Source file": rel_path,
                "Total generated": 0, "Valid count": 0, "Validity": 0.0,
                "Unique valid count": 0, "Reference active count": 0, "Novel unique count": 0,
                "Selected count": 0, "Selection mode": "N/A", "Status": "Failed",
                "Notes": f"Error reading file: {e}"
            })
            continue
            
        total_generated = len(smiles_series)
        if total_generated == 0:
            logger.warning("该文件为空，跳过处理。")
            report_rows.append({
                "Model": model_name, "Target": target, "Source file": rel_path,
                "Total generated": 0, "Valid count": 0, "Validity": 0.0,
                "Unique valid count": 0, "Reference active count": 0, "Novel unique count": 0,
                "Selected count": 0, "Selection mode": "N/A", "Status": "Failed",
                "Notes": "Empty input file"
            })
            continue
            
        # 3. RDKit Validity & Canonicalization
        valid_canonical_smiles = []
        valid_indices = []
        valid_mols = []
        
        for idx, smi in smiles_series.items():
            s_str = smi.strip()
            mol = Chem.MolFromSmiles(s_str)
            if mol:
                can_s = Chem.MolToSmiles(mol, canonical=True)
                if can_s:
                    valid_canonical_smiles.append(can_s)
                    valid_indices.append(idx)
                    valid_mols.append(mol)
                    
        valid_count = len(valid_canonical_smiles)
        validity = valid_count / total_generated
        logger.info(f"有效率分析: Total={total_generated} | Valid={valid_count} | Validity={validity:.2%}")
        
        if valid_count == 0:
            logger.warning("无有效 SMILES 分子，跳过后续分析。")
            report_rows.append({
                "Model": model_name, "Target": target, "Source file": rel_path,
                "Total generated": total_generated, "Valid count": 0, "Validity": 0.0,
                "Unique valid count": 0, "Reference active count": 0, "Novel unique count": 0,
                "Selected count": 0, "Selection mode": "N/A", "Status": "Failed",
                "Notes": "No valid SMILES molecules found"
            })
            continue
            
        # 4. 去重
        seen_canonical = {}
        unique_valid_mols = []
        unique_valid_canonical = []
        unique_original_idx = []
        
        for i, can_s in enumerate(valid_canonical_smiles):
            original_idx = valid_indices[i]
            if can_s not in seen_canonical:
                seen_canonical[can_s] = True
                unique_valid_canonical.append(can_s)
                unique_valid_mols.append(valid_mols[i])
                unique_original_idx.append(original_idx)
                
        unique_valid_count = len(unique_valid_canonical)
        logger.info(f"唯一性分析: Unique Valid={unique_valid_count} | 占有效比例={(unique_valid_count/valid_count):.2%}")
        
        # 5. 加载 reference actives 并计算最大 Tanimoto 相似度
        ref_file = find_reference_file(args.project_root, target)
        ref_fps = []
        if ref_file:
            if target not in ref_fps_cache:
                ref_fps_cache[target] = load_reference_actives(ref_file, logger)
            ref_fps = ref_fps_cache[target]
            
        ref_active_count = len(ref_fps)
        novel_unique_mols = []
        novel_unique_canonical = []
        novel_original_idx = []
        novel_max_similarities = []
        
        novelty_filter_applied = ref_active_count > 0
        
        # 计算生成分子的 fp 及其最大相似度
        unique_fps = [get_morgan_fp(m) for m in unique_valid_mols]
        
        for i, gen_fp in enumerate(unique_fps):
            can_s = unique_valid_canonical[i]
            orig_idx = unique_original_idx[i]
            mol = unique_valid_mols[i]
            
            if novelty_filter_applied:
                sims = DataStructs.BulkTanimotoSimilarity(gen_fp, ref_fps)
                max_sim = max(sims) if sims else 0.0
                is_novel = max_sim < args.novelty_threshold
            else:
                max_sim = np.nan
                is_novel = True
                
            novel_max_similarities.append(max_sim)
            
            if is_novel:
                novel_unique_mols.append(mol)
                novel_unique_canonical.append(can_s)
                novel_original_idx.append(orig_idx)
                
        novel_unique_count = len(novel_unique_canonical)
        
        if novelty_filter_applied:
            logger.info(f"新颖性分析: Reference Actives={ref_active_count} | Novel Unique={novel_unique_count} | 占唯一比例={(novel_unique_count/unique_valid_count):.2%}")
        else:
            logger.warning(f"未找到靶标 {target} 的已知活性分子参考文件。将不应用新颖性过滤！")
            notes_list.append("reference_missing=True; novelty_filter_applied=False")
            status = "Partial"
            
        # 6. 选择最多 100 个分子
        candidate_pool_mols = novel_unique_mols
        candidate_pool_canonical = novel_unique_canonical
        candidate_pool_orig_idx = novel_original_idx
        
        # 在候选池中进行筛选
        pool_size = len(candidate_pool_canonical)
        selected_indices = []
        applied_mode = args.selection_mode
        fallback_reason = None
        
        if pool_size == 0:
            logger.warning("经过过滤后候选分子池为空！将无法选择任何分子。")
            notes_list.append("Candidate pool is empty")
        elif pool_size <= args.max_selected:
            selected_indices = list(range(pool_size))
            applied_mode = "all_retained"
        else:
            if args.selection_mode == "diverse":
                pool_fps = [unique_fps[unique_valid_canonical.index(s)] for s in candidate_pool_canonical]
                selected_indices, applied_mode, fallback_reason = perform_diverse_selection(
                    candidate_pool_mols, pool_fps, args.max_selected, logger
                )
                if fallback_reason:
                    notes_list.append(f"diverse selection fallback to random: {fallback_reason}")
                    
            # 如果是 random 模式，或者是 diverse 失败后退回到 random
            if not selected_indices:
                selected_indices = random.sample(range(pool_size), args.max_selected)
                
        selected_count = len(selected_indices)
        logger.info(f"最终筛选: Selected={selected_count} / {args.max_selected} | 筛选机制: {applied_mode}")
        
        if selected_count < args.max_selected and pool_size > 0:
            notes_list.append(f"selected count ({selected_count}) less than max_selected ({args.max_selected})")
            
        # 7. 保存输出文件 (非 dry run 模式)
        if not args.dry_run:
            # 构建输出文件夹
            model_out_dir = os.path.join(out_dir_abs, "selected_candidates", model_name, target)
            os.makedirs(model_out_dir, exist_ok=True)
            
            # --- 保存全量中间过滤结果 (all_valid_unique_with_similarity.csv) ---
            # 整理数据框
            # 找到对应 index 的原始信息
            all_records = []
            for i, can_s in enumerate(unique_valid_canonical):
                orig_idx = unique_original_idx[i]
                orig_row = df_input.iloc[orig_idx].to_dict()
                
                max_sim = novel_max_similarities[i]
                is_novel = max_sim < args.novelty_threshold if novelty_filter_applied else True
                
                record = {
                    'canonical_smiles': can_s,
                    'max_tanimoto_to_target_actives': max_sim if novelty_filter_applied else "N/A",
                    'is_valid': True,
                    'is_unique': True,
                    'is_novel': is_novel,
                    'source_file': rel_path
                }
                # 合并原始列
                record.update(orig_row)
                all_records.append(record)
                
            df_all = pd.DataFrame(all_records)
            all_csv_path = os.path.join(model_out_dir, "all_valid_unique_with_similarity.csv")
            df_all.to_csv(all_csv_path, index=False)
            
            # --- 保存 100 个筛选候选 (selected_100_for_docking.csv) ---
            if selected_count > 0:
                selected_records = []
                for rank, p_idx in enumerate(selected_indices, 1):
                    can_s = candidate_pool_canonical[p_idx]
                    orig_idx = candidate_pool_orig_idx[p_idx]
                    orig_row = df_input.iloc[orig_idx].to_dict()
                    
                    # 查找在 unique 列表中对应的 max_sim
                    u_idx = unique_valid_canonical.index(can_s)
                    max_sim = novel_max_similarities[u_idx]
                    is_novel = max_sim < args.novelty_threshold if novelty_filter_applied else True
                    
                    record = {
                        'rank': rank,
                        'model_name': model_name,
                        'target': target,
                        'original_smiles': smiles_series.iloc[orig_idx],
                        'canonical_smiles': can_s,
                        'max_tanimoto_to_target_actives': max_sim if novelty_filter_applied else "N/A",
                        'is_valid': True,
                        'is_unique': True,
                        'is_novel': is_novel,
                        'selection_mode': applied_mode,
                        'source_file': rel_path
                    }
                    record.update(orig_row)
                    selected_records.append(record)
                    
                df_sel = pd.DataFrame(selected_records)
                sel_csv_path = os.path.join(model_out_dir, "selected_100_for_docking.csv")
                df_sel.to_csv(sel_csv_path, index=False)
                
        # 记录汇总数据
        report_rows.append({
            "Model": model_name,
            "Target": target,
            "Source file": rel_path,
            "Total generated": total_generated,
            "Valid count": valid_count,
            "Validity": f"{validity:.2%}",
            "Unique valid count": unique_valid_count,
            "Reference active count": ref_active_count,
            "Novel unique count": novel_unique_count,
            "Selected count": selected_count,
            "Selection mode": applied_mode,
            "Status": status,
            "Notes": "; ".join(notes_list) if notes_list else "None"
        })
        
    # 8. 生成汇总报告 (Markdown 和 JSON)
    if report_rows:
        df_report = pd.DataFrame(report_rows)
        
        # 报告保存路径
        report_dir = os.path.join(out_dir_abs, "reports")
        os.makedirs(report_dir, exist_ok=True)
        
        md_path = os.path.join(report_dir, "docking_candidate_selection_report.md")
        json_path = os.path.join(report_dir, "docking_candidate_selection_report.json")
        
        # 输出为 JSON
        if not args.dry_run:
            try:
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(report_rows, f, indent=4, ensure_ascii=False)
                logger.info(f"\n[✔ Success] 成功保存 JSON 汇总报告至: {json_path}")
            except Exception as e:
                logger.error(f"保存 JSON 报告失败: {e}")
                
        # 构筑 Markdown 表格
        md_content = "# 对接与结构预测候选分子筛选汇总报告 (Docking Candidate Selection Report)\n\n"
        md_content += f"本报告汇总了各生成模型在 12 个靶点上的分子筛选过滤状态。选择模式：`{args.selection_mode}`，相似度阈值：`Tanimoto < {args.novelty_threshold}`。\n\n"
        
        if args.dry_run:
            md_content += "> [!NOTE]\n> **当前运行模式为 Dry Run**。以下表格数据为预检测数据，未进行实际的候选分子 CSV 文件输出。\n\n"
            
        md_content += "| Model | Target | Source file | Total generated | Valid count | Validity | Unique valid count | Reference active count | Novel unique count | Selected count | Selection mode | Status | Notes |\n"
        md_content += "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|\n"
        
        for r in report_rows:
            md_content += f"| {r['Model']} | {r['Target']} | {r['Source file']} | {r['Total generated']} | {r['Valid count']} | {r['Validity']} | {r['Unique valid count']} | {r['Reference active count']} | {r['Novel unique count']} | {r['Selected count']} | {r['Selection mode']} | {r['Status']} | {r['Notes']} |\n"
            
        if not args.dry_run:
            try:
                with open(md_path, 'w', encoding='utf-8') as f:
                    f.write(md_content)
                logger.info(f"[✔ Success] 成功保存 Markdown 汇总报告至: {md_path}")
            except Exception as e:
                logger.error(f"保存 Markdown 报告失败: {e}")
                
        # 终端简易输出
        logger.info("\n========================================= 任务执行汇总 =========================================")
        print(df_report[['Model', 'Target', 'Total generated', 'Valid count', 'Unique valid count', 'Novel unique count', 'Selected count', 'Status']].to_string(index=False))
        logger.info("================================================================================================")
        
    logger.info("\n任务执行结束。")
    logger.info("==================================================================")

if __name__ == "__main__":
    main()
