import sys
import os
import json
import argparse
import logging
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# 1. Monkey-patch torch.compiler to support PyTorch 2.0.0 in transformers library
if not hasattr(torch, 'compiler'):
    from types import ModuleType
    compiler = ModuleType("compiler")
    compiler.disable = lambda func=None, recursive=False: (func if func is not None else (lambda f: f))
    torch.compiler = compiler
    sys.modules["torch.compiler"] = compiler

# 2. Monkey-patch nn.Module.load_state_dict to drop 'assign' keyword parameter
orig_load_state_dict = nn.Module.load_state_dict
def patched_load_state_dict(self, state_dict, *args, **kwargs):
    kwargs.pop('assign', None)
    return orig_load_state_dict(self, state_dict, *args, **kwargs)
nn.Module.load_state_dict = patched_load_state_dict

# Path configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
IMPROVED_DIFFUSION_DIR = os.path.join(PROJECT_ROOT, 'improved-diffusion', 'scripts')

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if IMPROVED_DIFFUSION_DIR not in sys.path:
    sys.path.insert(0, IMPROVED_DIFFUSION_DIR)

from improved_diffusion import gaussian_diffusion_self_cond as gd
from improved_diffusion.respace_self_cond import SpacedDiffusion
from improved_diffusion.transformer_model2_self_cond import TransformerNetModel2
from mytokenizers import ChemformerTokenizer
from process_train_csv import ESM2ProjectionLayer
from transformers import AutoModel, AutoTokenizer, set_seed
from datasets.ki_preference_dataset import KiPreferenceDataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("TrainDirectKiE2PO")

def extract_protein_embedding(seq, esm_model, esm_tokenizer, projection_layer, device):
    inputs = esm_tokenizer(
        [seq.strip().upper()],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256
    ).to(device)

    with torch.no_grad():
        outputs = esm_model(**inputs)
        token_embeddings = outputs.last_hidden_state
        token_embeddings = projection_layer(token_embeddings)

    attention_mask = inputs['attention_mask']
    states = token_embeddings[0] # [seq_len, 1024]
    mask = attention_mask[0]     # [seq_len]
    seq_len = states.shape[0]

    pad_len = 256 - seq_len
    if pad_len > 0:
        states = torch.nn.functional.pad(states, (0, 0, 0, pad_len))
        mask = torch.nn.functional.pad(mask, (0, pad_len))
    else:
        states = states[:256, :]
        mask = mask[:256]

    return states.unsqueeze(0), mask.unsqueeze(0)

def compute_length_norm_diff_loss(model, diffusion, x_ids, mask, desc_state, desc_mask, t, emb_noise, q_noise, self_cond_flags):
    raw_model = model.module if hasattr(model, 'module') else model
    x_start_mean = raw_model.get_embeds(x_ids)
    
    std = gd._extract_into_tensor(diffusion.sqrt_one_minus_alphas_cumprod, torch.zeros(1, dtype=torch.long, device=x_ids.device), x_start_mean.shape)
    x_start = x_start_mean + std * emb_noise
    
    x_t = diffusion.q_sample(x_start, t, noise=q_noise)
    
    self_cond = getattr(raw_model, 'self_cond', False)
    if self_cond:
        zero_self_cond = torch.zeros_like(x_t)
        self_cond_x0 = torch.zeros_like(x_t)
        
        if self_cond_flags.any():
            with torch.no_grad():
                x_t_input_prev = torch.cat([x_t, zero_self_cond], dim=-1)
                model_output_prev = model(x_t_input_prev, diffusion._scale_timesteps(t), desc_state, desc_mask)
                self_cond_x0_pred = model_output_prev.detach()
            
            for i in range(x_ids.shape[0]):
                if self_cond_flags[i]:
                    self_cond_x0[i] = self_cond_x0_pred[i]
                    
        x_t_input = torch.cat([x_t, self_cond_x0], dim=-1)
    else:
        x_t_input = x_t
        
    model_output = model(x_t_input, diffusion._scale_timesteps(t), desc_state, desc_mask)
    
    target = x_start
    token_err = ((target - model_output) ** 2).sum(dim=-1) # [B, 170]
    valid_err = token_err * mask # [B, 170]
    seq_lengths = mask.sum(dim=-1).clamp(min=1.0) # [B]
    loss = valid_err.sum(dim=-1) / seq_lengths # [B]
    return loss

def main():
    parser = argparse.ArgumentParser(description="Direct Ki-E2PO Preference Optimization Training")
    parser.add_argument('--csv_path', type=str, default=os.path.join(PROJECT_ROOT, 'data', 'bindingdb_ki', 'preference_pairs_delta1.0_no_test_targets.csv'))
    parser.add_argument('--model_path', type=str, default=os.path.join(PROJECT_ROOT, 'checkpoints_protein_170_self_cond_from_finetuned', 'PLAIN_ema_0.9999_050000.pt'))
    parser.add_argument('--reference_model_path', type=str, default="", help="Path to reference model checkpoint. If not provided, defaults to model_path.")
    parser.add_argument('--esm2_path', type=str, default=os.path.join(PROJECT_ROOT, 'esm2_t30_150M_UR50D'))
    parser.add_argument('--output_dir', type=str, default=os.path.join(PROJECT_ROOT, 'checkpoints_ki_e2po'))

    parser.add_argument('--lr', type=float, default=1e-6)
    parser.add_argument('--lambda_pref', type=float, default=0.05)
    parser.add_argument('--tau', type=float, default=1.0)
    parser.add_argument('--beta', type=float, default=None, help="If None, read from beta_calibration.json")
    parser.add_argument('--max_steps', type=int, default=5000)
    parser.add_argument('--projection_weights_path', type=str, default="", help="Path to saved projection layer weights. If not provided, initialize with seed 42.")
    parser.add_argument('--anchor_mode', type=str, default="winner_only", choices=["winner_only", "both"], help="Loss mode for anchor regularization.")

    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--max_proteins', type=int, default=None)
    parser.add_argument('--max_pairs_per_protein', type=int, default=None)
    parser.add_argument('--save_interval', type=int, default=1000)
    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--frozen_backbone', type=str, default='true', choices=['true', 'false'], help="Freeze all parameters except conditioning layers")
    parser.add_argument('--step_offset', type=int, default=0, help="Offset to add to step count for checkpoints and logs")
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Read calibrated beta if not provided
    if args.beta is None:
        calib_file = os.path.join(PROJECT_ROOT, 'data', 'bindingdb_ki', 'beta_calibration_v2.json')
        if not os.path.exists(calib_file):
            calib_file = os.path.join(PROJECT_ROOT, 'data', 'bindingdb_ki', 'beta_calibration.json')
        if os.path.exists(calib_file):
            with open(calib_file, 'r', encoding='utf-8') as f:
                calib_data = json.load(f)
            args.beta = float(calib_data.get('recommended_beta', 1.0))
            logger.info(f"Read recommended beta={args.beta:.4f} from {calib_file}")
        else:
            args.beta = 1.0
            logger.warning(f"Calibration file not found. Defaulting beta={args.beta}")

    if not os.path.exists(args.model_path):
        alt_path = os.path.join(PROJECT_ROOT, 'checkpoints_protein_170_self_cond_from_finetuned', 'PLAIN_model050000.pt')
        if os.path.exists(alt_path):
            args.model_path = alt_path
        else:
            logger.error(f"Model checkpoint not found at {args.model_path}")
            sys.exit(1)

    logger.info(f"Loading KiPreferenceDataset...")
    dataset = KiPreferenceDataset(
        csv_path=args.csv_path,
        max_proteins=args.max_proteins,
        max_pairs_per_protein=args.max_pairs_per_protein,
        seed=args.seed
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)

    tokenizer = dataset.tokenizer
    vocab_size = len(tokenizer)

    logger.info("Initializing trainable_model and reference_model...")
    def create_model(model_path):
        m = TransformerNetModel2(
            in_channels=32,
            model_channels=128,
            dropout=0.1,
            use_checkpoint=False,
            config_name='bert-base-uncased',
            training_mode='e2e',
            vocab_size=vocab_size,
            experiment_mode='lm',
            logits_mode=1,
            hidden_size=1024,
            num_attention_heads=16,
            num_hidden_layers=12,
            self_cond=True
        )
        ckpt = torch.load(model_path, map_location='cpu')
        m.load_state_dict(ckpt)
        return m

    ref_path = args.reference_model_path if args.reference_model_path else args.model_path
    trainable_model = create_model(args.model_path).to(device)
    reference_model = create_model(ref_path).to(device)

    
    trainable_model.train()
    reference_model.eval()
    for p in reference_model.parameters():
        p.requires_grad = False

    if args.frozen_backbone.lower() == 'true':
        logger.info("Frozen Backbone mode enabled. Freezing all parameters except conditioning layers ('desc').")
        for name, param in trainable_model.named_parameters():
            if 'desc' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
    else:
        logger.info("Full parameter fine-tuning enabled.")
        for p in trainable_model.parameters():
            p.requires_grad = True

    trainable_params = [p for p in trainable_model.parameters() if p.requires_grad]
    logger.info(f"Total parameters: {sum(p.numel() for p in trainable_model.parameters())}")
    logger.info(f"Trainable parameters: {sum(p.numel() for p in trainable_params)}")

    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.0)

    diffusion = SpacedDiffusion(
        use_timesteps=[i for i in range(2000)],
        betas=gd.get_named_beta_schedule('sqrt', 2000),
        model_mean_type=gd.ModelMeanType.START_X,
        model_var_type=gd.ModelVarType.FIXED_LARGE,
        loss_type=gd.LossType.E2E_MSE,
        rescale_timesteps=True,
        model_arch='transformer',
        training_mode='e2e',
    )

    logger.info(f"Loading ESM-2 model from {args.esm2_path}...")
    esm_tokenizer = AutoTokenizer.from_pretrained(args.esm2_path)
    esm_model = AutoModel.from_pretrained(args.esm2_path).to(device)
    esm_model.eval()
    for p in esm_model.parameters():
        p.requires_grad = False

    projection_layer = ESM2ProjectionLayer(input_dim=640, output_dim=1024).to(device)
    if args.projection_weights_path and os.path.exists(args.projection_weights_path):
        logger.info(f"[Confirm] Loading projection weights from: {args.projection_weights_path}")
        projection_layer.load_state_dict(torch.load(args.projection_weights_path, map_location=device))
    else:
        logger.info("[Confirm] No projection weights path provided/found. Initializing projection weights with seed 42 (matches inference).")
        torch.manual_seed(42)
        for m in projection_layer.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    projection_layer.eval()

    # Pre-build sequence to cache mapping for reusing first-stage precomputed protein embeddings
    seq_to_cache_path = {}
    try:
        mapping_file = os.path.join(PROJECT_ROOT, 'datasets', 'categorical_mappings.json')
        train_csv_file = os.path.join(PROJECT_ROOT, 'datasets', 'train.csv')
        if os.path.exists(mapping_file) and os.path.exists(train_csv_file):
            logger.info("Building sequence to precomputed cache mapping...")
            with open(mapping_file, 'r', encoding='utf-8') as f:
                mapping = json.load(f)
            seq_mapping = mapping.get('sequence', {})
            
            df_train = pd.read_csv(train_csv_file)
            train_seq_idx_to_row = {}
            for row_idx, seq_idx in enumerate(df_train['sequence']):
                if seq_idx not in train_seq_idx_to_row:
                    train_seq_idx_to_row[seq_idx] = row_idx
            
            # Map unique sequences in dataset
            unique_seqs = set(s['protein_sequence'] for s in dataset.samples)
            for seq in unique_seqs:
                if seq in seq_mapping:
                    seq_idx = seq_mapping[seq]
                    if seq_idx in train_seq_idx_to_row:
                        row_idx = train_seq_idx_to_row[seq_idx]
                        folder_idx = row_idx // 10000
                        file_path = os.path.join(PROJECT_ROOT, 'datasets', 'output', 'train_desc_states', f'{folder_idx:04d}', f'{row_idx}.pt')
                        if os.path.exists(file_path):
                            seq_to_cache_path[seq] = file_path
            logger.info(f"[Confirm] Built cache map. Found cache files for {len(seq_to_cache_path)} / {len(unique_seqs)} unique sequences.")
            if len(seq_to_cache_path) > 0:
                sample_seq = list(seq_to_cache_path.keys())[0]
                sample_path = seq_to_cache_path[sample_seq]
                sample_data = torch.load(sample_path, map_location='cpu')
                logger.info(f"[Confirm] Reusing embedding cache. Sample path: {sample_path}")
                logger.info(f"[Confirm] Protein embedding shape in cache: {sample_data['states'].shape} -> Padded to (1, 256, 1024)")
        else:
            logger.warning("Categorical mappings or train.csv not found. Re-caching not available, extracting online.")
    except Exception as e:
        logger.warning(f"Failed to build sequence cache mapping ({str(e)}), will compute online.")

    embeddings_cache = {}


    bce_loss_fn = nn.BCEWithLogitsLoss()

    logger.info(f"Starting Direct Ki-E2PO Training for max_steps={args.max_steps}...")
    step = 0
    start_time = time.time()
    
    # Metrics accumulators
    running_total_loss = 0.0
    running_anchor_loss = 0.0
    running_pref_loss = 0.0
    running_margin_theta = 0.0
    running_margin_ref = 0.0
    running_acc = 0.0
    running_w_loss = 0.0
    running_l_loss = 0.0
    
    data_iter = iter(dataloader)

    while step < args.max_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        step += 1
        optimizer.zero_grad()

        B = len(batch['winner_smiles'])
        w_ids = batch['winner_ids'].to(device)
        w_mask = batch['winner_mask'].to(device)
        l_ids = batch['loser_ids'].to(device)
        l_mask = batch['loser_mask'].to(device)
        delta_pKi = batch['delta_pKi'].to(device).float()

        # Dynamic protein embedding lookup/cache
        desc_states_list, desc_masks_list = [], []
        for seq in batch['protein_sequence']:
            if seq not in embeddings_cache:
                if seq in seq_to_cache_path:
                    cache_data = torch.load(seq_to_cache_path[seq], map_location='cpu')
                    states = cache_data['states'].float() # [seq_len, 1024]
                    mask = cache_data['mask']             # [seq_len]
                    
                    pad_len = 256 - states.shape[0]
                    if pad_len > 0:
                        states = torch.nn.functional.pad(states, (0, 0, 0, pad_len))
                        mask = torch.nn.functional.pad(mask, (0, pad_len))
                    else:
                        states = states[:256, :]
                        mask = mask[:256]
                    embeddings_cache[seq] = (states, mask.long())
                else:
                    st, mk = extract_protein_embedding(seq, esm_model, esm_tokenizer, projection_layer, device)
                    embeddings_cache[seq] = (st.squeeze(0).cpu(), mk.squeeze(0).cpu().long())
            st, mk = embeddings_cache[seq]
            desc_states_list.append(st)
            desc_masks_list.append(mk)

        desc_states = torch.stack(desc_states_list, dim=0).to(device)
        desc_masks = torch.stack(desc_masks_list, dim=0).to(device)

        # Timestep t, get_x_start noise, and q_sample noise shared across winner/loser/trainable/reference
        t = torch.randint(0, diffusion.num_timesteps, (B,), device=device)
        shared_emb_noise = torch.randn(B, 170, trainable_model.in_channels, device=device)
        shared_q_noise = torch.randn(B, 170, trainable_model.in_channels, device=device)
        self_cond_flags = (torch.rand(B, device=device) < 0.5)

        # 1. Trainable model forward pass
        loss_w_theta = compute_length_norm_diff_loss(
            trainable_model, diffusion, w_ids, w_mask, desc_states, desc_masks, t, 
            emb_noise=shared_emb_noise, q_noise=shared_q_noise, self_cond_flags=self_cond_flags
        )
        loss_l_theta = compute_length_norm_diff_loss(
            trainable_model, diffusion, l_ids, l_mask, desc_states, desc_masks, t, 
            emb_noise=shared_emb_noise, q_noise=shared_q_noise, self_cond_flags=self_cond_flags
        )

        # 2. Reference model forward pass (no_grad)
        with torch.no_grad():
            loss_w_ref = compute_length_norm_diff_loss(
                reference_model, diffusion, w_ids, w_mask, desc_states, desc_masks, t, 
                emb_noise=shared_emb_noise, q_noise=shared_q_noise, self_cond_flags=self_cond_flags
            )
            loss_l_ref = compute_length_norm_diff_loss(
                reference_model, diffusion, l_ids, l_mask, desc_states, desc_masks, t, 
                emb_noise=shared_emb_noise, q_noise=shared_q_noise, self_cond_flags=self_cond_flags
            )

        # 3. Margins and losses
        score_w_theta = -loss_w_theta
        score_l_theta = -loss_l_theta
        score_w_ref = -loss_w_ref
        score_l_ref = -loss_l_ref

        margin_theta = score_w_theta - score_l_theta
        margin_ref = score_w_ref - score_l_ref

        y_soft = torch.sigmoid(delta_pKi / args.tau)
        logit = args.beta * (margin_theta - margin_ref)
        
        loss_pref = bce_loss_fn(logit, y_soft)
        
        if args.anchor_mode == 'winner_only':
            loss_anchor = loss_w_theta.mean()
        else:
            loss_anchor = 0.5 * (loss_w_theta.mean() + loss_l_theta.mean())
            
        loss_total = loss_anchor + args.lambda_pref * loss_pref


        loss_total.backward()
        torch.nn.utils.clip_grad_norm_(trainable_model.parameters(), 1.0)
        optimizer.step()

        # Track metrics
        acc = (margin_theta > 0).float().mean().item()
        running_total_loss += loss_total.item()
        running_anchor_loss += loss_anchor.item()
        running_pref_loss += loss_pref.item()
        running_margin_theta += margin_theta.mean().item()
        running_margin_ref += margin_ref.mean().item()
        running_acc += acc
        running_w_loss += loss_w_theta.mean().item()
        running_l_loss += loss_l_theta.mean().item()

        if step % args.log_interval == 0:
            avg_tot = running_total_loss / args.log_interval
            avg_anc = running_anchor_loss / args.log_interval
            avg_prf = running_pref_loss / args.log_interval
            avg_m_th = running_margin_theta / args.log_interval
            avg_m_rf = running_margin_ref / args.log_interval
            avg_acc = running_acc / args.log_interval
            avg_w = running_w_loss / args.log_interval
            avg_l = running_l_loss / args.log_interval
            
            # Compute preference loss ratio and warning
            pref_contrib = args.lambda_pref * avg_prf
            pref_ratio = pref_contrib / avg_tot
            
            elapsed = time.time() - start_time
            logger.info(
                f"[Step {(step + args.step_offset):4d}/{(args.max_steps + args.step_offset)}] Loss: {avg_tot:.4f} (Anchor: {avg_anc:.4f}, Pref: {avg_prf:.4f}) | "
                f"Pref Contrib: {pref_contrib:.4f} (Ratio: {pref_ratio*100:.2f}%) | "
                f"Margin(th/ref): {avg_m_th:+.4f}/{avg_m_rf:+.4f} | Acc: {avg_acc*100:.1f}% | "
                f"W/L Loss: {avg_w:.4f}/{avg_l:.4f} | {elapsed:.1f}s"
            )
            
            if pref_ratio < 0.01:
                logger.warning(
                    f"Preference loss contribution ratio is below 1% ({pref_ratio*100:.2f}%). "
                    f"Consider increasing --lambda_pref or using a different anchor mode."
                )
            
            running_total_loss = 0.0
            running_anchor_loss = 0.0
            running_pref_loss = 0.0
            running_margin_theta = 0.0
            running_margin_ref = 0.0
            running_acc = 0.0
            running_w_loss = 0.0
            running_l_loss = 0.0

        if step % args.save_interval == 0 or step == args.max_steps:
            effective_step = step + args.step_offset
            ckpt_name = f"direct_ki_e2po_{effective_step // 1000}k.pt" if effective_step >= 1000 else f"direct_ki_e2po_step_{effective_step}.pt"
            ckpt_path = os.path.join(args.output_dir, ckpt_name)
            torch.save(trainable_model.state_dict(), ckpt_path)
            logger.info(f"Saved checkpoint to: {ckpt_path}")

    logger.info("Direct Ki-E2PO Training Finished Successfully!")

if __name__ == '__main__':
    main()
