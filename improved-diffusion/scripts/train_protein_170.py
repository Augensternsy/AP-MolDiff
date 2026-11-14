"""
Train a diffusion model on SMILES with protein sequence guidance using ChemformerTokenizer (170-length version).
支持自适应检查点断点续训，并加载预训练文本 Backbone 以 strict=False 进行跨模态微调，过滤词表尺寸不匹配的 Embedding 层。
"""

import argparse
import json, torch, os
import glob
import re
import copy
from improved_diffusion import gaussian_diffusion as gd
from improved_diffusion.respace import SpacedDiffusion, space_timesteps
import numpy as np
from improved_diffusion import logger
from improved_diffusion.transformer_model2 import TransformerNetModel2
from improved_diffusion.resample import create_named_schedule_sampler
from improved_diffusion.script_util import (
    model_and_diffusion_defaults,
    add_dict_to_argparser,
)
from improved_diffusion.train_util import TrainLoop
from transformers import set_seed
from mytokenizers import ChemformerTokenizer
from process_train_csv import TrainCSVDataset
from mydatasets import get_dataloader
import warnings
import swanlab

warnings.filterwarnings("ignore")


def find_latest_checkpoint(checkpoint_dir):
    """
    自动在指定目录下寻找步数最大的权重文件 (如 model040000.pt)
    """
    if not os.path.exists(checkpoint_dir):
        return ""

    files = glob.glob(os.path.join(checkpoint_dir, "*model*.pt"))
    if not files:
        files = glob.glob(os.path.join(checkpoint_dir, "*.pt"))

    if not files:
        return ""

    def extract_step(filepath):
        filename = os.path.basename(filepath)
        match = re.findall(r'\d+', filename)
        return int(match[-1]) if match else 0

    files.sort(key=extract_step, reverse=True)

    for f in files:
        basename = os.path.basename(f)
        if 'ema' in basename.lower():
            continue
        if 'model' in basename:
            return f

    return files[0] if files else ""


def main_worker(args):
    set_seed(args.seed)

    # 1. 动态寻找最新断点，防止断电或中断后前功尽弃
    latest_ckpt = find_latest_checkpoint(args.checkpoint_path)
    if latest_ckpt:
        print(f"\n[🚀 发现历史权重] 自动选择最新断点: {latest_ckpt}")
        args.resume_checkpoint = latest_ckpt
    else:
        print(f"\n[检查] 未在 {args.checkpoint_path} 发现历史权重，将从第 0 步开始全新训练。")

    # 初始化 swanlab
    swanlab.init(
        project="DiffusionLMProtein_170",
        config=args.__dict__,
    )
    print(swanlab.config)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("creating model and diffusion...")
    smtokenizer = ChemformerTokenizer(max_len=170)  # 强制对齐为 170 长度
    
    # 创建模型
    model = TransformerNetModel2(
        in_channels=args.model_in_channels,
        model_channels=args.model_model_channels,
        dropout=args.model_dropout,
        use_checkpoint=False, 
        config_name='bert-base-uncased',
        training_mode='e2e',
        vocab_size=len(smtokenizer),
        experiment_mode='lm',
        logits_mode=1,
        hidden_size=args.model_hidden_size,  
        num_attention_heads=args.model_num_attention_heads,
        num_hidden_layers=args.model_num_hidden_layers,
    )

    # ==================== 加载文本预训练权重并自动过滤不匹配的层 ====================
    # 仅在从头开始训练（没有发现断点）且提供了预训练模型路径时执行
    if not args.resume_checkpoint and args.pretrained_model_path:
        if os.path.exists(args.pretrained_model_path):
            print(f"\n[🔄 跨模态初始化] 从 {args.pretrained_model_path} 加载文本 Backbone 权重 (strict=False)...")
            pretrained_state = torch.load(args.pretrained_model_path, map_location="cpu")
            
            # 过滤掉与当前模型形状不匹配的层（如 word_embedding.weight，因为新分词器 vocab 大小由 312 变为 37）
            model_state = model.state_dict()
            filtered_state = {}
            mismatched_keys = []
            for k, v in pretrained_state.items():
                if k in model_state:
                    if v.shape == model_state[k].shape:
                        filtered_state[k] = v
                    else:
                        mismatched_keys.append(k)
            
            missing_keys, unexpected_keys = model.load_state_dict(filtered_state, strict=False)
            
            print(f"[🔄 跨模态初始化] 骨架权重加载成功！")
            print(f"  - 过滤掉尺寸不匹配的参数: {mismatched_keys}")
            print(f"  - 缺失/随机初始化的参数: {len(missing_keys)} 个 (包含词嵌入层、适配层和线性层)")
            print(f"  - 冗余忽略的参数: {len(unexpected_keys)} 个")
        else:
            print(f"[❌ 警告] 未能找到预训练模型路径: {args.pretrained_model_path}，将直接进行随机初始化训练。")
    # =========================================================================

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print('Model total parameter number is:', pytorch_total_params)
    print('Smiles tokenizer vocab length:', len(smtokenizer))

    model.to(device)

    diffusion = SpacedDiffusion(
        use_timesteps=[i for i in range(2000)],
        betas=gd.get_named_beta_schedule('sqrt', 2000),
        model_mean_type=(gd.ModelMeanType.START_X),
        model_var_type=(gd.ModelVarType.FIXED_LARGE),
        loss_type=gd.LossType.E2E_MSE,
        rescale_timesteps=True,
        model_arch='transformer',
        training_mode='e2e',
    )

    schedule_sampler = create_named_schedule_sampler('uniform', diffusion)
    print('load data', '*' * 50)

    # 加载 170 长度限制的数据集，蛋白质序列长度上限保持 256
    train_dataset = TrainCSVDataset(
        dir='../../datasets/',
        smi_tokenizer=smtokenizer,
        csv_path=args.csv_path,
        replace_desc=False,
        load_state=True,
        corrupt_prob=args.corrupt_prob,
        mask_desc=False,
        max_desc_length=256
    )
    print('In total', len(train_dataset), 'for training....')

    train_loader = get_dataloader(train_dataset, args.batch_size, rank=0, world_size=1)

    # 训练循环
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=train_loader,
        batch_size=args.batch_size,
        microbatch=args.microbatch, 
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16, 
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        checkpoint_path=args.checkpoint_path,
        gradient_clipping=args.gradient_clipping,
        eval_data=None,
        eval_interval=args.eval_interval
    ).run_loop()


def create_argparser():
    defaults = dict()
    text_defaults = dict(
        attention_resolutions='16,8',
        batch_size=8,  
        microbatch=2,  
        cache_mode='no',
        checkpoint_path='../../checkpoints_protein_170',  # 默认保存路径改为 170 分类文件夹
        pretrained_model_path='../../checkpoints/PLAIN_model200000.pt', # 文本预训练权重路径
        class_cond=False,
        commonGen_train='diffusion_lm/common-gen/commongen_data',
        config='ll',
        config_name='bert-base-uncased',
        data_dir='',
        dataset_config_name='wikitext-2-raw-v1',
        dataset_name='wikitext',
        diffusion_steps=2000,
        dropout=0.1,
        e2e_train='',
        ema_rate='0.9999',
        emb_scale_factor=1.0,
        eval_interval=2000,
        experiment='protein',
        experiment_mode='lm',
        fp16_scale_growth=0.001,
        gradient_clipping=2.4,
        image_size=8,
        in_channel=16,
        learn_sigma=False,
        log_interval=20,
        logits_mode=1,
        lr=0.00005,
        lr_anneal_steps=200000,
        modality='e2e-tgt',
        model_arch='transformer',
        model_name_or_path='predictability/diff_models/compress_e=5_b=60_m=gpt2_wikitext-103-raw-v1_None',
        noise_level=0.0,
        noise_schedule='sqrt',
        num_channels=128,
        num_heads=4,
        num_heads_upsample=-1,
        num_res_blocks=2,
        out_channel=16,
        padding_mode='pad',
        predict_xstart=True,
        preprocessing_num_workers=1,
        rescale_learned_sigmas=True,
        rescale_timesteps=True,
        resume_checkpoint='',  # 留空，由 main_worker 动态检索覆盖
        roc_train='diffusion_lm/ROCstory',
        save_interval=10000,
        schedule_sampler='uniform',
        seed=19991009,
        sigma_small=False,
        timestep_respacing='',
        training_mode='e2e',
        use_bert_tokenizer='no',
        use_checkpoint=False,
        use_fp16=False,
        use_kl=False,
        use_scale_shift_norm=True,
        weight_decay=0.0,
        model_in_channels=32,
        model_model_channels=128,
        model_dropout=0.1,
        model_hidden_size=1024,
        model_num_attention_heads=16,
        model_num_hidden_layers=12,
        corrupt_prob=0.4,
        csv_path='../../datasets/train.csv',
    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(text_defaults)
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    import os
    os.environ['CUDA_DEVICES_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

    print("Starting training in Pure Single-GPU auto-resume transfer mode (170 max length)...")
    args = create_argparser().parse_args()
    main_worker(args)
