# Licensed base: Diffusion-LM 759889d58ef38e2eed41a8c34db8032e072826f4 (Apache-2.0).
# OpenAI-derived portions retain their upstream MIT notices; see ../THIRD_PARTY_NOTICES.md.
# AP-MolDiff adapters: newly expressed from the frozen interface/behavior contract.
# Wsy-authorized local contributions remain subject to their MIT authorization.
# No assertion of full historical-model or paper reproduction is made.
import torch
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.utils.rnn import pad_sequence

def custom_collate_fn(batch):
    """
    自定义批处理拼接函数，自动检测使用的分词器类别（regexTokenizer 还是 ChemformerTokenizer），
    自适应调整 MAX_LEN 与 MAX_TOKEN_ID 以防止 CUDA 越界和长度溢出。
    """
    pad_idx = 0
    
    # 动态检测是否存在大于 36 的 token id，判断是否为 regexTokenizer
    has_large_tokens = False
    for b in batch:
        t = b['tok_smiles']
        if isinstance(t, torch.Tensor):
            max_val = t.max().item()
        elif hasattr(t, 'max'):
            max_val = t.max()
        else:
            max_val = max(t) if len(t) > 0 else 0
            if hasattr(max_val, 'item'):
                max_val = max_val.item()
        if max_val > 36:
            has_large_tokens = True
            break
            
    if has_large_tokens:
        MAX_LEN = 256
        MAX_TOKEN_ID = 311
    else:
        MAX_LEN = 170
        MAX_TOKEN_ID = 36
    
    # 提取 tok_smiles 并动态补齐
    toked_smis = []
    for b in batch:
        t = torch.tensor(b['tok_smiles']) if not isinstance(b['tok_smiles'], torch.Tensor) else b['tok_smiles']
        # 1. 强制截断超长序列，防止位置编码越界
        t = t[:MAX_LEN]
        # 2. 强制限制 token ID 的最大值，防止 Embedding 越界！
        t = torch.clamp(t, min=0, max=MAX_TOKEN_ID)
        toked_smis.append(t)
    toked_smis = pad_sequence(toked_smis, batch_first=True, padding_value=pad_idx)
    
    # 提取 corrupted_toked_smis 并动态补齐
    corrupted_toked_smis = []
    for b in batch:
        t = torch.tensor(b['corrupted_toked_smis']) if not isinstance(b['corrupted_toked_smis'], torch.Tensor) else b['corrupted_toked_smis']
        # 1. 强制截断超长序列，防止位置编码越界
        t = t[:MAX_LEN]
        # 2. 强制限制 token ID 的最大值，防止 Embedding 越界！
        t = torch.clamp(t, min=0, max=MAX_TOKEN_ID)
        corrupted_toked_smis.append(t)
    corrupted_toked_smis = pad_sequence(corrupted_toked_smis, batch_first=True, padding_value=pad_idx)
    
    # 提取 desc_state 并对齐到 MAX_LEN 长度
    desc_states = []
    for b in batch:
        state = b['desc_state']
        if isinstance(state, torch.Tensor):
            state = state.clone().detach()
        else:
            state = torch.tensor(state)
        # 去掉 batch 维度
        if state.dim() == 3:
            state = state.squeeze(0)
        # 裁剪或填充到 MAX_LEN
        L = state.shape[0]
        if L > MAX_LEN:
            state = state[:MAX_LEN]
        elif L < MAX_LEN:
            state = torch.nn.functional.pad(state, (0, 0, 0, MAX_LEN - L))
        desc_states.append(state)
    desc_states = torch.stack(desc_states)
    
    # 提取 desc_mask 并对齐到 MAX_LEN 长度
    desc_mask = []
    for b in batch:
        mask = b['desc_mask']
        if isinstance(mask, torch.Tensor):
            mask = mask.clone().detach()
        else:
            mask = torch.tensor(mask)
        # 去掉 batch 维度
        if mask.dim() == 2:
            mask = mask.squeeze(0)
        elif mask.dim() == 3:
            mask = mask.squeeze(0).squeeze(0)
        # 裁剪或填充到 MAX_LEN
        L = mask.shape[0]
        if L > MAX_LEN:
            mask = mask[:MAX_LEN]
        elif L < MAX_LEN:
            mask = torch.nn.functional.pad(mask, (0, MAX_LEN - L))
        desc_mask.append(mask)
    desc_mask = torch.stack(desc_mask)
    
    return toked_smis, desc_states, desc_mask, corrupted_toked_smis

def get_dataloader(dataset, batchsize, rank, world_size):
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    loader = DataLoader(dataset, batch_size=batchsize, shuffle=False,
                        collate_fn=custom_collate_fn, sampler=sampler,
                        num_workers=4, pin_memory=True)
    def batches():
        epoch = 0
        while True:
            sampler.set_epoch(epoch)
            yield from loader
            epoch += 1
    return batches()
