# Licensed base: Diffusion-LM 759889d58ef38e2eed41a8c34db8032e072826f4 (Apache-2.0).
# OpenAI-derived portions retain their upstream MIT notices; see ../../licenses/THIRD_PARTY_NOTICES.md.
# AP-MolDiff adapters: newly expressed from the frozen interface/behavior contract.
# Wsy-authorized local contributions remain subject to their MIT authorization.
# No assertion of full historical-model or paper reproduction is made.
import torch as th
import numpy as np

def compute_logp(args, model, x, input_ids):
    word_emb = model.weight
    sigma = 0.1
    if args.model_arch == '1d-unet':
        x = x.permute(0, 2, 1)

    bsz, seqlen, dim = x.shape

    x_flat = x.reshape(-1, x.size(-1)).unsqueeze(0)  # 1, bsz*sample*seqlen, dim
    word_emb_flat = word_emb.unsqueeze(1)  # vocab, 1,  dim
    diff = (x_flat - word_emb_flat) ** 2  # vocab, seqlen, dim

    logp_expanded = -diff.sum(dim=-1) / (2 * sigma ** 2)  # vocab, seqlen
    logp_expanded = logp_expanded.permute((1, 0))
    # print(th.topk(logp_expanded.view(bsz, seqlen, -1), k=5, dim=-1)[0])
    # print(input_ids[0])
    ce = th.nn.CrossEntropyLoss(reduction='none')
    loss = ce(logp_expanded, input_ids.view(-1)).view(bsz, seqlen)
    # print(loss[0])

    # print(loss.shape)
    return loss

def get_weights(model, args):
    if hasattr(model, 'transformer'):
        input_embs = model.transformer.wte  # input_embs
        down_proj = model.down_proj
        down_proj_emb = down_proj(input_embs.weight)
        print(down_proj_emb.shape)
        # model = th.nn.Embedding(down_proj_emb.shape[1], down_proj_emb.shape[0])
        model = th.nn.Embedding(down_proj_emb.size(0), down_proj_emb.size(1))
        print(args.emb_scale_factor)
        model.weight.data = down_proj_emb * args.emb_scale_factor

    elif hasattr(model, 'weight'):
        pass
    else:
        assert NotImplementedError
        
    model.weight.requires_grad = False
    return model

def load_results(json_path, load_dict):
    import json
    with open(json_path, 'w') as f:
        json.dump(load_dict, f, indent=2)

def denoised_fn_round(args, model, text_emb, t):
    threshold = getattr(args, 'clamp_thresh', 350)
    if threshold is not None and t[0] > threshold:
        return text_emb
    count = getattr(args, 'clamp_top_k', 5)
    temperature = getattr(args, 'clamp_temp', 0.1)
    soft = getattr(args, 'clamp_soft', True)
    points = text_emb.reshape(-1, text_emb.shape[-1]).to(model.device)
    # Squared Euclidean distances, using the historical matmul arithmetic order.
    distances = model.square().sum(-1, keepdim=True) + points.square().sum(-1)[None, :]
    distances = (distances - 2.0 * model.mm(points.T)).clamp_min(0)
    values, ids = (-distances).topk(count, dim=0)
    selected = model[ids]
    if count == 1:
        replacement = selected[0]
    else:
        replacement = (selected * (values / temperature).softmax(0)[..., None]).sum(0)
    if soft:
        # Deliberately do not silently fix the legacy threshold=None/0 failure cases.
        fraction = ((threshold - t[0].float()) / threshold).clamp(0, 1)
        replacement = fraction * replacement + (1 - fraction) * points
    return replacement.reshape(text_emb.shape).to(text_emb.device)
