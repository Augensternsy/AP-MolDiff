# AP-MolDiff network adapter, independently expressed from the frozen module/forward contract.
# Architecture foundations: Diffusion-LM 759889d... Apache-2.0 and HF BertEncoder Apache-2.0.
# Actual BERT implementation is imported from the external Transformers dependency.
# Original module names, order and shared embedding registration are compatibility requirements.
import os
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoConfig
from transformers.models.bert.modeling_bert import BertEncoder
from .nn import SiLU, timestep_embedding

class TransformerNetModel2(nn.Module):
    def __init__(self, in_channels, model_channels, dropout=0.1, num_classes=None,
                 use_checkpoint=False, config=None, config_name='bert-base-uncased',
                 training_mode='emb', vocab_size=None, experiment_mode='lm',
                 init_pretrained=False, logits_mode=1, hidden_size=768,
                 num_attention_heads=12, num_hidden_layers=12, mask=False):
        super().__init__()
        directory = Path(os.environ.get('AP_MOLDIFF_BERT_CONFIG_DIR',
                         str(Path(__file__).resolve().parents[2] / 'configs' / 'bert-base-uncased'))).expanduser().resolve()
        if not (directory / 'config.json').is_file():
            raise FileNotFoundError(f'Local Bert config missing: {directory / "config.json"}')
        # Historical implementation also loaded the local config even when config was passed.
        settings = AutoConfig.from_pretrained(str(directory), local_files_only=True)
        for key, value in dict(_attn_implementation='eager', is_decoder=True,
                               add_cross_attention=True, hidden_dropout_prob=.1,
                               hidden_size=hidden_size, num_attention_heads=num_attention_heads,
                               num_hidden_layers=num_hidden_layers).items():
            setattr(settings, key, value)
        self.mask = mask
        self.in_channels, self.model_channels = in_channels, model_channels
        self.dropout = dropout
        self.num_classes, self.use_checkpoint = None, False
        self.num_heads_upsample, self.logits_mode = 4, 1
        # Non-SC model has no self_cond flag.
        # Keep registration/initialization order, including unused desc_down_proj.
        self.word_embedding = nn.Embedding(312, in_channels)
        self.lm_head = nn.Linear(in_channels, 312)
        self.lm_head.weight = self.word_embedding.weight
        self.conditional_gen = False
        self.desc_dim_adapter = nn.Linear(1024, hidden_size)
        self.desc_down_proj = nn.Sequential(nn.Linear(hidden_size, hidden_size), SiLU(), nn.Linear(hidden_size, hidden_size))
        self.time_embed = nn.Sequential(nn.Linear(model_channels, 4 * model_channels), SiLU(), nn.Linear(4 * model_channels, hidden_size))
        width = in_channels
        self.input_up_proj = nn.Sequential(nn.Linear(width, hidden_size), nn.Tanh(), nn.Linear(hidden_size, hidden_size))
        self.input_transformers = BertEncoder(settings)
        self.register_buffer('position_ids', torch.arange(settings.max_position_embeddings).unsqueeze(0))
        self.position_embeddings = nn.Embedding(settings.max_position_embeddings, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=settings.layer_norm_eps)
        self.dropout = nn.Dropout(settings.hidden_dropout_prob)
        self.output_down_proj = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.Tanh(), nn.Linear(hidden_size, in_channels))

    def get_embeds(self, input_ids):
        return self.word_embedding(input_ids)

    def get_logits_deep(self, hidden_repr):
        value = hidden_repr.float()
        if not hasattr(self, 'deep_head'):
            return value
        layer = self.deep_head
        return F.linear(value, layer.weight.float(), None if layer.bias is None else layer.bias.float())

    def get_logits(self, hidden_repr):
        value, weights = hidden_repr.float(), self.lm_head.weight.float()
        bias = None if self.lm_head.bias is None else self.lm_head.bias.float()
        if self.logits_mode == 1:
            return F.linear(value, weights, bias)
        if self.logits_mode != 2:
            raise NotImplementedError
        rows = value.reshape(-1, value.shape[-1])
        distance = weights.square().sum(-1, keepdim=True) + rows.square().sum(-1)[None, :]
        distance = distance - 2.0 * weights.mm(rows.T)
        return -distance.clamp_min(0).sqrt().reshape(weights.shape[0], value.shape[0], value.shape[1]).permute(1, 2, 0).contiguous()

    @property
    def inner_dtype(self):
        return next(self.input_transformers.parameters()).dtype

    def convert_to_fp16(self):
        self.to(torch.float16)

    def convert_to_fp32(self):
        self.to(torch.float32)

    def forward(self, x, timesteps, desc_state, desc_mask, y=None, src_ids=None, src_mask=None):
        assert (y is not None) == (self.num_classes is not None), 'must specify y if and only if the model is class-conditional'
        x = x.to(dtype=self.inner_dtype)
        condition = desc_state.to(dtype=self.inner_dtype)
        if condition.ndim == 4:
            condition = condition.squeeze(1)
        condition = self.desc_dim_adapter(condition)
        additive_mask = (1.0 - desc_mask.to(condition.dtype)) * -10000.0
        for _ in range(max(0, 4 - additive_mask.ndim)):
            additive_mask = additive_mask.unsqueeze(1)
        time = self.time_embed(timestep_embedding(timesteps, self.model_channels).to(self.inner_dtype))
        positions = self.position_embeddings(self.position_ids[:, :x.shape[-2]])
        projected = self.input_up_proj(x)
        if x.ndim == 4:
            batch, copies, length, hidden = projected.shape
            projected = projected.reshape(batch * copies, length, hidden)
            positions = positions.expand(batch * copies, -1, -1)
            time = time[:, None, :].expand(-1, copies, -1).reshape(batch * copies, -1)[:, None, :].expand(-1, length, -1)
            # Preserve the historical repeat ordering, even for batch > 1.
            condition = condition.repeat(copies, 1, 1)
            additive_mask = additive_mask.repeat(copies, 1, 1, 1)
        else:
            time = time[:, None, :].expand(-1, x.shape[-2], -1)
        inputs = self.dropout(self.LayerNorm(positions + projected + time))
        encoded = self.input_transformers(inputs, encoder_hidden_states=condition,
                                          encoder_attention_mask=additive_mask).last_hidden_state
        output = self.output_down_proj(encoded)
        if x.ndim == 4:
            output = output.reshape(x.shape[0], x.shape[1], x.shape[2], output.shape[-1])
        return output.to(x.dtype)

    # Exact licensed legacy auxiliary API; its old block-based assumptions remain.
    def get_feature_vectors(self, x, timesteps, y=None):
        """
        Apply the model and return all of the intermediate tensors.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: a dict with the following keys:
                 - 'down': a list of hidden state tensors from downsampling.
                 - 'middle': the tensor of the output of the lowest-resolution
                             block in the model.
                 - 'up': a list of hidden state tensors from upsampling.
        """
        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)
        result = dict(down=[], up=[])
        h = x.type(self.inner_dtype)
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
            result["down"].append(h.type(x.dtype))
        h = self.middle_block(h, emb)
        result["middle"] = h.type(x.dtype)
        for module in self.output_blocks:
            cat_in = th.cat([h, hs.pop()], dim=-1)
            h = module(cat_in, emb)
            result["up"].append(h.type(x.dtype))
        return result