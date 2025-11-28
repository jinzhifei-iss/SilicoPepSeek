
import math
import numpy as np
import openpyxl.comments.shape_writer
import torch
import torch.nn as nn
import random
import einops


from torch.nn import functional as F
from torch.utils.data import dataset
from typing import Tuple, Optional, Literal
from dataclasses import dataclass
from rotary_embedding_torch import RotaryEmbedding
from st_moe_pytorch import MoE, SparseMoEBlock


n_embd = 64
max_len = 2600
n_layer = 4
dropout = 0.1#0.5
n_head = 2

world_size = 1
# block_size = 128
gemm_impl: Literal["bf16", "fp8"] = "bf16"
attn_impl: Literal["naive", "absorb"] = "absorb"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch.manual_seed(42)
random.seed(42)


class Head(nn.Module):

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias= False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(max_len, max_len)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2,-1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] ==0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out


class MultiHeadAttention(nn.Module):

    def __init__(self, num_head, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_head)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedFoward(nn.Module):

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4* n_embd),
            nn.LeakyReLU(),
            nn.Linear(4* n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)



class MLA(nn.Module):

    def __init__(self, d_model, attention_head_num):
        super().__init__()
        self.attention_head_num = attention_head_num
        self.d_model = d_model
        assert d_model % attention_head_num == 0
        self.scale = d_model ** -0.5
        # self.softcap_value = 50.
        self.per_head_dmodel = d_model // attention_head_num # each attention head dim

        # Q linear layer and normalization layer
        self.q_rope_dense = torch.nn.Linear(self.per_head_dmodel, self.per_head_dmodel * 2)
        self.q_norm = torch.nn.RMSNorm(self.per_head_dmodel * 2)
        # QK dim in low-rank latent space
        self.qk_nope_dim = self.per_head_dmodel
        self.qk_rope_dim = self.per_head_dmodel
        # KV projection dim and relevant layer
        self.kv_proj_dim = self.d_model
        self.kv_proj_dim_VS_qk_rope_dim = (self.kv_proj_dim + self.qk_rope_dim)
        self.kv_layernorm = torch.nn.RMSNorm(self.kv_proj_dim)
        self.kv_dense = torch.nn.Linear(self.kv_proj_dim, (self.d_model + self.attention_head_num * self.qk_nope_dim))

        # linear layer for QKV initial representation
        self.qkv_layer = torch.nn.Linear(d_model, (d_model + self.kv_proj_dim_VS_qk_rope_dim))
        self.rotary_embedding = RotaryEmbedding(self.per_head_dmodel // 2)
        self.out_layer = torch.nn.Linear(d_model, d_model)

    def forward(self, embedding, past_length=0):
        B, S, D = embedding.shape
        # get initial representation for QKV by linear layer
        qky_x = self.qkv_layer(embedding)
        # split q and compressed kv
        q, compressed_kv = torch.split(qky_x, split_size_or_sections=[self.d_model, self.kv_proj_dim_VS_qk_rope_dim], dim=-1)
        # rearrange Q and linear transformation, normalization
        q = einops.rearrange(q, "b s (h d) -> b h s d", h=self.attention_head_num)
        q = self.q_norm(self.q_rope_dense(q))

        # separate Q to 2 parts, apply rotary embedding for one part
        q, q_for_rope = torch.split(q, [self.qk_nope_dim, self.qk_rope_dim], dim=-1)
        q_for_rope = self.rotary_embedding.rotate_queries_or_keys(q_for_rope)

        # split compressed KV, normalization and linear transformation
        KV_for_lora, K_for_rope = torch.split(compressed_kv, [self.kv_proj_dim, self.qk_rope_dim], dim=-1)
        KV_for_lora = self.kv_layernorm(KV_for_lora)
        KV = self.kv_dense(KV_for_lora)
        KV = einops.rearrange(KV, "b s (h d) -> b h s d", h=self.attention_head_num)
        K, V = torch.split(KV, [self.qk_nope_dim, self.per_head_dmodel], dim=-1)

        # expand K_for_rope to match attention head size
        K_for_rope = einops.repeat(K_for_rope, "b s d -> b h s d", h=self.attention_head_num)
        # combine Q, K heads for attention score calculation
        q_heads = torch.cat([q, q_for_rope], dim=-1)
        k_heads = torch.cat([K, K_for_rope], dim=-1)
        v_heads = V # has been rearranged previously

        # scale Q for attention score calculation
        q_heads = q_heads * self.scale
        sim = einops.einsum(q_heads, k_heads, 'b h i d, b h j d -> b h i j')
        # causal mask, calculate softmax attention weight
        mask_value = -torch.finfo(sim.dtype).max
        i, j = sim.shape[-2:]
        causal_mask = torch.ones((i, j), dtype=torch.bool).triu(past_length).to(embedding.device)
        sim = sim.masked_fill(causal_mask, mask_value)
        attn = sim.softmax(dim=-1)

        # attention weight for V, get embedding
        out = einops.einsum(attn, v_heads, 'b h i j, b h j d -> b h i d')
        embedding = einops.rearrange(out, "b h s d -> b s (h d)")
        embedding = self.out_layer(embedding)
        return embedding




def swiglu(x):
    x = torch.chunk(x, 2, dim=-1)
    return nn.functional.silu(x[0]) * x[1]


class Swiglu(nn.Module):
    def __init__(self, hidden_size = n_embd, add_bias_linear=False):
        super().__init__()
        self.add_bias = add_bias_linear
        self.hidden_size = hidden_size
        self.dense_h_to_4h = nn.Linear(
            hidden_size,
            hidden_size * 4,
            bias = self.add_bias
        )

        self.activation_func = swiglu

        self.dense_4h_to_h = nn.Linear(
            hidden_size * 2,
            hidden_size,
            bias = self.add_bias
        )

    def forward(self, hidden_states):
        intermediate_parallel = self.dense_h_to_4h(hidden_states)
        intermediate_parallel = self.activation_func(intermediate_parallel)
        out = self.dense_4h_to_h(intermediate_parallel)

        return out


class SwiGLU(nn.Module):
    def __init__(self, hidden_size = n_embd):
        super().__init__()
        self.WG = nn.Linear(hidden_size, hidden_size * 2)
        self.W1 = nn.Linear(hidden_size, hidden_size * 2)
        self.W2 = nn.Linear(hidden_size * 2, hidden_size)


    def forward(self, hidden_states):
        g = F.silu(self.WG(hidden_states))
        z = self.W1(hidden_states)
        out = self.W2(g * z)

        return out


class MOE(nn.Module):
    def __init__(self, n_embd=n_embd):
        super().__init__()

        self.moe = MoE(
            dim = n_embd,
            num_experts=4,
            gating_top_n=2,
            threshold_train=0.1,
            threshold_eval=0.1,
            capacity_factor_train=1.0,
            capacity_factor_eval=1.5,
            balance_loss_coef=5e-2,
            router_z_loss_coef=1e-4
        )

        self.moe_block = SparseMoEBlock(
            self.moe,
            add_ff_before=True,
            add_ff_after=True
        )

        self.norm = nn.RMSNorm(n_embd)
        self.moe_linear = nn.Linear(n_embd, n_embd, bias=False)
        self.activity_layer = Swiglu(hidden_size = n_embd)

    def forward(self, x):
        x = self.norm(x)
        out = self.moe(x)[0]
        out = self.activity_layer(out)
        out = self.moe_linear(out)

        return out



class Block(nn.Module):

    def __init__(self, d_model = 64, attention_head_num = 2):
        super().__init__()
        # head_size = n_embd // n_head
        # self.sa = MultiHeadAttention(n_head, head_size)
        self.sa = MLA(d_model, attention_head_num)
        # self.ffwd = FeedFoward(d_model)
        # self.ffwd = Swiglu(d_model)
        self.ffwd = MOE(d_model)
        self.ln1 = nn.RMSNorm(d_model)
        self.ln2 = nn.RMSNorm(d_model)

    def forward(self, x: torch.Tensor, past_length = 0):#, start_pos: int, freqs_cis: torch.Tensor):
        # args = ModelArgs()
        # freqs_cis = precompute_freqs_cis(args).to(device)

        x = x + self.sa(self.ln1(x), past_length)#, start_pos, freqs_cis)
        x = x + self.ffwd((self.ln2(x)))
        return x



class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(max_len, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd)
                                      for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        # self.register_buffer("freqs_cis", precompute_freqs_cis(args), persistent=False)


    def forward(self, idx, targets=None):#, start_pos: int = 0 ):
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx) #(B, T, C)
        # pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb #+ pos_emb

        # freqs_cis = self.freqs_cis[start_pos:start_pos+T]

        x = self.blocks(x)#, start_pos, freqs_cis)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets, ignore_index=0)

        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, eos_token=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -max_len:]
            logits, loss = self(idx_cond)
            logits = logits[:,-1,:]/temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:,[-1]]] = -float('inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

            # if eos_token is reached, stop generation # new!!!
            if eos_token is not None and idx_next.item() == eos_token:
                break

        return idx

    def generate_ensemble(self, idx, max_new_tokens, temperature=1.0, top_k=None, models=None, eos_token=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -max_len:]
            all_logits = []
            for model in models:
                for param in model.parameters():
                    param.requires_grad = False
                model.eval()
                logits, loss = model(idx_cond)
                logits = logits[:,-1,:] #提取最后一个位置的logits
                all_logits.append(logits)
            avg_logits = torch.mean(torch.stack(all_logits), dim=0)
            avg_logits = avg_logits/temperature

            if top_k is not None:
                v, _ = torch.topk(avg_logits, min(top_k, avg_logits.size(-1)))
                avg_logits[avg_logits < v[:,[-1]]] = -float('inf')
            probs = F.softmax(avg_logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

            # if eos_token is reached, stop generation # new!!!
            if eos_token is not None and idx_next.item() == eos_token:
                break
        return idx
