import math
import torch
import torch.nn as nn
from torch.nn import functional as F

class MHAAttention(nn.Module):
    """Multi-Head Attention: num_q_heads = num_kv_heads"""
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        
        self.n_embd = config.n_embd
        self.n_head = config.n_head
        self.d_head = config.n_embd // config.n_head
        self.dropout = config.dropout
        self.block_size = config.block_size
        
        # Q, K, V projections for all heads
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
        # Flash attention support
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))
    
    def forward(self, x):
        B, T, C = x.size()
        
        # Calculate Q, K, V for all heads
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, self.d_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, self.d_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, self.d_head).transpose(1, 2)  # (B, nh, T, hs)
        
        # Attention
        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True
            )
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.d_head))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v
        
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class GQAAttention(nn.Module):
    """Grouped-Query Attention: num_q_heads > num_kv_heads"""
    def __init__(self, config, num_kv_heads=2):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        assert config.n_head % num_kv_heads == 0, "n_head must be divisible by num_kv_heads"
        
        self.n_embd = config.n_embd
        self.n_head = config.n_head
        self.num_kv_heads = num_kv_heads
        self.d_head = config.n_embd // config.n_head
        self.dropout = config.dropout
        self.block_size = config.block_size
        
        # Q projection for all heads
        self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # K, V projections for fewer heads
        self.k_proj = nn.Linear(config.n_embd, num_kv_heads * self.d_head, bias=config.bias)
        self.v_proj = nn.Linear(config.n_embd, num_kv_heads * self.d_head, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
        # Flash attention support
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))
    
    def forward(self, x):
        B, T, C = x.size()
        
        # Q for all heads
        q = self.q_proj(x).view(B, T, self.n_head, self.d_head).transpose(1, 2)  # (B, nh, T, hs)
        
        # K, V for fewer heads
        k = self.k_proj(x).view(B, T, self.num_kv_heads, self.d_head).transpose(1, 2)  # (B, num_kv_heads, T, hs)
        v = self.v_proj(x).view(B, T, self.num_kv_heads, self.d_head).transpose(1, 2)  # (B, num_kv_heads, T, hs)
        
        # Repeat K, V to match number of Q heads
        # Each KV head is shared across n_head // num_kv_heads query heads
        k = k.repeat_interleave(self.n_head // self.num_kv_heads, dim=1)  # (B, nh, T, hs)
        v = v.repeat_interleave(self.n_head // self.num_kv_heads, dim=1)  # (B, nh, T, hs)
        
        # Attention
        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True
            )
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.d_head))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v
        
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class MQAAttention(nn.Module):
    """Multi-Query Attention: num_kv_heads = 1"""
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        
        self.n_embd = config.n_embd
        self.n_head = config.n_head
        self.d_head = config.n_embd // config.n_head
        self.dropout = config.dropout
        self.block_size = config.block_size
        
        # Q projection for all heads
        self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # K, V projections for single head
        self.k_proj = nn.Linear(config.n_embd, self.d_head, bias=config.bias)
        self.v_proj = nn.Linear(config.n_embd, self.d_head, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
        # Flash attention support
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))
    
    def forward(self, x):
        B, T, C = x.size()
        
        # Q for all heads
        q = self.q_proj(x).view(B, T, self.n_head, self.d_head).transpose(1, 2)  # (B, nh, T, hs)
        
        # K, V for single head
        k = self.k_proj(x).view(B, T, 1, self.d_head).transpose(1, 2)  # (B, 1, T, hs)
        v = self.v_proj(x).view(B, T, 1, self.d_head).transpose(1, 2)  # (B, 1, T, hs)
        
        # Repeat K, V to match number of Q heads
        k = k.repeat(1, self.n_head, 1, 1)  # (B, nh, T, hs)
        v = v.repeat(1, self.n_head, 1, 1)  # (B, nh, T, hs)
        
        # Attention
        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True
            )
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.d_head))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v
        
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class StaticMoASAttention(nn.Module):
    """Static Mixture of Attention Schemes - averages MHA, GQA, MQA outputs"""
    def __init__(self, config):
        super().__init__()
        self.mha = MHAAttention(config)
        self.gqa = GQAAttention(config, num_kv_heads=2)
        self.mqa = MQAAttention(config)
        
    def forward(self, x):
        o_mha = self.mha(x)
        o_gqa = self.gqa(x)
        o_mqa = self.mqa(x)
        return (o_mha + o_gqa + o_mqa) / 3.0


class MoASAttention(nn.Module):
    """Mixture of Attention Schemes with learned per-token routing"""
    def __init__(self, config):
        super().__init__()
        self.n_embd = config.n_embd
        
        # Three attention branches
        self.mha = MHAAttention(config)
        self.gqa = GQAAttention(config, num_kv_heads=2)
        self.mqa = MQAAttention(config)
        
        # Router: 2-layer MLP
        router_hidden = config.n_embd // 4
        self.router = nn.Sequential(
            nn.Linear(config.n_embd, router_hidden, bias=config.bias),
            nn.GELU(),
            nn.Linear(router_hidden, 3, bias=config.bias)  # 3 attention types
        )
        
        self.gate_dropout = nn.Dropout(config.dropout)
        
    def forward(self, x, return_gate_stats=False):
        B, T, C = x.size()
        
        # Compute all attention outputs
        o_mha = self.mha(x)  # (B, T, C)
        o_gqa = self.gqa(x)  # (B, T, C)
        o_mqa = self.mqa(x)  # (B, T, C)
        
        # Stack outputs: (B, T, 3, C)
        outputs = torch.stack([o_mha, o_gqa, o_mqa], dim=2)
        
        # Compute routing logits for each token
        router_logits = self.router(x)  # (B, T, 3)
        gates = F.softmax(router_logits, dim=-1)  # (B, T, 3)
        gates = self.gate_dropout(gates)
        
        # Mix outputs per token: (B, T, 3, 1) * (B, T, 3, C) -> (B, T, 3, C) -> (B, T, C)
        y = (gates.unsqueeze(-1) * outputs).sum(dim=2)
        
        if return_gate_stats:
            # Return average gate values for logging
            avg_gates = gates.mean(dim=(0, 1))  # (3,)
            return y, avg_gates
        
        return y
    
    def get_load_balancing_loss(self, x):
        """Compute load balancing loss to encourage using all attention types"""
        B, T, C = x.size()
        
        # Compute gates
        router_logits = self.router(x)  # (B, T, 3)
        gates = F.softmax(router_logits, dim=-1)  # (B, T, 3)
        
        # Average gate per type across all tokens
        avg_gates = gates.mean(dim=(0, 1))  # (3,)
        
        # Target: uniform distribution (1/3 for each type)
        target = torch.ones_like(avg_gates) / 3.0
        
        # MSE loss
        loss = F.mse_loss(avg_gates, target)
        
        return loss
