import torch
import torch.nn as nn
from torch import Tensor

import transformers
from transformers import PretrainedConfig


class MHAttention(nn.Module):
    def __init__(self, d_model:int, num_heads:int):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

    def forward(self, x:Tensor):
        # x: [batch_size, sequence_len, d_model] 
        batch_size, sequence_len, d_model = x.shape

        # q,k,v: [batch_size, num_heads, sequence_len, head_dim]
        q = self.w_q(x).view(batch_size, sequence_len, self.num_heads, self.head_dim).transpose(1,2)
        k = self.w_k(x).view(batch_size, sequence_len, self.num_heads, self.head_dim).transpose(1,2)
        v = self.w_v(x).view(batch_size, sequence_len, self.num_heads, self.head_dim).transpose(1,2)

        # attention_scores: [batch_size, num_heads, sequence_len, sequence_len]
        attention_scores = torch.matmul(q,k.transpose(-2,-1)) / (self.head_dim ** 0.5)
        attention_weight = torch.softmax(attention_scores,dim=-1)
        # attention_output: [batch_size, num_heads, sequence_len, head_dim]->[batch_size, sequence_len, num_heads, head_dim]->[batch_size, sequence_len, d_model]
        # transpose() 会让张量在内存中变成不连续的，而 view() 只能操作连续的张量，
        # 所以必须用 .contiguous() 把内存重新整理成连续的，否则直接 view() 会报错
        attention_output = torch.matmul(attention_weight,v).transpose(1,2).contiguous().view(batch_size, sequence_len, d_model)

        # output: [batch_size, sequence_len, d_model] = x.shape
        output = self.w_o(attention_output)
        return output


def repeat_kv(x:Tensor, n_rep: int) -> torch.Tensor:
    """
    沿 num_kv_heads 维度重复 K/V 头，使其与 Q 头数匹配
    Args:
        x: [batch_size, sequence_len, num_kv_heads, head_dim]
        n_rep: int: 重复次数 num_q_heads // num_kv_heads
    Returns:
        [batch_size, sequence_len, num_kv_heads * n_rep, head_dim]
    """
    batch_size, sequence_len, num_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return x[:, :, :, None, :].expand(batch_size, sequence_len, num_kv_heads, n_rep, head_dim).reshape(batch_size, sequence_len, num_kv_heads * n_rep, head_dim)
   

class GQAttention(nn.Module):
    def __init__(
            self, 
            d_model:int, 
            num_q_heads:int,
            num_kv_heads:int,
            dropout: float = 0.1
        ):
        super().__init__()
        self.d_model = d_model
        self.num_q_heads = num_q_heads
        self.head_dim = d_model // num_q_heads
        self.num_kv_heads = num_kv_heads
        self.n_rep = num_q_heads // num_kv_heads
        assert num_q_heads % num_kv_heads == 0, "num_q_heads 必须是 num_kv_heads 的整数倍"

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, num_kv_heads * self.head_dim)
        self.w_v = nn.Linear(d_model, num_kv_heads * self.head_dim)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.dropout = dropout

    def forward(self, x:Tensor):
        # x: [batch_size, sequence_len, d_model] 
        batch_size, sequence_len, d_model = x.shape

        # q: [batch_size, sequence_len, d_model]->[batch_size, sequence_len, num_q_heads, head_dim]->[batch_size, num_q_heads, sequence_len, head_dim]
        q = self.w_q(x).view(batch_size, sequence_len, self.num_q_heads, self.head_dim).transpose(1,2)
        # k,v: [batch_size, sequence_len, num_kv_heads * head_dim]->[batch_size, sequence_len, num_kv_heads, head_dim]->[batch_size, sequence_len, num_kv_heads * n_rep, head_dim]->[batch_size, num_q_heads, sequence_len, head_dim]
        k = repeat_kv(self.w_k(x).view(batch_size, sequence_len, self.num_kv_heads, self.head_dim),n_rep=self.n_rep).transpose(1,2)
        v = repeat_kv(self.w_v(x).view(batch_size, sequence_len, self.num_kv_heads, self.head_dim),n_rep=self.n_rep).transpose(1,2)

        # attention_scores: [batch_size, num_heads, sequence_len, sequence_len]
        attention_scores = torch.matmul(q,k.transpose(-2,-1)) / (torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32)))
        attention_weight = self.attn_dropout(torch.softmax(attention_scores,dim=-1))

        # attention_output: [batch_size, num_heads, sequence_len, head_dim]->[batch_size, sequence_len, num_heads, head_dim]->[batch_size, sequence_len, d_model]
        # transpose() 会让张量在内存中变成不连续的，而 view() 只能操作连续的张量，
        # 所以必须用 .contiguous() 把内存重新整理成连续的，否则直接 view() 会报错
        attention_output = torch.matmul(attention_weight,v).transpose(1,2).contiguous().view(batch_size, sequence_len, d_model)

        # output: [batch_size, sequence_len, d_model] = x.shape
        output = self.resid_dropout(self.w_o(attention_output))
        return output

