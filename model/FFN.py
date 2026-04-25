import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F


class FeedForward(nn.Module):
    def __init__(self, d_model, intermediate_size):
        super().__init__()
        self.d_model = d_model
        self.intermediate_size = intermediate_size

        self.up_proj = nn.Linear(self.d_model,self.intermediate_size, bias=False)
        self.gate_proj = nn.Linear(self.d_model,self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size,self.d_model, bias=False)

        self.act_fn = F.silu  
    
    def forward(self,x:Tensor):
        # x: [batch_size, sequence_len, d_model] 
        return self.down_proj(self.up_proj(x) * self.act_fn(self.gate_proj(x)))
        

