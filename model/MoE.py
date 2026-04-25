import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

from .FFN import FeedForward


class MoEFeedForward(nn.Module):
    def __init__(
            self, 
            d_model,
            intermediate_size,
            num_experts
        ):
        super().__init__()

        self.d_model = d_model
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts

        self.gate = nn.Linear(self.d_model,self.num_experts, bias=False)
        self.experts = nn.ModuleList(
            [FeedForward(self.d_model, self.intermediate_size) for _ in range(self.num_experts)]
        )


        self.act_fn = F.silu  
    
    def forward(self,x:Tensor):
        # x: [batch_size, sequence_len, d_model] 
        # self.gate(x): [batch_size, sequence_len, num_experts] 
        torch.softmax(self.gate(x),dim=-1)
        
        return 
        

