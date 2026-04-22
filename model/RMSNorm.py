import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self,dim:int,eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim))  # nn.Parameter可训练参数
    
    def _norm(self,x:torch.Tensor,):
        return x * torch.rsqrt(x.pow(2).mean(-1,keepdim=True) + self.eps)

    def forward(self,x:torch.Tensor):   
        # x.float() 将输入转换32位浮点数防止溢出或精度问题 type_as(x)将结果转换回输入的类型
        return self.gamma * self._norm(x.float()).type_as(x)
    
    
