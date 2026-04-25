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
    

if __name__ == "__main__":
    # x = torch.randn(2,3,4)
    x = torch.Tensor(
        [[[10000,  30, -0.6288,  1.2145],
         [ 0.1785,  1.4688, 872,  0.8699],
         [ 2.0121, -0.1344,  0.1555, -1.7043]],

        [[ 0.0519, 343, -0.5836,  2.5901],
         [ 1.2780,  423, -0.3882, -1.1333],
         [ 0.3795,  1.5145, 1119,  4026]]]
    ) 
    norm = RMSNorm(4)
    print(x)
    out = norm(x)
    
    a,b,c = out.shape
    print(f"out shape: {a} {b} {c}")
    print(out.shape)
    print(out)
