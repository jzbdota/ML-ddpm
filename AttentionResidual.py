"""
This module defined the attention with residual layer in pytorch.
See Umich EECS course slide 13-83 for details
"""

from torch import nn
import torch.nn.functional as F

class AttentionResidual(nn.Module):
    """
    Output has the same shape of input.
    """
    def __init__(self,in_channel, num_heads=4, head_channel=None) -> None:
        super().__init__()
        if not head_channel:
            head_channel = in_channel // num_heads
        self.head_channel = head_channel
        self.num_heads = num_heads
        hidden_channel = num_heads*head_channel
        self.wqkv = nn.Conv2d(in_channel,hidden_channel*3,1)

        self.softmax = nn.Softmax(dim=-1)

        self.final_conv = nn.Conv2d(hidden_channel,in_channel,1)

####################################################################################
# x: [N,C,H,W]
    def forward(self,x):
        if x.dim == 3:
            x = x.unsqueeze(0)
        H,W = x.shape[2:]
        # [N,3*num_heads*head_channel,H,W]
        # [N,num_heads,3*head_channel,H,W]
        # [N*num_heads,3*head_channel,H,W]
        # [N*num_heads,3*head_channel,H*W]
        # 3*[N*num_heads,head_channel,H*W]
        Q,K,V = self.wqkv(x) \
                .unflatten(1,(self.num_heads,3*self.head_channel)) \
                .flatten(0,1) \
                .flatten(2) \
                .chunk(3,dim=1)
        # [N*num_heads,H*W,H*W]
        attention_weights = self.softmax(Q.permute(0,2,1).bmm(K).div(self.head_channel**.5))
        # [N*num_heads,head_channel,H*W]
        # [N*num_heads,head_channel,H,W]
        # [N,num_heads,head_channel,H,W]
        # [N,num_heads*head_channel,H,W]
        output = V.bmm(attention_weights) \
                .unflatten(-1,(H,W)) \
                .unflatten(0,(-1,self.num_heads)) \
                .flatten(1,2)
        output = self.final_conv(output)
        return F.relu(output+x).squeeze()
    
if __name__=="__main__":
    print('Attention and Residual')