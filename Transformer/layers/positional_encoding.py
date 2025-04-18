import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F
import math
import copy

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len):
        """
        x: [batch_size, seq_len, d_model]
        
        """
        super(PositionalEncoding,self).__init__()
        pe=torch.zeros(max_seq_len, d_model)

        #create 1D tensor -> 2D, [1,2,3]->[[1],[2],[3]]
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)


        denominator=torch.exp(-math.log(1000)/d_model*torch.arrange(0,d_model,2,dtype=torch.float))
        pe[:,0::2]=torch.sin(position*denominator)
        pe[:,1::2]=torch.cos(position*denominator)

        #2D->3D, [max_seq_len,d_model]->[1,max_seq_len,d_model]
        pe=pe.unsqueeze(0)
        self.register_buffer('pe',pe)

    def forward(self,x):
        x=x+self.pe[:, :x.size(1)]
        return x
        
