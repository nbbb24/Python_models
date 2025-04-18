import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import math
import copy

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        #check if d_model is divisible by num_heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads

        #calculate the dimension of each head
        self.head_dim = d_model // num_heads

        #setup linear d_model x d_model matrices for the queries, keys, and values
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)

        #setup linear d_model x d_model matrix for the output
        self.out = nn.Linear(d_model, d_model)

    def dot_product_attention(self, Q, K, V, mask=None):
        product_result=Q@K.transpose(-2,-1)/math.sqrt(self.head_dim)

        #set the masked positions to -infinity
        #so that the softmax of these positions will be 0
        if mask is not None:
            attention_score=attention_score.masked_fill(mask==0,-1e9)
        attention_score=nn.softmax(attention_score, dim=-1)@V
        return  attention_score


    def forward(self, query, key, value, mask=None):
        
