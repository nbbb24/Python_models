import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F
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
        '''
        Q: [batch_size, num_heads, seq_len, head_dim]
        K: [batch_size, num_heads, seq_len, head_dim]
        V: [batch_size, num_heads, seq_len, head_dim]
        mask: [batch_size, num_heads, seq_len, seq_len]

        output: [batch_size, num_heads, seq_len, head_dim]
        '''

        attention_score=Q@K.transpose(-2,-1)/math.sqrt(self.head_dim)

        #set the masked positions to -infinity
        #so that the softmax of these positions will be 0
        if mask is not None:
            attention_score=attention_score.masked_fill(mask==0,-1e9)

        
        #The softmax of these scores is the attention weights
        #The attention weights are then used to weight the values
        #These scores indicate how relevant each key is to the current query
        #The last dimension represents: How uch attention each query position pays to each key position
        output=F.softmax(attention_score, dim=-1)@V
        return output
    
    def split_heads(self, x):
        '''
        x: [batch_size, seq_len, d_model]
        '''
        batch_size, seq_length, d_model = x.size()
        #split the last dimension into num_heads and head_dim
        x=x.view(batch_size, seq_length, self.num_heads, self.head_dim)

        #   batch_size, num_heads, seq_len, head_dim
        return x.transpose(-2,-3)
    
    def merge_heads(self, x):
        '''
        x: [batch_size, num_heads, seq_len, head_dim]
        output: [batch_size, seq_len, d_model]
        '''
        batch_size, num_heads, seq_length, head_dim = x.size()
        #switch num_heads and seq_length
        #concatenate the heads along num_heads and head_dim
        x=x.transpose(1,2).contiguous().view(batch_size, seq_length, self.d_model)
    
        return x


    def forward(self, query, key, value, mask=None):
        Q=self.Wq(query)
        K=self.Wk(key)
        V=self.Wv(value)

        Q=self.split_heads(Q)
        K=self.split_heads(K)
        V=self.split_heads(V)   

        attention_output_i=self.dot_product_attention(Q,K,V,mask)
        #[batch_size, seq_len, d_model]
        attention_output=self.merge_heads(attention_output_i)
        output=self.out(attention_output)
        return output
        
