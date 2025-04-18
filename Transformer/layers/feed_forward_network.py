import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F
import math
import copy


class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model, d_ff):
        """
        d_ff is the dimension of the feed forward network
        d_model is the dimension of the model
        """

        super(FeedForwardNetwork, self).__init__()
        self.linear1=nn.Linear(d_model, d_ff)
        self.linear2=nn.Linear(d_ff, d_model)
        self.Relu=nn.ReLU()

    def forward(self, x):
        x=self.linear1(x)
        x=self.Relu(x)
        x=self.linear2(x)
        return x
