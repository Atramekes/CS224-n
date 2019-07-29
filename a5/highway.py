#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1h
import torch
import torch.nn as nn
import torch.nn.functional as F

class Highway(torch.nn.Module):
    def __init__(self, e_word=256):
        """ Initialize the highway model.
        @param e_word (int): number of convolution output units and hidden units
        """
        super(Highway, self).__init__()
        self.conv_size = e_word
        self.hidden_size = e_word
        self.conv_out_to_proj_hidden = nn.Linear(self.conv_size, self.hidden_size)
        self.conv_out_to_gate_hidden = nn.Linear(self.conv_size, self.hidden_size)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        """ Run the model forward.
        @param x (Tensor): input tensor of tokens (batch_size, e_word)
        @return highway_out (Tensor): tensor of highway (output after applying the layers of the network)
                                 without applying dropout (batch_size, e_word)
        """
        proj_hidden = self.conv_out_to_proj_hidden(x)
        gate_hidden = self.conv_out_to_gate_hidden(x)
        x_proj = F.relu(proj_hidden)
        x_gate = torch.sigmoid(gate_hidden)
        return torch.mul(x_gate, x_proj) + torch.mul((1 - x_gate), x)
### END YOUR CODE 