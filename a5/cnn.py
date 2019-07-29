#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1i
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(torch.nn.Module):
    def __init__(self, m_word=21, k=5, e_char=50, e_word=256):
        """ Initialize the highway model.
        @param k (int): kernel size
        @param e_char, m_word, e_word (int): number of convolution input and output units
        """
        super(CNN, self).__init__()
        self.conv = nn.Conv1d(e_char, e_word, kernel_size=k)
        self.maxpooling = nn.MaxPool1d(m_word - k + 1)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        """ Run the model forward.
        @param x (Tensor): input tensor of tokens (batch_size, e_char * m_word)
        @return cnn_out (Tensor): tensor of cnn (output after applying the layers of the network) (batch_size, e_word)
        """
        x_conv = self.conv(x)
        return self.maxpooling(F.relu(x_conv)).squeeze(2)
        
### END YOUR CODE

