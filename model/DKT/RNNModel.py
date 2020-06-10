# -*- coding: utf-8 -*-
# @Author: jarvis.zhang
# @Date:   2020-05-10 00:29:34
# @Last Modified by:   jarvis.zhang
# @Last Modified time: 2020-05-10 13:14:50
import torch
import torch.nn as nn


class RNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, device):
        super(RNNModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.output_dim = output_dim
        self.rnn = nn.RNN(input_dim,
                          hidden_dim,
                          layer_dim,
                          batch_first=True,
                          nonlinearity='tanh')
        self.fc = nn.Linear(self.hidden_dim, self.output_dim)
        self.sig = nn.Sigmoid()
        self.device = device

    def forward(self, x):  # shape of input: [batch_size, length, questions * 2]
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim, device=self.device)  # shape: [num_layers * num_directions, batch_size, hidden_size]
        out, hn = self.rnn(x, h0)  # shape of out: [batch_size, length, hidden_size]
        res = self.sig(self.fc(out))  # shape of res: [batch_size, length, question]
        return res
