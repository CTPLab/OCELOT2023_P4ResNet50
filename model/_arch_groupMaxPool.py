# -*- coding: utf-8 -*-
"""
    __author__ = "Maxime Lafarge, maxime.lafarge[at]usz[dot]ch"
    __creation__ = "2023"
    __copyright__ = "Maxime Lafarge - All Rights Reserved"
"""
import torch

class GroupMaxPool(torch.nn.Module):
    def __init__(self, down_factor=2):
        super(GroupMaxPool, self).__init__()
        self.pool = torch.nn.MaxPool2d(down_factor, down_factor)
        
    def forward(self, x, *args, **kwargs):
        x = self.pool(x)
        return x