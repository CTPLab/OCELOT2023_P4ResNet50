# -*- coding: utf-8 -*-
"""
    __author__ = "Maxime Lafarge, maxime.lafarge[at]usz[dot]ch"
    __creation__ = "2023"
    __copyright__ = "Maxime Lafarge - All Rights Reserved"
"""

import torch
import numpy as np

from ._arch_p4_BLOCK import p4_BLOCK


""" DEFINE A CONVOLUTIONAL BLOCK
"""
class p4_RES_BLOCK(torch.nn.Module):
    def __init__(self,
            channels_in    = 1,
            channels_inter = 1,
            channels_out   = 1,
            kernel_size    = 3):
        super(p4_RES_BLOCK, self).__init__()
        
        #-- Store kernel size
        self.kernel_size = kernel_size
        
        #-- Calculate input/output channel ratio
        self.channel_ratio = channels_out // channels_in
        
        #-- Operations in Residual Block
        kwargs = {
            "channels_in":  channels_in,
            "channels_out": channels_inter,
            
            "kernel_size":  1,
            "relu":    True,
            "pre_act": True,
            
            "bn":      True,
            "bias":    True
        }
        self.module_A = p4_BLOCK(**kwargs)
        
        kwargs = {
            "channels_in":  channels_inter,
            "channels_out": channels_out,
            
            "kernel_size":  kernel_size,
            "relu":    True,
            "pre_act": True,
            
            "bn":      False,
            "bias":    True
        }
        self.module_B = p4_BLOCK(**kwargs)

        kwargs = {
            "channels_in":  channels_out,
            "channels_out": channels_out,
            
            "kernel_size":  1,
            "relu":    True,
            "pre_act": True,
            
            "bn":      False,
            "bias":    False
        }
        self.module_C = p4_BLOCK(**kwargs)

    def forward(self, x, *args, **kwargs):
        x_res = x #-- Copy input
        
        conv_margin = (self.kernel_size - 1) // 2 #-- Assuming a odd-sized kernel_size
        if conv_margin > 0:
            crop_slice = slice(conv_margin, -1*conv_margin)
            x_res = x_res[:, :, crop_slice, crop_slice]
        
        #-- Convolutional Layers
        x = self.module_A(x)
        x = self.module_B(x)
        x = self.module_C(x)
        
        #-- Duplicate input dimensions if needed
        if self.channel_ratio > 1:
            shape_in = list(x_res.shape) # [b,c,h,w]
            group_size = 4
            c_sub = shape_in[1] // group_size
            
            x_res = x_res.view(shape_in[0], group_size, c_sub, shape_in[2], shape_in[3])
            x_res = torch.cat([x_res]*self.channel_ratio, dim=2) #! Concatenate along channel dimension
            x_res = x_res.view(shape_in[0], shape_in[1]*self.channel_ratio, shape_in[2], shape_in[3])
        
        #-- Residual Block
        x = x_res + x
        return x