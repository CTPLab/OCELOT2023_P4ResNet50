# -*- coding: utf-8 -*-
"""
    __author__ = "Maxime Lafarge, maxime.lafarge[at]usz[dot]ch"
    __creation__ = "2023"
    __copyright__ = "Maxime Lafarge - All Rights Reserved"
"""
import torch


""" DEFINE A CONVOLUTIONAL BLOCK
"""
class p4_BLOCK(torch.nn.Module):
    def __init__(self,
            channels_in    = 1,
            channels_out   = 1,
            kernel_size    = 3,
            
            bias    = False,
            relu    = True,
            bn      = True,
            pre_act = False):
        super(p4_BLOCK, self).__init__()
        
        #-- Kernel initialization
        kernel_shape = (channels_out, channels_in, kernel_size, kernel_size)
        self.kernel = torch.normal(0, 1, size=kernel_shape)
        if torch.cuda.is_available(): self.kernel = self.kernel.cuda()
        self.kernel = torch.nn.Parameter(self.kernel)
        
        #-- Pre-Activation Parameter
        self.pre_act = pre_act
        
        #-- Additional bias operator
        self.do_bias = bias
        self.bias = None
        if bias:
            bias_shape = [1, channels_out, 1, 1]
            self.bias = torch.ones(bias_shape)
            if torch.cuda.is_available(): self.bias = self.bias.cuda()
            self.bias = torch.nn.Parameter(self.bias)
            
        #-- BN operator
        self.do_bn = bn
        self.bn = None
        if bn:
            param_shape = [1, channels_in, 1, 1]
            if not pre_act:
                param_shape = [1, channels_out, 1, 1]
                
            self.alpha = torch.ones(param_shape)
            self.beta  = torch.zeros(param_shape)
            
            self.alpha = torch.nn.Parameter(self.alpha)
            self.beta  = torch.nn.Parameter(self.beta)
        
        #-- Non-linerarity operator
        self.do_relu = relu
        self.relu = None
        if relu:
            self.relu = torch.nn.ReLU()


    def forward(self, x, *args, **kwargs):
        if self.pre_act:
            if self.do_bn:
                x = self.alpha * x + self.beta
                
            if self.do_relu:
                x = self.relu(x)
                
        x = torch.nn.functional.conv2d(
            input    = x,
            weight   = self.kernel,
            bias     = None,
            stride   = 1,
            padding  = 0,
            dilation = 1,
            groups   = 1)
        
        if self.do_bias:
            x = x + self.bias
            
        if not self.pre_act:
            if self.do_bn:
                x = self.alpha * x + self.beta
                
            if self.do_relu:
                x = self.relu(x)
                
        return x