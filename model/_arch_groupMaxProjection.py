# -*- coding: utf-8 -*-
"""
    __author__ = "Maxime Lafarge, maxime.lafarge[at]usz[dot]ch"
    __creation__ = "2023"
    __copyright__ = "Maxime Lafarge - All Rights Reserved"
"""
import torch

class GroupMaxProjection(torch.nn.Module):
    def __init__(self, group_size=4):
        super(GroupMaxProjection, self).__init__()
        self.group_size = group_size
        
    def forward(self, x, *args, **kwargs):
        
        """ RESHAPE INPUT: FLATTEN GROUP AXIS IN SPATIAL DIMENSIONS
        """
        ## x: #[batch, groupSize * channels, height, width]
        shape_in = list(x.shape) # [b,c,h,w]
        c_sub = shape_in[1] // self.group_size
        x = x.view(shape_in[0], self.group_size, c_sub, shape_in[2], shape_in[3])
        
        x = torch.amax(x, axis=1, keepdim=False)
        return x