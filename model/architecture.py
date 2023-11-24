# -*- coding: utf-8 -*-
"""
    __author__ = "Maxime Lafarge, maxime.lafarge[at]usz[dot]ch"
    __creation__ = "2023"
    __copyright__ = "Maxime Lafarge - All Rights Reserved"
"""
import torch

from ._arch_p4_BLOCK import p4_BLOCK
from ._arch_p4_RES_BLOCK import p4_RES_BLOCK
from ._arch_groupMaxPool import GroupMaxPool
from ._arch_groupMaxProjection import GroupMaxProjection

class p4_ResNet50_net(torch.nn.Module):
    def __init__(self):
        super(p4_ResNet50_net, self).__init__()
        
        #-- Units
        group_size = 4        
        nb_units_A = group_size * 24
        nb_units_B = group_size * 24
        nb_units_C = group_size * 48
        nb_units_D = group_size * 48
        nb_units_E = group_size * 96
        nb_units_Ep = 96
        nb_units_out = 3
        
        #-- Groups of blocks
        nb_blocks_B = 4
        nb_blocks_C = 5
        nb_blocks_D = 6
        
        #-- Lists of parameters
        self.block_list   = torch.nn.ModuleList()
        
        #-- Module counter
        module_idx = 0
        
        
        #====================================================================
        #== DOWN 1 (130 -> 128 > 64)
        kwargs = {
            "channels_in":  3, #-- RGB
            "channels_out": nb_units_A,
            "kernel_size":  3,
            "pre_act": False
        }
        setattr(self, f"module_{module_idx}", p4_BLOCK(**kwargs))
        self.block_list.append(getattr(self, f"module_{module_idx}"))
        module_idx += 1
        
        setattr(self, f"module_{module_idx}", GroupMaxPool(down_factor=2))
        self.block_list.append(getattr(self, f"module_{module_idx}"))
        module_idx += 1
        
        
        #====================================================================
        #== DOWN 2 (64 -> 62 > 31)
        kwargs = {
            "channels_in":  nb_units_A,
            "channels_out": nb_units_B,
            "kernel_size":  3,
            "pre_act": False,
            "relu": False,
            "bn":   False,
            "bias":  True
            
        }
        setattr(self, f"module_{module_idx}", p4_BLOCK(**kwargs))
        self.block_list.append(getattr(self, f"module_{module_idx}"))
        module_idx += 1
        
        setattr(self, f"module_{module_idx}", GroupMaxPool(down_factor=2))
        self.block_list.append(getattr(self, f"module_{module_idx}"))
        module_idx += 1
        
        
        #====================================================================
        #== RESIDUAL BLOCKS [PART_B] (31 --> 23)
        for idx in range(nb_blocks_B-1):
            kwargs = {
                "channels_in":    nb_units_B,
                "channels_inter": nb_units_B,
                "channels_out":   nb_units_B,
                "kernel_size":   3
                }
            setattr(self, f"module_{module_idx}", p4_RES_BLOCK(**kwargs))
            self.block_list.append(getattr(self, f"module_{module_idx}"))
            module_idx += 1
        kwargs = {
            "channels_in":    nb_units_B,
            "channels_inter": nb_units_B,
            "channels_out":   nb_units_C,
            "kernel_size":   3
            }
        setattr(self, f"module_{module_idx}", p4_RES_BLOCK(**kwargs))
        self.block_list.append(getattr(self, f"module_{module_idx}"))
        module_idx += 1
        
        
        #====================================================================
        #== RESIDUAL BLOCKS [PART_C] (23 --> 13)
        for idx in range(nb_blocks_C-1):
            kwargs = {
                "channels_in":    nb_units_C,
                "channels_inter": nb_units_C,
                "channels_out":   nb_units_C,
                "kernel_size":   3
                }
            setattr(self, f"module_{module_idx}", p4_RES_BLOCK(**kwargs))
            self.block_list.append(getattr(self, f"module_{module_idx}"))
            module_idx += 1
        kwargs = {
            "channels_in":    nb_units_C,
            "channels_inter": nb_units_C,
            "channels_out":   nb_units_D,
            "kernel_size":   3
            }
        setattr(self, f"module_{module_idx}", p4_RES_BLOCK(**kwargs))
        self.block_list.append(getattr(self, f"module_{module_idx}"))
        module_idx += 1
        
        
        #====================================================================
        #== RESIDUAL BLOCKS [PART_D] (13 --> 1)
        for idx in range(nb_blocks_D-1):
            kwargs = {
                "channels_in":    nb_units_D,
                "channels_inter": nb_units_D,
                "channels_out":   nb_units_D,
                "kernel_size":   3
                }
            setattr(self, f"module_{module_idx}", p4_RES_BLOCK(**kwargs))
            self.block_list.append(getattr(self, f"module_{module_idx}"))
            module_idx += 1
        kwargs = {
            "channels_in":    nb_units_D,
            "channels_inter": nb_units_D,
            "channels_out":   nb_units_E,
            "kernel_size":   3
            }
        setattr(self, f"module_{module_idx}", p4_RES_BLOCK(**kwargs))
        self.block_list.append(getattr(self, f"module_{module_idx}"))
        module_idx += 1
        
        
        #====================================================================
        #== PRE-PROJECTION LAYER
        kwargs = {
            "channels_in":  nb_units_E,
            "channels_out": nb_units_E,
            "kernel_size":  1,
            "relu": True,
            "bn":   True,
            "pre_act": True
        }
        setattr(self, f"module_{module_idx}", p4_BLOCK(**kwargs))
        self.block_list.append(getattr(self, f"module_{module_idx}"))
        module_idx += 1
        
        
        #====================================================================
        #== PROJECTION LAYER
        setattr(self, f"module_{module_idx}", GroupMaxProjection(group_size))
        self.block_list.append(getattr(self, f"module_{module_idx}"))
        module_idx += 1
        
        
        #====================================================================
        #== MODEL HEAD
        kwargs = {
            "channels_in":  nb_units_Ep,
            "channels_out": nb_units_Ep,            
            "kernel_size":  1,
            "pre_act": True
        }
        setattr(self, f"module_{module_idx}", p4_BLOCK(**kwargs))
        self.block_list.append(getattr(self, f"module_{module_idx}"))
        module_idx += 1
        
        kwargs = {
            "channels_in":  nb_units_Ep,
            "channels_out": nb_units_out,            
            "kernel_size":  1,
            "pre_act": True,
            "bias":    True
        }
        setattr(self, f"module_{module_idx}", p4_BLOCK(**kwargs))
        self.block_list.append(getattr(self, f"module_{module_idx}"))
        module_idx += 1


    def forward(self, x):
        x = x / 255.0
        for block_c in self.block_list:
            x = block_c(x)
        
        return x