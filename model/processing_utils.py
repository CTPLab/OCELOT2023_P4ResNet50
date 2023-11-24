# -*- coding: utf-8 -*-
"""
    __author__ = "Maxime Lafarge, maxime.lafarge[at]usz[dot]ch"
    __creation__ = "2023"
    __copyright__ = "Maxime Lafarge - All Rights Reserved"
"""
import numpy as np


""" PROCESSING UTILS
"""
def draw_AonB(A, B, y, x):
    A_h, A_w = A.shape[:2]
    B_h, B_w = B.shape[:2]
    
    offset_top    = -1 * min(0, y)
    offset_bottom = -1 * min(0, B_h - (y+A_h))
    offset_left   = -1 * min(0, x)
    offset_right  = -1 * min(0, B_w - (x+A_w))
    
    slice_read_y = slice(offset_top,  A_h-offset_bottom)
    slice_read_x = slice(offset_left, A_w-offset_right)

    slice_write_y = slice(y+offset_top,  y+A_h-offset_bottom)
    slice_write_x = slice(x+offset_left, x+A_w-offset_right)
    
    B[slice_write_y, slice_write_x, ...] = np.logical_or(
        B[slice_write_y, slice_write_x, ...],
        A[slice_read_y, slice_read_x, ...])
    return B


def _generate_circle_mask(circle_radius=1):
    circle_mask = np.zeros([2*circle_radius + 1]*2, dtype=bool)
    for yc in range(circle_mask.shape[0]):
        for xc in range(circle_mask.shape[1]):
            dc =  (yc - circle_radius) ** 2
            dc += (xc - circle_radius) ** 2
            if dc <= circle_radius ** 2:
                circle_mask.itemset((yc, xc), 1)
    return circle_mask


def mapToLocalMaxima(image_c, th_cutoff=0, local_radius=1):
    image_c = np.copy(image_c)
    image_c = image_c.astype(np.float32)
    
    #-- Generate a mask for non-maximum suppression
    circle_m = _generate_circle_mask(local_radius)
    
    #-- 1) Extract candidate locations
    candidate_locs = np.where(image_c > th_cutoff)
    
    #-- 2) Rank candidate locations
    scores = image_c[candidate_locs]
    indices_ranked = np.argsort(scores)[::-1]
    
    #-- 3) Scan candidate locations
    buffer_read = np.zeros(image_c.shape, dtype=bool)
    
    localMaxima = []
    for idx_c in indices_ranked:
        y_c = candidate_locs[0].item(idx_c)
        x_c = candidate_locs[1].item(idx_c)
        
        if not buffer_read.item(y_c, x_c):
            localMaxima.append([y_c, x_c])
            yyc = y_c - local_radius
            xxc = x_c - local_radius
            buffer_read = draw_AonB(circle_m, buffer_read, yyc, xxc)
    
    return localMaxima
    