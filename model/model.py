# -*- coding: utf-8 -*-
"""
    __author__ = "Maxime Lafarge, maxime.lafarge[at]usz[dot]ch"
    __creation__ = "2023"
    __copyright__ = "Maxime Lafarge - All Rights Reserved"
"""
import os
import numpy as np
import torch
from .architecture import p4_ResNet50_net
from . import processing_utils as utils

""" =======================================================================================================
""" 
""" CHECKPOINTS OF ENSEMBLE MODEL
"""
current_dir = os.path.dirname(os.path.abspath(__file__))
checkpoint_path_list = []
cutoff_dict_list = []

base_organ_to_label_to_cutoff = {}
LIST_ORGANS = ["default", "kidney", "endometrium", "bladder", "prostate", "stomach", "head-and-neck"]
LIST_LABELS = [1,2]

#-- MODEL A
checkpoint_path_list.append(f"{current_dir}/checkpoint_A.pt")

a_org_to_label_to_cutoff = {}
for organ_c in LIST_ORGANS:
    a_org_to_label_to_cutoff[organ_c] = {k:0.5 for k in LIST_LABELS}

a_org_to_label_to_cutoff["default"][0] = 0.5661
a_org_to_label_to_cutoff["kidney"][0] = 0.5970
a_org_to_label_to_cutoff["endometrium"][0] = 0.5449
a_org_to_label_to_cutoff["bladder"][0] = 0.5553
a_org_to_label_to_cutoff["prostate"][0] = 0.5861
a_org_to_label_to_cutoff["stomach"][0] = 0.5580
a_org_to_label_to_cutoff["head-and-neck"][0] = 0.4948

a_org_to_label_to_cutoff["default"][1] = 0.5095
a_org_to_label_to_cutoff["kidney"][1] = 0.3866
a_org_to_label_to_cutoff["endometrium"][1] = 0.4974
a_org_to_label_to_cutoff["bladder"][1] = 0.5797
a_org_to_label_to_cutoff["prostate"][1] = 0.5333
a_org_to_label_to_cutoff["stomach"][1] = 0.5864
a_org_to_label_to_cutoff["head-and-neck"][1] = 0.4432

a_org_to_label_to_cutoff["default"][2] = 0.3401
a_org_to_label_to_cutoff["kidney"][2] = 0.4347
a_org_to_label_to_cutoff["endometrium"][2] = 0.3779
a_org_to_label_to_cutoff["bladder"][2] = 0.3454
a_org_to_label_to_cutoff["prostate"][2] = 0.4557
a_org_to_label_to_cutoff["stomach"][2] = 0.3439
a_org_to_label_to_cutoff["head-and-neck"][2] = 0.3334
cutoff_dict_list.append(a_org_to_label_to_cutoff)

#-- MODEL B
checkpoint_path_list.append(f"{current_dir}/checkpoint_B.pt")

b_org_to_label_to_cutoff = {}
for organ_c in LIST_ORGANS:
    b_org_to_label_to_cutoff[organ_c] = {k:0.5 for k in LIST_LABELS}
    
b_org_to_label_to_cutoff["default"][0] = 0.5108
b_org_to_label_to_cutoff["kidney"][0] = 0.6127
b_org_to_label_to_cutoff["endometrium"][0] = 0.4577
b_org_to_label_to_cutoff["bladder"][0] = 0.5922
b_org_to_label_to_cutoff["prostate"][0] = 0.5420
b_org_to_label_to_cutoff["stomach"][0] = 0.5915
b_org_to_label_to_cutoff["head-and-neck"][0] = 0.5888

b_org_to_label_to_cutoff["default"][1] = 0.4112
b_org_to_label_to_cutoff["kidney"][1] = 0.5580
b_org_to_label_to_cutoff["endometrium"][1] = 0.3733
b_org_to_label_to_cutoff["bladder"][1] = 0.4048
b_org_to_label_to_cutoff["prostate"][1] = 0.4647
b_org_to_label_to_cutoff["stomach"][1] = 0.4341
b_org_to_label_to_cutoff["head-and-neck"][1] = 0.2684

b_org_to_label_to_cutoff["default"][2] = 0.4609
b_org_to_label_to_cutoff["kidney"][2] = 0.3286
b_org_to_label_to_cutoff["endometrium"][2] = 0.3382
b_org_to_label_to_cutoff["bladder"][2] = 0.4684
b_org_to_label_to_cutoff["prostate"][2] = 0.4222
b_org_to_label_to_cutoff["stomach"][2] = 0.5001
b_org_to_label_to_cutoff["head-and-neck"][2] = 0.6899
cutoff_dict_list.append(b_org_to_label_to_cutoff)

#-- MODEL C
checkpoint_path_list.append(f"{current_dir}/checkpoint_C.pt")

c_org_to_label_to_cutoff = {}
for organ_c in LIST_ORGANS:
    c_org_to_label_to_cutoff[organ_c] = {k:0.5 for k in LIST_LABELS}
    
c_org_to_label_to_cutoff["default"][0] = 0.5494
c_org_to_label_to_cutoff["kidney"][0] = 0.5930
c_org_to_label_to_cutoff["endometrium"][0] = 0.4988
c_org_to_label_to_cutoff["bladder"][0] = 0.6143
c_org_to_label_to_cutoff["prostate"][0] = 0.5067
c_org_to_label_to_cutoff["stomach"][0] = 0.5626
c_org_to_label_to_cutoff["head-and-neck"][0] = 0.5926
   
c_org_to_label_to_cutoff["default"][1] = 0.5693
c_org_to_label_to_cutoff["kidney"][1] = 0.5928
c_org_to_label_to_cutoff["endometrium"][1] = 0.5704
c_org_to_label_to_cutoff["bladder"][1] = 0.62737
c_org_to_label_to_cutoff["prostate"][1] = 0.5398
c_org_to_label_to_cutoff["stomach"][1] = 0.5212
c_org_to_label_to_cutoff["head-and-neck"][1] = 0.5253

c_org_to_label_to_cutoff["default"][2] = 0.3342
c_org_to_label_to_cutoff["kidney"][2] = 0.3245
c_org_to_label_to_cutoff["endometrium"][2] = 0.3784
c_org_to_label_to_cutoff["bladder"][2] = 0.2991
c_org_to_label_to_cutoff["prostate"][2] = 0.2836
c_org_to_label_to_cutoff["stomach"][2] = 0.2447
c_org_to_label_to_cutoff["head-and-neck"][2] = 0.4389
cutoff_dict_list.append(c_org_to_label_to_cutoff)



def flip(x):
    x = np.transpose(x, (0,1,3,2))
    return x

class Model():

    def __init__(self):
        # Model's parameters        
        self.TILE_SIZE   = 130
        self.TILE_MARGIN = 63
        self.TILE_STRIDE = 4
        self.OBJECT_DIST_40x = 15
        self.OBJECT_DIST_5x  = 4

        self.WINDOW_TILE_NB = 600 #-- Number of tiles to process at once
        
        # Initialize model
        self.net = p4_ResNet50_net()
        
        checkpoint_kwargs = {}
        if not torch.cuda.is_available():
            checkpoint_kwargs["map_location"] = lambda storage, loc: storage
        self.checkpoint_list = []
        for chkpt_path_c in checkpoint_path_list:
            self.checkpoint_list.append(torch.load(chkpt_path_c, **checkpoint_kwargs))
        
        # Weight Transfer
        self.net.load_state_dict(self.checkpoint_list[0], strict=False)
        
        # Activate CUDA-based computation if available
        if torch.cuda.is_available(): self.net.to("cuda")
    
    def process_image(self, input_image):
        """ 0) GET INPUT DIMENSIONS
        """
        image_c = input_image[:,:,:3] # Keep RGB channels
        image_c = image_c.astype(np.float32)
        height_base, width_base = image_c.shape[:2]
 
        """ 1) FULL IMAGE MUST HAVE SIZE [MARGIN + k * STRIDE + EXTENSION + MARGIN]
        """
        nb_tiles_h = (height_base - 1) // self.TILE_STRIDE + 1
        nb_tiles_w = (width_base - 1) // self.TILE_STRIDE + 1
        
        extension_h = nb_tiles_h * self.TILE_STRIDE - height_base
        extension_w = nb_tiles_w * self.TILE_STRIDE - width_base
        
        extension_h_top  = int(extension_h // 2)
        extension_w_left = int(extension_w // 2)
        
        extension_h_bottom = extension_h - extension_h_top
        extension_w_right  = extension_w - extension_w_left

        """ >> INITIALIZE BUFFERS FOR ENSEMBLE RESULTS
        """
        model_to_scores = []
        model_to_locs   = []
        
        """ >> SCAN ENSMEBLE
        """
        prediction_canvas_list = []
        for checkpoint_c in self.checkpoint_list:
            self.net.load_state_dict(checkpoint_c, strict=False) # Weight Transfer
            
            """ PREPARE OUTPUT CANVAS
            """
            prediction_canvas = np.zeros([nb_tiles_h, nb_tiles_w, 3])

            """ 2) DIVIDE FULL IMAGE IN SCANNING WINODWS
            """
            nb_windows_h = (nb_tiles_h - 1) // self.WINDOW_TILE_NB + 1
            nb_windows_w = (nb_tiles_w - 1) // self.WINDOW_TILE_NB + 1
            
            for window_y in range(nb_windows_h):
                for window_x in range(nb_windows_w):
                    #-- Number of tiles in the current window
                    nb_tile_h = min(nb_tiles_h, (window_y+1)*self.WINDOW_TILE_NB) - window_y * self.WINDOW_TILE_NB
                    nb_tile_w = min(nb_tiles_w, (window_x+1)*self.WINDOW_TILE_NB) - window_x * self.WINDOW_TILE_NB
                    
                    input_h = 2*self.TILE_MARGIN + nb_tile_h * self.TILE_STRIDE
                    input_w = 2*self.TILE_MARGIN + nb_tile_w * self.TILE_STRIDE
                    
                    #-- Coordinates of the core input region
                    win_y = window_y * self.WINDOW_TILE_NB * self.TILE_STRIDE
                    win_x = window_x * self.WINDOW_TILE_NB * self.TILE_STRIDE
                    
                    #-- Shift top-left of the region to account for margin + extension
                    win_y = win_y - extension_h_top  - self.TILE_MARGIN
                    win_x = win_x - extension_w_left - self.TILE_MARGIN
                    
                    #-- Prepare input canvas
                    canvas_input = 255 * np.ones([input_h, input_w, 3])
                    
                    #-- Define offsets to read raw image
                    offset_top    = -1 * min(0, win_y)
                    offset_bottom = -1 * min(0, height_base - (win_y+input_h))
                    offset_left  = -1 * min(0, win_x)
                    offset_right = -1 * min(0, width_base - (win_x+input_w))
                    
                    #-- Define extraction slices
                    slice_read_y = slice(win_y+offset_top, win_y+input_h-offset_bottom)
                    slice_read_x = slice(win_x+offset_left, win_x+input_w-offset_right)
                    
                    #-- Define inpainting slides
                    slice_write_y = slice(offset_top, input_h-offset_bottom)
                    slice_write_x = slice(offset_left, input_w-offset_right)
                    
                    #-- Transfer image on input canvas
                    canvas_input[slice_write_y, slice_write_x, :] = image_c[slice_read_y, slice_read_x, :]
                
                    """ TEST-TIME AUGMENTATION LOOP
                    """
                    augmentation_stack = []
                    for augm_idx in range(2):
                    
                        """ >> PROCESS INPUT CANVAS
                        """
                        with torch.no_grad():
                            batch_stack  = np.stack([canvas_input], axis=0)
                            batch_tensor = np.transpose(batch_stack, (0,3,1,2))
                            batch_tensor = batch_tensor.astype(np.float32)
                            
                            """ >>> FORWARD-AUGMENTATION
                            """
                            if augm_idx == 1:
                                batch_tensor = flip(batch_tensor)
                                
                            batch_tensor = torch.from_numpy(batch_tensor)
                            if torch.cuda.is_available(): batch_tensor = batch_tensor.cuda()
                            
                            predictions_c = self.net.forward(batch_tensor)
                            predictions_c = predictions_c.cpu()
                            predictions_c = predictions_c.data.numpy()
                            
                            """ >>> BACKWARD-AUGMENTATION
                            """
                            if augm_idx == 1:
                                predictions_c = flip(predictions_c)
                                
                        predictions_c = predictions_c[0, ...]# [c, h, w]
                        predictions_c = predictions_c.transpose(1,2,0) #[c, h, w] -> [h, w, c]
                        
                        # Softmax activation
                        predictions_c = predictions_c.astype(np.float32)
                        predictions_c = predictions_c - np.amax(predictions_c, axis=2, keepdims=True) #-- Numerical stability
                        predictions_exp    = np.exp(predictions_c)
                        predictions_expsum = np.sum(predictions_exp, axis=2, keepdims=True)
                        predictions_c = predictions_exp / predictions_expsum
                        augmentation_stack.append(predictions_c)
                    
                    """ >> TEST-TIME AUGMENTATION AVERAGING
                    """
                    augmentation_stack = np.stack(augmentation_stack, axis=0) 
                    predictions_c = np.mean(augmentation_stack, axis=0)
                    
                    """ >> WRITE PREDICTIONS IN BUFFER
                    """
                    slice_y = slice(window_y * self.WINDOW_TILE_NB, window_y * self.WINDOW_TILE_NB + nb_tile_h)
                    slice_x = slice(window_x * self.WINDOW_TILE_NB, window_x * self.WINDOW_TILE_NB + nb_tile_w)
                    prediction_canvas[slice_y, slice_x, :] = predictions_c
            
            """ STORE CURRENT CANVAS
            """
            prediction_canvas_list.append(prediction_canvas)
        
        return prediction_canvas_list
        
        
    def __call__(self, cell_patch, metadata_c):
        ## >> cell-patch shape: (H, W, 3)
        
        """ >> RETREIVE ORGAN METADATA
        """
        organ_c = "default"
        if "organ" in metadata_c:
            if metadata_c["organ"] in cutoff_dict_list[0]:
                organ_c = metadata_c["organ"]
                
        """ >> GENERATE DETECTION MAP FROM PROBABILITY MAP
        """
        canvas_list = self.process_image(cell_patch)
        model_to_channel_to_yx_list = {}
        for model_idx, canvas_c in enumerate(canvas_list):
        
            """ >> RETREIVE ORGAN-SPECIFIC CUTOFF VALUES
            """
            label_to_cutoff_list = cutoff_dict_list[model_idx][organ_c]
        
            model_to_channel_to_yx_list[model_idx] = {}
            for channel_c in LIST_LABELS:
                cutoff_c = label_to_cutoff_list[channel_c]
                
                #-- Locate local maxima
                local_radius = self.OBJECT_DIST_5x
                localMaxima_yx = utils.mapToLocalMaxima(
                    canvas_c[...,channel_c],
                    th_cutoff = cutoff_c,
                    local_radius = local_radius)
                
                model_to_channel_to_yx_list[model_idx][channel_c] = localMaxima_yx
        
        """ >> AGGREGATE TOUCHING DETECTED POINTS
        """
        obj_to_yx_list = []
        obj_to_label_list = []
        obj_to_score_list = []
        
        dist_cutoff = self.OBJECT_DIST_5x ** 2 + 1
        for model_idx, channel_to_yx_list in model_to_channel_to_yx_list.items():
            for channel_c, yx_list in channel_to_yx_list.items():
                for yq, xq in yx_list:
                    score_c = canvas_list[model_idx].item(yq,xq,channel_c)
                    
                    dist_min      = dist_cutoff
                    obj_match_idx = None
                    for obj_idx, (yt, xt) in enumerate(obj_to_yx_list):
                        dist_q = (yt-yq) ** 2 + (xt-xq) ** 2
                        if dist_q < dist_cutoff:
                            dist_min      = dist_q
                            obj_match_idx = obj_idx
                    
                    if dist_min >= dist_cutoff: #-- New point
                        obj_to_yx_list.append([yq, xq])
                        obj_to_label_list.append([channel_c])
                        obj_to_score_list.append([score_c])
                        
                    else: #-- Match found
                        #-- Update object location via running average
                        yb, xb = obj_to_yx_list[obj_match_idx]
                        obj_weight = len(obj_to_label_list[obj_match_idx])
                        obj_weight = float(obj_weight)
                        yu = (obj_weight * yb + yq) / (obj_weight+1)
                        xu = (obj_weight * xb + xq) / (obj_weight+1)
                        
                        obj_to_yx_list[obj_match_idx] = [yu, xu]
                        obj_to_label_list[obj_match_idx].append(channel_c)
                        obj_to_score_list[obj_match_idx].append(score_c)

        #-- Integer conversion
        obj_to_yx_list = [[int(v+0.5) for v in yx] for yx in obj_to_yx_list]
        
        """ >> MAKE A DECISION FOR EACH DETECTED POINT
        """
        output_list = []
        for obj_idx, yx in enumerate(obj_to_yx_list):
            label_list = obj_to_label_list[obj_idx]
            score_list = obj_to_score_list[obj_idx]
            label_to_scores = {i:[] for i in LIST_LABELS}
            for c,s in zip(label_list, score_list):
                label_to_scores[c].append(s)
            
            #-- Summarize results
            label_to_vote = {k:len(v) for k,v in label_to_scores.items()}
            label_to_score = {}
            for k,v in label_to_scores.items():
                label_to_score[k] = 0
                if len(v) > 0: label_to_score[k] = np.mean(v)
            
            #-- Final call
            label_c = 1 #-- Default
            if label_to_vote[1] == label_to_vote[2]: #-- [Scenario A] no majority vote
                label_c = 2 #-- In case of doubt: label as tumor
            else: #-- [Scenario B] majority vote
                if label_to_vote[2] > label_to_vote[1]:
                    label_c = 2
            
            """ >> RESCALING STEP
            """
            a = self.TILE_STRIDE
            b = self.TILE_STRIDE // 2
            
            output_entry = [
                a * yx[1] + b,
                a * yx[0] + b,
                label_c,
                label_to_score[label_c]
            ]
            output_list.append(output_entry)
        
        
        # RETURN RESULS PER SAMPLE (as required by the OCELOT2023 challenge)
        # - int: cell's x-coordinate in the cell patch
        # - int: cell's y-coordinate in the cell patch
        # - int: class id of the cell, either 1 (BC) or 2 (TC)
        # - float: confidence score
        return output_list
        