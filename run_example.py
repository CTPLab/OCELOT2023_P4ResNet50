# -*- coding: utf-8 -*-
"""
    __author__ = "Maxime Lafarge, maxime.lafarge[at]usz[dot]ch"
    __creation__ = "2023"
    __copyright__ = "Maxime Lafarge - All Rights Reserved"
"""

import numpy as np
import PIL.Image

from model import Model

def process():
    """ Main function of an example of application of the inference pipeline described
        in Lafarge et al. 2023 "Detecting Cells in Histopathology Images with a ResNet Ensemble Model" 
    """
    # Instantiate the inferring model
    model = Model()
    
    # Import an image to process
    # (cell-level field-of-view; resolution ~0.25um/px; H&E-stained tissue specimen)
    source_path = "./example/000.tif"
    cell_patch = PIL.Image.open(source_path)
    cell_patch = np.array(cell_patch)
    
    # Apply model to input image
    metadata_dict = {} #-- Optional metadata
    result_list = model(cell_patch, metadata_dict)
    
    # Print the list of detected objects
    # Format (as required by the OCELOT2023 challenge): [x, y, label(1:non-tumor cell; 2:tumor cell), confidence score]
    print("Detected cells:")
    for i, v in enumerate(result_list):
        print("Object [{}] Coordinates-xy[{},{}] Label={} Confidence={}".format(i,*v))

if __name__ == "__main__":
    process()
