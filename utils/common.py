#!/usr/bin/env python3
# Author: hiro
# Date: 2022-06-30 22:59:40
# LastEditors: hiro
# Copyright (c) 2022 by hiro jinguang.tong@anu.edu.au, All Rights Reserved. 

import torch
import numpy as np

def compute_iou(occ1, occ2):
    """ Computes the Intersection over Union (IoU) value for two sets of occupancy values.

    Args:
        occ1 (tensor): first set of occupancy values
        occ2 (tensor): second set of occupancy values
    """
    occ1 = np.asarray(occ1)
    occ2 = np.asarray(occ2)
    
    # Put all data in second dimension
    # Also works for 1-dimensional data
    if occ1.ndim >= 2:
        occ1 = occ1.reshape(occ1.shape[0], -1)
    if occ2.ndim >= 2:
        occ2 = occ2.reshape(occ2.shape[0], -1)
    
    # Convert to boolean values
    occ1 = (occ1 >= 0.5)
    occ2 = (occ2 >= 0.5)
    
    # Compute IoU
    area_union = (occ1 | occ2).astype(np.float32).sum(axis=-1)
    area_intersect = (occ1 & occ2).astype(np.float32).sum(axis=-1)

    iou = (area_intersect / area_union)

    return iou