"""
calculate_iou.py

Author: Wanqing Xia
Email: wxia612@aucklanduni.ac.nz

This script calculates the Intersection over Union (IoU) of two binary masks.

"""
import numpy as np

def calculate_iou(mask1, mask2):
    """
    Calculate the Intersection over Union (IoU) of two binary masks.

    Parameters:
        mask1 (np.array): First binary mask.
        mask2 (np.array): Second binary mask.

    Returns:
        float: The IoU between the two masks.
    """
    # Ensure the masks are boolean
    mask1 = mask1.astype(bool)
    mask2 = mask2.astype(bool)

    # Calculate intersection and union
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)

    # Calculate IoU
    iou = np.sum(intersection) / np.sum(union)

    return iou
