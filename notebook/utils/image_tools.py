import numpy as np
import scipy.ndimage as nd
import pandas as pd

from cellcomplex.utils import array_dict


def labelled_image_projection(seg_img, axis=2, direction=1, background_label=1, return_coords=False):
    """Compute the 2D projection of a labelled image along the specified  axis

    Parameters
    ----------
    seg_img : LabelledImage
        Labelled image to project in 2D
    axis : int
        The axis along which to project the labelled image
    direction : int
        On which side of the image to project (-1 or 1)
    background_label
        Ignored value for projection

    Returns
    -------
    np.ndarray
        2D Projected labelled image

    """
    xxx, yyy, zzz = np.mgrid[0:seg_img.shape[0], 0:seg_img.shape[1], 0:seg_img.shape[2]].astype(float)

    if axis == 0:
        y = np.arange(seg_img.shape[1])
        z = np.arange(seg_img.shape[2])
        yy, zz = map(np.transpose, np.meshgrid(y, z))
        proj = xxx * (seg_img.get_array() != background_label)
    elif axis == 1:
        x = np.arange(seg_img.shape[0])
        z = np.arange(seg_img.shape[2])
        xx, zz = map(np.transpose, np.meshgrid(x, z))
        proj = yyy * (seg_img.get_array() != background_label)
    elif axis == 2:
        x = np.arange(seg_img.shape[0])
        y = np.arange(seg_img.shape[1])
        xx, yy = map(np.transpose, np.meshgrid(x, y))
        proj = zzz * (seg_img.get_array() != background_label)

    proj[proj == 0] = np.nan 
    if direction == 1:
        proj = np.nanmax(proj, axis=axis)
        proj[np.isnan(proj)] = seg_img.shape[axis] - 1
    elif direction == -1:
        proj = np.nanmin(proj, axis=axis)
        proj[np.isnan(proj)] = 0

    if axis == 0:
        xx = proj
    elif axis == 1:
        yy = proj
    elif axis == 2:
        zz = proj

    # coords = tuple(np.transpose(np.concatenate(np.transpose([xx, yy, zz], (1, 2, 0)).astype(int))))
    coords = tuple(np.transpose(np.concatenate(np.transpose([xx, yy, zz], (1, 2, 0)).astype(int))))
    projected_img = seg_img.get_array()[coords].reshape(xx.shape)
    
    if return_coords:
        return projected_img, coords
    else:
        return projected_img
