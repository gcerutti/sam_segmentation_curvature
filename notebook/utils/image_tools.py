import numpy as np
import scipy.ndimage as nd
import pandas as pd

from timagetk.components import SpatialImage, LabelledImage
from timagetk.plugins import h_transform, region_labeling, segmentation

from cellcomplex.utils import array_dict

def membrane_image_segmentation(img, gaussian_sigma=0.75, h_min=None, segmentation_gaussian_sigma=0.5, volume_threshold=None):

    voxelsize = np.array(img.voxelsize)

    if h_min is None:
        h_min = 2 if img.dtype==np.uint8 else 1000

    smooth_image = nd.gaussian_filter(img, sigma=gaussian_sigma / voxelsize).astype(img.dtype)
    smooth_img = SpatialImage(smooth_image, voxelsize=voxelsize)

    ext_img = h_transform(smooth_img, h=h_min, method='min')

    seed_img = region_labeling(ext_img, low_threshold=1, high_threshold=h_min, method='connected_components')

    seg_smooth_image = nd.gaussian_filter(img, sigma=segmentation_gaussian_sigma / voxelsize).astype(img.dtype)
    seg_smooth_img = SpatialImage(seg_smooth_image, voxelsize=voxelsize)

    seg_img = segmentation(seg_smooth_img, seed_img, control='first', method='seeded_watershed')

    if volume_threshold is not None:
        seg_volumes = dict(zip(np.arange(seg_img.max()) + 1,
                               nd.sum(np.prod(voxelsize) * np.ones_like(seg_img),
                                      seg_img,
                                      index=np.arange(seg_img.max()) + 1)))

        labels_to_remove = np.array(list(seg_volumes.keys()))[np.array(list(seg_volumes.values())) > volume_threshold]
        print("--> Removing too large labels :", labels_to_remove)
        for l in labels_to_remove:
            seg_img[seg_img == l] = 1

    return seg_img

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
