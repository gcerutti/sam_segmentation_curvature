from copy import deepcopy

import numpy as np
import scipy.ndimage as nd

from timagetk.components import SpatialImage, LabelledImage
from timagetk.plugins.resampling import isometric_resampling

from cellcomplex.utils.array_dict import array_dict

from cellcomplex.property_topomesh.analysis import compute_topomesh_property, compute_topomesh_vertex_property_from_faces

from tissue_nukem_3d.nuclei_mesh_tools import nuclei_image_surface_topomesh, up_facing_surface_topomesh


def segmentation_surface_topomesh(seg_img, background=1, resampling_voxelsize=1., maximal_length=5., microscope_orientation=None, compute_curvature=False):

    if microscope_orientation is None:
        microscope_orientation = 1 - 2*(np.mean(seg_img[:,:,0]) > np.mean(seg_img[:,:,-1]))

    binary_img = SpatialImage(nd.gaussian_filter(255*(seg_img != background), sigma=1./np.array(seg_img.voxelsize)).astype(np.uint8), voxelsize=seg_img.voxelsize)

    if resampling_voxelsize is not None:
        binary_img = isometric_resampling(binary_img, method=resampling_voxelsize, option='linear')
    else:
        resampling_voxelsize = np.array(binary_img.voxelsize)

    surface_topomesh = nuclei_image_surface_topomesh(binary_img,
                                                     nuclei_sigma=resampling_voxelsize,
                                                     density_voxelsize=resampling_voxelsize,
                                                     maximal_length=maximal_length,
                                                     intensity_threshold=64,
                                                     padding=False,
                                                     decimation=100)

    surface_topomesh = up_facing_surface_topomesh(surface_topomesh, normal_method='orientation',
                                                  upwards=microscope_orientation==1,
                                                  down_facing_threshold=0)

    compute_topomesh_property(surface_topomesh, 'normal', 2, normal_method='orientation')
    compute_topomesh_vertex_property_from_faces(surface_topomesh, 'normal', neighborhood=3, adjacency_sigma=1.2)

    if compute_curvature:
        curvature_properties = ['mean_curvature', 'gaussian_curvature', 'principal_curvature_min', 'principal_curvature_max']
        compute_topomesh_property(surface_topomesh, 'mean_curvature', 2)
        for property_name in curvature_properties:
            compute_topomesh_vertex_property_from_faces(surface_topomesh, property_name, neighborhood=3, adjacency_sigma=1.2)

    return surface_topomesh


def tissue_surface_topomesh(tissue, surface_topomesh=None, background=1, resampling_voxelsize=1., maximal_length=5., microscope_orientation=None, compute_curvature=False):
    assert tissue.label.get_property('surface_center') is not None

    cell_labels = np.array([l for l in tissue.get_label_ids()])

    if surface_topomesh is None:
        seg_img = tissue.label.image
        surface_topomesh = segmentation_surface_topomesh(seg_img,
                                                         background=background,
                                                         resampling_voxelsize=resampling_voxelsize,
                                                         maximal_length=maximal_length,
                                                         microscope_orientation=microscope_orientation,
                                                         compute_curvature=compute_curvature)

    cell_surface_center = array_dict(tissue.label.get_property('surface_center'))

    cell_points = np.array([cell_surface_center[c]
                            if c in cell_surface_center.keys()
                            else np.nan * np.ones(3)
                            for c in cell_labels])
    surface_points = surface_topomesh.wisp_property('barycenter', 0).values(list(surface_topomesh.wisps(0)))
    surface_distances = np.linalg.norm(cell_points[:, np.newaxis] - surface_points[np.newaxis], axis=2)

    surface_vertex_cell = cell_labels[np.nanargmin(surface_distances, axis=0)]
    surface_topomesh.update_wisp_property('cell', 0, dict(zip(surface_topomesh.wisps(0), surface_vertex_cell)))
    surface_topomesh.update_wisp_property('label', 0, dict(zip(surface_topomesh.wisps(0), surface_vertex_cell % 256)))

    return surface_topomesh


def tissue_layer_images(tissue):
    assert tissue.label.get_property('layer') is not None

    cell_layer = array_dict(tissue.label.get_property('layer'))
    cell_layer[1] = 0

    seg_img = tissue.label.image

    all_layer_img = LabelledImage(SpatialImage(array_dict(cell_layer).values(seg_img.get_array()),voxelsize=seg_img.voxelsize),no_label_id=0)

    layer_images = {}
    for layer in [1, 2, 3, 4]:
        layer_images[layer] = LabelledImage(SpatialImage((all_layer_img.get_array()>layer-1).astype(np.uint8),voxelsize=seg_img.voxelsize),no_label_id=0)

    return layer_images


def tissue_layer_surface_meshes(tissue, surface_topomesh=None, background=1, resampling_voxelsize=1., maximal_length=5., microscope_orientation=None, compute_curvature=False):
    assert tissue.label.get_property('layer') is not None
    assert tissue.label.get_property('layer_surface_center') is not None

    cell_labels = np.array([l for l in tissue.get_label_ids()])

    cell_layer = tissue.label.get_property('layer')
    cell_layer_surface_center = tissue.label.get_property('layer_surface_center')

    seg_img = tissue.label.image
    layer_images = tissue_layer_images(tissue)

    layer_surface_meshes = {}
    for layer in [1, 2, 3]:
        if layer==1 and surface_topomesh is not None:
            layer_surface_topomesh = surface_topomesh
        else:
            layer_img = layer_images[layer]
            layer_seg_img = deepcopy(seg_img)
            layer_seg_img[layer_img==0] = 1

            layer_surface_topomesh = segmentation_surface_topomesh(layer_seg_img,
                                                                   background = background,
                                                                   resampling_voxelsize = resampling_voxelsize,
                                                                   maximal_length = maximal_length,
                                                                   microscope_orientation = microscope_orientation,
                                                                   compute_curvature = compute_curvature)

        layer_cells = np.array([c for c in cell_labels if cell_layer[c]==layer])
        layer_cell_points = np.array([cell_layer_surface_center[c]
                                      if c in cell_layer_surface_center.keys()
                                      else np.nan * np.ones(3)
                                      for c in layer_cells])
        layer_surface_points = layer_surface_topomesh.wisp_property('barycenter', 0).values(list(layer_surface_topomesh.wisps(0)))
        layer_surface_distances = np.linalg.norm(layer_cell_points[:, np.newaxis] - layer_surface_points[np.newaxis], axis=2)

        layer_surface_vertex_cell = layer_cells[np.nanargmin(layer_surface_distances, axis=0)]
        layer_surface_topomesh.update_wisp_property('cell', 0, dict(zip(layer_surface_topomesh.wisps(0), layer_surface_vertex_cell)))
        layer_surface_topomesh.update_wisp_property('label', 0, dict(zip(layer_surface_topomesh.wisps(0), layer_surface_vertex_cell % 256)))

        layer_surface_meshes[layer] = layer_surface_topomesh

    return layer_surface_meshes

