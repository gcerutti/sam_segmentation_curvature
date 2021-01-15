import numpy as np
import scipy.ndimage as nd
import pandas as pd

from cellcomplex.utils.array_dict import array_dict

from cellcomplex.property_topomesh.analysis import compute_topomesh_property, compute_topomesh_vertex_property_from_faces
from cellcomplex.property_topomesh.utils.triangulation_tools import triangles_from_adjacency_edges

from tissue_analysis.array_tools import find_geometric_median
from .labelled_image_tools import topological_elements_extraction3D


def compute_tissue_cell_centers(tissue):
    cell_labels = np.array([l for l in tissue.get_label_ids()])
    seg_img = tissue.label.image

    cell_center = dict(zip(cell_labels,
                           nd.center_of_mass(np.ones_like(seg_img), seg_img, index=cell_labels) * np.array(seg_img.voxelsize)))
    tissue.label._property['center'] = cell_center


def compute_tissue_cell_surface_centers(tissue):
    cell_labels = np.array([l for l in tissue.get_label_ids()])
    cell_surfels = tissue.wall.get_property('coordinates')

    cell_surface_center = {c: s[find_geometric_median(s)]
                           for (n,c), s in cell_surfels.items()
                           if n == 1}

    for c in tissue.get_label_ids():
        if not c in cell_surface_center.keys():
            cell_surface_center[c] = np.nan*np.ones(3)
    cell_neighbors = tissue.label.get_property('neighbors')
    if cell_neighbors is not None:
        for c in cell_surface_center.keys():
            if not c in cell_neighbors[1]:
                cell_surface_center[c] = np.nan * np.ones(3)

    tissue.label._property['surface_center'] = cell_surface_center


def compute_tissue_cell_volumes(tissue):
    cell_labels = np.array([l for l in tissue.get_label_ids()])
    seg_img = tissue.label.image

    cell_volumes = dict(zip(cell_labels,
                            nd.sum(np.ones_like(seg_img), seg_img, index=cell_labels) * np.prod(seg_img.voxelsize)))
    tissue.label._property['volume'] = cell_volumes


def compute_tissue_cell_neighbors_from_surfels(tissue, min_surfel_number=6):
    seg_img = tissue.label.image

    if len(tissue.get_wall_ids()) == 0:
        cell_surfels = topological_elements_extraction3D(seg_img, elem_order=[2])[2]
        #cell_surfels = seg_img.topological_elements(element_order=[2])[2]
        cell_surfels = {c:s*np.array(seg_img.voxelsize) for c,s in cell_surfels.items()}
        tissue.wall._ids = set(cell_surfels.keys())
        tissue.wall._property['coordinates'] = cell_surfels
    cell_surfels = tissue.wall.get_property('coordinates')

    cell_adjacency_edges = np.array([[c,n] for (c,n),s in cell_surfels.items() if len(s)>=min_surfel_number])

    cell_neighbors = {l:set() for l in tissue.get_label_ids()}
    cell_neighbors[1] = set()
    for c1,c2 in cell_adjacency_edges:
        cell_neighbors[c1] |= {c2}
        cell_neighbors[c2] |= {c1}
    tissue.label._property['neighbors'] = cell_neighbors


def compute_tissue_cell_layer_from_surface_mesh(tissue, surface_topomesh):
    assert tissue.label.get_property('neighbors') is not None
    assert surface_topomesh.has_wisp_property('cell', 0, is_computed=True)

    cell_labels = np.array([l for l in tissue.get_label_ids()])
    cell_neighbors = tissue.label.get_property('neighbors')

    surface_vertex_cell = surface_topomesh.wisp_property('cell',0).values(list(surface_topomesh.wisps(0)))

    l1_cells = np.unique(surface_vertex_cell)
    l2_cells = [c for c in cell_labels
                if np.any([n in l1_cells for n in cell_neighbors[c]])
                and not c in l1_cells]
    l3_cells = [c for c in cell_labels
                if np.any([n in l2_cells for n in cell_neighbors[c]])
                and not (c in l1_cells or c in l2_cells)]

    cell_layer = {c: 1 if c in l1_cells else 2 if c in l2_cells else 3 if c in l3_cells else 4 for c in cell_labels}

    tissue.label._property['layer'] = cell_layer


def compute_tissue_cell_layer_surface_centers(tissue):
    assert tissue.label.get_property('layer') is not None
    assert tissue.wall.get_property('coordinates') is not None

    cell_labels = np.array([l for l in tissue.get_label_ids()])
    cell_surfels = tissue.wall.get_property('coordinates')
    cell_layer = tissue.label.get_property('layer')

    cell_layer_surface_center = {}
    for layer in [1, 2, 3]:
        layer_cells = [c for c in cell_labels if cell_layer[c] == layer]
        if layer == 1:
            if tissue.label.get_property('surface_center') is None:
                cell_surface_center = {c: s[find_geometric_median(s)]
                                       for (n,c), s in cell_surfels.items()
                                       if n == 1 and c in layer_cells}
            else:
                cell_surface_center = tissue.label.get_property('surface_center')
            for c in layer_cells:
                cell_layer_surface_center[c] = cell_surface_center[c]
        else:
            previous_layer_cells = [c for c in cell_labels if cell_layer[c] == layer - 1]
            layer_surface_surfels = [(c, n) for (c, n) in cell_surfels.keys()
                                     if c in previous_layer_cells and n in layer_cells
                                     or c in layer_cells and n in previous_layer_cells]

            for c in layer_cells:
                layer_cell_surface_surfels = [k for k in layer_surface_surfels if c in k]
                if len(layer_cell_surface_surfels) > 0:
                    layer_cell_surfels = np.concatenate([cell_surfels[k] for k in layer_cell_surface_surfels])
                    cell_layer_surface_center[c] = layer_cell_surfels[find_geometric_median(layer_cell_surfels)]

    for c in tissue.get_label_ids():
        if not c in cell_layer_surface_center.keys():
            cell_layer_surface_center[c] = np.nan*np.ones(3)

    tissue.label._property['layer_surface_center'] = cell_layer_surface_center


def compute_tissue_cell_property_from_surface_mesh(tissue, property_name, surface_topomesh):
    assert surface_topomesh.has_wisp_property(property_name, 0, is_computed=True)
    assert surface_topomesh.has_wisp_property('cell', 0, is_computed=True)

    cell_labels = np.array([l for l in tissue.get_label_ids()])

    compute_topomesh_vertex_property_from_faces(surface_topomesh, 'area', weighting='uniform', neighborhood=1)
    vertex_cells = surface_topomesh.wisp_property('cell', 0).values(list(surface_topomesh.wisps(0)))
    vertex_areas = surface_topomesh.wisp_property('area', 0).values(list(surface_topomesh.wisps(0)))

    vertex_property = surface_topomesh.wisp_property(property_name, 0).values(list(surface_topomesh.wisps(0)))

    if vertex_property.ndim == 1:
        cell_property = nd.sum(vertex_areas * vertex_property, vertex_cells, index=cell_labels)
        cell_property /= nd.sum(vertex_areas, vertex_cells, index=cell_labels)
    elif vertex_property.ndim == 2:
        cell_property = np.transpose([nd.sum(vertex_areas * vertex_property[:, k], vertex_cells, index=cell_labels) for k in range(3)])
        cell_property /= nd.sum(vertex_areas, vertex_cells, index=cell_labels)[:, np.newaxis]
        if property_name in ['normal']:
            cell_property /= np.linalg.norm(cell_property, axis=1)[:, np.newaxis]

    tissue.label._property[property_name] = dict(zip(cell_labels, cell_property))


def compute_tissue_cell_property_from_layer_surface_meshes(tissue, property_name, layer_surface_meshes):
    assert tissue.label.get_property('layer') is not None

    cell_labels = np.array([l for l in tissue.get_label_ids()])
    cell_layer = tissue.label.get_property('layer')

    cell_layer_property = {}
    for layer in [1, 2, 3]:
        layer_cells = [c for c in cell_labels if cell_layer[c]==layer]
        layer_surface_topomesh = layer_surface_meshes[layer]

        assert layer_surface_topomesh.has_wisp_property(property_name, 0, is_computed=True)
        assert layer_surface_topomesh.has_wisp_property('cell', 0, is_computed=True)

        compute_topomesh_vertex_property_from_faces(layer_surface_topomesh, 'area', weighting='uniform', neighborhood=1)
        vertex_cells = layer_surface_topomesh.wisp_property('cell', 0).values(list(layer_surface_topomesh.wisps(0)))
        vertex_areas = layer_surface_topomesh.wisp_property('area', 0).values(list(layer_surface_topomesh.wisps(0)))

        vertex_property = layer_surface_topomesh.wisp_property(property_name, 0).values(list(layer_surface_topomesh.wisps(0)))

        if vertex_property.ndim == 1:
            cell_property = nd.sum(vertex_areas * vertex_property, vertex_cells, index=layer_cells)
            cell_property /= nd.sum(vertex_areas, vertex_cells, index=layer_cells)
        elif vertex_property.ndim == 2:
            cell_property = np.transpose([nd.sum(vertex_areas * vertex_property[:, k], vertex_cells, index=layer_cells) for k in range(3)])
            cell_property /= nd.sum(vertex_areas, vertex_cells, index=layer_cells)[:, np.newaxis]
            if property_name in ['normal']:
                cell_property /= np.linalg.norm(cell_property, axis=1)[:, np.newaxis]

        cell_layer_property.update(dict(zip(layer_cells, cell_property)))

    for c in tissue.get_label_ids():
        if not c in cell_layer_property.keys():
            if vertex_property.ndim == 1:
                cell_layer_property[c] = np.nan
            elif vertex_property.ndim == 2:
                cell_layer_property[c] = np.nan*np.ones(3)

    tissue.label._property[property_name] = cell_layer_property


def compute_tissue_cell_heights(tissue):
    assert tissue.label.get_property('layer') is not None
    assert tissue.label.get_property('normal') is not None

    cell_labels = np.array([l for l in tissue.get_label_ids()])
    cell_center = tissue.label.get_property('center')
    cell_surfels = tissue.wall.get_property('coordinates')
    cell_layer = tissue.label.get_property('layer')
    cell_normals = tissue.label.get_property('normal')

    cell_heights = {}
    for layer in [1, 2, 3]:
        layer_cells = [c for c in cell_labels if cell_layer[c]==layer]

        for c in layer_cells:
            cell_points = np.concatenate([cell_surfels[k] for k in cell_surfels.keys() if c in k])
            #cell_points =  np.transpose(np.where(seg_img==c))*np.array(seg_img.voxelsize)
            cell_point_vectors = [p - cell_center[c] for p in cell_points]
            cell_normal = cell_normals[c]
            cell_point_vector_dot_products = np.einsum("...ij,...ij->...i",cell_point_vectors,cell_normal[np.newaxis])
            cell_heights[c] = np.max(cell_point_vector_dot_products) - np.min(cell_point_vector_dot_products)

    for c in tissue.get_label_ids():
        if not c in cell_heights.keys():
            cell_heights[c] = np.nan

    tissue.label._property['height'] = cell_heights


def tissue_cell_binary_property_connected_components(tissue, property_name, layer_restriction=1):
    assert tissue.label.get_property('neighbors') is not None
    assert tissue.label.get_property(property_name) is not None

    cell_labels = np.array([l for l in tissue.get_label_ids()])
    cell_neighbors = tissue.label.get_property('neighbors')
    if layer_restriction is not None:
        assert tissue.label.get_property('layer') is not None
        cell_layer = tissue.label.get_property('layer')
        layer_labels = [l for l in cell_labels if cell_layer[l] == layer_restriction]
    else:
        layer_labels = cell_labels

    cell_property = array_dict(tissue.label.get_property(property_name))
    property_labels = [l for l in layer_labels if cell_property[l]]

    considered_labels = set()
    component_cells = []
    for label in property_labels:
        if not label in considered_labels:
            considered_labels |= {label}
            component_labels = [label]
            neighbor_labels = [l for l in cell_neighbors[label] if l in property_labels]
            while len(neighbor_labels) > 0:
                n_label = neighbor_labels.pop()
                if not n_label in considered_labels:
                    considered_labels |= {n_label}
                    component_labels += [n_label]
                    neighbor_neighbors = [l for l in cell_neighbors[n_label] if l in property_labels]
                    neighbor_labels += list(set(neighbor_neighbors).difference(considered_labels))
            component_cells += [component_labels]

    return component_cells


def tissue_cell_binary_property_morphological_operation(tissue, property_name, method='erosion', iterations=1, layer_restriction=1):
    assert tissue.label.get_property('neighbors') is not None
    assert tissue.label.get_property(property_name) is not None

    cell_labels = [l for l in tissue.get_label_ids()]
    cell_neighbors = tissue.label.get_property('neighbors')
    if layer_restriction is not None:
        assert tissue.label.get_property('layer') is not None
        cell_layer = array_dict(tissue.label.get_property('layer'))
        layer_labels = [l for l in cell_labels if cell_layer[l] == layer_restriction]
    else:
        layer_labels = cell_labels

    cell_property = array_dict(tissue.label.get_property(property_name))

    label_neighbors = [n for c in cell_labels for n in cell_neighbors[c] if n in layer_labels]
    label_neighbor_labels = [c for c in cell_labels for n in cell_neighbors[c] if n in layer_labels]

    for iteration in range(iterations):
        label_neighbor_properties = cell_property.values(cell_labels + label_neighbors)
        if method == 'erosion':
            morpho_property = nd.minimum(label_neighbor_properties != 0, cell_labels + label_neighbor_labels, index=cell_labels)
        elif method == 'dilation':
            morpho_property = nd.maximum(label_neighbor_properties != 0, cell_labels + label_neighbor_labels, index=cell_labels)
        elif method == 'opening':
            morpho_property = nd.minimum(label_neighbor_properties != 0, cell_labels + label_neighbor_labels, index=cell_labels)
            label_neighbor_properties = array_dict(morpho_property, keys=cell_labels).values(cell_labels + label_neighbors)
            morpho_property = nd.maximum(label_neighbor_properties != 0, cell_labels + label_neighbor_labels, index=cell_labels)
        elif method == 'closing':
            morpho_property = nd.maximum(label_neighbor_properties != 0, cell_labels + label_neighbor_labels, index=cell_labels)
            label_neighbor_properties = array_dict(morpho_property, keys=cell_labels).values(cell_labels + label_neighbors)
            morpho_property = nd.minimum(label_neighbor_properties != 0, cell_labels + label_neighbor_labels, index=cell_labels)
        else:
            morpho_property = cell_property.values(cell_labels)

        if layer_restriction is not None:
            morpho_property *= cell_layer.values(cell_labels) == layer_restriction
        tissue.label._property[property_name] = dict(zip(cell_labels, morpho_property))


def compute_tissue_cell_binary_property_from_surface_mesh(tissue, property_name, surface_topomesh, method='any'):
    assert surface_topomesh.has_wisp_property(property_name, 0, is_computed=True)
    assert surface_topomesh.has_wisp_property('cell', 0, is_computed=True)

    cell_labels = np.array([l for l in tissue.get_label_ids()])

    vertex_cells = surface_topomesh.wisp_property('cell', 0).values(list(surface_topomesh.wisps(0)))
    vertex_property = surface_topomesh.wisp_property(property_name, 0).values(list(surface_topomesh.wisps(0)))

    assert vertex_property.ndim == 1

    if method == 'any':
        cell_property = (nd.mean(vertex_property, vertex_cells, index=cell_labels) > 0.).astype(int)
    elif method == 'most':
        cell_property = (nd.mean(vertex_property, vertex_cells, index=cell_labels) > 0.5).astype(int)
    elif method == 'all':
        cell_property = (nd.mean(vertex_property, vertex_cells, index=cell_labels) == 1.).astype(int)

    tissue.label._property[property_name] = dict(zip(cell_labels, cell_property))


def tissue_cell_binary_property_layer_propagation(tissue, property_name):
    assert tissue.label.get_property('neighbors') is not None
    assert tissue.label.get_property('layer') is not None
    assert tissue.label.get_property(property_name) is not None

    cell_labels = [l for l in tissue.get_label_ids()]
    cell_neighbors = tissue.label.get_property('neighbors')
    cell_layer = array_dict(tissue.label.get_property('layer'))

    cell_labels = [l for l in tissue.get_label_ids()]

    for layer in [2, 3, 4]:
        cell_property = array_dict(tissue.label.get_property(property_name))

        layer_cells = np.array([l for l in cell_labels if cell_layer[l] == layer])
        prev_layer_cells = np.array([l for l in cell_labels if cell_layer[l] == layer - 1])

        prev_layer_cell_neighbors = np.array([n for l in prev_layer_cells
                                              for n in cell_neighbors[l]
                                              if n in prev_layer_cells])
        prev_layer_cell_neighbor_cells = np.array([l for l in prev_layer_cells
                                                   for n in cell_neighbors[l]
                                                   if n in prev_layer_cells])

        prev_layer_neighborhood_edges = np.transpose([prev_layer_cell_neighbor_cells, prev_layer_cell_neighbors])
        prev_layer_neighborhood_edges = np.unique(np.sort(prev_layer_neighborhood_edges), axis=0)

        prev_layer_neighborhood_triangles = triangles_from_adjacency_edges(prev_layer_neighborhood_edges)
        prev_layer_neighborhood_property_triangles = prev_layer_neighborhood_triangles[np.all(cell_property.values(prev_layer_neighborhood_triangles), axis=1)]

        prev_layer_property_tetrahedras = np.array([list(t) + [n]
                                                    for t in prev_layer_neighborhood_property_triangles
                                                    for n in layer_cells
                                                    if np.all([n in cell_neighbors[l] for l in t])])

        layer_property_cells = np.unique([t[-1] for t in prev_layer_property_tetrahedras])

        for c in layer_property_cells:
            cell_property[c] = True

        tissue.label._property[property_name] = cell_property

        tissue_cell_binary_property_morphological_operation(tissue, property_name, method='closing', iterations=1, layer_restriction=None)


def tissue_analysis_to_dataframe(tissue, element='cell'):
    if element == 'cell':
        cell_labels = [l for l in tissue.get_label_ids()]

        cell_df = pd.DataFrame({'label':cell_labels})

        for property_name in tissue.label.list_properties():
            cell_df[property_name] = [tissue.label.get_property(property_name)[c]
                                      if c in tissue.label.get_property(property_name).keys()
                                      else np.nan for c in cell_labels]

        return cell_df
