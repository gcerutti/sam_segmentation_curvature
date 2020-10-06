import numpy as np
import scipy.ndimage as nd
import pandas as pd

from cellcomplex.utils import array_dict


def image_property_morphological_operation(p_img, property_name, method='erosion', iterations=1, layer_restriction=1):
    labels = list(p_img.labels)
    if layer_restriction is not None:
        layer_labels = [l for l in labels if p_img.image_property('layer')[l]==layer_restriction]
    else:
        layer_labels = labels
    
    label_neighbors = [n for c in labels for n in p_img._image_graph.neighbors(c) if n in layer_labels]
    label_neighbor_labels = [c for c in labels for n in p_img._image_graph.neighbors(c) if n in layer_labels]
    
    for iteration in range(iterations):
        label_neighbor_properties = p_img.image_property(property_name).values(labels+label_neighbors)
        if method == 'erosion':
            morpho_property = nd.minimum(label_neighbor_properties!=0, labels+label_neighbor_labels,index=labels)
        elif method == 'dilation':
            morpho_property = nd.maximum(label_neighbor_properties!=0, labels+label_neighbor_labels,index=labels)
        elif method == 'opening':
            morpho_property = nd.minimum(label_neighbor_properties!=0, labels+label_neighbor_labels, index=labels)
            label_neighbor_properties = array_dict(morpho_property,keys=labels).values(labels+label_neighbors)
            morpho_property = nd.maximum(label_neighbor_properties!=0, labels+label_neighbor_labels, index=labels)
        elif method == 'closing':
            morpho_property = nd.maximum(label_neighbor_properties!=0, labels+label_neighbor_labels, index=labels)
            label_neighbor_properties = array_dict(morpho_property,keys=labels).values(labels+label_neighbors)
            morpho_property = nd.minimum(label_neighbor_properties!=0, labels+label_neighbor_labels, index=labels)
        else:
            morpho_property = p_img.image_property(property_name).values(labels)
        
        if layer_restriction is not None:
            morpho_property *= p_img.image_property('layer').values(labels) == layer_restriction
        p_img.update_image_property(property_name,dict(zip(labels,morpho_property)))

        
def property_image_to_dataframe(p_img, labels=None):

    cell_labels = list(p_img.labels)
    if labels is not None:
        labels = list(set(cell_labels) & set(list(labels)))
    else:
        labels = cell_labels

    dataframe = pd.DataFrame()
    dataframe['id'] = np.array(list(labels))
    dataframe['label'] = np.array(list(labels))

    for property_name in p_img.image_property_names():
        if np.array(p_img.image_property(property_name).values()[0]).ndim == 0:
            #print "  --> Adding column ",property_name
            dataframe[property_name] = np.array([p_img.image_property(property_name)[v] for v in labels])
        elif property_name == 'barycenter':
            for i, axis in enumerate(['x','y','z']):
                dataframe[property_name+"_"+axis] = np.array([p_img.image_property(property_name)[v][i] for v in labels])

    dataframe = dataframe.set_index('id')
    dataframe.index.name = None

    return dataframe
