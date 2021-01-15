import time
import bisect

import numpy as np
from scipy.cluster.vq import vq

from timagetk.components.labelled_image import elapsed_time


def topological_elements_extraction3D(img, elem_order=None, use_vq=False):
    """
    Extract the topological elements of order 2 (ie. wall), 1 (ie. wall-edge)
    and 0 (ie. cell vertex) by returning their coordinates grouped by pair,
    triplet and quadruplet of labels.

    Parameters
    ----------
    img : numpy.array
        numpy array representing a labelled image
    elem_order : list, optional
        list of dimensional order of the elements to return, should be in [2, 1, 0]

    Returns
    -------
    topological_elements : dict
        dictionary with topological elements order as key, each containing
        dictionaries of n-uplets as keys and coordinates array as values

    Example
    -------
    >>> import numpy as np
    >>> from timagetk.components.labelled_image import topological_elements_extraction3D
    >>> a = np.array([[2, 2, 2, 3, 3, 3, 3, 3],
                      [2, 2, 2, 3, 3, 3, 3, 3],
                      [2, 2, 2, 2, 3, 3, 3, 3],
                      [2, 2, 2, 2, 3, 3, 3, 3],
                      [2, 2, 2, 2, 3, 3, 3, 3],
                      [4, 4, 4, 4, 4, 4, 4, 4],
                      [4, 4, 4, 4, 4, 4, 4, 4],
                      [4, 4, 4, 4, 4, 4, 4, 4]])
    >>> bkgd_im = np.ones_like(a)
    >>> # Create an image by adding a background and repeat the previous array 6 times as a Z-axis:
    >>> im = np.array([bkgd_im, a, a, a, a, a, a]).transpose(1, 2, 0)
    >>> im.shape
    (8, 8, 7)
    >>> # Extract topological elements coordinates:
    >>> elem = topological_elements_extraction3D(im)
    >>> # Get the cell-vertex coordinates between labels 1, 2, 3 and 4
    >>> elem[0]
    {(1, 2, 3, 4): array([[ 4.5,  3.5,  0.5]])}
    >>> # Get the wall-edge voxel coordinates between labels 1, 2 and 3:
    >>> elem[1][(1, 2, 3)]
    array([[ 0.5,  2.5,  0.5],
           [ 1.5,  2.5,  0.5],
           [ 1.5,  3.5,  0.5],
           [ 2.5,  3.5,  0.5],
           [ 3.5,  3.5,  0.5]])
    >>> # Get the wall voxel coordinates between labels 1 and 4:
    >>> elem[2][(1, 4)]
    array([[ 5.5,  0.5,  0.5],
           [ 5.5,  1.5,  0.5],
           [ 5.5,  2.5,  0.5],
           [ 5.5,  3.5,  0.5],
           [ 5.5,  4.5,  0.5],
           [ 5.5,  5.5,  0.5],
           [ 5.5,  6.5,  0.5],
           [ 6.5,  0.5,  0.5],
           [ 6.5,  1.5,  0.5],
           [ 6.5,  2.5,  0.5],
           [ 6.5,  3.5,  0.5],
           [ 6.5,  4.5,  0.5],
           [ 6.5,  5.5,  0.5],
           [ 6.5,  6.5,  0.5]])
    """

    print("# - Detecting cell topological elements:")
    sh = np.array(img.shape)

    if elem_order is None:
        elem_order = range(3)

    neighborhood_images = {}
    element_coords = {}
    if 0 in elem_order:
        n_nei_vox = 8
        n_elements = (sh[0] - 1) * (sh[1] - 1) * (sh[2] - 1)
        msg ="  - Computing the cell neighborhood of image pointels (n={})..."
        print(msg.format(n_elements))
        start_time = time.time()
        neighborhood_img = []
        for x in np.arange(-1, 1):
            for y in np.arange(-1, 1):
                for z in np.arange(-1, 1):
                    neighborhood_img.append(
                        img[1 + x:sh[0] + x, 1 + y:sh[1] + y, 1 + z:sh[2] + z])

        # - Reshape the neighborhood matrix in a N_voxel x n_nei_vox:
        neighborhood_img = np.sort(np.transpose(neighborhood_img, (1, 2, 3, 0))).reshape((sh - 1).prod(), n_nei_vox)
        print("\tComputed pointel neighborhoods: {}".format(elapsed_time(start_time)))

        print("  - Removing pointels surrounded only by 1 cell...")
        start_time = time.time()
        # - Detect the voxels that are not alone (only neighbors to themself, or same label around):
        non_flat = np.sum(
            neighborhood_img == np.tile(neighborhood_img[:, :1], (1, n_nei_vox)),
            axis=1) != n_nei_vox
        # - Keep only these "non flat" neighborhood:
        neighborhood_img = neighborhood_img[non_flat]

        neighborhood_images[0] = neighborhood_img
        n_pointels = neighborhood_img.shape[0]
        pc = float(n_elements - n_pointels) / n_elements * 100
        print("\tRemoved {}% of the initial pointels: {}".format(round(pc, 3), elapsed_time(start_time)))

        print("  - Creating the associated coordinate array...")
        start_time = time.time()
        # - Create the coordinate matrix associated to each voxels of the neighborhood matrix:
        pointel_coords = np.transpose(
            np.mgrid[0:sh[0] - 1, 0:sh[1] - 1, 0:sh[2] - 1], (1, 2, 3, 0)).reshape(
            (sh - 1).prod(), 3) + 0.5
        # - Keep only these "non flat" coordinates:
        pointel_coords = pointel_coords[non_flat]
        element_coords[0] = pointel_coords
        print("\tComputed {} pointel coordinates: {}".format(len(pointel_coords), elapsed_time(start_time)))

    if 1 in elem_order:
        n_nei_vox = 4
        n_elements = 0
        n_elements += sh[0] * (sh[1] - 1) * (sh[2] - 1)
        n_elements += (sh[0] - 1) * sh[1] * (sh[2] - 1)
        n_elements += (sh[0] - 1) * (sh[1] - 1) * sh[2]
        total = 0
        msg = "  - Computing the cell neighborhood of image linels (n={})..."
        print(msg.format(n_elements))
        start_time = time.time()
        # - Extract x-oriented linels
        x_neighborhood_img = []
        for y in np.arange(-1, 1):
            for z in np.arange(-1, 1):
                x_neighborhood_img.append(img[:, 1 + y:sh[1] + y, 1 + z:sh[2] + z])
        x_neighborhood_img = np.sort(np.transpose(x_neighborhood_img, (1, 2, 3, 0))).reshape(sh[0] * (sh[1] - 1) * (sh[2] - 1), n_nei_vox)
        total += len(x_neighborhood_img)

        non_flat = np.sum(
            x_neighborhood_img == np.tile(x_neighborhood_img[:, :1], (1, n_nei_vox)),
            axis=1) != n_nei_vox
        # - Keep only these "non flat" neighborhood:
        x_neighborhood_img = x_neighborhood_img[non_flat]

        x_linel_coords = np.transpose(
            np.mgrid[0:sh[0], 0:sh[1] - 1, 0:sh[2] - 1], (1, 2, 3, 0)).reshape(
            sh[0] * (sh[1] - 1) * (sh[2] - 1), 3) + np.array([0., 0.5, 0.5])
        # - Keep only these "non flat" coordinates:
        x_linel_coords = x_linel_coords[non_flat]

        # - Extract y-oriented linels
        y_neighborhood_img = []
        for x in np.arange(-1, 1):
            for z in np.arange(-1, 1):
                y_neighborhood_img.append(img[1 + x:sh[0] + x, :, 1 + z:sh[2] + z])
        y_neighborhood_img = np.sort(np.transpose(y_neighborhood_img, (1, 2, 3, 0))).reshape((sh[0] - 1) * sh[1] * (sh[2] - 1), n_nei_vox)
        total += len(y_neighborhood_img)

        non_flat = np.sum(
            y_neighborhood_img == np.tile(y_neighborhood_img[:, :1], (1, n_nei_vox)),
            axis=1) != n_nei_vox
        # - Keep only these "non flat" neighborhood:
        y_neighborhood_img = y_neighborhood_img[non_flat]

        y_linel_coords = np.transpose(
            np.mgrid[0:sh[0] - 1, 0:sh[1], 0:sh[2] - 1], (1, 2, 3, 0)).reshape(
            (sh[0] - 1) * sh[1] * (sh[2] - 1), 3) + np.array([0.5, 0., 0.5])
        # - Keep only these "non flat" coordinates:
        y_linel_coords = y_linel_coords[non_flat]

        # - Extract z-oriented linels
        z_neighborhood_img = []
        for x in np.arange(-1, 1):
            for y in np.arange(-1, 1):
                z_neighborhood_img.append(img[1 + x:sh[0] + x, 1 + y:sh[1] + y, :])
        z_neighborhood_img = np.sort(np.transpose(z_neighborhood_img, (1, 2, 3, 0))).reshape((sh[0] - 1) * (sh[1] - 1) * sh[2], n_nei_vox)
        total += len(z_neighborhood_img)

        non_flat = np.sum(
            z_neighborhood_img == np.tile(z_neighborhood_img[:, :1], (1, n_nei_vox)),
            axis=1) != n_nei_vox
        # - Keep only these "non flat" neighborhood:
        z_neighborhood_img = z_neighborhood_img[non_flat]

        z_linel_coords = np.transpose(
            np.mgrid[0:sh[0] - 1, 0:sh[1] - 1, 0:sh[2]], (1, 2, 3, 0)).reshape(
            (sh[0] - 1) * (sh[1] - 1) * sh[2], 3) + np.array([0.5, 0.5, 0.])
        # - Keep only these "non flat" coordinates:
        z_linel_coords = z_linel_coords[non_flat]
        print("\tComputed linel neighborhoods: {}".format(elapsed_time(start_time)))

        start_time = time.time()
        neighborhood_img = []
        neighborhood_img += list(x_neighborhood_img)
        neighborhood_img += list(y_neighborhood_img)
        neighborhood_img += list(z_neighborhood_img)
        neighborhood_img = np.array(neighborhood_img)

        neighborhood_images[1] = neighborhood_img
        n_linels = neighborhood_img.shape[0]
        pc = float(n_elements - n_linels) / n_elements * 100
        print("\tRemoved {}% of the initial linels: {}".format(round(pc, 3), elapsed_time(start_time)))

        print("  - Creating the associated coordinate array...")
        start_time = time.time()
        linel_coords = []
        linel_coords += list(x_linel_coords)
        linel_coords += list(y_linel_coords)
        linel_coords += list(z_linel_coords)
        linel_coords = np.array(linel_coords)
        element_coords[1] = linel_coords
        print("\tComputed {} linel coordinates: {}".format(len(linel_coords), elapsed_time(start_time)))

    if 2 in elem_order:
        n_nei_vox = 2
        n_elements = 0
        n_elements += sh[0] * sh[1] * (sh[2] - 1)
        n_elements += sh[0] * (sh[1] - 1) * sh[2]
        n_elements += (sh[0] - 1) * sh[1] * sh[2]
        total = 0
        msg = "  - Computing the cell neighborhood of image surfels (n={})..."
        print(msg.format(n_elements))
        start_time = time.time()
        # - Extract xy-oriented surfels
        xy_neighborhood_img = []
        for z in np.arange(-1, 1):
            xy_neighborhood_img.append(img[:, :, 1 + z:sh[2] + z])
        xy_neighborhood_img = np.sort(np.transpose(xy_neighborhood_img, (1, 2, 3, 0))).reshape(sh[0] * sh[1] * (sh[2] - 1), n_nei_vox)
        total += len(xy_neighborhood_img)

        non_flat = np.sum(
            xy_neighborhood_img == np.tile(xy_neighborhood_img[:, :1], (1, n_nei_vox)),
            axis=1) != n_nei_vox
        # - Keep only these "non flat" neighborhood:
        xy_neighborhood_img = xy_neighborhood_img[non_flat]

        xy_surfel_coords = np.transpose(
            np.mgrid[0:sh[0], 0:sh[1], 0:sh[2] - 1], (1, 2, 3, 0)).reshape(
            sh[0] * sh[1] * (sh[2] - 1), 3) + np.array([0., 0., 0.5])
        # - Keep only these "non flat" coordinates:
        xy_surfel_coords = xy_surfel_coords[non_flat]

        # - Extract xz-oriented surfels
        xz_neighborhood_img = []
        for y in np.arange(-1, 1):
            xz_neighborhood_img.append(img[:, 1 + y:sh[1] + y, :])
        xz_neighborhood_img = np.sort(np.transpose(xz_neighborhood_img, (1, 2, 3, 0))).reshape(sh[0] * (sh[1] - 1) * sh[2], n_nei_vox)
        total += len(xz_neighborhood_img)

        non_flat = np.sum(
            xz_neighborhood_img == np.tile(xz_neighborhood_img[:, :1], (1, n_nei_vox)),
            axis=1) != n_nei_vox
        # - Keep only these "non flat" neighborhood:
        xz_neighborhood_img = xz_neighborhood_img[non_flat]

        xz_surfel_coords = np.transpose(
            np.mgrid[0:sh[0], 0:sh[1] - 1, 0:sh[2]], (1, 2, 3, 0)).reshape(
            sh[0] * (sh[1] - 1) * sh[2], 3) + np.array([0., 0.5, 0.])
        # - Keep only these "non flat" coordinates:
        xz_surfel_coords = xz_surfel_coords[non_flat]

        # - Extract yz-oriented surfels
        yz_neighborhood_img = []
        for x in np.arange(-1, 1):
            yz_neighborhood_img.append(img[1 + x:sh[0] + x, :, :])
        yz_neighborhood_img = np.sort(np.transpose(yz_neighborhood_img, (1, 2, 3, 0))).reshape((sh[0] - 1) * sh[1] * sh[2], n_nei_vox)
        total += len(yz_neighborhood_img)

        non_flat = np.sum(
            yz_neighborhood_img == np.tile(yz_neighborhood_img[:, :1], (1, n_nei_vox)),
            axis=1) != n_nei_vox
        # - Keep only these "non flat" neighborhood:
        yz_neighborhood_img = yz_neighborhood_img[non_flat]

        yz_surfel_coords = np.transpose(
            np.mgrid[0:sh[0] - 1, 0:sh[1], 0:sh[2]], (1, 2, 3, 0)).reshape(
            (sh[0] - 1) * sh[1] * sh[2], 3) + np.array([0.5, 0., 0.])
        # - Keep only these "non flat" coordinates:
        yz_surfel_coords = yz_surfel_coords[non_flat]
        print("\tComputed surfel neighborhoods: {}".format(elapsed_time(start_time)))

        start_time = time.time()
        neighborhood_img = []
        neighborhood_img += list(xy_neighborhood_img)
        neighborhood_img += list(xz_neighborhood_img)
        neighborhood_img += list(yz_neighborhood_img)
        neighborhood_img = np.array(neighborhood_img)

        neighborhood_images[2] = neighborhood_img
        n_surfels = neighborhood_img.shape[0]
        pc = float(n_elements - n_surfels) / n_elements * 100
        print("\tRemoved {}% of the initial surfels: {}".format(round(pc, 3), elapsed_time(start_time)))

        print("  - Creating the associated coordinate array...")
        start_time = time.time()
        surfel_coords = []
        surfel_coords += list(xy_surfel_coords)
        surfel_coords += list(xz_surfel_coords)
        surfel_coords += list(yz_surfel_coords)
        surfel_coords = np.array(surfel_coords)

        element_coords[2] = surfel_coords
        print("\tComputed {} surfel coordinates: {}".format(len(surfel_coords), elapsed_time(start_time)))

    # n_nei_vox = 8
    # n_voxels = (sh[0] - 1) * (sh[1] - 1) * (sh[2] - 1)
    # msg ="  - Computing the cell neighborhood of image pointels (n={})..."
    # print(msg.format(n_voxels))
    # start_time = time.time()
    # # - Create the neighborhood matrix of each pointel:
    # neighborhood_img = []
    # for x in np.arange(-1, 1):
    #     for y in np.arange(-1, 1):
    #         for z in np.arange(-1, 1):
    #             neighborhood_img.append(
    #                 img[1 + x:sh[0] + x, 1 + y:sh[1] + y, 1 + z:sh[2] + z])
    #
    # # - Reshape the neighborhood matrix in a N_voxel x n_nei_vox:
    # neighborhood_img = np.sort(
    #     np.transpose(neighborhood_img, (1, 2, 3, 0))).reshape((sh - 1).prod(),
    #                                                           n_nei_vox)
    # print("\tComputed pointel neighborhoods: {}".format(elapsed_time(start_time)))
    #
    # print("  - Removing pointels surrounded only by 1 cell...")
    # start_time = time.time()
    # # - Detect the voxels that are not alone (only neighbors to themself, or same label around):
    # non_flat = np.sum(
    #     neighborhood_img == np.tile(neighborhood_img[:, :1], (1, n_nei_vox)),
    #     axis=1) != n_nei_vox
    # # - Keep only these "non flat" neighborhood:
    # neighborhood_img = neighborhood_img[non_flat]
    #
    # n_voxels_elem = neighborhood_img.shape[0]
    # pc = float(n_voxels - n_voxels_elem) / n_voxels * 100
    # print("\tRemoved {}% of the initial voxel matrix: {}".format(round(pc, 3),elapsed_time(start_time)))
    #
    # print("  - Creating the associated coordinate array...")
    # start_time = time.time()
    # # - Create the coordinate matrix associated to each voxels of the neighborhood matrix:
    # vertex_coords = np.transpose(
    #     np.mgrid[0:sh[0] - 1, 0:sh[1] - 1, 0:sh[2] - 1], (1, 2, 3, 0)).reshape(
    #     (sh - 1).prod(), 3) + 0.5
    # # - Keep only these "non flat" coordinates:
    # vertex_coords = vertex_coords[non_flat]
    # elapsed_time(start_time)
    # print("\tComputed {} pointel coordinates: {}".format(len(vertex_coords),elapsed_time(start_time)))

    # print("  - Computing the pointel neighborhood size...")
    # start_time = time.time()
    # # - Keep the "unique values" in each neighborhood:
    # #neighborhoods = Pool().map(np.unique, neighborhood_img)
    # neighborhoods = list(map(np.unique, neighborhood_img))
    # neighborhoods = np.array(neighborhoods)
    # # - Compute the neighborhood size of each voxel:
    # #neighborhood_size = Pool().map(len, neighborhoods)
    # neighborhood_size = list(map(len, neighborhoods))
    # neighborhood_size = np.array(neighborhood_size)
    # print("\tComputed pointel neighborhood sizes: {}".format(elapsed_time(start_time)))

    # - Separate the voxels depending on the size of their neighborhood:
    #   "wall" is a dimension 2 element with a neighborhood size == 2;
    #   "wall-edge" is a dimension 1 element with a neighborhood size == 3;
    #   "cell-vertex" is a dimension 0 element with a neighborhood size == 4+;
    topological_elements = {}
    element_types = {0: 'pointels', 1: 'linels', 2: 'surfels'}
    for dimension in elem_order:
        try:
            assert dimension in range(3)
        except AssertionError:
            print("Given element orders should be in [0, 1, 2]!")
            continue

        neighborhood_img = neighborhood_images[dimension]
        elem_type = element_types[dimension]

        msg = "  - Creating dictionary of {} coordinates detected as topological elements (n={})..."
        print(msg.format(elem_type, len(neighborhood_img)))

        if dimension < 2:
            print("  -- Computing the {} neighborhood size...".format(elem_type))
            start_time = time.time()
            # - Keep the "unique values" in each neighborhood:
            # neighborhoods = Pool().map(np.unique, neighborhood_img)
            neighborhoods = list(map(np.unique, neighborhood_img))
            neighborhoods = np.array(neighborhoods)
            # - Compute the neighborhood size of each voxel:
            # neighborhood_size = Pool().map(len, neighborhoods)
            neighborhood_size = list(map(len, neighborhoods))
            neighborhood_size = np.array(neighborhood_size)

            # - Make a mask indexing desired 'neighborhood_size':
            dim_mask = neighborhood_size == 4 - dimension
            print("\tComputed {} neighborhood sizes: {}".format(elem_type, elapsed_time(start_time)))
        else:
            # - No need to compute for (non-flat) surfels: always 2 cells
            neighborhoods = np.sort(neighborhood_img)
            neighborhood_size = 2 * np.ones(len(neighborhood_img))
            dim_mask = np.ones(len(neighborhood_img)).astype(bool)

        # - Get all coordinates corresponding to selected 'neighborhood_size':
        element_points = element_coords[dimension][dim_mask]
        # - Get labels list for this given 'neighborhood_size':
        element_cells = np.array([p for p in neighborhoods[dim_mask]], int)

        # - In case of "cell-vertex" try to find 5-neighborhood:
        if (dimension == 0) & ((neighborhood_size >= 5).sum() > 0):
            start_time = time.time()
            # - Make a mask indexing 'neighborhood_size == 5':
            mask_5 = neighborhood_size == 5
            # - Get all coordinates for 'neighborhood_size == 5':
            clique_pointel_points = np.concatenate(
                [(p, p, p, p, p) for p in element_coords[dimension][mask_5]])
            # - Get labels list for 'neighborhood_size == 5':
            clique_pointel_cells = np.concatenate(
                [[np.concatenate([p[:i], p[i + 1:]]) for i in range(5)] for p in
                 neighborhoods[mask_5]]).astype(int)
            # - Add them to the 4-neighborhood arrays of coordinates and labels:
            element_points = np.concatenate(
                [element_points, clique_pointel_points])
            element_cells = np.concatenate([element_cells, clique_pointel_cells])
            msg = "\tAdded {} overconnected {}: {}"
            print(msg.format(len(clique_pointel_cells), elem_type, elapsed_time(start_time)))

        if (dimension == 1) & ((neighborhood_size == 4).sum() > 0):
            start_time = time.time()
            # - Make a mask indexing 'neighborhood_size == 4':
            mask_4 = neighborhood_size == 4
            # - Get all coordinates for 'neighborhood_size == 4':
            clique_linel_points = np.concatenate(
                [(p, p, p, p) for p in element_coords[dimension][mask_4]])
            # - Get labels list for 'neighborhood_size == 4':
            clique_linel_cells = np.concatenate(
                [[np.concatenate([p[:i], p[i + 1:]]) for i in range(4)] for p in
                 neighborhoods[mask_4]]).astype(int)
            # - Add them to the 3-neighborhood arrays of coordinates and labels:
            element_points = np.concatenate(
                [element_points, clique_linel_points])
            element_cells = np.concatenate([element_cells, clique_linel_cells])
            msg = "\tAdded {} overconnected {}: {}"
            print(msg.format(len(clique_linel_cells), elem_type, elapsed_time(start_time)))

        msg = "  -- Sorting {} {} as topological elements of order {}..."
        print(msg.format(len(element_cells), elem_type, dimension))
        start_time = time.time()
        if element_cells != np.array([]):
            # - Remove duplicate of labels n-uplets, with 'n = 4 - dim':
            unique_cell_elements = np.unique(element_cells, axis=0)
            # - ??
            if use_vq:
                match_start_time = time.time()
                element_matching = vq(element_cells, unique_cell_elements)[0]
                # - Make a dictionary of all {(n-uplet) : np.array(coordinates)}:

                match_start_time = time.time()
                topological_elements[dimension] = dict(
                    zip([tuple(e) for e in unique_cell_elements],
                        [element_points[element_matching == e] for e, _ in
                         enumerate(unique_cell_elements)]))
            else:
                match_start_time = time.time()
                element_sorting = np.lexsort(element_cells.T[::-1])
                sorted_element_cells = element_cells[element_sorting]
                sorted_element_points = element_points[element_sorting]

                match_start_time = time.time()
                topological_elements[dimension] = {}
                for element in unique_cell_elements:
                    i_left = 0
                    i_right = len(sorted_element_cells)
                    for k in range(4 - dimension):
                        i_left_k = bisect.bisect_left(sorted_element_cells[i_left:i_right, k], element[k])
                        i_right_k = bisect.bisect_right(sorted_element_cells[i_left:i_right, k], element[k])
                        i_right = i_left + i_right_k
                        i_left = i_left + i_left_k
                    topological_elements[dimension][tuple(element)] = sorted_element_points[i_left:i_right]
        else:
            msg = "WARNING: Could not find topological elements of order {}!"
            print(msg.format(dimension))
            topological_elements[dimension] = None
        msg = "\tSorted topological elements of order {} ({}): {}"
        print(msg.format(dimension, elem_type, elapsed_time(start_time)))

    return topological_elements
