"""
This module is an example of a barebones numpy reader plugin for napari.

It implements the Reader specification, but your plugin may choose to
implement multiple readers or even other plugin contributions. see:
https://napari.org/stable/plugins/building_a_plugin/guides.html#readers
"""
import numpy as np
import openslide
import openslide.deepzoom
import dask
import dask.array as da


def napari_get_reader(path):
    """A basic implementation of a Reader contribution.

    Parameters
    ----------
    path : str or list of str
        Path to file, or list of paths.

    Returns
    -------
    function or None
        If the path is a recognized format, return a function that accepts the
        same path or list of paths, and returns a list of layer data tuples.
    """
    if isinstance(path, list):
        # reader plugins may be handed single path, or a list of paths.
        # if it is a list, it is assumed to be an image stack...
        # so we are only going to look at the first file.
        path = path[0]

    # if we know we cannot read the file, we immediately return None.
    if not path.endswith(".svs"):
        return None

    # otherwise we return the *function* that can read ``path``.
    return reader_function

def svs2dask(path, tile_size = 256, overlap = 2: str) -> list[da.Array]:
    """Convert an svs image to a dask array with delayed loading.

    Parameters:
    -----------
        path : str 
            Path to the svs file to load and convert
    Returns:
    --------
        List[da.Array]:
            List of lazy loaded arrays. Each list index is a zoom level. 
    """

    opr = openslide.open_slide(path)
    dzi = openslide.deepzoom.DeepZoomGenerator(
        opr, 
        tile_size=tile_size,
        overlap=overlap,
        limit_bounds=False
    )
    n_levels = len(dzi.level_dimensions)
    n_t_x = [t[0] for t in dzi.level_tiles]
    n_t_y = [t[1] for t in dzi.level_tiles]


    @dask.delayed(pure=True)
    def delayed_padded_tile(dzi, level, col, row, tile_size=256, overlap=2):
        target_h = tile_size + overlap
        target_w = tile_size + overlap

        tile = dzi.get_tile(level, (col, row)).convert("RGB")
        tile_np = da.array(tile)
        h, w = tile_np.shape[:2]

        padded = da.zeros((target_h, target_w, 3), dtype=np.uint8)

        crop_h = min(h, target_h)
        crop_w = min(w, target_w)

        padded[:crop_h, :crop_w, :] = tile_np[:crop_h, :crop_w, :]

        return padded

    def get_padded_tile_array(dzi, level, col, row, tile_size=256, overlap=2):
        target_h = tile_size + overlap 
        target_w = tile_size + overlap 

        return da.from_delayed(
            delayed_padded_tile(dzi, level, col, row, tile_size, overlap),
            shape=(target_h, target_w, 3),
            dtype=np.uint8
        )


    arr = []

    for level in range(n_levels):
    
        cols, rows = n_t_x[level], n_t_y[level]

        row_blocks = []
        for row in range(rows):
            row_tiles = [
                get_padded_tile_array(dzi, level, col, row)
                for col in range(cols)
            ]
            row_block = da.concatenate(row_tiles, axis=1)
            row_blocks.append(row_block)

        level_array = da.concatenate(row_blocks, axis=0)
        arr.append(level_array)

    arr.reverse()


    return arr

def reader_function(path):
    """Take a path or list of paths and return a list of LayerData tuples.

    Readers are expected to return data as a list of tuples, where each tuple
    is (data, [add_kwargs, [layer_type]]), "add_kwargs" and "layer_type" are
    both optional.

    Parameters
    ----------
    path : str or list of str
        Path to file, or list of paths.

    Returns
    -------
    layer_data : list of tuples
        A list of LayerData tuples where each tuple in the list contains
        (data, metadata, layer_type), where data is a numpy array, metadata is
        a dict of keyword arguments for the corresponding viewer.add_* method
        in napari, and layer_type is a lower-case string naming the type of
        layer. Both "meta", and "layer_type" are optional. napari will
        default to layer_type=="image" if not provided
    """
    if type(path) is not str:
        return None
    # load all files into array
    try:
        arrays = svs2dask(path)
    except Exception as e:
        print(e)
        return [([np.array([1,1,1])], {})]

    # optional kwargs for the corresponding viewer.add_* method
    add_kwargs = {
        "contrast_limits": [0,255],
        "multiscale": True,
    }

    layer_type = "image"  # optional, default is "image"
    return [(arrays, add_kwargs, layer_type)]
