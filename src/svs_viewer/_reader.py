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

def svs2dask(path: str) -> list[da.Array]:
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
        tile_size=256,
        overlap=0,
        limit_bounds=True
    )
    n_levels = len(dzi.level_dimensions)
    n_t_x = [t[0] for t in dzi.level_tiles]
    n_t_y = [t[1] for t in dzi.level_tiles]

    @dask.delayed(pure=True)
    def get_tile(level, c, r):
        tile = dzi.get_tile(level, (c, r))
        return np.array(tile).transpose((1,0,2))

    arr = [da.concatenate([
        da.concatenate([
            da.from_delayed(
                get_tile(level, col, row), 
                shape=dzi.get_tile_dimensions(level, (col, row))+(3,),
                dtype=np.uint8
                )
                for row in range(n_t_y[level] - (1 if n_t_y[level] > 1 else 0))
        ], axis=1)
        for col in range(n_t_x[level] - (1 if n_t_x[level] > 1 else 0))
    ]) for level in range(n_levels)]
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
    arrays = svs2dask(path)

    # optional kwargs for the corresponding viewer.add_* method
    add_kwargs = {
        "contrast_limits": [0,255],
        "multiscale": True,
    }

    return [(arrays, add_kwargs, "image")]
