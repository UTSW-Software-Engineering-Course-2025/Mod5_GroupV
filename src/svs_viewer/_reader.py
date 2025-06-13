# %%
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
from dask import delayed

from .utils import rgb2hed, threshold_nuclei_dask
from .stardist_he import predict_nucleus
# Import LabelColormap for dynamic colormap assignment

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
    # Fix: Check if path is a list using isinstance(path, list)
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

# We will store the OpenSlide object and DeepZoomGenerator outside
# the reader_function scope, or pass them appropriately to delayed functions.
# For simplicity in this plugin, we'll open it once in reader_function
# and pass the necessary information down.

def _svs2dask(svs_path: str, openslide_object: openslide.OpenSlide, dzi_generator: openslide.deepzoom.DeepZoomGenerator) -> list[da.Array]:
    """Convert an svs image to a dask array with delayed loading. 
    This is the version that pads the arrays to avoid cropping the edges.
    Left in here just for future bugfixes, but it has problems when we try to add layers in the ui.

    Parameters:
    -----------
        svs_path : str
            Path to the svs file (for context, though not directly used for opening in this function).
        openslide_object : openslide.OpenSlide
            The already opened OpenSlide object.
        dzi_generator : openslide.deepzoom.DeepZoomGenerator
            The already created DeepZoomGenerator object.
    Returns:
    --------
        List[da.Array]:
            List of lazy loaded arrays. Each list index is a zoom level.
    """
    @delayed
    def delayed_padded_tile(dzi, level, col, row, tile_size=256, overlap=0):
        target_h = tile_size + 2 * overlap
        target_w = tile_size + 2 * overlap
        tile = dzi.get_tile(level, (col, row)).convert("RGB")
        tile_np = np.array(tile)
        h, w = tile_np.shape[:2]

        padded = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        crop_h = min(h, target_h)
        crop_w = min(w, target_w)
        padded[:crop_h, :crop_w, :] = tile_np[:crop_h, :crop_w, :]
        return padded

    def get_padded_tile_array(dzi, level, col, row, tile_size=256, overlap=0):
        target_h = tile_size + 2 * overlap
        target_w = tile_size + 2 * overlap
        delayed_tile = delayed_padded_tile(dzi, level, col, row, tile_size, overlap)
        return da.from_delayed(delayed_tile, shape=(target_h, target_w, 3), dtype=np.uint8)

    opr = openslide_object
    dzi = dzi_generator

    n_levels = len(dzi.level_dimensions)
    n_t_x = [t[0] for t in dzi.level_tiles]
    n_t_y = [t[1] for t in dzi.level_tiles]

    # Dask array for each level, concatenated from tiles
    arr = []
    for level in range(n_levels):
        rows_of_tiles = []
        for row in range(n_t_y[level]):
            cols_of_tiles = []
            for col in range(n_t_x[level]):
                # Determine shape of an individual tile for from_delayed
                # get_tile_dimensions returns (width, height)
                tile_dims_wh = dzi.get_tile_dimensions(level, (col, row))
                # Dask array expects (height, width, channels)
                tile_shape_hwc = (tile_dims_wh[1], tile_dims_wh[0], 3)

                delayed_tile = get_padded_tile_array(dzi, level, col, row)
                    
                cols_of_tiles.append(delayed_tile)
            if cols_of_tiles: # Only concatenate if there are tiles
                rows_of_tiles.append(da.concatenate(cols_of_tiles, axis=1))
        if rows_of_tiles: # Only concatenate if there are rows
            arr.append(da.concatenate(rows_of_tiles, axis=0))
        else: # Handle case of empty level (though unlikely with DeepZoomGenerator)
            arr.append(da.zeros((0,0,3), dtype=np.uint8)) 

    arr.reverse() # Napari expects highest resolution at index 0
    return arr

def svs2dask(svs_path: str, openslide_object: openslide.OpenSlide, dzi_generator: openslide.deepzoom.DeepZoomGenerator) -> list[da.Array]:
    """Convert an svs image to a dask array with delayed loading.

    Parameters:
    -----------
        svs_path : str
            Path to the svs file (for context, though not directly used for opening in this function).
        openslide_object : openslide.OpenSlide
            The already opened OpenSlide object.
        dzi_generator : openslide.deepzoom.DeepZoomGenerator
            The already created DeepZoomGenerator object.
    Returns:
    --------
        List[da.Array]:
            List of lazy loaded arrays. Each list index is a zoom level.
    """
    opr = openslide_object
    dzi = dzi_generator

    n_levels = len(dzi.level_dimensions)
    n_t_x = [t[0] for t in dzi.level_tiles]
    n_t_y = [t[1] for t in dzi.level_tiles]

    @dask.delayed(pure=True)
    def get_tile(level, c, r, generator_ref): # Pass the generator as a reference
        # OpenSlide get_tile returns a PIL Image. Convert to numpy array.
        # PIL Image is (width, height), so array is (height, width, channels) by default.
        # Assuming RGB (3 channels). OpenSlide guarantees RGB for its tiles.
        tile = generator_ref.get_tile(level, (c, r))
        return np.array(tile)

    # Dask array for each level, concatenated from tiles
    arr = []
    for level in range(n_levels):
        rows_of_tiles = []
        for row in range(n_t_y[level] - (1 if n_t_y[level] > 1 else 0)):
            cols_of_tiles = []
            for col in range(n_t_x[level] - (1 if n_t_x[level] > 1 else 0)):
                # Determine shape of an individual tile for from_delayed
                # get_tile_dimensions returns (width, height)
                tile_dims_wh = dzi.get_tile_dimensions(level, (col, row))
                # Dask array expects (height, width, channels)
                tile_shape_hwc = (tile_dims_wh[1], tile_dims_wh[0], 3)

                delayed_tile = da.from_delayed(
                    get_tile(level, col, row, dzi), # Pass dzi_generator here
                    shape=tile_shape_hwc,
                    dtype=np.uint8
                )
                cols_of_tiles.append(delayed_tile)
            if cols_of_tiles: # Only concatenate if there are tiles
                rows_of_tiles.append(da.concatenate(cols_of_tiles, axis=1))
        if rows_of_tiles: # Only concatenate if there are rows
            arr.append(da.concatenate(rows_of_tiles, axis=0))
        else: # Handle case of empty level (though unlikely with DeepZoomGenerator)
            arr.append(da.zeros((0,0,3), dtype=np.uint8)) # Or handle appropriately

    arr.reverse() # Napari expects highest resolution at index 0
    return arr

def nuclei_label_pyramid(
    image_pyramid: list[da.Array],
    openslide_object: openslide.OpenSlide,
    dzi_generator: openslide.deepzoom.DeepZoomGenerator,
    labels: da.Array 
) -> list[da.Array]:
    """
    Creates a Dask-based label pyramid with nuclei identified and labelled
    but only at the highest resolution level.

    Parameters:
    -----------
    image_pyramid : list of da.Array
        The original multiscale image pyramid (from svs2dask). (Kept for original signature)
    openslide_object : openslide.OpenSlide
        The already opened OpenSlide object. (Kept for original signature)
    dzi_generator : openslide.deepzoom.DeepZoomGenerator
        The DeepZoomGenerator object used to get level dimensions.
    labels : da.Array
        A Dask array containing the segmented nuclei labels at the highest resolution.

    Returns:
    --------
    list of da.Array
        A list representing the label pyramid. Lower resolution levels will be
        empty Dask arrays (containing background value 0), and the highest
        resolution level will contain the segmented nuclei labels.
    """
    dzi = dzi_generator
    n_levels = len(dzi.level_dimensions)
    label_pyramid = [None] * n_levels

    # Assign the pre-computed Dask array with the segmentation result to the highest resolution level
    label_pyramid[0] = labels

    # For lower resolution levels, create Dask arrays filled with zeros (background)
    for i in range(1, n_levels):
        original_level_idx = (n_levels - 1) - i
        level_dims_wh = dzi.level_dimensions[original_level_idx]
        label_pyramid[i] = da.zeros(
            (level_dims_wh[1], level_dims_wh[0]), # (height, width)
            dtype=labels.dtype, 
            chunks=(256, 256) 
        )
    return label_pyramid


### dealing with stardist (kept as is)
def predict_on_tile(image_tile, prob_thresh=0.8, nms_thresh=0.2):
    # This function should expect a numpy array chunk
    # and return a numpy array of labels for that chunk.
    labels, _ = predict_nucleus(image_tile, prob_thresh=prob_thresh, nms_thresh=nms_thresh)
    labels = labels.astype(np.uint16)
    labels = np.nan_to_num(labels, nan=0, posinf=0, neginf=0)
    labels[labels > 0] = 255 # Binarize if that's your goal (nuclei vs. background) -- typically for visualization
    return labels

def get_nuclei_labels_from_stardist(
    image_pyramid: list[da.Array],
    **kwargs
) -> da.Array:
    """
    Performs nuclei segmentation using StarDist on the highest resolution image
    from the Dask image pyramid and returns a Dask array of the segmented labels.
    """
    # The highest resolution image is at index 0 in the napari-ordered image_pyramid
    highest_res_image_dask = image_pyramid[0]

    labels = highest_res_image_dask.map_blocks(
        predict_on_tile,
        dtype=np.uint16,
        chunks=(256, 256),
        drop_axis=2,
        **kwargs)

    return labels

### NEW: HED Thresholding functions
def get_nuclei_labels_from_hed_threshold(
    image_pyramid: list[da.Array]
) -> da.Array:
    """
    Performs nuclei segmentation using HED deconvolution and thresholding
    on the highest resolution image from the Dask image pyramid.
    """
    highest_res_image_dask = image_pyramid[0]

    # Ensure it's RGB by slicing if it happens to be RGBA
    if highest_res_image_dask.shape[-1] == 4:
        highest_res_image_dask = highest_res_image_dask[..., :3]

    hed_image = rgb2hed(highest_res_image_dask)
    thresholded_labels = threshold_nuclei_dask(hed_image)

    return thresholded_labels.astype(np.uint8) # Ensure binary 0/1 output as uint8

# Main reader function
def reader_function(path, label_method='StarDist'): # Default method remains StarDist
    """Take a path or list of paths and return a list of LayerData tuples.

    Readers are expected to return data as a list of tuples, where each tuple
    is (data, [add_kwargs, [layer_type]]), "add_kwargs" and "layer_type" are
    both optional.

    Parameters
    ----------
    path : str or list of str
        Path to file, or list of paths.

    label_method: str. Default: 'StarDist'
        Which method to use to generate nuclei annotations. Can be 'StarDist' or 'hed_threshold'.

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
    if not isinstance(path, str):
        return None

    opr = openslide.open_slide(path)
    dzi = openslide.deepzoom.DeepZoomGenerator(
        opr,
        tile_size=256,
        overlap=0,
        limit_bounds=True
    )

    image_arrays = svs2dask(path, opr, dzi)

    image_add_kwargs = {
        "contrast_limits": [0,255],
        "multiscale": True,
        "name": "Original Image"
    }
    
    # Initialize labels and their kwargs
    highest_res_computed_labels = None
    label_add_kwargs = {
        "multiscale": True,
        "opacity": 0.8 # Default opacity for labels
    }

    if label_method != None:
        if label_method == 'StarDist':
            # Get the segmented nuclei labels at the highest resolution
            highest_res_labels = get_nuclei_labels_from_stardist(image_arrays, prob_thresh=0.8, nms_thresh=0.2)
            label_add_kwargs["name"] = "Nuclei Labels"
        elif label_method == 'hed_threshold':
            highest_res_labels = get_nuclei_labels_from_hed_threshold(image_arrays)
            
            # Colormap for binary HED threshold labels (0 is background, 1 is nuclei)
            nuclei_label_colormap = {
                0: [1.0, 1.0, 1.0, 0.0],  # Label 0 (background): Transparent black
                1: [0.0, 1.0, 0.5, 0.8],  # Label 1 (nuclei): Semi-transparent green (adjust R,G,B,A as desired)
            }
            label_add_kwargs["name"] = "Nuclei Labels (HED Threshold)"
            label_add_kwargs["colormap"] = nuclei_label_colormap
        else:
            # If no valid label_method is provided, only return the image
            print(f"Warning: Unsupported label_method '{label_method}'. Only original image will be loaded.")
            return [(image_arrays, image_add_kwargs, "image")]

        # Create the nuclei label pyramid using the pre-computed highest-resolution labels
        label_arrays = nuclei_label_pyramid(image_arrays, opr, dzi, highest_res_labels)

        # Important: Do NOT close opr here. The Dask arrays still hold references
        # that will trigger computations later. Napari manages the lifecycle of
        # data objects provided by readers.
        return [
            (image_arrays, image_add_kwargs, "image"),
            (label_arrays, label_add_kwargs, "labels")]
    else:
        # Fallback if somehow label_method was processed but labels weren't generated
        return [(image_arrays, image_add_kwargs, "image")]

# # %%
# %%
