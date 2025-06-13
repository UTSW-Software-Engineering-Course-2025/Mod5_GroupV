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

# from .utils import rgb2hed, threshold_nuclei_dask
from .stardist_he import predict_nucleus

# %%
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
    Creates a Dask-based label pyramid with nuclei identifed and labelled
    but only at the highest resolution level.

    Parameters:
    -----------
    image_pyramid : list of da.Array
        The original multiscale image pyramid (from svs2dask).
    openslide_object : openslide.OpenSlide
        The already opened OpenSlide object (not directly used in this function,
        but kept for consistency with reader_function signature).
    dzi_generator : openslide.deepzoom.DeepZoomGenerator
        The already created DeepZoomGenerator object.
    labels : da.Array
        A Dask array containing the segmented nuclei labels at the highest resolution.

    Returns:
    --------
    list of da.Array
        A list representing the label pyramid. Lower resolution levels will be
        empty Dask arrays (containing background value 0), and the highest
        resolution level will contain the StarDist-segmented nuclei labels.
    """
    dzi = dzi_generator
    n_levels = len(dzi.level_dimensions)
    label_pyramid = [None] * n_levels

    # Assign the pre-computed Dask array with the segmentation result to the highest resolution level
    label_pyramid[0] = labels

    # For lower resolution levels, create Dask arrays filled with zeros (background)
    # The image_pyramid is reversed (napari expects highest resolution at index 0)
    # so image_pyramid[i] corresponds to original_level (n_levels - 1) - i
    for i in range(1, n_levels):
        original_level_idx = (n_levels - 1) - i
        level_dims_wh = dzi.level_dimensions[original_level_idx]
        # Labels are 2D, so shape is (height, width)
        label_pyramid[i] = da.zeros(
            (level_dims_wh[1], level_dims_wh[0]), # (height, width)
            dtype=np.uint16, # Labels are typically unsigned integers
            chunks=(256, 256) # Use chunks appropriate for deepzoom tiles
        )
    return label_pyramid


def reader_function(path, label_method='StarDist'):
    """Take a path or list of paths and return a list of LayerData tuples.

    Readers are expected to return data as a list of tuples, where each tuple
    is (data, [add_kwargs, [layer_type]]), "add_kwargs" and "layer_type" are
    both optional.

    Parameters
    ----------
    path : str or list of str
        Path to file, or list of paths.

    label_method: str. Default: 'StarDist'
        Which method to use to generate neclei annoatations.

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
        # This reader expects a single path, not a list of paths
        return None

    # Open the slide once and create the DeepZoomGenerator
    # This ensures the OpenSlide object remains open for all Dask computations
    opr = openslide.open_slide(path)
    dzi = openslide.deepzoom.DeepZoomGenerator(
        opr,
        tile_size=256,
        overlap=0,
        limit_bounds=True
    )

    # Load the original SVS image as a Dask array pyramid
    # Pass the opened OpenSlide object and DeepZoomGenerator
    image_arrays = svs2dask(path, opr, dzi)

    # Optional kwargs for the original image layer
    image_add_kwargs = {
        "contrast_limits": [0,255],
        "multiscale": True,
        "name": "Original Image"
    }
    
    # Custom colormap for labels: 0 for transparent background, 1 for nuclei
    # custom_labels_colormap_dict = {
    #     0: [1.0, 1.0, 1.0, 0.0],  # Label 0 (background): transparent 
    #     1: [0.0, 0.0, 0.0, 1.0],  # Label 1 (nuclei): black
    # }

    # Optional kwargs for the labels layer
    label_add_kwargs = {
        "multiscale": True,
        "name": "Nuclei Labels",
        # "colormap": custom_labels_colormap_dict,
        "opacity": 0.8 # Adjust opacity as needed
    }

    if label_method != None:
        if label_method == 'StarDist':
            # Get the segmented nuclei labels at the highest resolution
            highest_res_labels = get_nuclei_labels_from_stardist(image_arrays, prob_thresh=0.8, nms_thresh=0.2)
        # Create the nuclei label pyramid using the pre-computed highest-resolution labels
        label_arrays = nuclei_label_pyramid(image_arrays, opr, dzi, highest_res_labels)

        # Important: Do NOT close opr here. The Dask arrays still hold references
        # that will trigger computations later. Napari manages the lifecycle of
        # data objects provided by readers.
        return [
            (image_arrays, image_add_kwargs, "image"),
            (label_arrays, label_add_kwargs, "image")]
    else:
        return [(image_arrays, image_add_kwargs, "image")]

    
### dealing with stardist
def predict_on_tile(image_tile, prob_thresh=0.8, nms_thresh=0.2):
    # This function should expect a numpy array chunk
    # and return a numpy array of labels for that chunk.

    labels, _ = predict_nucleus(image_tile, prob_thresh=prob_thresh, nms_thresh=nms_thresh)
    labels = labels.astype(np.uint16)
    labels = np.nan_to_num(labels, nan=0, posinf=0, neginf=0)
    labels[labels > 0] = 255 # Binarize if that's your goal (nuclei vs. background)
    return labels

def get_nuclei_labels_from_stardist(
    image_pyramid: list[da.Array],
    **kwargs
) -> da.Array:
    """
    Performs nuclei segmentation using StarDist on the highest resolution image
    from the Dask image pyramid and returns a Dask array of the segmented labels.

    Parameters:
    -----------
    image_pyramid : list of da.Array
        The original multiscale image pyramid (from svs2dask).

    Returns:
    --------
    da.Array
        A Dask array representing the segmented nuclei labels at the highest resolution.
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





# def thresholded_label_pyramid(
#     svs_path: str,
#     image_pyramid: list[da.Array],
#     openslide_object: openslide.OpenSlide,
#     dzi_generator: openslide.deepzoom.DeepZoomGenerator,
#     threshold_value: float = 0.001
# ) -> list[da.Array]:
#     """
#     Creates a Dask-based label pyramid where thresholding is applied only
#     at the highest resolution level.

#     Parameters:
#     -----------
#     svs_path : str
#         Path to the SVS file, used to get level dimensions.
#     image_pyramid : list of da.Array
#         The original multiscale image pyramid (from svs2dask).
#     openslide_object : openslide.OpenSlide
#         The already opened OpenSlide object.
#     dzi_generator : openslide.deepzoom.DeepZoomGenerator
#         The already created DeepZoomGenerator object.
#     threshold_value : float
#         The threshold value for identifying nuclei.

#     Returns:
#     --------
#     list of da.Array
#         A list representing the label pyramid. Lower resolution levels will be
#         empty Dask arrays, and the highest resolution level will contain the
#         thresholded Dask array.
#     """
#     opr = openslide_object
#     dzi = dzi_generator

#     n_levels = len(dzi.level_dimensions)
#     label_pyramid = [None] * n_levels # Initialize the pyramid list

#     # The highest resolution level is at index 0 in the reversed image_pyramid
#     highest_res_image_dask = image_pyramid[0]

#     # Apply rgb2hed and then threshold_nuclei_dask to the highest resolution
#     # OpenSlide tiles are typically RGB, so we don't strictly need to slice for 4 channels
#     # But keeping it robust in case of future changes or unusual SVS files
#     if highest_res_image_dask.shape[-1] == 4:
#         highest_res_image_dask = highest_res_image_dask[..., :3]

#     hed_image = rgb2hed(highest_res_image_dask)
#     thresholded_labels = threshold_nuclei_dask(hed_image, threshold_value=threshold_value)

#     # Assign the computed Dask array to the highest resolution level
#     label_pyramid[0] = thresholded_labels

#     # For lower resolution levels, create empty dask arrays of the correct shape (2D for labels)
#     # OpenSlide's dzi.level_dimensions are (width, height) at each level
#     # Since image_pyramid is reversed, highest resolution is at index 0 of the original
#     # dzi levels. So the lower resolutions correspond to indices 1 to n_levels-1 of dzi.level_dimensions
#     # in reverse order.
#     # We need to map the image_pyramid index to the original OpenSlide level index.
#     # image_pyramid[i] corresponds to original_level = (n_levels - 1) - i
#     for i in range(1, n_levels):
#         original_level_idx = (n_levels - 1) - i
#         level_dims_wh = dzi.level_dimensions[original_level_idx]
#         # Labels are 2D, so shape is (height, width)
#         label_pyramid[i] = da.zeros(
#             (level_dims_wh[1], level_dims_wh[0]), # (height, width)
#             dtype=np.uint8,
#             chunks=(256, 256) # Use chunks appropriate for deepzoom tiles
#         )
#     return label_pyramid



# # %%
