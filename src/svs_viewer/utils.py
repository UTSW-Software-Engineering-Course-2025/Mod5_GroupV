import dask.array as da
import numpy as np
rgb_from_hed = np.array([
    [0.65, 0.70, 0.29],
    [0.07, 0.99, 0.11], 
    [0.27, 0.57, 0.78]
])
hed_from_rgb = np.linalg.inv(rgb_from_hed)

def rgb2hed(rgb):
    if rgb.ndim != 3:
        return 
    rgb = rgb.astype(np.float32) / 255.0 # Convert to [0,1] float image
    da.maximum(rgb, 1e-6, out=rgb) # Remove 0s to prevent div by zero
    log_adjust = da.log(1e-6) 
    stains = (da.log(rgb) / log_adjust) @ hed_from_rgb
    da.maximum(stains, 0, out=stains) # clip zeros
    return stains

def threshold_nuclei_dask(hed_image_dask_array,
                           hematoxylin_min_threshold=0.01, # Renamed for clarity: minimum H value
                           eosin_max_threshold=None,        # Maximum E value to be considered nucleus
                           dab_max_threshold=None):         # Maximum D value to be considered nucleus
    """
    Thresholds the Hematoxylin channel and optionally filters by Eosin/DAB channels
    of an HED Dask array to identify nuclei.
    """
    H = hed_image_dask_array[:, :, 0]  # Hematoxylin
    E = hed_image_dask_array[:, :, 1]  # Eosin
    D = hed_image_dask_array[:, :, 2]  # DAB

    # Start with the primary Hematoxylin condition
    # Using H > hematoxylin_min_threshold for positive concentration values
    nuclei_condition = (H > hematoxylin_min_threshold)

    # Apply additional filtering conditions based on max thresholds for Eosin and DAB
    if eosin_max_threshold is not None:
        nuclei_condition = nuclei_condition & (E < eosin_max_threshold)

    if dab_max_threshold is not None:
        nuclei_condition = nuclei_condition & (D < dab_max_threshold)

    nuclei_mask = nuclei_condition.astype(np.uint8)
    return nuclei_mask