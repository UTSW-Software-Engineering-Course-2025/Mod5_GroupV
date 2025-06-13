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

def threshold_nuclei_dask(hed_image_dask_array, threshold_value=0.001):
    """
    Thresholds the Hematoxylin channel of an HED Dask array to identify nuclei.
    """
    hematoxylin_channel = hed_image_dask_array[:, :, 0] # Assuming Hematoxylin is the first channel
    nuclei_mask = (hematoxylin_channel > threshold_value).astype(np.uint8)
    return nuclei_mask