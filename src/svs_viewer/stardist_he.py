# %%
import numpy as np
from stardist.models import StarDist2D
from stardist.plot import render_label
from csbdeep.utils import normalize
from skimage.color import rgb2gray
import dask.array as da
import warnings
warnings.filterwarnings('ignore')

STARDIST_HE_MODEL = StarDist2D.from_pretrained('2D_versatile_he')

def predict_nucleus(image, model_name='2D_versatile_he', prob_thresh=0.8, nms_thresh=0.2):
    """
    Performs nuclei segmentation using StarDist.

    Parameters:
    -----------
    image : da.Array or np.array
        The original image.
    model_name : str. Default: '2D_versatile_he'
        Which stardist model to use.
    prob_thresh : float. Default: 0.6
        Probability threshold for StarDist.
    nms_thresh : float. Default: 0.4
        Non-maximum suppression (boundary detection) threshold for StarDist.

    Returns:
    --------
    np.array
        A numpy array representing the segmented nuclei labels.
    """
    if model_name != '2D_versatile_he':
        model = StarDist2D.from_pretrained(model_name)
    else:
        model = STARDIST_HE_MODEL

    # print(f"Original image shape before processing: {image.shape}")

    if isinstance(image, da.Array):
        image = image.compute()

    # Ensure the image is 2D or 2D with a single channel for StarDist2D
    # If the image is (H, W, 1), squeeze the last dimension to make it (H, W)
    if image.ndim == 3 and image.shape[-1] == 1:
        image = np.squeeze(image, axis=-1)
        # print(f"Image shape adjusted to (H, W): {image.shape}")
    elif image.ndim == 3 and image.shape[-1] in [3, 4]: # Handle RGB or RGBA images
        pass
    elif image.ndim == 2: 
        pass
    else:
        print(f"Warning: Unexpected image dimensionality {image.ndim}. StarDist2D expects 2D (H, W) or 3D (H, W, C).")
        return None   

    labels, _ = model.predict_instances(normalize(image), prob_thresh=prob_thresh, nms_thresh=nms_thresh)
    # full_results = render_label(labels, img=image)
    return labels, image


