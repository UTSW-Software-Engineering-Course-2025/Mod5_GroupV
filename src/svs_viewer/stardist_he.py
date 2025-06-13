# %%
import numpy as np
from stardist.models import StarDist2D
from stardist.plot import render_label
from csbdeep.utils import normalize
from skimage.color import rgb2gray
from skimage import exposure, img_as_float
import dask.array as da
import warnings
warnings.filterwarnings('ignore')

STARDIST_HE_MODEL = StarDist2D.from_pretrained('2D_versatile_he')

def format_image(image):
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
    return image

def adjust_image_contrast(image):
    if image.dtype != np.float32 and image.dtype != np.float64:
        image = img_as_float(image) 

    # Apply CLAHE to each channel if it's an RGB image
    if image.ndim == 3:
        enhanced_image = np.empty_like(image)
        for i in range(image.shape[-1]):
            enhanced_image[:, :, i] = exposure.equalize_adapthist(image[:, :, i])
    else: # Grayscale image
        enhanced_image = exposure.equalize_adapthist(image)

    # Convert back to original dtype if needed, or keep as float for normalize
    return enhanced_image

    
def predict_nucleus(image, model_name='2D_versatile_he', prob_thresh=None, nms_thresh=None):
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

    image = format_image(image)
    image = adjust_image_contrast(image)

    labels, _ = model.predict_instances(normalize(image), prob_thresh=prob_thresh, nms_thresh=nms_thresh)
    # full_results = render_label(labels, img=image)
    return labels, image


# # %%
# import openslide
# import numpy as np
# import os
# from tqdm import trange

# filepath = '/project/GCRB/Hon_lab/s440862/courses/se/Mod5_GroupV/datasets'
# output_dir = '/project/GCRB/Hon_lab/s440862/courses/se/Mod5_GroupV/datasets/slices'
# batch_size = 256
# chunk_size = (batch_size, batch_size)

# for file in os.listdir(filepath):
#     svs_filepath = os.path.join(filepath, file)
#     file_name = file[:7]
#     os.makedirs(os.path.join(output_dir, file_name), exist_ok=True)

#     slide = openslide.OpenSlide(svs_filepath)
#     width, height = slide.dimensions
#     chunk_width, chunk_height = chunk_size

#     chunk_count = 0
#     for y in trange(0, height, chunk_height):
#         for x in range(0, width, chunk_width):
#             read_width = min(chunk_width, width - x)
#             read_height = min(chunk_height, height - y)

#             if read_width <= 0 or read_height <= 0:
#                 continue # Skip if no valid region to read

#             try:
#                 region_pil = slide.read_region((x, y), 0, (read_width, read_height))

#                 region_np = np.array(region_pil)[:, :, :3] # Take only RGB channels
                
#                 padded_chunk = np.full((chunk_height, chunk_width, 3), 255, dtype=np.uint8) # Pad with white
#                 padded_chunk[0:read_height, 0:read_width, :] = region_np

#                 output_filename = os.path.join(output_dir, file_name, f"chunk_{x}_{y}.npy")
#                 np.save(output_filename, padded_chunk)
#                 chunk_count += 1
            
#             except openslide.OpenSlideError as e:
#                 print(f"Error reading region at ({x}, {y}) with size ({read_width}, {read_height}): {e}")
#             except Exception as e:
#                 print(f"An unexpected error occurred processing chunk at ({x}, {y}): {e}")

#     slide.close()


# from glob import glob
# from matplotlib import pyplot as plt
# images = {}
# for filename in os.listdir(output_dir):
#     images[filename] = []
#     for filepath in glob(os.path.join(output_dir, filename, "*.npy")):
#         images[filename].append(np.load(filepath))

# # %%

# model = STARDIST_HE_MODEL
# prob_thresh = 0.6
# nms_thresh = 0.9
# from skimage import img_as_float

# idx = 50
# print(f"prob_thresh: {prob_thresh}, nms_thresh: {nms_thresh}")
# cnt = 1
# for file_name in images.keys():
#     image = images[file_name][idx]

#     if image.ndim == 3 and image.shape[-1] == 1:
#         image = np.squeeze(image, axis=-1) 

#     if image.dtype != np.float32 and image.dtype != np.float64:
#         image = img_as_float(image) # From skimage.util

#     # Apply CLAHE to each channel if it's an RGB image
#     if image.ndim == 3:
#         enhanced_image = np.empty_like(image)
#         for i in range(image.shape[-1]):
#             image[:, :, i] = exposure.equalize_adapthist(image[:, :, i])
#     else: # Grayscale image
#         image = exposure.equalize_adapthist(image)

#     labels, _ = model.predict_instances(normalize(image), prob_thresh=prob_thresh, nms_thresh=nms_thresh, sparse=False)

#     plt.subplot(2,2,cnt)
#     plt.imshow(image, cmap="gray")
#     plt.axis("off")
#     plt.title(f"{file_name} input image")

#     plt.subplot(2,2,cnt+1)
#     plt.imshow(render_label(labels, img=image, alpha=1))
#     plt.axis("off")
#     plt.title("prediction + input overlay")

#     cnt += 2
# # %%
