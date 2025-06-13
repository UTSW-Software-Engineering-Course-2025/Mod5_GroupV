
try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"

from ._reader import napari_get_reader
from ._widget import ExampleQWidget, ImageThreshold, threshold_autogenerate_widget, threshold_magic_widget
from ._writer import write_multiple, write_single_image
from .utils import rgb2hed, threshold_nuclei_dask
from .stardist_he import predict_nucleus

__all__ = (
    "napari_get_reader",
    "write_single_image",
    "write_multiple",
    "ExampleQWidget",
    "ImageThreshold",
    "threshold_autogenerate_widget",
    "threshold_magic_widget",
    "rgb2hed",
    "threshold_nuclei_dask",
    "predict_nucleus"
)
