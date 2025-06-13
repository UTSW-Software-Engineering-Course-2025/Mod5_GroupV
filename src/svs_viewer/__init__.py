
try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"

from ._reader import napari_get_reader, svs2dask
from ._widget import Segment
from .utils import rgb2hed, threshold_nuclei_dask
from .stardist_he import predict_nucleus

__all__ = (
    "napari_get_reader",
    "Segment",
    "rgb2hed",
    "svs2dask",
    "threshold_nuclei_dask",
    "predict_nucleus"
)
