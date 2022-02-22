__all__ = ["cvt_lowpoly", "fast_cvt_lowpoly"]

from .cvt_lowpoly import lowpoly_mesh, lowpoly_image
from .fast_cvt_lowpoly import lowpoly_image as fast_lowpoly_image
from .fast_cvt_lowpoly import lowpoly_mesh as fast_lowpoly_mesh