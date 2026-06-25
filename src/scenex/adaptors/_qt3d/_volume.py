from __future__ import annotations

from typing import TYPE_CHECKING, Any

from scenex.adaptors._base import VolumeAdaptor

from ._image import Image

if TYPE_CHECKING:
    from scenex import model


class Volume(Image, VolumeAdaptor):
    """Qt3D backend adaptor for a Volume node.

    Volume rendering (ray-marching) is not supported in V1.  The volume is
    rendered as a maximum-intensity projection onto a 2D image using numpy on
    the CPU, then displayed as a flat image via the Image adaptor.

    A Qt RHI-based backend would be required to support full 3D volumetric
    rendering.
    """

    def __init__(self, volume: model.Volume, **backend_kwargs: Any) -> None:
        import logging

        logging.getLogger("scenex.adaptors.qt3d").warning(
            "Volume rendering is not supported in the qt3d backend. "
            "A maximum-intensity projection (MIP) is shown instead."
        )
        # Show MIP along last axis as a 2D image
        import numpy as np

        data = np.asarray(volume.data)
        if data.ndim == 3:
            mip: Any = data.max(axis=0)
        else:
            mip = data

        # Build a fake Image model with the MIP data to reuse Image machinery
        import copy

        fake = copy.copy(volume)
        object.__setattr__(fake, "data", mip)  # bypass pydantic validation
        super().__init__(fake, **backend_kwargs)  # type: ignore[arg-type]
        self._model = volume  # type: ignore[assignment]

    def _snx_set_render_mode(self, arg: model.RenderMode) -> None:
        pass  # Not supported; ignored
