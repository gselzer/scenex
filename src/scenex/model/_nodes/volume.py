from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import Field

from .image import Image

if TYPE_CHECKING:
    import numpy as np

RenderMode = Literal["iso", "mip"]


class Volume(Image):
    """A dense 3-dimensional array of intensity values."""

    render_mode: RenderMode = Field(
        default="mip",
        description="The method to use in rendering the volume.",
    )

    def passes_through(
        self, ray_origin: np.ndarray, ray_direction: np.ndarray
    ) -> float | None:
        # This should somewhat resemble the image check, but we will have to check each
        # plane on the bounding cube around the volume.
        raise NotImplementedError("TODO")
