from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import Field

from .image import Image, _passes_through_parallelogram

if TYPE_CHECKING:
    from scenex.events.events import Ray

RenderMode = Literal["iso", "mip"]


class Volume(Image):
    """A dense 3-dimensional array of intensity values."""

    render_mode: RenderMode = Field(
        default="mip",
        description="The method to use in rendering the volume.",
    )

    def passes_through(self, ray: Ray) -> float | None:
        d, w, h = self.data.shape

        tlf = self.transform.map((0, 0, 0, 1))[:3]
        brb = self.transform.map((w, h, d, 1))[:3]

        u = self.transform.map((w, 0, 0, 0))[:3]
        v = self.transform.map((0, h, 0, 0))[:3]
        w = self.transform.map((0, 0, d, 0))[:3]

        faces = [
            (tlf, u, v),  # front face
            (tlf, v, w),  # left face
            (tlf, w, u),  # top face
            (brb, -u, -v),  # back face
            (brb, -v, -w),  # right face
            (brb, -w, -u),  # bottom face
        ]
        results = [_passes_through_parallelogram(ray, o, a, b) for o, a, b in faces]
        depths = [r for r in results if r is not None]
        return min(depths) if depths else None
