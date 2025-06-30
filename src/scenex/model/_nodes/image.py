from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any, Literal

import numpy as np
from annotated_types import Interval
from cmap import Colormap
from pydantic import Field

from .node import Node

if TYPE_CHECKING:
    from scenex.events.events import Ray

InterpolationMode = Literal["nearest", "linear", "bicubic"]


class Image(Node):
    """A dense array of intensity values."""

    node_type: Literal["image"] = Field(default="image", repr=False)

    # NB: we may want this to be a pure `set_data()` method, rather than a field
    # on the model that stores state.
    data: Any = Field(
        default=None, repr=False, exclude=True, description="The current image data."
    )
    cmap: Colormap = Field(
        default_factory=lambda: Colormap("gray"),
        description="The colormap to apply when rendering the image.",
    )
    clims: tuple[float, float] | None = Field(
        default=None,
        description="The min and max values to use when normalizing the image.",
    )
    gamma: Annotated[float, Interval(gt=0, le=2)] = Field(
        default=1.0, description="Gamma correction applied after normalization."
    )
    interpolation: InterpolationMode = Field(
        default="nearest", description="Interpolation mode."
    )

    def passes_through(self, ray: Ray) -> float | None:
        # Math graciously adapted from:
        # https://raytracing.github.io/books/RayTracingTheNextWeek.html#quadrilaterals

        # Step 1 - Determine where the ray intersects the image plane

        # The image plane is defined by the normal vector n=(a, b, c) and an offset (d)
        # such that any point p=(x, y, z) on the plane satisfies np.dot(v, p) = d, or
        # ax + by + cz + -d = 0.

        # In this case, the normal vector n is the node's transformation of (0, 0, 1),
        # since by default images are displayed in the XY plane.
        tformed = self.transform.map((0, 0, 1, 0))[:3]
        normal = tformed / np.linalg.norm(tformed)
        # And we know that the node's transformation of (0, 0, 0) is on the plane.
        # This is the origin point of the image.
        image_origin = self.transform.map((0, 0, 0, 1))[:3]
        # So if we find the value of d...
        d = np.dot(normal, image_origin)
        # ...we can find the depth t at which the ray would intersect the plane.
        #
        # Note that our ray is defined by (ray.origin + ray.direction * t).
        # This is just np.dot(normal, ray.origin + ray.direction * t) = d,
        # rearranged to solve for t.
        t = (d - np.dot(normal, ray.origin)) / np.dot(normal, ray.direction)
        # With our value of t, we can find the intersection point
        intersection = tuple(
            a + t * b for a, b in zip(ray.origin, ray.direction, strict=False)
        )

        # Step 2 - Determine whether the ray hits the image.

        # We need to determine whether the planar intersection is within the image
        # interval bounds. In other words, the intersection point should be within
        # [0, self.data.shape[0]] units away from the image origin along the X axis and
        # [0, self.data.shape[1]] units away from the image origin along the Y axis.
        offset = intersection - image_origin
        # We transform the X and Y normal vectors...
        u = self.transform.map((1, 0, 0, 0))[:3]
        v = self.transform.map((0, 1, 0, 0))[:3]

        # And use some fancy math derived from the link above to convert offset into...
        n = np.cross(u, v)
        w = n / np.dot(n, n)
        # ...the component of offset in direction of u...
        alpha = np.dot(w, np.cross(offset, v))
        # ...and the component of offset in direction of v
        beta = np.dot(w, np.cross(u, offset))
        # Our ray passes through the image if alpha and beta are positive and within
        # the data dimensions
        is_inside = (
            alpha >= 0
            and alpha <= self.data.shape[0]
            and beta >= 0
            and beta <= self.data.shape[1]
        )

        # If the ray passes through node, return the depth of the intersection.
        return t if is_inside else None
