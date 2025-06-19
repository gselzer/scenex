from __future__ import annotations

from typing import Annotated, Any, Literal

import numpy as np
from annotated_types import Interval
from cmap import Color
from pydantic import Field

from .node import Node

SymbolName = Literal[
    "disc",
    "arrow",
    "ring",
    "clobber",
    "square",
    "x",
    "diamond",
    "vbar",
    "hbar",
    "cross",
    "tailed_arrow",
    "triangle_up",
    "triangle_down",
    "star",
    "cross_lines",
]
ScalingMode = Literal[True, False, "fixed", "scene", "visual"]


class Points(Node):
    """Coordinates that can be represented in a scene."""

    node_type: Literal["points"] = "points"

    # numpy array of 2D/3D point centers, shape (N, 2) or (N, 3)
    coords: Any = Field(default=None, repr=False, exclude=True)
    size: Annotated[float, Interval(ge=0.5, le=100)] = Field(
        default=10.0, description="The diameter of the points."
    )
    face_color: Color | None = Field(
        default=Color("white"), description="The color of the faces."
    )
    edge_color: Color | None = Field(
        default=Color("black"), description="The color of the edges."
    )
    edge_width: float | None = Field(default=1.0, description="The width of the edges.")
    symbol: SymbolName = Field(
        default="disc", description="The symbol to use for the points."
    )
    # TODO: these are vispy-specific names.  Determine more general names
    scaling: ScalingMode = Field(
        default=True, description="Determines how points scale when zooming."
    )

    antialias: float = Field(default=1, description="Anti-aliasing factor, in px.")

    def passes_through(
        self, ray_origin: np.ndarray, ray_direction: np.ndarray
    ) -> float | None:
        # Math graciously adapted from:
        # https://raytracing.github.io/books/RayTracingInOneWeekend.html#addingasphere/ray-sphereintersection

        # TODO: This could be overly restrictive
        if not isinstance(self.coords, np.ndarray):
            return None

        coords = self.coords
        if coords.ndim < len(ray_origin):
            coords = np.pad(
                self.coords, ((0, 0), (0, 1)), mode="constant", constant_values=0
            )
        coords = self.transform.map(coords)[:, :3]

        ray_diff = coords - ray_origin

        r = self.size / 2 + (self.edge_width if self.edge_width else 0)
        a = np.dot(ray_direction, ray_direction)
        b = -2 * np.dot(ray_diff, ray_direction)
        c = np.sum(ray_diff * ray_diff, axis=1) - r**2

        discriminant = b**2 - 4 * a * c

        intersecting_indices = np.where(discriminant >= 0)[0]
        if intersecting_indices.size:
            print(intersecting_indices)

        # Step 1 - Determine where the ray intersects the image plane

        # Step 1 - Determine where the ray intersects the image plane

        return None
