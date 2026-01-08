from __future__ import annotations

import logging
from abc import abstractmethod

from cmap import Color
from pydantic import ConfigDict, Field

from ._base import EventedBase

logger = logging.getLogger(__name__)


class Resizer(EventedBase):
    @abstractmethod
    def resize(
        self,
        layout: Layout,
        canvas_size: tuple[float, float],
    ) -> None:
        """Resize the given layout based on the canvas size."""
        ...


class ProportionalResizer(Resizer):
    start: int | tuple[int, int] = Field(default=0)
    end: int | tuple[int, int] = Field(default=1)
    total: int | tuple[int, int] = Field(default=1)

    def resize(
        self,
        layout: Layout,
        canvas_size: tuple[float, float],
    ) -> None:
        cw, ch = canvas_size
        sw, sh = (
            self.start if isinstance(self.start, tuple) else (self.start, self.start)
        )
        ew, eh = self.end if isinstance(self.end, tuple) else (self.end, self.end)
        tw, th = (
            self.total if isinstance(self.total, tuple) else (self.total, self.total)
        )
        layout.x = (sw / tw) * cw
        layout.y = (sh / th) * ch
        layout.width = ((ew - sw) / tw) * cw
        layout.height = ((eh - sh) / th) * ch


class AnchorResizer(Resizer):
    """Anchor the layout to a fixed pixel position on the canvas.

    Primarily useful for overlays that should stay in a corner.
    """

    anchor: tuple[float, float] = Field(default=(0, 0))

    def resize(
        self,
        layout: Layout,
        canvas_size: tuple[float, float],
    ) -> None:
        sw = self.anchor[0]
        if sw < 0:
            sw += canvas_size[0]
        layout.x = sw

        sh = self.anchor[1]
        if sh < 0:
            sh += canvas_size[1]
        layout.y = sh


class Layout(EventedBase):
    """Rectangular layout model with positioning and styling.

    The Layout model defines the position, size, and visual styling of rectangular
    areas. It uses a box model with margin, border, padding, and content areas,
    similar to CSS.

    Examples
    --------
    Create a layout at position (100, 100) with size 400x300:
        >>> layout = Layout(x=100, y=100, width=400, height=300)

    Create a layout with border and padding:
        >>> layout = Layout(
        ...     width=200,
        ...     height=200,
        ...     border_width=2,
        ...     border_color=Color("white"),
        ...     padding=10,
        ... )

    Notes
    -----
    The layout follows this box model::

            y
            |
            v
        x-> +--------------------------------+  ^
            |            margin              |  |
            |  +--------------------------+  |  |
            |  |         border           |  |  |
            |  |  +--------------------+  |  |  |
            |  |  |      padding       |  |  |  |
            |  |  |  +--------------+  |  |  |   height
            |  |  |  |   content    |  |  |  |  |
            |  |  |  |              |  |  |  |  |
            |  |  |  +--------------+  |  |  |  |
            |  |  +--------------------+  |  |  |
            |  +--------------------------+  |  |
            +--------------------------------+  v

            <------------ width ------------->
    """

    x: float = Field(
        default=0, description="The x-coordinate of the left edge of the layout"
    )
    y: float = Field(
        default=0, description="The y-coordinate of the top edge of the layout"
    )
    width: float = Field(
        default=600, description="The total width (including margin, border, padding)"
    )
    height: float = Field(
        default=600, description="The total height (including margin, border, padding)"
    )
    resizer: Resizer | None = Field(default_factory=ProportionalResizer)
    background_color: Color | None = Field(
        default=Color("black"),
        description="The background color (inside of the border). "
        "None implies transparent",
    )
    border_width: float = Field(
        default=0, description="The width of the border in pixels."
    )
    border_color: Color | None = Field(
        default=Color("black"), description="The color of the border."
    )
    padding: int = Field(
        default=0,
        description="Number of pixels between border and content",
    )
    margin: int = Field(
        default=0,
        description="Number of pixels between top/left edge and border",
    )

    model_config = ConfigDict(extra="forbid")

    @property
    def position(self) -> tuple[float, float]:
        """Return the x, y position of the layout as a tuple."""
        return self.x, self.y

    @property
    def size(self) -> tuple[float, float]:
        """Return the width, height of the layout as a tuple."""
        return self.width, self.height

    @property
    def content_rect(self) -> tuple[float, float, float, float]:
        """Return the (x, y, width, height) of the content area."""
        offset = self.padding + self.border_width + self.margin
        return (
            self.x + offset,
            self.y + offset,
            self.width - 2 * offset,
            self.height - 2 * offset,
        )

    def __contains__(self, pos: tuple[float, float]) -> bool:
        offset = self.padding + self.border_width + self.margin

        left = self.x + offset
        right = self.x + self.width - offset
        bottom = self.y + offset
        top = self.y + self.height - offset
        return left <= pos[0] and pos[0] <= right and bottom <= pos[1] and pos[1] <= top
