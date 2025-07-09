from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import Field, PrivateAttr

from scenex.events.events import MouseButton, MouseEvent, WheelEvent
from scenex.model._transform import Transform

from .node import Node

if TYPE_CHECKING:
    from collections.abc import Callable

    from scenex.events.events import Event, Ray

CameraType = Literal["panzoom", "perspective"]
Position2D = tuple[float, float]
Position3D = tuple[float, float, float]
Position = Position2D | Position3D


class _DefaultCameraFilter:
    def __init__(self) -> None:
        self.drag_pos: tuple[float, float] | None = None

    def __call__(self, event: Event, node: Node) -> bool:
        assert isinstance(node, Camera)
        handled = False

        # FIXME: Probably doesn't work outside of panzoom camera
        if isinstance(event, MouseEvent):
            new_pos = event.world_ray.origin[:2]

            # Panning involves keeping a particular position underneath the cursor.
            # That position is recorded on a left mouse button press.
            if event.type == "press" and MouseButton.LEFT in event.buttons:
                self.drag_pos = new_pos
            # Every time the cursor is moved, until the left mouse button is released,
            # We translate the camera such that the position is back under the cursor
            # (i.e. under the world ray origin)
            elif (
                event.type == "move"
                and MouseButton.LEFT in event.buttons
                and self.drag_pos
            ):
                dx = self.drag_pos[0] - new_pos[0]
                dy = self.drag_pos[1] - new_pos[1]
                node.transform = node.transform.translated((dx, dy))
                handled = True

            elif isinstance(event, WheelEvent):
                dx, dy = event.angle_delta
                if dy:
                    fy = 2 ** (dy * -0.001)  # Magnifier stolen from pygfx
                    node.projection = node.projection.scaled((fy, fy, 1.0))
                    print(node.projection)
                    # TODO: Pan to compensate zoom
                    handled = True

        return handled


class Camera(Node):
    """A camera that defines the view and perspective of a scene.

    The camera lives in, and is a child of, a scene graph.  It defines the view
    transformation for the scene, mapping it onto a 2D surface.

    Cameras have two different Transforms. Like all Nodes, it has a transform
    `transform`, describing its location in the world. Its other transform,
    `projection`, describes how 2D normalized device coordinates
    {(x, y) | x in [-1, 1], y in [-1, 1]} map to a ray in 3D world space. The inner
    product of these matrices can convert a 2D canvas position to a 3D ray eminating
    from the camera node into the world.
    """

    node_type: Literal["camera"] = "camera"

    type: CameraType = Field(default="panzoom", description="Camera type.")
    interactive: bool = Field(
        default=True,
        description="Whether the camera responds to user interaction, "
        "such as mouse and keyboard events.",
    )
    zoom: float = Field(default=1.0, description="Zoom factor of the camera.")
    center: Position = Field(
        default=(0, 0, 0), description="Center position of the view."
    )
    projection: Transform = Field(
        default_factory=Transform,
        description="Describes how 3D points are mapped to a 2D canvas",
    )
    _filter: Callable[[Event, Node], bool] | None = PrivateAttr(
        default_factory=_DefaultCameraFilter
    )

    def passes_through(self, ray: Ray) -> float | None:
        return None
