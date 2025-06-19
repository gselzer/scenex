from __future__ import annotations

from dataclasses import dataclass
from enum import IntFlag, auto
from typing import TYPE_CHECKING

import numpy as np
import pylinalg as la

from scenex.model import Camera

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any

    from scenex.model import Canvas, Node, View


@dataclass
class Event:
    """A general interaction event."""

    # TODO: Enum?
    type: str


class MouseButton(IntFlag):
    """A general mouse interaction event."""

    LEFT = auto()
    MIDDLE = auto()
    RIGHT = auto()
    NONE = auto()


@dataclass
class MouseEvent(Event):
    """A general mouse interaction event."""

    type: str
    # TODO: Maybe a 3D vector/ray?
    # TODO: Named tuple? e.g. event.pos.x, event.pos.y, ...
    pos: tuple[float, float]
    # TODO: Enum?
    # TODO: Just a MouseButton, you can AND the MouseButtons
    buttons: set[MouseButton]

    @property
    def x(self) -> float:
        """The x-coordinate of the mouse event."""
        return self.pos[0]

    @property
    def y(self) -> float:
        """The y-coordinate of the mouse event."""
        return self.pos[1]


def _handle_event(canvas: Canvas, event: Event) -> bool:
    handled = False
    if isinstance(event, MouseEvent):
        if view := _containing_view(event, canvas.views):
            # FIXME: This only works for pan/zoom
            ray_origin: np.ndarray = np.asarray(_canvas_to_world(view, event.pos))
            # FIXME: Account for camera transform?
            ray_direction = np.asarray((0, 0, 1))

            through: list[tuple[Node, float]] = []
            for child in view.scene.children:
                if (d := child.passes_through(ray_origin, ray_direction)) is not None:
                    through.append((child, d))

            # FIXME: Consider only reporting the first?
            # Or do we only report until we hit a node with opacity=1?
            for node, _depth in sorted(through, key=lambda e: e[1]):
                # Filter through parent scenes to child
                handled |= _filter_through(event, node, node)
            # No nodes in the view handled the event - pass it to the camera
            if not handled:
                # FIXME: To move the camera around, we need the world position.
                # _move_camera(view.camera, ray_origin)
                view.camera.filter_event(event, view.camera)

    return handled


def _containing_view(event: MouseEvent, views: Sequence[View]) -> View | None:
    for view in views:
        if event.pos in view.layout:
            return view
    return None


def _filter_through(event: Any, node: Node, target: Node) -> bool:
    """Filter the event through the scene graph to the target node."""
    # First give this node a chance to filter the event.
    if node.filter_event(event, target):
        # Node filtered out the event, so we stop here.
        return True
    if (parent := node.parent) is None:
        # Node did not filter out the event, and we've reached the top of the graph.
        return False
    # Recursively filter the event through node's parent.
    return _filter_through(event, parent, target)


def _canvas_to_world(
    model_view: View, pos_xy: tuple[float, float]
) -> tuple[float, float, float]:
    """Map XY canvas position (pixels) to XYZ coordinate in world space."""
    # Code adapted from:
    # https://github.com/pygfx/pygfx/pull/753/files#diff-173d643434d575e67f8c0a5bf2d7ea9791e6e03a4e7a64aa5fa2cf4172af05cdR395
    x, y = pos_xy[0], pos_xy[1]
    if (
        x < model_view.layout.x
        or x > model_view.layout.x + model_view.layout.width
        or y < model_view.layout.y
        or y > model_view.layout.y + model_view.layout.height
    ):
        return (-1, -1, -1)

    # Get position relative to viewport
    pos_rel = (
        pos_xy[0] - model_view.layout.x,
        pos_xy[1] - model_view.layout.y,
    )

    width, height = model_view.layout.size

    # Convert position to Normalized Device Coordinates (NDC) - i.e., within [-1, 1]
    x = pos_rel[0] / width * 2 - 1
    y = -(pos_rel[1] / height * 2 - 1)
    pos_ndc = (x, y, 0)

    # Note that the camera matrix is the matrix multiplication of:
    # * The projection matrix, which projects local space (the rectangular
    #   bounds of the perspective camera) into NDC.
    # * The view matrix, i.e. the transform positioning the camera in the world.
    # The result is a matrix mapping world coordinates
    camera_matrix = model_view.camera.projection @ model_view.camera.transform.inv().T
    # TODO: Is this addition to pos_ndc ever (functionally) nonzero?
    camera_position = model_view.camera.transform.root[3, :3]
    pos_diff = la.vec_transform(camera_position, camera_matrix)
    # Unproject the canvas NDC coordinates into world space.
    pos_world = la.vec_unproject(pos_ndc[:2] + pos_diff[:2], camera_matrix)
    # print(f"NDC pos: {pos_ndc}, NDC diff: {pos_diff}, world_pos: {pos_world}")

    # NB In vispy, (0.5,0.5) is a center of an image pixel, while in pygfx
    # (0,0) is the center. We conform to vispy's standard.
    return (pos_world[0] + 0.5, pos_world[1] + 0.5, pos_world[2] + 0.5)


class _DefaultCameraFilter:
    # TODO: This is an IntFlag - set not necessary
    _last_pos: tuple[float, float] | None = None

    def __call__(self, event: Event, node: Node) -> bool:
        assert isinstance(node, Camera)

        if isinstance(event, MouseEvent):
            if (
                event.type == "move"
                and MouseButton.LEFT in event.buttons
                and self._last_pos is not None
            ):
                # FIXME: Event position needs to be converted into world positions
                p1 = node.transform.imap(self._last_pos)
                p2 = node.transform.imap(event.pos)
                node.transform = node.transform.translated(
                    (p1[0] - p2[0], p2[1] - p1[1])
                )

            self._last_pos = event.pos

        return False
