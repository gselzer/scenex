from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeGuard, cast

import numpy as np

from scenex.adaptors._base import CanvasAdaptor
from scenex.events import MouseEvent
from scenex.events._auto import app
from scenex.events.events import MouseButton

from ._adaptor_registry import get_adaptor

if TYPE_CHECKING:
    from cmap import Color
    from rendercanvas.base import BaseRenderCanvas

    from scenex import model

    class SupportsHideShow(BaseRenderCanvas):
        def show(self) -> None: ...
        def hide(self) -> None: ...


def supports_hide_show(obj: Any) -> TypeGuard[SupportsHideShow]:
    return hasattr(obj, "show") and hasattr(obj, "hide")


class Canvas(CanvasAdaptor):
    """Canvas interface for vispy Backend."""

    def __init__(self, canvas: model.Canvas, **backend_kwargs: Any) -> None:
        from vispy.scene import Grid, SceneCanvas, VisualNode

        self._canvas = SceneCanvas(
            title=canvas.title, size=(canvas.width, canvas.height)
        )
        # Qt RenderCanvas calls show() in its __init__ method, so we need to hide it
        if supports_hide_show(self._canvas.native):
            self._canvas.native.hide()
        self._grid = cast("Grid", self._canvas.central_widget.add_grid())
        for view in canvas.views:
            self._snx_add_view(view)
        self._views = canvas.views
        self._filter = app().install_event_filter(
            self._canvas.native, self._handle_event
        )

        self._visual_to_node: dict[VisualNode, model.Node | None] = {}
        self._last_canvas_pos: tuple[float, float] | None = None

    def _filter_through(self, event: Any, node: model.Node, target: model.Node) -> bool:
        """Filter the event through the scene graph to the target node."""
        # First give this node a chance to filter the event.
        if node.filter_event(event, target):
            # Node filtered out the event, so we stop here.
            return True
        if (parent := node.parent) is None:
            # Node did not filter out the event, and we've reached the top of the graph.
            return False
        # Recursively filter the event through node's parent.
        return self._filter_through(event, parent, target)

    def _handle_event(self, event: Any) -> bool:
        from vispy.scene import ViewBox

        # Pass the event to the view
        if isinstance(event, MouseEvent):
            handled = False
            # Find the visual under the mouse
            visuals = self._canvas.visuals_at(event.pos)
            visual = next(filter(lambda v: not isinstance(v, ViewBox), visuals), None)
            if not handled and visual:
                # Find the scenex node associated with the visual
                if visual not in self._visual_to_node:
                    for view in self._views:
                        for child in view.scene.children:
                            if get_adaptor(child)._snx_get_native() == visual:
                                self._visual_to_node[visual] = child
                    self._visual_to_node.setdefault(visual, None)
                if node := self._visual_to_node.get(visual, None):
                    # Filter through parent scenes to child
                    self._filter_through(event, node, node)
            # No visual selected - move the camera
            # FIXME: Move this to the camera, somehow?
            if (
                not handled
                and event.type == "move"
                and MouseButton.LEFT in event.buttons
                and self._last_canvas_pos is not None
            ):
                # FIXME: Which camera to use?
                cam = self._views[0].camera
                if cam.type == "panzoom":
                    # PanZoomCamera
                    p1 = cam.transform.imap(self._last_canvas_pos)
                    p2 = cam.transform.imap(event.pos)
                    cam.transform = cam.transform.translated(
                        (p1[0] - p2[0], p2[1] - p1[1])
                    )

            self._last_canvas_pos = event.pos

        return True

    def _snx_get_native(self) -> Any:
        return self._canvas.native

    def _snx_set_visible(self, arg: bool) -> None:
        # show the qt canvas we patched earlier in __init__
        app().show(self._canvas.native, arg)

    def _draw(self) -> None:
        self._canvas.update()

    def _snx_add_view(self, view: model.View) -> None:
        self._grid.add_widget(get_adaptor(view)._snx_get_native())

    def _snx_set_width(self, arg: int) -> None:
        self._canvas.size = (self._canvas.size[0], arg)

    def _snx_set_height(self, arg: int) -> None:
        self._canvas.size = (arg, self._canvas.size[1])

    def _snx_set_background_color(self, arg: Color | None) -> None:
        if arg is None:
            self._canvas.bgcolor = "black"
        else:
            self._canvas.bgcolor = arg.rgba

    def _snx_set_title(self, arg: str) -> None:
        self._canvas.title = arg

    def _snx_close(self) -> None:
        """Close canvas."""
        self._canvas.close()

    def _snx_render(
        self,
        region: tuple[int, int, int, int] | None = None,
        size: tuple[int, int] | None = None,
        bgcolor: Color | None = None,
        crop: np.ndarray | tuple[int, int, int, int] | None = None,
        alpha: bool = True,
    ) -> np.ndarray:
        """Render to screenshot."""
        vispy_bgcolor = None
        if bgcolor and bgcolor.name:
            from vispy.color import Color as _Color

            vispy_bgcolor = _Color(bgcolor.name)
        return np.asarray(
            self._canvas.render(
                region=region, size=size, bgcolor=vispy_bgcolor, crop=crop, alpha=alpha
            )
        )
