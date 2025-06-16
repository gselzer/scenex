from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeGuard, cast

import numpy as np
import pygfx
import pylinalg as la

from scenex.adaptors._base import CanvasAdaptor
from scenex.events._auto import app
from scenex.events.events import Event, MouseEvent

from ._adaptor_registry import adaptors

if TYPE_CHECKING:
    from cmap import Color
    from rendercanvas.base import BaseRenderCanvas

    from scenex import model

    from ._view import View

    class SupportsHideShow(BaseRenderCanvas):
        def show(self) -> None: ...
        def hide(self) -> None: ...


def supports_hide_show(obj: Any) -> TypeGuard[SupportsHideShow]:
    return hasattr(obj, "show") and hasattr(obj, "hide")


class Canvas(CanvasAdaptor):
    """Canvas interface for pygfx Backend."""

    def __init__(self, canvas: model.Canvas, **backend_kwargs: Any) -> None:
        from rendercanvas.auto import RenderCanvas

        canvas_cls = RenderCanvas
        # HACK: Qt
        if canvas_cls.__module__.startswith("rendercanvas.qt"):
            from qtpy.QtCore import QSize
            from rendercanvas.auto import loop
            from rendercanvas.qt import QRenderWidget

            class _QRenderWidget(QRenderWidget):
                def sizeHint(self) -> QSize:
                    return QSize(self.width(), self.height())

            loop._rc_init()
            canvas_cls = _QRenderWidget
        self._wgpu_canvas = canvas_cls()

        self._wgpu_canvas.set_logical_size(canvas.width, canvas.height)
        self._wgpu_canvas.set_title(canvas.title)
        self._views = canvas.views
        self._filter = app().install_event_filter(self._wgpu_canvas, self._handle_event)

    def _handle_event(self, event: Event) -> bool:
        if isinstance(event, MouseEvent):
            for view in self._views:
                # FIXME: This only works for pan/zoom
                ray_origin = np.asarray(
                    self.canvas_to_world(view._get_adaptor(), event.pos)
                )
                # FIXME: Account for camera transform?
                ray_direction = np.asarray((0, 0, 1))

                through = []
                for child in view.scene.children:
                    if (
                        d := child.passes_through(ray_origin, ray_direction)
                    ) is not None:
                        through.append((child, d))

        return False

    def canvas_to_world(
        self, view: View, pos_xy: tuple[float, float]
    ) -> tuple[float, float, float]:
        """Map XY canvas position (pixels) to XYZ coordinate in world space."""
        # Code adapted from:
        # https://github.com/pygfx/pygfx/pull/753/files#diff-173d643434d575e67f8c0a5bf2d7ea9791e6e03a4e7a64aa5fa2cf4172af05cdR395
        viewport = pygfx.Viewport.from_viewport_or_renderer(view._renderer)
        if not viewport.is_inside(*pos_xy):
            return (-1, -1, -1)

        # Get position relative to viewport
        pos_rel = (
            pos_xy[0] - viewport.rect[0],
            pos_xy[1] - viewport.rect[1],
        )

        vs = viewport.logical_size

        # Convert position to NDC
        x = pos_rel[0] / vs[0] * 2 - 1
        y = -(pos_rel[1] / vs[1] * 2 - 1)
        pos_ndc = (x, y, 0)

        if cam := view._pygfx_cam:
            pos_ndc += la.vec_transform(cam.world.position, cam.camera_matrix)
            pos_world = la.vec_unproject(pos_ndc[:2], cam.camera_matrix)

            # NB In vispy, (0.5,0.5) is a center of an image pixel, while in pygfx
            # (0,0) is the center. We conform to vispy's standard.
            return (pos_world[0] + 0.5, pos_world[1] + 0.5, pos_world[2] + 0.5)
        else:
            return (-1, -1, -1)

    def _snx_get_native(self) -> BaseRenderCanvas:
        return self._wgpu_canvas

    def _snx_set_visible(self, arg: bool) -> None:
        # show the qt canvas we patched earlier in __init__
        if supports_hide_show(self._wgpu_canvas):
            self._wgpu_canvas.show()
        self._wgpu_canvas.request_draw(self._draw)

    def _draw(self) -> None:
        for view in self._views:
            adaptor = cast("View", adaptors.get_adaptor(view, create=True))
            adaptor._draw()

    def _snx_add_view(self, view: model.View) -> None:
        pass
        # adaptor = cast("View", view.backend_adaptor())
        # adaptor._pygfx_cam.set_viewport(self._viewport)
        # self._views.append(adaptor)

    def _snx_set_width(self, arg: int) -> None:
        _, height = cast("tuple[float, float]", self._wgpu_canvas.get_logical_size())
        self._wgpu_canvas.set_logical_size(arg, height)

    def _snx_set_height(self, arg: int) -> None:
        width, _ = cast("tuple[float, float]", self._wgpu_canvas.get_logical_size())
        self._wgpu_canvas.set_logical_size(width, arg)

    def _snx_set_background_color(self, arg: Color | None) -> None:
        # not sure if pygfx has both a canavs and view background color...
        pass

    def _snx_set_title(self, arg: str) -> None:
        self._wgpu_canvas.set_title(arg)

    def _snx_close(self) -> None:
        """Close canvas."""
        self._wgpu_canvas.close()

    def _snx_render(self) -> np.ndarray:
        """Render to offscreen buffer."""
        from rendercanvas.offscreen import OffscreenRenderCanvas

        # not sure about this...
        # w, h = self._wgpu_canvas.get_logical_size()
        canvas = OffscreenRenderCanvas(size=(640, 480), pixel_ratio=2)
        canvas.request_draw(self._draw)
        canvas.force_draw()
        return cast("np.ndarray", canvas.draw())
