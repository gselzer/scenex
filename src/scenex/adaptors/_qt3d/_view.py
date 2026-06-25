from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from scenex.adaptors._base import ViewAdaptor

from ._adaptor_registry import get_adaptor

if TYPE_CHECKING:
    from cmap import Color

    from scenex import model

    from ._camera import Camera
    from ._canvas import Canvas
    from ._scene import Scene

logger = logging.getLogger("scenex.adaptors.qt3d")


class View(ViewAdaptor):
    """View adaptor for the Qt3D backend.

    A View wires a scenex Scene (entity hierarchy) and a scenex Camera (QCamera
    configuration) to the shared Qt3DWindow.  The canvas calls
    `_attach_to_canvas` when this view is added; until then the adaptor is inert.
    """

    def __init__(self, view: model.View, **backend_kwargs: Any) -> None:
        self._model = view
        self._canvas: Canvas | None = None

        # Listen for layout changes so we can propagate background color.
        view.layout.events.background_color.connect(self._set_background_color)

    # ------------------------------------------------------------------
    # Called by Canvas._snx_add_view
    # ------------------------------------------------------------------

    def _attach_to_canvas(self, canvas: Canvas) -> None:
        """Wire this view to a canvas after creation."""
        self._canvas = canvas

        # Attach the scene entity tree to the canvas root.
        self._snx_set_scene(self._model.scene)

        # Configure the canvas's camera from the model camera.
        self._snx_set_camera(self._model.camera)

        # Auto-fit: position the Qt3D camera so the full scene is visible by
        # default.  This mirrors vispy's behaviour for new views.
        self._fit_camera_to_scene(self._model.scene)

        # Apply background color (on the canvas framegraph since Qt3D has one
        # clear color per framegraph, not per viewport in V1).
        self._set_background_color(self._model.layout.background_color)

    # ------------------------------------------------------------------
    # ViewAdaptor interface
    # ------------------------------------------------------------------

    def _snx_set_visible(self, arg: bool) -> None:
        pass

    def _snx_set_scene(self, scene: model.Scene) -> None:
        if self._canvas is None:
            return
        scene_adaptor = cast("Scene", get_adaptor(scene))
        scene_adaptor._qt_entity.setParent(self._canvas._root)

    def _snx_set_camera(self, cam: model.Camera) -> None:
        if self._canvas is None:
            return
        cam_adaptor = cast("Camera", get_adaptor(cam))
        qt_camera = self._canvas._window.camera()  # type: ignore[attr-defined]
        cam_adaptor._configure(qt_camera)

    def _snx_render(self) -> np.ndarray:
        if self._canvas is None:
            h = int(self._model.layout.height or 480)
            w = int(self._model.layout.width or 640)
            return np.zeros((h, w, 4), dtype=np.uint8)
        return self._canvas._snx_render()

    def _set_background_color(self, color: Color | None) -> None:
        if self._canvas is None:
            return
        self._canvas._snx_set_background_color(color)

    # ------------------------------------------------------------------
    # Auto-fit helpers
    # ------------------------------------------------------------------

    def _fit_camera_to_scene(self, scene: model.Scene) -> None:
        """Compute scene bounds and zoom the Qt3D camera to show everything.

        This is a one-time default view; subsequent model-driven camera updates
        (e.g. from a PanZoom controller) override this via _snx_set_projection.
        """
        if self._canvas is None:
            return

        try:
            from qtpy.QtGui import QVector3D
        except Exception:
            return

        bounds = self._compute_scene_bounds(scene)
        if bounds is None:
            return

        x_min, x_max, y_min, y_max = bounds

        # Guard against degenerate (empty / point) geometry.
        if x_max <= x_min:
            x_min -= 0.5
            x_max += 0.5
        if y_max <= y_min:
            y_min -= 0.5
            y_max += 0.5

        # Add 5 % margin on each side.
        pad_x = (x_max - x_min) * 0.05
        pad_y = (y_max - y_min) * 0.05
        x_min -= pad_x
        x_max += pad_x
        y_min -= pad_y
        y_max += pad_y

        # Adjust y-range to match canvas aspect ratio so geometry is not
        # stretched.
        canvas_w = self._canvas._model.width
        canvas_h = self._canvas._model.height
        if canvas_h > 0 and canvas_w > 0:
            scene_aspect = (x_max - x_min) / (y_max - y_min)
            canvas_aspect = canvas_w / canvas_h
            if scene_aspect > canvas_aspect:
                # Scene is wider than canvas: expand y to match.
                y_center = (y_min + y_max) / 2
                half_h = (x_max - x_min) / canvas_aspect / 2
                y_min = y_center - half_h
                y_max = y_center + half_h
            else:
                # Scene is taller than canvas: expand x to match.
                x_center = (x_min + x_max) / 2
                half_w = (y_max - y_min) * canvas_aspect / 2
                x_min = x_center - half_w
                x_max = x_center + half_w

        cx = (x_min + x_max) / 2
        cy = (y_min + y_max) / 2
        # Eye-space half-extents (camera is centred on cx,cy).
        half_w = (x_max - x_min) / 2.0
        half_h = (y_max - y_min) / 2.0

        qt_camera = self._canvas._window.camera()  # type: ignore[attr-defined]
        # Camera sits 1 unit in front of the scene plane (z=1 → z=0).
        # setOrthographicProjection takes eye-space bounds, so use ±half_w/h.
        qt_camera.lens().setOrthographicProjection(
            -float(half_w),
            float(half_w),
            -float(half_h),
            float(half_h),
            0.01,
            10_000.0,
        )
        qt_camera.setPosition(QVector3D(float(cx), float(cy), 1.0))
        qt_camera.setViewCenter(QVector3D(float(cx), float(cy), 0.0))
        qt_camera.setUpVector(QVector3D(0.0, 1.0, 0.0))

    @staticmethod
    def _compute_scene_bounds(
        scene: model.Scene,
    ) -> tuple[float, float, float, float] | None:
        """Return (x_min, x_max, y_min, y_max) across all nodes with vertices."""
        verts_list: list[np.ndarray] = []
        View._collect_vertices(scene, verts_list)
        if not verts_list:
            return None
        verts = np.vstack(verts_list)
        return (
            float(verts[:, 0].min()),
            float(verts[:, 0].max()),
            float(verts[:, 1].min()),
            float(verts[:, 1].max()),
        )

    @staticmethod
    def _collect_vertices(node: model.Node, out: list[np.ndarray]) -> None:
        """Recursively collect vertex arrays from all nodes that have them."""
        if hasattr(node, "vertices") and node.vertices is not None:
            v = np.asarray(node.vertices, dtype=np.float32)
            if v.ndim == 1:
                v = v.reshape(-1, 3 if v.shape[0] % 3 == 0 else 2)
            if v.shape[0] > 0:
                out.append(v)
        for child in getattr(node, "children", []):
            View._collect_vertices(child, out)
