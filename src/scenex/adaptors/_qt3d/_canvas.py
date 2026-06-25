from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np

from scenex.adaptors._base import CanvasAdaptor
from scenex.app import app

from ._adaptor_registry import get_adaptor

if TYPE_CHECKING:
    from cmap import Color

    from scenex import model

    from ._view import View

try:
    from qtpy.Qt3DCore import QEntity
    from qtpy.Qt3DExtras import Qt3DWindow
    from qtpy.Qt3DInput import QInputSettings
    from qtpy.QtGui import QColor
    from qtpy.QtWidgets import QWidget
except Exception as e:
    raise ImportError(
        "Qt3D is required for the qt3d backend. "
        "Install PySide6 (includes Qt3D) or PyQt6 + PyQt6-3D."
    ) from e


class Canvas(CanvasAdaptor):
    """Canvas adaptor for the Qt3D backend.

    Wraps a Qt3DExtras.Qt3DWindow embedded inside a QWidget container so it
    can be used as a normal widget in any Qt layout.
    """

    def __init__(self, canvas: model.Canvas, **backend_kwargs: Any) -> None:
        self._model = canvas

        # Ensure a QApplication is running
        app()

        # Qt3D's RHI/DirectX backend on Windows fails to build graphics
        # pipelines with custom geometry (COM error 0x80070057 in
        # CreateInputLayout).  Force the legacy OpenGL renderer which is
        # stable and handles QGeometry/QAttribute correctly.
        import os

        os.environ.setdefault("QT3D_RENDERER", "opengl")

        self._window = Qt3DWindow()  # type: ignore[no-untyped-call]

        # Embed the QWindow inside a QWidget so the rest of scenex's Qt machinery
        # (event filters, show/hide, etc.) operates on a regular widget.
        self._container = QWidget.createWindowContainer(self._window)  # type: ignore[no-untyped-call]
        self._container.setMinimumSize(canvas.width, canvas.height)
        self._container.resize(canvas.width, canvas.height)
        # Enable mouse tracking so MouseMove events arrive even without buttons held.
        self._container.setMouseTracking(True)

        # Root entity that owns everything in the scene
        self._root = QEntity()

        # Qt3D input settings must be attached to the root entity and point to the
        # window so that camera controllers and other input handlers receive events.
        self._input_settings = QInputSettings()
        self._input_settings.setEventSource(self._window)  # type: ignore[arg-type]
        self._root.addComponent(self._input_settings)

        # Disable frustum culling: without setCount() on every attribute Qt3D
        # may fail to compute a valid bounding volume before the first frame,
        # causing entities to be silently dropped.  We set setCount() now, but
        # disabling culling is a cheap belt-and-suspenders safety net.
        fg = self._window.defaultFrameGraph()  # type: ignore[attr-defined]
        fg.setFrustumCullingEnabled(False)

        self._window.setRootEntity(self._root)

        if canvas.background_color is not None:
            self._snx_set_background_color(canvas.background_color)

        self._views: list[model.View] = []
        for view in canvas.views:
            self._snx_add_view(view)

        # On Windows, createWindowContainer forwards all input to the embedded
        # QWindow rather than the container QWidget, so install the filter on the
        # Qt3DWindow itself (installEventFilter works on any QObject).
        self._filter = app().install_event_filter(self._window, canvas.handle)  # type: ignore[arg-type]

    # ------------------------------------------------------------------
    # CanvasAdaptor interface
    # ------------------------------------------------------------------

    def _snx_get_native(self) -> Any:
        return self._container

    def _snx_set_visible(self, arg: bool) -> None:
        app().show(self._container, arg)

    def _snx_add_view(self, view: model.View) -> None:
        if view in self._views:
            return
        view_adaptor = cast("View", get_adaptor(view))
        view_adaptor._attach_to_canvas(self)
        self._views.append(view)

    def _snx_set_width(self, arg: int) -> None:
        self._container.resize(self._model.width, self._model.height)

    def _snx_set_height(self, arg: int) -> None:
        self._container.resize(self._model.width, self._model.height)

    def _snx_set_background_color(self, arg: Color | None) -> None:
        if arg is None:
            color = QColor("black")
        else:
            r, g, b, a = (int(c * 255) for c in arg.rgba)
            color = QColor(r, g, b, a)
        self._window.defaultFrameGraph().setClearColor(color)  # type: ignore[attr-defined]

    def _snx_set_title(self, arg: str) -> None:
        self._window.setTitle(arg)  # type: ignore[arg-type]
        self._container.setWindowTitle(arg)

    def _snx_close(self) -> None:
        self._container.close()

    def _snx_render(self) -> np.ndarray:
        # Grab the container widget's rendered pixels.
        from qtpy.QtWidgets import QApplication

        QApplication.processEvents()
        pixmap = self._container.grab()
        qimage = pixmap.toImage()
        w, h = qimage.width(), qimage.height()
        # Qt stores ARGB in native byte order; convert to RGBA uint8.
        ptr = qimage.bits()
        try:
            arr = np.frombuffer(ptr, dtype=np.uint8).reshape((h, w, 4)).copy()
        except Exception:
            return np.zeros((h, w, 4), dtype=np.uint8)
        # Qt uses BGRA (or ARGB depending on format) — convert to RGBA
        arr = arr[:, :, [2, 1, 0, 3]]
        return arr
