from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from scenex.adaptors._base import TextAdaptor

from ._node import Node

if TYPE_CHECKING:
    from cmap import Color

    from scenex import model

try:
    from qtpy.Qt3DExtras import QText2DEntity
    from qtpy.QtGui import QColor
except Exception as e:
    raise ImportError(
        "Qt3D is required for the qt3d backend. "
        "Install PySide6 (includes Qt3D) or PyQt6 + PyQt6-3D."
    ) from e

logger = logging.getLogger("scenex.adaptors.qt3d")


class Text(Node, TextAdaptor):
    """Qt3D backend adaptor for a Text node.

    Uses QText2DEntity which provides raster-based text rendering.  SDF
    antialiasing is not available; the ``antialias`` flag is accepted but has
    no effect.
    """

    _qt_text: QText2DEntity

    def __init__(self, text: model.Text, **backend_kwargs: Any) -> None:
        self._model = text
        self._init_entity()

        self._qt_text = QText2DEntity(self._qt_entity)
        self._qt_text.setText(text.text)
        self._qt_text.setWidth(200.0)  # default width; content determines layout
        self._qt_text.setHeight(50.0)

        self._snx_set_color(text.color)
        self._snx_set_size(text.size)

    def _snx_set_text(self, arg: str) -> None:
        self._qt_text.setText(arg)

    def _snx_set_color(self, arg: Color) -> None:
        rgba = arg.rgba
        r, g, b, a = (int(c * 255) for c in rgba)
        self._qt_text.setColor(QColor(r, g, b, a))

    def _snx_set_size(self, arg: int) -> None:
        self._qt_text.setHeight(float(arg))

    def _snx_set_antialias(self, arg: bool) -> None:
        # QText2DEntity uses raster rendering; antialiasing is always applied
        # internally by Qt's text engine and cannot be toggled from Python.
        pass
