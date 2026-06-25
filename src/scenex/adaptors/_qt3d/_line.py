from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from scenex.adaptors._base import LineAdaptor

from ._geometry import make_line_geometry, update_buffer
from ._material import expand_colors, make_material
from ._node import Node

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from scenex import model

logger = logging.getLogger("scenex.adaptors.qt3d")


class Line(Node, LineAdaptor):
    """Qt3D backend adaptor for a Line node.

    Renders as a GL_LINE_STRIP using QPerVertexColorMaterial.  Line widths > 1
    are not guaranteed on OpenGL core profile; the ``width`` field is accepted
    but may be silently clamped to 1 by the GPU driver.
    """

    def __init__(self, line: model.Line, **backend_kwargs: Any) -> None:
        self._model = line
        self._init_entity()

        verts = self._coerce_verts(np.asarray(line.vertices, dtype=np.float32))
        self._n_verts = len(verts)

        colors = expand_colors(line.color, self._n_verts)
        self._renderer, self._vbuf, self._cbuf = make_line_geometry(verts, colors)
        self._material = make_material()

        self._qt_entity.addComponent(self._renderer)
        self._qt_entity.addComponent(self._material)

        if line.width > 1:
            logger.warning(
                "Line width > 1 is not supported on OpenGL core profile; "
                "rendering with width=1."
            )

    @staticmethod
    def _coerce_verts(verts: np.ndarray) -> np.ndarray:
        if verts.ndim == 1:
            verts = verts.reshape(-1, 2)
        if verts.shape[-1] == 2:
            verts = np.column_stack([verts, np.zeros(len(verts), dtype=np.float32)])
        return np.ascontiguousarray(verts, dtype=np.float32)

    def _snx_set_vertices(self, arg: NDArray) -> None:
        verts = self._coerce_verts(np.asarray(arg, dtype=np.float32))
        self._n_verts = len(verts)
        update_buffer(self._vbuf, verts)
        self._renderer.setVertexCount(self._n_verts)
        # Keep colours consistent with the new vertex count
        if self._cbuf is not None:
            colors = expand_colors(self._model.color, self._n_verts)
            update_buffer(self._cbuf, np.array(colors, dtype=np.float32))

    def _snx_set_color(self, arg: model.ColorModel) -> None:
        if self._cbuf is not None:
            colors = expand_colors(arg, self._n_verts)
            update_buffer(self._cbuf, np.array(colors, dtype=np.float32))

    def _snx_set_width(self, arg: float) -> None:
        if arg > 1:
            logger.warning(
                "Line width > 1 is not supported on OpenGL core profile; "
                "rendering with width=1."
            )

    def _snx_set_antialias(self, arg: bool) -> None:
        pass
