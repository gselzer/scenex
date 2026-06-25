from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from scenex.adaptors._base import ImageAdaptor

from ._node import Node

if TYPE_CHECKING:
    from cmap import Colormap
    from numpy.typing import ArrayLike

    from scenex import model

try:
    from qtpy.Qt3DCore import QAttribute, QBuffer, QGeometry
    from qtpy.Qt3DExtras import QTextureMaterial
    from qtpy.Qt3DRender import (
        QGeometryRenderer,
        QPaintedTextureImage,
        QTexture2D,
    )
    from qtpy.QtCore import QByteArray, QSize
    from qtpy.QtGui import QImage, QPainter
except Exception as e:
    raise ImportError(
        "Qt3D is required for the qt3d backend. "
        "Install PySide6 (includes Qt3D) or PyQt6 + PyQt6-3D."
    ) from e

logger = logging.getLogger("scenex.adaptors.qt3d")


class _RGBATextureImage(QPaintedTextureImage):
    """A Qt3D painted texture that renders a pre-computed RGBA QImage."""

    def __init__(self, rgba: np.ndarray) -> None:
        super().__init__()
        self._qimage: QImage | None = None
        self._update_qimage(rgba)

    def update_rgba(self, rgba: np.ndarray) -> None:
        self._update_qimage(rgba)
        self.update()

    def _update_qimage(self, rgba: np.ndarray) -> None:
        rgba = np.ascontiguousarray(rgba, dtype=np.uint8)
        h, w = rgba.shape[:2]
        self._qimage = QImage(
            rgba.tobytes(), w, h, w * 4, QImage.Format.Format_RGBA8888
        ).copy()
        self.setSize(QSize(w, h))

    def paint(self, painter: QPainter) -> None:
        if self._qimage is not None:
            painter.drawImage(0, 0, self._qimage)


def _make_quad_geometry(w: float, h: float) -> tuple[QGeometry, int]:
    """Build a 2-triangle quad in the XY plane from (0,0) to (w,h).

    Returns the QGeometry and vertex count.
    """
    # 4 vertices: (x, y, z) + (u, v)
    vertices = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [w, 0.0, 0.0, 1.0, 0.0],
            [w, h, 0.0, 1.0, 1.0],
            [0.0, h, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    indices = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32)

    stride = 5 * 4  # 5 floats × 4 bytes

    vertex_buffer = QBuffer()
    vertex_buffer.setData(QByteArray(vertices.tobytes()))

    index_buffer = QBuffer()
    index_buffer.setData(QByteArray(indices.tobytes()))

    pos_attr = QAttribute()
    pos_attr.setAttributeType(QAttribute.AttributeType.VertexAttribute)
    pos_attr.setVertexBaseType(QAttribute.VertexBaseType.Float)
    pos_attr.setVertexSize(3)
    pos_attr.setByteOffset(0)
    pos_attr.setByteStride(stride)
    pos_attr.setBuffer(vertex_buffer)
    pos_attr.setName(QAttribute.defaultPositionAttributeName())

    uv_attr = QAttribute()
    uv_attr.setAttributeType(QAttribute.AttributeType.VertexAttribute)
    uv_attr.setVertexBaseType(QAttribute.VertexBaseType.Float)
    uv_attr.setVertexSize(2)
    uv_attr.setByteOffset(3 * 4)
    uv_attr.setByteStride(stride)
    uv_attr.setBuffer(vertex_buffer)
    uv_attr.setName(QAttribute.defaultTextureCoordinateAttributeName())

    idx_attr = QAttribute()
    idx_attr.setAttributeType(QAttribute.AttributeType.IndexAttribute)
    idx_attr.setVertexBaseType(QAttribute.VertexBaseType.UnsignedInt)
    idx_attr.setBuffer(index_buffer)
    idx_attr.setCount(len(indices))

    geom = QGeometry()
    geom.addAttribute(pos_attr)
    geom.addAttribute(uv_attr)
    geom.addAttribute(idx_attr)

    return geom, len(indices)


def _apply_colormap(image: model.Image) -> np.ndarray:
    """Apply cmap colormap, clims and gamma to produce an RGBA uint8 array."""
    data = np.asarray(image.data)

    if data.ndim == 3 and data.shape[-1] in (3, 4):
        # RGB / RGBA: just ensure uint8
        if data.dtype != np.uint8:
            data = (
                np.clip(data, 0, 1) * 255
                if data.dtype.kind == "f"
                else data.astype(np.uint8)
            )
        if data.shape[-1] == 3:
            h, w = data.shape[:2]
            alpha = np.full((h, w, 1), 255, dtype=np.uint8)
            data = np.concatenate([data, alpha], axis=-1)
        return data.astype(np.uint8)

    # Grayscale: apply clims → gamma → colormap
    clims = image.clims
    lo, hi = (float(data.min()), float(data.max())) if clims is None else clims
    denom = hi - lo
    if denom == 0:
        denom = 1.0
    normalized = np.clip((data.astype(np.float64) - lo) / denom, 0.0, 1.0)
    if image.gamma != 1.0:
        normalized = normalized ** float(image.gamma)

    rgba_f = np.asarray(image.cmap(normalized))  # (H, W, 4) float64 0-1
    return (rgba_f * 255).astype(np.uint8)


class Image(Node, ImageAdaptor):
    """Qt3D backend adaptor for an Image node.

    Renders a textured quad in the XY plane.  Colormap, clims, and gamma are
    applied on the CPU using the `cmap` library — the resulting RGBA uint8
    array is uploaded to a QPaintedTextureImage.
    """

    def __init__(self, image: model.Image, **backend_kwargs: Any) -> None:
        self._model = image
        self._init_entity()

        rgba = _apply_colormap(image)
        h, w = rgba.shape[:2]

        # Texture
        self._tex_image = _RGBATextureImage(rgba)
        self._texture = QTexture2D(self._qt_entity)
        self._texture.addTextureImage(self._tex_image)

        # Geometry
        geom, n_indices = _make_quad_geometry(float(w), float(h))

        renderer = QGeometryRenderer(self._qt_entity)
        renderer.setGeometry(geom)
        renderer.setPrimitiveType(QGeometryRenderer.PrimitiveType.Triangles)
        renderer.setVertexCount(n_indices)  # number of indices in the index buffer
        renderer.setIndexOffset(0)
        renderer.setInstanceCount(1)

        # Material
        self._material = QTextureMaterial(self._qt_entity)
        self._material.setTexture(self._texture)

        self._qt_entity.addComponent(renderer)
        self._qt_entity.addComponent(self._material)

    def _snx_set_data(self, data: ArrayLike) -> None:
        rgba = _apply_colormap(self._model)
        self._tex_image.update_rgba(rgba)

    def _snx_set_cmap(self, arg: Colormap) -> None:
        rgba = _apply_colormap(self._model)
        self._tex_image.update_rgba(rgba)

    def _snx_set_clims(self, arg: tuple[float, float] | None) -> None:
        rgba = _apply_colormap(self._model)
        self._tex_image.update_rgba(rgba)

    def _snx_set_gamma(self, arg: float) -> None:
        rgba = _apply_colormap(self._model)
        self._tex_image.update_rgba(rgba)

    def _snx_set_interpolation(self, arg: model.InterpolationMode) -> None:
        from qtpy.Qt3DRender import QAbstractTexture

        match arg:
            case "nearest":
                self._texture.setMagnificationFilter(QAbstractTexture.Filter.Nearest)
                self._texture.setMinificationFilter(QAbstractTexture.Filter.Nearest)
            case _:  # "linear" / "bicubic" → fall back to linear
                if arg == "bicubic":
                    logger.warning(
                        "Bicubic interpolation not supported by qt3d backend — "
                        "falling back to linear"
                    )
                self._texture.setMagnificationFilter(QAbstractTexture.Filter.Linear)
                self._texture.setMinificationFilter(QAbstractTexture.Filter.Linear)
