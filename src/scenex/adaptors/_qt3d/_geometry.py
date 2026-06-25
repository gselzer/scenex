"""Reusable helpers for building Qt3D geometry from numpy arrays."""

from __future__ import annotations

import numpy as np

try:
    from qtpy.Qt3DCore import QAttribute, QBuffer, QGeometry
    from qtpy.Qt3DRender import QGeometryRenderer
    from qtpy.QtCore import QByteArray
except Exception as e:
    raise ImportError(
        "Qt3D is required for the qt3d backend. "
        "Install PySide6 (includes Qt3D) or PyQt6 + PyQt6-3D."
    ) from e


# ---------------------------------------------------------------------------
# Internal attribute builders — do NOT call addAttribute inside these; the
# caller is responsible for both addAttribute and setParent(geom).
# ---------------------------------------------------------------------------


def _pos_attr(buf: QBuffer, stride: int, count: int, offset: int = 0) -> QAttribute:
    a = QAttribute()
    a.setAttributeType(QAttribute.AttributeType.VertexAttribute)
    a.setVertexBaseType(QAttribute.VertexBaseType.Float)
    a.setVertexSize(3)
    a.setByteStride(stride)
    a.setByteOffset(offset)
    a.setCount(count)  # required for bounding-volume computation
    a.setBuffer(buf)
    a.setName(QAttribute.defaultPositionAttributeName())
    return a


def _color_attr(
    buf: QBuffer, stride: int, offset: int, name: str | None = None
) -> QAttribute:
    a = QAttribute()
    a.setAttributeType(QAttribute.AttributeType.VertexAttribute)
    a.setVertexBaseType(QAttribute.VertexBaseType.Float)
    a.setVertexSize(4)
    a.setByteStride(stride)
    a.setByteOffset(offset)
    a.setBuffer(buf)
    a.setName(name if name is not None else QAttribute.defaultColorAttributeName())
    return a


def _idx_attr(buf: QBuffer, count: int) -> QAttribute:
    a = QAttribute()
    a.setAttributeType(QAttribute.AttributeType.IndexAttribute)
    a.setVertexBaseType(QAttribute.VertexBaseType.UnsignedInt)
    a.setCount(count)
    a.setBuffer(buf)
    return a


def _attach(geom: QGeometry, attr: QAttribute) -> None:
    """Parent *attr* to *geom* (C++ ownership) then register it."""
    attr.setParent(geom)  # geom owns attr; safe to drop Python ref
    geom.addAttribute(attr)


# ---------------------------------------------------------------------------
# Public geometry builders
#
# Ownership chain established here:
#   renderer  →owns→  geom  →owns→  pos_attr, color_attr (index_attr, ibuf)
#
# vbuf and cbuf are NOT parented to geom because callers keep Python refs to
# them in order to call update_buffer() later.  Those Python refs are the
# sole thing keeping vbuf/cbuf alive.
#
# Return value: (renderer, vbuf, cbuf_or_None)
# ---------------------------------------------------------------------------


def make_points_geometry(
    positions: np.ndarray,
    colors: list[tuple[float, float, float, float]] | None,
    edge_colors: list[tuple[float, float, float, float]] | None = None,
) -> tuple[QGeometryRenderer, QBuffer, QBuffer | None, QBuffer | None]:
    """Build geometry for a point cloud.

    Returns (renderer, vbuf, cbuf, ecbuf) — ecbuf is the edge-color buffer
    (attribute name "vertexEdgeColor") or None if edge_colors was None.
    """
    pts = np.ascontiguousarray(positions, dtype=np.float32)
    if pts.ndim == 1:
        pts = pts.reshape(-1, 3)
    elif pts.shape[-1] == 2:
        pts = np.column_stack([pts, np.zeros(len(pts), dtype=np.float32)])
    n = len(pts)

    geom = QGeometry()

    vbuf = QBuffer()
    vbuf.setData(QByteArray(pts.tobytes()))
    _attach(geom, _pos_attr(vbuf, 12, n))

    cbuf: QBuffer | None = None
    if colors is not None:
        carr = np.array(colors, dtype=np.float32).reshape(-1, 4)
        cbuf = QBuffer()
        cbuf.setData(QByteArray(carr.tobytes()))
        _attach(geom, _color_attr(cbuf, 16, 0))

    ecbuf: QBuffer | None = None
    if edge_colors is not None:
        earr = np.array(edge_colors, dtype=np.float32).reshape(-1, 4)
        ecbuf = QBuffer()
        ecbuf.setData(QByteArray(earr.tobytes()))
        _attach(geom, _color_attr(ecbuf, 16, 0, "vertexEdgeColor"))

    renderer = QGeometryRenderer()
    renderer.setGeometry(geom)
    geom.setParent(renderer)  # renderer now owns geom (and all its children)
    renderer.setPrimitiveType(QGeometryRenderer.PrimitiveType.Points)
    renderer.setVertexCount(n)

    return renderer, vbuf, cbuf, ecbuf


def make_line_geometry(
    positions: np.ndarray,
    colors: list[tuple[float, float, float, float]] | None,
) -> tuple[QGeometryRenderer, QBuffer, QBuffer | None]:
    """Build geometry for a polyline (LineStrip)."""
    pts = np.ascontiguousarray(positions, dtype=np.float32)
    if pts.ndim == 1:
        pts = pts.reshape(-1, 3)
    elif pts.shape[-1] == 2:
        pts = np.column_stack([pts, np.zeros(len(pts), dtype=np.float32)])
    n = len(pts)

    geom = QGeometry()

    vbuf = QBuffer()
    vbuf.setData(QByteArray(pts.tobytes()))
    _attach(geom, _pos_attr(vbuf, 12, n))

    cbuf: QBuffer | None = None
    if colors is not None:
        carr = np.array(colors, dtype=np.float32).reshape(-1, 4)
        cbuf = QBuffer()
        cbuf.setData(QByteArray(carr.tobytes()))
        _attach(geom, _color_attr(cbuf, 16, 0))

    renderer = QGeometryRenderer()
    renderer.setGeometry(geom)
    geom.setParent(renderer)  # renderer now owns geom (and all its children)
    renderer.setPrimitiveType(QGeometryRenderer.PrimitiveType.LineStrip)
    renderer.setVertexCount(n)

    return renderer, vbuf, cbuf


def make_mesh_geometry(
    vertices: np.ndarray,
    faces: np.ndarray,
    colors: list[tuple[float, float, float, float]] | None,
) -> tuple[QGeometryRenderer, QBuffer, QBuffer | None, QBuffer]:
    """Build triangle mesh geometry (flat/expanded — no index buffer).

    Vertices and colors are expanded by face index so each triangle gets its
    own three vertex slots.  This avoids Qt3D indexed-rendering issues on some
    drivers while keeping the public API identical.

    Returns (renderer, vbuf, cbuf, ibuf) where ibuf is a placeholder kept for
    interface compatibility; face updates rewrite vbuf/cbuf instead.
    """
    verts = np.ascontiguousarray(vertices, dtype=np.float32)
    if verts.shape[-1] == 2:
        verts = np.column_stack([verts, np.zeros(len(verts), dtype=np.float32)])
    idx = np.ascontiguousarray(
        faces, dtype=np.int64
    ).flatten()  # int64 for numpy indexing
    flat_verts = np.ascontiguousarray(verts[idx], dtype=np.float32)  # (n_idx, 3)
    n_draw = len(idx)

    geom = QGeometry()

    vbuf = QBuffer()
    vbuf.setData(QByteArray(flat_verts.tobytes()))
    _attach(geom, _pos_attr(vbuf, 12, n_draw))

    cbuf: QBuffer | None = None
    if colors is not None:
        carr = np.array(colors, dtype=np.float32).reshape(-1, 4)
        flat_colors = np.ascontiguousarray(carr[idx])  # (n_idx, 4)
        cbuf = QBuffer()
        cbuf.setData(QByteArray(flat_colors.tobytes()))
        _attach(geom, _color_attr(cbuf, 16, 0))

    # Placeholder buffer — not used for rendering but returned for API compat.
    ibuf = QBuffer()

    renderer = QGeometryRenderer()
    renderer.setGeometry(geom)
    geom.setParent(renderer)
    renderer.setPrimitiveType(QGeometryRenderer.PrimitiveType.Triangles)
    renderer.setVertexCount(n_draw)

    return renderer, vbuf, cbuf, ibuf


def update_buffer(buf: QBuffer, data: np.ndarray) -> None:
    """Replace the data in an existing QBuffer."""
    buf.setData(QByteArray(np.ascontiguousarray(data).tobytes()))
