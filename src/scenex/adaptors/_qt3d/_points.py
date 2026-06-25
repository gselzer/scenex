from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from scenex.adaptors._base import PointsAdaptor

from ._geometry import make_points_geometry, update_buffer
from ._material import expand_colors, make_material
from ._node import Node

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from scenex import model

logger = logging.getLogger("scenex.adaptors.qt3d")

_SUPPORTED_SYMBOLS = {"disc", "o", "circle", "square", "s"}


# ---------------------------------------------------------------------------
# Shader builders — edgeRatio is baked in as a literal to avoid relying on
# QParameter uniform binding across patched QPerVertexColorMaterial shaders.
# QPointSize(Fixed) handles point size via render state (proven to work).
# ---------------------------------------------------------------------------


def _vert_src(core: bool) -> bytes:
    version = b"#version 150 core" if core else b"#version 130"
    return (
        version
        + b"""
in vec3 vertexPosition;
in vec4 vertexColor;
in vec4 vertexEdgeColor;
uniform mat4 modelViewProjection;
out vec4 vFaceColor;
out vec4 vEdgeColor;
void main() {
    gl_Position = modelViewProjection * vec4(vertexPosition, 1.0);
    vFaceColor  = vertexColor;
    vEdgeColor  = vertexEdgeColor;
}
"""
    )


def _frag_src(inner_thresh: float, core: bool) -> bytes:
    """Fragment shader with edge threshold baked in as a literal."""
    version = b"#version 150 core" if core else b"#version 130"
    thresh = f"{inner_thresh:.6f}".encode()
    return (
        version
        + b"""
in  vec4 vFaceColor;
in  vec4 vEdgeColor;
out vec4 fragColor;
void main() {
    float dist = length(gl_PointCoord - vec2(0.5)) * 2.0;
    if (dist > 1.0) discard;
    fragColor = dist > """
        + thresh
        + b""" ? vEdgeColor : vFaceColor;
}
"""
    )


def _patch_disc_shaders(
    mat: Any,
    size: float,
    inner_thresh: float,
) -> tuple[list[Any], list[tuple[Any, bool]]]:
    """Replace QPerVertexColorMaterial's shaders with disc+edge variants.

    Uses QPointSize(Fixed) for size (no gl_PointSize in shader) and bakes the
    edge threshold into the fragment source to avoid QParameter binding issues.

    Returns
    -------
        point_size_states  – list of QPointSize objects (one per render pass)
        disc_rp_infos      – list of (render_pass, is_core) for shader updates
    """
    from qtpy.Qt3DRender import QGraphicsApiFilter, QPointSize

    ps_states: list[Any] = []
    rp_infos: list[tuple[Any, bool]] = []

    try:
        effect = mat.effect()
        for technique in effect.techniques():
            f = technique.graphicsApiFilter()
            if f.api() != QGraphicsApiFilter.Api.OpenGL:
                continue  # skip OpenGL ES techniques
            is_core = f.profile() == QGraphicsApiFilter.OpenGLProfile.CoreProfile
            vert = _vert_src(is_core)
            frag = _frag_src(inner_thresh, is_core)
            for rp in technique.renderPasses():
                shader = rp.shaderProgram()
                if shader is not None:
                    shader.setVertexShaderCode(vert)
                    shader.setFragmentShaderCode(frag)
                ps = QPointSize()
                ps.setSizeMode(QPointSize.SizeMode.Fixed)
                ps.setValue(float(size))
                ps.setParent(rp)
                rp.addRenderState(ps)
                ps_states.append(ps)
                rp_infos.append((rp, is_core))
    except Exception:
        logger.exception("Could not patch disc shaders onto QPerVertexColorMaterial")

    return ps_states, rp_infos


class Points(Node, PointsAdaptor):
    """Qt3D backend adaptor for a Points node.

    Renders GL_POINTS as disc sprites by patching QPerVertexColorMaterial's
    shaders.  The edge threshold is baked into the fragment shader source;
    point size uses QPointSize(Fixed) render state.  Both update dynamically.
    """

    def __init__(self, points: model.Points, **backend_kwargs: Any) -> None:
        self._model = points
        self._init_entity()

        verts = self._coerce_verts(np.asarray(points.vertices, dtype=np.float32))
        self._n_verts = len(verts)
        self._size = float(points.size)
        self._edge_width = float(points.edge_width)

        colors = expand_colors(points.face_color, self._n_verts)
        edge_colors = expand_colors(points.edge_color, self._n_verts)
        self._renderer, self._vbuf, self._cbuf, self._ecbuf = make_points_geometry(
            verts, colors, edge_colors
        )

        self._material = make_material()
        self._ps_states, self._rp_infos = _patch_disc_shaders(
            self._material, self._size, self._inner_thresh()
        )

        self._qt_entity.addComponent(self._renderer)
        self._qt_entity.addComponent(self._material)

        if points.symbol not in _SUPPORTED_SYMBOLS:
            logger.warning(
                "Symbol %r is not supported by the qt3d backend — rendering as default",
                points.symbol,
            )

    def _inner_thresh(self) -> float:
        """1 - edge_ratio: distance from centre where face/edge boundary sits.

        edge_width is treated as a stroke centred on the disc boundary; only the
        inner half falls inside the disc, giving ratio = edge_width / size.
        With size=20, edge_width=10 → ratio=0.5 (inner half face, outer half edge).
        """
        if self._size <= 0:
            return 1.0
        ratio = min(1.0, self._edge_width / self._size)
        return 1.0 - ratio

    def _update_frag_shaders(self) -> None:
        thresh = self._inner_thresh()
        for rp, is_core in self._rp_infos:
            shader = rp.shaderProgram()
            if shader is not None:
                shader.setFragmentShaderCode(_frag_src(thresh, is_core))

    @staticmethod
    def _coerce_verts(verts: np.ndarray) -> np.ndarray:
        if verts.ndim == 1:
            verts = verts.reshape(-1, 3 if verts.shape[0] % 3 == 0 else 2)
        if verts.shape[-1] == 2:
            verts = np.column_stack([verts, np.zeros(len(verts), dtype=np.float32)])
        return np.ascontiguousarray(verts, dtype=np.float32)

    def _snx_set_vertices(self, vertices: NDArray) -> None:
        verts = self._coerce_verts(np.asarray(vertices, dtype=np.float32))
        self._n_verts = len(verts)
        update_buffer(self._vbuf, verts)
        self._renderer.setVertexCount(self._n_verts)
        if self._cbuf is not None:
            colors = expand_colors(self._model.face_color, self._n_verts)
            update_buffer(self._cbuf, np.array(colors, dtype=np.float32))
        if self._ecbuf is not None:
            edge_colors = expand_colors(self._model.edge_color, self._n_verts)
            update_buffer(self._ecbuf, np.array(edge_colors, dtype=np.float32))

    def _snx_set_size(self, arg: float) -> None:
        self._size = max(1.0, float(arg))
        for ps in self._ps_states:
            ps.setValue(self._size)
        # edge ratio depends on size, so regenerate frag shader too
        self._update_frag_shaders()

    def _snx_set_face_color(self, arg: model.ColorModel) -> None:
        if self._cbuf is not None:
            colors = expand_colors(arg, self._n_verts)
            update_buffer(self._cbuf, np.array(colors, dtype=np.float32))

    def _snx_set_edge_color(self, arg: model.ColorModel) -> None:
        if self._ecbuf is not None:
            edge_colors = expand_colors(arg, self._n_verts)
            update_buffer(self._ecbuf, np.array(edge_colors, dtype=np.float32))

    def _snx_set_edge_width(self, arg: float) -> None:
        self._edge_width = max(0.0, float(arg))
        self._update_frag_shaders()

    def _snx_set_symbol(self, arg: str) -> None:
        if arg not in _SUPPORTED_SYMBOLS:
            logger.warning(
                "Symbol %r is not supported by the qt3d backend — rendering as default",
                arg,
            )

    def _snx_set_scaling(self, arg: model.ScalingMode) -> None:
        pass

    def _snx_set_antialias(self, arg: bool) -> None:
        pass
