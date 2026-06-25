from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from scenex.adaptors._base import MeshAdaptor

from ._geometry import make_mesh_geometry, update_buffer
from ._material import expand_colors, make_material
from ._node import Node

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from scenex import model

logger = logging.getLogger("scenex.adaptors.qt3d")


def _vert_src(core: bool) -> bytes:
    version = b"#version 150 core" if core else b"#version 130"
    return (
        version
        + b"""
in vec3 vertexPosition;
in vec4 vertexColor;
uniform mat4 modelViewProjection;
out vec4 vColor;
void main() {
    gl_Position = modelViewProjection * vec4(vertexPosition, 1.0);
    vColor = vertexColor;
}
"""
    )


def _frag_src(core: bool) -> bytes:
    version = b"#version 150 core" if core else b"#version 130"
    return (
        version
        + b"""
in vec4 vColor;
out vec4 fragColor;
void main() {
    fragColor = vColor;
}
"""
    )


def _setup_mesh_material(mat: Any) -> None:
    """Patch QPerVertexColorMaterial for mesh rendering.

    - Replaces default shaders with per-vertex-color variants that compile on
      both GL Core Profile (GLSL 1.50) and GL NoProfile (GLSL 1.30).
    - Adds QCullFace(NoCulling) to every OpenGL render pass so that meshes
      render regardless of triangle winding order.  Qt3D's QForwardRenderer
      enables back-face culling by default, which silently drops triangles
      wound clockwise from the camera's perspective.
    """
    from qtpy.Qt3DRender import QCullFace, QGraphicsApiFilter

    try:
        effect = mat.effect()
        for technique in effect.techniques():
            f = technique.graphicsApiFilter()
            if f.api() != QGraphicsApiFilter.Api.OpenGL:
                continue
            is_core = f.profile() == QGraphicsApiFilter.OpenGLProfile.CoreProfile
            for rp in technique.renderPasses():
                shader = rp.shaderProgram()
                if shader is not None:
                    shader.setVertexShaderCode(_vert_src(is_core))
                    shader.setFragmentShaderCode(_frag_src(is_core))
                cull = QCullFace()
                cull.setMode(QCullFace.CullingMode.NoCulling)
                cull.setParent(rp)
                rp.addRenderState(cull)
    except Exception:
        logger.exception("Could not set up mesh material")


class Mesh(Node, MeshAdaptor):
    """Qt3D backend adaptor for a Mesh node."""

    def __init__(self, mesh: model.Mesh) -> None:
        self._model = mesh
        self._init_entity()

        verts = self._coerce_verts(np.asarray(mesh.vertices, dtype=np.float32))
        self._n_verts = len(verts)

        colors = expand_colors(mesh.color, self._n_verts)
        self._renderer, self._vbuf, self._cbuf, _ = make_mesh_geometry(
            verts,
            np.asarray(mesh.faces, dtype=np.uint32),
            colors,
        )

        self._material = make_material()
        _setup_mesh_material(self._material)

        self._qt_entity.addComponent(self._renderer)
        self._qt_entity.addComponent(self._material)

    @staticmethod
    def _coerce_verts(verts: np.ndarray) -> np.ndarray:
        if verts.shape[-1] == 2:
            verts = np.column_stack([verts, np.zeros(len(verts), dtype=np.float32)])
        return np.ascontiguousarray(verts, dtype=np.float32)

    def _snx_set_vertices(self, arg: NDArray) -> None:
        verts = self._coerce_verts(np.asarray(arg, dtype=np.float32))
        self._n_verts = len(verts)
        faces = self._model.faces
        if faces is None:
            return
        idx = np.ascontiguousarray(faces, dtype=np.int64).flatten()
        update_buffer(self._vbuf, np.ascontiguousarray(verts[idx], dtype=np.float32))

    def _snx_set_faces(self, arg: NDArray | None) -> None:
        if arg is None:
            return
        idx = np.ascontiguousarray(arg, dtype=np.int64).flatten()
        verts = self._coerce_verts(np.asarray(self._model.vertices, dtype=np.float32))
        update_buffer(self._vbuf, np.ascontiguousarray(verts[idx], dtype=np.float32))
        if self._cbuf is not None:
            colors = expand_colors(self._model.color, self._n_verts)
            carr = np.array(colors, dtype=np.float32).reshape(-1, 4)
            update_buffer(self._cbuf, np.ascontiguousarray(carr[idx]))
        self._renderer.setVertexCount(len(idx))

    def _snx_set_color(self, arg: model.ColorModel) -> None:
        if self._cbuf is None:
            return
        faces = self._model.faces
        if faces is None:
            return
        idx = np.ascontiguousarray(faces, dtype=np.int64).flatten()
        colors = expand_colors(arg, self._n_verts)
        carr = np.array(colors, dtype=np.float32).reshape(-1, 4)
        update_buffer(self._cbuf, np.ascontiguousarray(carr[idx]))
