"""Material helpers using Qt3D built-in QPerVertexColorMaterial.

All geometry types (Line, Points, Mesh) use per-vertex colours so that a single
material class works with every rendering backend Qt3D supports (OpenGL and RHI).
For "uniform" colour nodes the single RGBA is replicated across all vertices at the
geometry level; dynamic colour changes update the per-vertex buffer directly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scenex import model

try:
    from qtpy.Qt3DExtras import QPerVertexColorMaterial
except Exception as e:
    raise ImportError(
        "Qt3D is required for the qt3d backend. "
        "Install PySide6 (includes Qt3D) or PyQt6 + PyQt6-3D."
    ) from e


def make_material() -> QPerVertexColorMaterial:
    """Return a new unlit per-vertex-colour material."""
    return QPerVertexColorMaterial()


def color_model_to_rgba(
    color_model: model.ColorModel,
) -> tuple[
    tuple[float, float, float, float] | None,
    list[tuple[float, float, float, float]] | None,
]:
    """Return ``(uniform_rgba, per_vertex_colors)`` — exactly one is non-None."""
    from scenex.model._color import UniformColor, VertexColors

    if isinstance(color_model, UniformColor):
        rgba = tuple(float(c) for c in color_model.color.rgba)
        return rgba, None  # type: ignore[return-value]
    if isinstance(color_model, VertexColors):
        colors = [tuple(float(c) for c in col.rgba) for col in color_model.color]
        return None, colors  # type: ignore[return-value]
    # FaceColors: fall back to a neutral grey
    return (0.5, 0.5, 0.5, 1.0), None


def expand_colors(
    color_model: model.ColorModel,
    n_verts: int,
) -> list[tuple[float, float, float, float]]:
    """Return exactly *n_verts* RGBA tuples for the given ColorModel.

    Uniform colours are replicated; per-vertex colours are returned as-is.
    """
    uniform_rgba, per_vertex = color_model_to_rgba(color_model)
    if per_vertex is not None:
        return per_vertex
    rgba = uniform_rgba or (1.0, 1.0, 1.0, 1.0)
    return [rgba] * n_verts
