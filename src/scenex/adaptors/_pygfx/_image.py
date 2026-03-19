from __future__ import annotations

import logging
from functools import lru_cache
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np
import pygfx

from scenex.adaptors._base import ImageAdaptor

from ._node import Node

if TYPE_CHECKING:
    from cmap import Colormap
    from numpy.typing import ArrayLike

    from scenex import model

logger = logging.getLogger("scenex.adaptors.pygfx")

# Certain numpy data types are not supported by pygfx. We downcast them to another type
# defined in this dict.
DOWNCASTS = {
    np.dtype("float64"): np.dtype("float32"),
    np.dtype("int64"): np.dtype("int32"),
    np.dtype("uint64"): np.dtype("uint32"),
}


class _TextureParams(NamedTuple):
    """Parameters that uniquely describe a pygfx Texture's structure."""

    shape: tuple[int, ...]
    dtype: np.dtype
    dim: int


@lru_cache(maxsize=1)
def _get_max_texture_sizes() -> tuple[int | None, int | None]:
    """Return (max_2d, max_3d) texture dimensions from the wgpu adapter."""
    try:
        import wgpu

        adapter = wgpu.gpu.request_adapter_sync()
        limits = adapter.limits
        return limits.get("max-texture-dimension-2d"), limits.get(
            "max-texture-dimension-3d"
        )
    except Exception:
        return None, None


def _downsample_data(
    data: np.ndarray, max_size: int
) -> tuple[np.ndarray, tuple[int, int]]:
    """Downsample spatial axes of *data* so none exceeds *max_size*.

    Uses a strided NumPy view (no copy). For 2-D arrays the shape is
    ``(rows, cols)``; for 3-D arrays with a trailing colour channel
    (shape[-1] <= 4) only the two spatial axes are checked.

    Returns the (possibly downsampled) array and a ``(row_factor, col_factor)``
    tuple of per-spatial-axis stride factors.
    """
    has_channel = data.ndim == 3 and data.shape[-1] <= 4
    spatial_shape = data.shape[:-1] if has_channel else data.shape
    row_f, col_f = (
        int(np.ceil(s / max_size)) if s > max_size else 1 for s in spatial_shape
    )
    if row_f > 1 or col_f > 1:
        logger.warning(
            "Data shape %s exceeds max texture dimension (%d) and will be "
            "downsampled for rendering (strides: (%d, %d)).",
            data.shape,
            max_size,
            row_f,
            col_f,
        )
        slices: tuple[slice, ...] = (slice(None, None, row_f), slice(None, None, col_f))
        if has_channel:
            slices = (*slices, slice(None))
        data = data[slices]
    return data, (row_f, col_f)


class Image(Node, ImageAdaptor):
    """pygfx backend adaptor for an Image node."""

    _pygfx_node: pygfx.Image
    _material: pygfx.ImageBasicMaterial
    _geometry: pygfx.Geometry
    _downsample_factors: tuple[int, int]

    def __init__(self, image: model.Image, **backend_kwargs: Any) -> None:
        self._model = image
        self._downsample_factors = (1, 1)
        self._material = pygfx.ImageBasicMaterial(
            clim=image.clims,
            # This value has model render order win for coplanar objects
            depth_compare="<=",
        )
        self._snx_set_data(image.data)
        self._pygfx_node = pygfx.Image(self._geometry, self._material)

    def _snx_set_cmap(self, arg: Colormap) -> None:
        if np.asarray(self._model.data).ndim == 3:
            self._material.map = None
        else:
            self._material.map = arg.to_pygfx()

    def _snx_set_clims(self, arg: tuple[float, float] | None) -> None:
        self._material.clim = arg

    def _snx_set_gamma(self, arg: float) -> None:
        self._material.gamma = arg

    def _snx_set_interpolation(self, arg: model.InterpolationMode) -> None:
        if arg == "bicubic":
            logger.warning(
                "Bicubic interpolation not supported by pygfx - falling back to linear",
            )
            self._model.interpolation = "linear"
            return
        self._material.interpolation = arg

    def _snx_set_transform(self, arg: model.Transform) -> None:
        matrix = arg.root.T
        row_f, col_f = self._downsample_factors
        if row_f > 1 or col_f > 1:
            # Pre-multiply: scale local space before applying the user transform.
            # factors are (row=y, col=x) -> diagonal is [x_scale, y_scale, 1, 1]
            scale_mat = np.diag([float(col_f), float(row_f), 1.0, 1.0])
            matrix = matrix @ scale_mat
        self._pygfx_node.local.matrix = matrix

    def _compute_texture_params(
        self, data: np.ndarray
    ) -> tuple[np.ndarray, _TextureParams, tuple[int, int]]:
        """Derive the pygfx Texture parameters needed for *data*.

        Applies dtype downcasting and spatial downsampling (both as views where
        possible).  Returns a 3-tuple of:

        - ``processed``: the (view of) data ready for upload — **not** a copy,
          so callers that need to own the buffer must ``.copy()`` it.
        - ``params``: a :class:`_TextureParams` describing the texture structure.
        - ``factors``: ``(row_factor, col_factor)`` downsample stride multipliers.
        """
        dim = data.ndim
        if dim > 2 and data.shape[-1] <= 4:
            dim -= 1  # last axis is colour channels, not a spatial dimension

        if data.dtype in DOWNCASTS:
            cast_to = DOWNCASTS[data.dtype]
            logger.warning(
                "Downcasting image data from %s to %s for pygfx compatibility",
                data.dtype.name,
                cast_to.name,
            )
            data = data.astype(cast_to)  # astype always returns a copy

        factors: tuple[int, int] = (1, 1)
        max_size = _get_max_texture_sizes()[0]
        if max_size is not None:
            data, factors = _downsample_data(data, max_size)

        return data, _TextureParams(data.shape, data.dtype, dim), factors

    @staticmethod
    def _can_reuse(current_texture: pygfx.Texture, params: _TextureParams) -> bool:
        return (
            current_texture.data is not None
            and _TextureParams(
                current_texture.data.shape,
                current_texture.data.dtype,
                current_texture.dim,
            )
            == params
        )

    def _snx_set_data(self, data: ArrayLike) -> None:
        arr = np.asanyarray(data)
        processed, params, factors = self._compute_texture_params(arr)

        current: pygfx.Texture | None = getattr(self, "_texture", None)
        if current is not None and self._can_reuse(current, params):
            # Reuse the existing texture: overwrite its buffer and mark dirty.
            assert current.data is not None  # guaranteed by _can_reuse
            current.data[:] = processed
            current.update_range((0,) * params.dim, params.shape[: params.dim])
        else:
            # Copy so later in-place reuse won't mutate the originally-passed array.
            self._texture = pygfx.Texture(processed.copy(), dim=params.dim)
            self._geometry = pygfx.Geometry(grid=self._texture)
            if hasattr(self, "_pygfx_node"):
                self._pygfx_node.geometry = self._geometry
            # Only update the material map when the array dimensionality changes
            # (grayscale ↔ RGB/RGBA) or on first creation; otherwise the existing
            # map is still correct and recreating it is wasteful.
            prev_ndim = (
                current.data.ndim
                if current is not None and current.data is not None
                else None
            )
            if prev_ndim != arr.ndim:
                self._material.map = (
                    None if arr.ndim == 3 else self._model.cmap.to_pygfx()
                )

        self._downsample_factors = factors
        if hasattr(self, "_pygfx_node"):
            # Keep transform compensation in sync whenever data or factors change.
            self._snx_set_transform(self._model.transform)
