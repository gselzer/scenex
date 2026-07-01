from __future__ import annotations

from typing import TYPE_CHECKING, Any, Generic, TypeVar, cast

import numpy as np

from scenex import model
from scenex.adaptors._base import NodeAdaptor, TNode

from ._adaptor_registry import get_adaptor

if TYPE_CHECKING:
    from scenex.model import Transform

try:
    from qtpy.Qt3DCore import QEntity
    from qtpy.Qt3DCore import QTransform as QTransform3D
    from qtpy.QtGui import QMatrix4x4
except Exception as e:
    raise ImportError(
        "Qt3D is required for the qt3d backend. "
        "Install PySide6 (includes Qt3D) or PyQt6 + PyQt6-3D."
    ) from e

TObj = TypeVar("TObj", bound="QEntity")

BLEND_MODES = {
    model.BlendMode.OPAQUE: "solid",
    model.BlendMode.ALPHA: "auto",
    model.BlendMode.ADDITIVE: "add",
}

# Z step per order unit for depth-based render ordering.
# With far=10_000 and a 24-bit depth buffer the minimum distinguishable world-Z
# difference is ≈ 6e-4; 1e-2 gives ~16 depth-buffer steps of separation.
_ORDER_Z_STEP: float = 1e-2


def _to_qmatrix(transform: Transform, z_offset: float = 0.0) -> QMatrix4x4:
    """Convert a scenex Transform to a QMatrix4x4.

    scenex uses row-vector convention (v @ M); Qt3D uses column-vector (M @ v),
    so we transpose before packing into QMatrix4x4 (which takes row-major values).
    z_offset is added to the Z translation component (column 3, row 2) before
    conversion so depth-based render ordering works without Qt API calls.
    """
    mat = np.asarray(transform.root, dtype=np.float64).T
    if z_offset:
        mat = mat.copy()
        mat[2, 3] += z_offset  # Z translation in column-vector convention
    return QMatrix4x4(*mat.flatten().tolist())


class Node(NodeAdaptor[TNode, TObj], Generic[TNode, TObj]):
    """Base node adaptor for the Qt3D backend.

    Each node corresponds to a QEntity with a QTransform component attached.
    Visual components (geometry renderer, material) are added by subclasses.
    """

    _qt_entity: TObj
    _qt_transform: QTransform3D

    def _init_entity(self, parent: QEntity | None = None) -> None:
        self._qt_entity = cast("TObj", QEntity(parent) if parent else QEntity())
        self._qt_transform = QTransform3D()
        self._qt_entity.addComponent(self._qt_transform)

    def _snx_set_name(self, arg: str) -> None:
        self._qt_entity.setObjectName(arg)

    def _snx_add_child(self, child: model.Node) -> None:
        child_adaptor = cast("Node[Any, QEntity]", get_adaptor(child, create=True))
        child_adaptor._qt_entity.setParent(self._qt_entity)

    def _snx_remove_child(self, child: model.Node) -> None:
        child_adaptor = cast("Node[Any, QEntity]", get_adaptor(child, create=True))
        child_adaptor._qt_entity.setParent(None)

    def _snx_set_visible(self, arg: bool) -> None:
        self._qt_entity.setEnabled(arg)

    def _snx_set_opacity(self, arg: float) -> None:
        # Opacity is handled per-material in subclasses; store for reference.
        pass

    def _snx_set_order(self, arg: int) -> None:
        self._order = int(arg)
        # Re-apply the stored transform with the new Z offset.
        if hasattr(self, "_last_transform"):
            self._snx_set_transform(self._last_transform)

    def _snx_set_interactive(self, arg: bool) -> None:
        pass

    def _snx_set_transform(self, arg: Transform) -> None:
        self._last_transform = arg
        order_z = getattr(self, "_order", 0) * _ORDER_Z_STEP
        self._qt_transform.setMatrix(_to_qmatrix(arg, order_z))

    def _snx_set_blending(self, arg: model.BlendMode) -> None:
        pass  # Override in subclasses that use materials with blend state

    def _snx_add_node(self, node: model.Node) -> None:
        adaptor = cast("Node[Any, QEntity]", get_adaptor(node))
        adaptor._qt_entity.setParent(self._qt_entity)

    def _snx_force_update(self) -> None:
        pass

    def _snx_block_updates(self) -> None:
        self._qt_entity.blockSignals(True)

    def _snx_unblock_updates(self) -> None:
        self._qt_entity.blockSignals(False)
