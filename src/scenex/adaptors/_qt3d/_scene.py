from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ._node import Node

if TYPE_CHECKING:
    from scenex import model


class Scene(Node):
    """Qt3D backend adaptor for a Scene node (root container entity)."""

    def __init__(self, scene: model.Scene, **backend_kwargs: Any) -> None:
        self._init_entity()
        self._qt_entity.setObjectName(scene.name or "scene")
        self._qt_entity.setEnabled(scene.visible)
