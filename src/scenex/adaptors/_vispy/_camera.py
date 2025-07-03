from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import vispy.scene

from scenex.adaptors._base import CameraAdaptor
from scenex.model import Transform

from ._node import Node

if TYPE_CHECKING:
    from scenex import model


class Camera(Node, CameraAdaptor):
    """Adaptor for pygfx camera."""

    _vispy_node: vispy.scene.BaseCamera

    def __init__(self, camera: model.Camera, **backend_kwargs: Any) -> None:
        self._camera_model = camera
        if camera.type == "panzoom":
            self._vispy_node = vispy.scene.PanZoomCamera()
            self._vispy_node.interactive = True
        elif camera.type == "perspective":
            # TODO: These settings were copied from the pygfx camera.
            # Unify these values?
            self._vispy_node = vispy.scene.ArcballCamera(70)

        self._snx_zoom_to_fit(0.1)

    def _snx_set_zoom(self, zoom: float) -> None:
        self._vispy_node.zoom_factor = zoom

    def _snx_set_center(self, arg: tuple[float, ...]) -> None:
        self._vispy_node.center = arg

    def _snx_set_type(self, arg: model.CameraType) -> None:
        raise NotImplementedError()

    def _snx_set_transform(self, arg: Transform) -> None:
        # FIXME: Transform
        return
        # if isinstance(self._vispy_node, vispy.scene.PanZoomCamera):
        #     self._vispy_node.center = tuple(np.asarray(arg)[3, :3])
        # else:
        #     super()._snx_set_transform(arg)

    def _snx_set_projection(self, arg: Transform) -> None:
        pass
        # self._pygfx_node.projection_matrix = arg.root

    def _view_size(self) -> tuple[float, float] | None:
        """Return the size of first parent viewbox in pixels."""
        raise NotImplementedError

    def _snx_zoom_to_fit(self, margin: float) -> None:
        # reset camera to fit all objects
        self._vispy_node.set_range()
        projection = self._vispy_node.transform
        if isinstance(projection, vispy.scene.transforms.STTransform):
            self._camera_model.transform = (
                Transform()
                .scaled(projection.scale)
                .translated(projection.translate)
                .inv()
            )
            if vb := self._vispy_node.viewbox:
                w, h = cast("tuple[float, float]", vb.size)
                projection = (
                    Transform().translated((-w / 2, -h / 2)).scaled((2 / w, 2 / h, 1)).T
                )
                self._camera_model.projection = projection
