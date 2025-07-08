from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np
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
            self._vispy_node.flip = (False, True, False)
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
        # self._vispy_node.transform = vispy.scene.transforms.STTransform(
        #     np.diag(arg),
        #     arg.root[3, :3],
        # )
        # self._vispy_node.view_changed()
        # tform = self._vispy_node.transform
        # if isinstance(tform, vispy.scene.transforms.STTransform):
            # center = arg.root[3, :3]
            # print(center)
            # self._vispy_node.center = tuple(arg.root[3, :3])
            # print(self._vispy_node)
            # tform.scale = np.diag(arg)
            # tform.translate = arg.root[3, :3]
        # else:
        #     raise NotImplementedError(f"Unsupported transform type: {type(tform)}.")
        before = self._vispy_node.transform
        if isinstance(before, vispy.scene.transforms.STTransform):
            before.translate = arg.root[3, :3]

        self._vispy_node._set_scene_transform(before)
        print(f"Before: {before}, After: {self._vispy_node.transform}")

    def _snx_set_projection(self, arg: Transform) -> None:
        pass
        # self._pygfx_node.projection_matrix = arg.root

    def _view_size(self) -> tuple[float, float] | None:
        """Return the size of first parent viewbox in pixels."""
        raise NotImplementedError

    def _snx_zoom_to_fit(self, margin: float) -> None:
        # reset camera to fit all objects
        self._vispy_node.set_range()
        vis_tform = self._vispy_node.transform

        tform = Transform()
        if isinstance(vis_tform, vispy.scene.transforms.STTransform):
            tform = Transform(vis_tform.as_matrix().matrix).translated(self._vispy_node.center)

        # Vispy's camera transforms map canvas coordinates to world coordinates.
        # Thus the projection matrix should map NDC coordinates to canvas
        # coordinates, to obtain the desired effect of mapping NDC coordinates in
        # scenex to world coordinates through the projection and transform matrices.
        if vb := self._vispy_node.viewbox:
            w, h = cast("tuple[float, float]", vb.size)
            # This transform maps NDC coordinates to canvas position
            de_NDC = (
                Transform().translated((-w / 2, -h / 2)).scaled((2 / w, 2 / h, 1)).T
            )
            tform = de_NDC @ vis_tform.as_matrix().matrix.T

        self._camera_model.transform = (
            Transform()
        )

        scale = np.diag(tform)
        self._camera_model.projection = tform
