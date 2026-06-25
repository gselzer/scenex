from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from scenex.adaptors._base import CameraAdaptor

from ._node import Node

if TYPE_CHECKING:
    from qtpy.Qt3DRender import QCamera

    from scenex import model
    from scenex.model import Transform

try:
    from qtpy.Qt3DExtras import QOrbitCameraController
    from qtpy.QtGui import QMatrix4x4, QVector3D
except Exception as e:
    raise ImportError(
        "Qt3D is required for the qt3d backend. "
        "Install PySide6 (includes Qt3D) or PyQt6 + PyQt6-3D."
    ) from e

logger = logging.getLogger("scenex.adaptors.qt3d")


class Camera(Node, CameraAdaptor):
    """Camera adaptor for the Qt3D backend.

    The scenex Camera stores its viewing state as a pair of Transform matrices
    (view and projection).  We configure the Qt3DRender.QCamera that lives on
    the Qt3DWindow by pushing these matrices directly.

    Lifecycle:
    - ``__init__`` builds the adaptor but does NOT yet touch any QCamera.
    - ``_configure(qt_camera)`` is called once by the View adaptor after the
      canvas is known; it stores the QCamera reference and syncs state.
    - Subsequent ``_snx_set_*`` calls push updates to the stored QCamera.
    """

    def __init__(self, camera: model.Camera, **backend_kwargs: Any) -> None:
        self._model = camera
        self._qt_camera: QCamera | None = None
        self._controller: QOrbitCameraController | None = None
        # Use a dummy entity so the base Node methods don't crash before
        # the camera is attached to a real Qt3D window.
        self._init_entity()

    def _configure(self, qt_camera: QCamera) -> None:
        """Wire this adaptor to the canvas's QCamera."""
        self._qt_camera = qt_camera
        self._sync_projection(self._model.projection)
        self._sync_view_matrix(self._model.transform)
        self._snx_set_controller(self._model.controller)

    # ------------------------------------------------------------------
    # CameraAdaptor interface
    # ------------------------------------------------------------------

    def _snx_set_projection(self, arg: Transform) -> None:
        self._sync_projection(arg)

    def _snx_set_transform(self, arg: Transform) -> None:
        self._sync_view_matrix(arg)

    def _snx_set_controller(self, arg: model.CameraController | None) -> None:
        if self._qt_camera is None:
            return

        # Remove any existing controller.
        if self._controller is not None:
            self._controller.setParent(None)  # type: ignore[arg-type]
            self._controller = None

        if arg is None:
            return

        controller_type = getattr(arg, "type", None)

        if controller_type == "orbit":
            ctrl = QOrbitCameraController(self._qt_camera)
            ctrl.setCamera(self._qt_camera)
            ctrl.setLinearSpeed(50.0)
            ctrl.setLookSpeed(180.0)
            self._controller = ctrl

        elif controller_type == "pan_zoom":
            # PanZoom is implemented entirely in the scenex model (Python), so
            # the adaptor only needs to relay model-driven updates (which arrive
            # via the normal _snx_set_transform / _snx_set_projection path).
            # No Qt3D-level controller is required.
            pass

        else:
            logger.warning(
                "Unknown camera controller type %r — ignoring", controller_type
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _sync_projection(self, proj: Transform) -> None:
        """Push the scenex camera projection to the Qt3D camera lens.

        scenex stores the projection as NDC→local (the INVERSE of the standard
        projection matrix).  The diagonal elements of the row-vector projection
        root give us the visible half-extents directly.  We map those into Qt3D's
        built-in orthographic lens so Qt3D handles near/far clipping correctly.
        """
        if self._qt_camera is None:
            return

        mat = np.asarray(proj.root, dtype=np.float64)

        is_diag = np.allclose(mat - np.diag(np.diagonal(mat)), 0, atol=1e-9)
        px, py = mat[0, 0], mat[1, 1]

        if is_diag and px > 1e-9 and py > 1e-9:
            half_w = 1.0 / px
            half_h = 1.0 / py
            self._qt_camera.lens().setOrthographicProjection(
                -half_w,
                half_w,
                -half_h,
                half_h,
                0.01,
                10_000.0,
            )
        else:
            col_mat = mat.T
            qmat = QMatrix4x4(*col_mat.flatten().tolist())
            self._qt_camera.setProjectionMatrix(qmat)

    def _sync_view_matrix(self, transform: Transform) -> None:
        """Push the scenex camera transform to the Qt3D camera.

        scenex positions the camera at the scene centre (transform translation),
        but that would place the camera *in* the geometry plane (z_eye = 0).
        We offset the Qt3D camera 1 unit backward along the forward direction so
        the scene is always in front of the near clip plane.
        """
        if self._qt_camera is None:
            return
        T = np.asarray(transform.root, dtype=np.float64)
        scene_center = T[3, :3]  # translation = scenex camera position ≈ scene centre
        up = T[1, :3]  # local Y
        forward = -T[2, :3]  # camera looks down -Z in local space

        # Place Qt3D camera 1 unit behind the scene centre so the scene is
        # 1 unit in front (z_eye ≈ -1), safely past the 0.01 near clip.
        qt_pos = scene_center - forward  # subtract forward = go back
        view_center = scene_center  # look at the scene centre

        self._qt_camera.setPosition(QVector3D(*qt_pos.tolist()))
        self._qt_camera.setViewCenter(QVector3D(*view_center.tolist()))
        self._qt_camera.setUpVector(QVector3D(*up.tolist()))
