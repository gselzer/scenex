from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from cmap import Color
from pydantic import Field

from scenex.model._color import UniformColor, VertexColors

from .node import AABB, Node, ScalingMode

if TYPE_CHECKING:
    from scenex.app.events._events import Ray
    from scenex.model._view import View


class Line(Node):
    """A polyline defined by connected vertices.

    Line renders a sequence of connected line segments by drawing from each vertex to
    the next. The line can be colored uniformly or with per-vertex colors that smoothly
    interpolate along the path. Lines support width control and anti-aliasing for
    smooth rendering.

    Vertices can be 2D or 3D coordinates. For 2D vertices, the z-coordinate is assumed
    to be 0, placing the line in the xy-plane.

    Examples
    --------
    Create a simple line connecting several points:
        >>> import numpy as np
        >>> vertices = np.array([[0, 0], [10, 5], [20, 0]])
        >>> line = Line(
        ...     vertices=vertices,
        ...     color=UniformColor(color=Color("red")),
        ... )

    Create a line with per-vertex colors:
        >>> vertices = np.array([[0, 0], [10, 10], [20, 0]])
        >>> colors = [Color("red"), Color("green"), Color("blue")]
        >>> line = Line(
        ...     vertices=vertices,
        ...     color=VertexColors(color=colors),
        ...     width=2.0,
        ... )

    Create a 3D line:
        >>> vertices = np.array([[0, 0, 0], [10, 5, 3], [20, 0, 6]])
        >>> line = Line(vertices=vertices, width=3.0)
    """

    node_type: Literal["line"] = "line"

    # numpy array of 2D/3D vertices, shape (N, 2) or (N, 3)
    vertices: Any = Field(
        default=None,
        repr=False,
        exclude=True,
        description="Array of N vertex positions, of shape (N, 2) or (N, 3)",
    )

    color: UniformColor | VertexColors = Field(
        default_factory=lambda: UniformColor(color=Color("white")),
        description="Color specification; uniform or per-vertex colors",
    )
    width: float = Field(default=1.0, ge=0.0, description="Width of the line")
    antialias: bool = Field(
        default=True, description="Whether to apply anti-aliasing to line rendering"
    )
    scaling: ScalingMode = Field(
        default="fixed",
        description=(
            "Scaling mode: "
            "'fixed' keeps line width constant in screen pixels, "
            "'scene' keeps line width constant in world units, "
            "'visual' is like 'fixed' but also scales with this node's transform."
        ),
    )

    @property  # TODO: Cache?
    def bounding_box(self) -> AABB:
        arr = np.asarray(self.vertices)

        min_vals = tuple(float(d) for d in np.min(arr, axis=0))
        max_vals = tuple(float(d) for d in np.max(arr, axis=0))

        # Ensure we have at least 3 dimensions by padding with zeros if needed
        if len(min_vals) == 2:
            min_vals = (*min_vals, 0.0)
            max_vals = (*max_vals, 0.0)

        return (min_vals, max_vals)  # type: ignore

    def passes_through(self, ray: Ray) -> float | None:
        """Check if the ray passes through this line.

        Parameters
        ----------
        ray : Ray
            The ray to test for intersection.

        Returns
        -------
        float | None
            The parameter t of the closest intersection (i.e. the point
            ``ray.origin + t * ray.direction``), or None if no intersection.
        """
        if self.scaling == "fixed":
            return self._passes_through_screen(ray)
        elif self.scaling == "scene":
            return self._passes_through_world(ray)
        else:  # "visual"
            raise NotImplementedError(
                "Lines with 'visual' scaling mode do not (yet) support "
                "ray intersection tests."
            )

    def _passes_through_screen(self, ray: Ray) -> float | None:
        """Test ray intersection in screen/canvas space for fixed-size lines."""
        verts = np.asarray(self.vertices)
        # Convert vertices to canvas space
        canvas_vertices = self._node_to_canvas(ray.source)
        # Convert ray to canvas space
        canvas_ray = Line._world_to_canvas(ray, np.array([ray.origin]))[0]

        starts = canvas_vertices[:-1]
        ends = canvas_vertices[1:]

        # Compute the distance from the ray ON THE CANVAS to the closest point the line
        # associated with each line segment.
        #
        # Equation loaned from https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line#Line_defined_by_two_points
        num = np.abs(
            (ends[:, 1] - starts[:, 1]) * canvas_ray[0]
            - (ends[:, 0] - starts[:, 0]) * canvas_ray[1]
            + ends[:, 0] * starts[:, 1]
            - ends[:, 1] * starts[:, 0]
        )
        den = np.sqrt(
            (ends[:, 1] - starts[:, 1]) ** 2 + (ends[:, 0] - starts[:, 0]) ** 2
        )
        den[den == 0] = float("inf")  # Avoid division by zero
        distance = num / den

        # Determine the corresponding point in world space corresponding to that closest
        # point. Note that this point is only on the line segment if 0 <= t <= 1.
        # (We check this at the end.)
        a = np.subtract(canvas_ray, starts)
        b = np.subtract(ends, starts)
        # Vectorized version of dot product
        t = np.sum(a * b, axis=1) / np.sum(b * b, axis=1)
        intersect_world = verts[1:] + t[:, np.newaxis] * (verts[:-1] - verts[1:])

        # Calculate the distance along the ray to the intersection point
        # The ray is defined as: ray.origin + d * ray.direction
        # We need to find d such that ray.origin + d * ray.direction = intersect_world
        # This gives us: d * ray.direction = intersect_world - ray.origin
        # Solving for d: d = dot(intersect_world - ray.origin, ray.direction) /
        #                     dot(ray.direction, ray.direction)
        ray_to_intersect = np.subtract(intersect_world, ray.origin)
        ray_dir_squared = np.dot(ray.direction, ray.direction)

        if ray_dir_squared == 0:
            return None  # Degenerate ray

        d = np.dot(ray_to_intersect, ray.direction) / ray_dir_squared

        # Our ray intersects the line if:
        # 1. The distance from the ray to the line is less than the line width
        # 2. The intersection point is within the line segment (0 <= t <= 1)
        # 3. The intersection point is in front of the ray origin (d >= 0)
        condition = (distance <= self.width) & (t >= 0) & (t <= 1) & (d >= 0)
        valid_intersections = d[condition]
        if len(valid_intersections):
            return float(np.min(valid_intersections))
        else:
            return None

    def _passes_through_world(self, ray: Ray) -> float | None:
        """Test ray intersection in world space for scene-scaled lines.

        Treats each segment as a capsule of radius ``width / 2`` in world units
        and returns the ray parameter t of the closest hit.
        """
        if self.vertices is None or len(self.vertices) < 2:
            return None

        verts = np.asarray(self.vertices, dtype=float)
        if verts.shape[1] == 2:
            verts = np.pad(verts, ((0, 0), (0, 1)), mode="constant", constant_values=0)

        tform = self.transform_to_node(ray.source.scene)
        world_verts = tform.map(verts)[:, :3]

        starts = world_verts[:-1]  # (M, 3)
        ends = world_verts[1:]  # (M, 3)

        O = np.asarray(ray.origin, dtype=float)
        D = np.asarray(ray.direction, dtype=float)

        E = ends - starts  # (M, 3) segment direction vectors
        a = O[np.newaxis, :] - starts  # (M, 3) ray-origin relative to seg starts

        dd = float(np.dot(D, D))
        if dd == 0:
            return None  # Degenerate ray

        de = E @ D  # (M,) D · E
        da = a @ D  # (M,) D · a
        ee = np.sum(E * E, axis=1)  # (M,) |E|²
        ea = np.sum(E * a, axis=1)  # (M,) E · a

        # determinant; zero when ray and segment are parallel
        det = dd * ee - de**2  # (M,)
        nonparallel = det > 1e-10
        safe_det = np.where(nonparallel, det, 1.0)
        safe_ee = np.where(ee > 0, ee, 1.0)

        # Unclamped closest-point parameters
        t_seg = np.where(nonparallel, (dd * ea - de * da) / safe_det, ea / safe_ee)
        t_seg = np.clip(t_seg, 0.0, 1.0)

        # Ray parameter for the (clamped) segment point
        t_ray = (t_seg * de - da) / dd

        # When t_ray < 0 the closest approach is behind the ray origin; clamp
        # and reproject t_seg onto the segment closest to the ray origin.
        at_origin = t_ray < 0
        t_ray = np.maximum(t_ray, 0.0)
        t_seg_reproject = np.clip(ea / safe_ee, 0.0, 1.0)
        t_seg = np.where(at_origin, t_seg_reproject, t_seg)

        # Closest points on ray and segment
        ray_pts = O[np.newaxis, :] + t_ray[:, np.newaxis] * D[np.newaxis, :]  # (M, 3)
        seg_pts = starts + t_seg[:, np.newaxis] * E  # (M, 3)

        distances = np.linalg.norm(ray_pts - seg_pts, axis=1)  # (M,)

        r = self.width / 2
        valid_t_rays = t_ray[distances <= r]
        if len(valid_t_rays) == 0:
            return None
        return float(np.min(valid_t_rays))

    @staticmethod
    def _world_to_canvas(ray: Ray, points: np.ndarray) -> np.ndarray:
        view = ray.source
        if (rect := view.content_rect) is None:
            raise ValueError(
                f"Ray source {ray.source} must be displayed on a canvas for "
                "line intersections."  # TODO: when scaling=\"fixed\""
            )
        cam = view.camera
        ndc_points = cam.projection.map(cam.transform.imap(points))[:, :2]
        _, _, w, h = rect
        return (ndc_points + 1) / 2 * (w, h)

    def _node_to_canvas(self, view: View) -> np.ndarray:
        if (rect := view.content_rect) is None:
            raise ValueError(
                f"Ray source {view} must be displayed on a canvas for "
                "line intersections."  # TODO: when scaling=\"fixed\""
            )
        cam = view.camera
        tform_to_root_scene = self.transform_to_node(view.scene)
        ndc_points = cam.projection.map(
            cam.transform.imap(tform_to_root_scene.map(self.vertices))
        )[:, :2]
        _, _, w, h = rect
        return (ndc_points + 1) / 2 * (w, h)
