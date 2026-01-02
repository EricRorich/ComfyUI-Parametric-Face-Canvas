"""Defines a simple parametric face model in 3D.

The model is composed of a handful of analytic curves (ellipses and lines).
Each curve is defined in three dimensional space and can be modified by
changing a handful of scalar parameters.

The coordinate system used throughout this module follows these conventions:

* **x**: horizontal axis – positive values extend to the right ear.
* **y**: depth axis – positive values extend towards the viewer (nose tip).
* **z**: vertical axis – positive values extend upwards (towards the forehead).

When rendering the wireframe the camera looks towards the origin from a
positive *y* position.  Changing the yaw and pitch rotates the face prior
to projection.
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np


def _ellipse_points(
    center: Tuple[float, float, float],
    radii: Tuple[float, float],
    axis: str,
    n: int = 40,
) -> List[Tuple[float, float, float]]:
    """Generate points on an ellipse in 3D.

    Parameters
    ----------
    center : tuple(float, float, float)
        The centre of the ellipse in 3D space.
    radii : tuple(float, float)
        Radii of the ellipse along its principal axes.
    axis : {'xy', 'xz', 'yz'}
        Plane in which to draw the ellipse.  For example 'xz' draws an
        ellipse parallel to the XZ plane (constant depth y coordinate).
    n : int
        Number of points to generate.

    Returns
    -------
    list of tuple(float, float, float)
        Sequence of 3D points tracing the ellipse.
    """
    cx, cy, cz = center
    rx, ry = radii
    points: List[Tuple[float, float, float]] = []
    for i in range(n + 1):
        theta = 2.0 * math.pi * i / n
        u = math.cos(theta) * rx
        v = math.sin(theta) * ry
        if axis == "xy":
            x, y, z = cx + u, cy + v, cz
        elif axis == "xz":
            x, y, z = cx + u, cy, cz + v
        elif axis == "yz":
            x, y, z = cx, cy + u, cz + v
        else:
            raise ValueError(f"Invalid axis '{axis}'. Must be one of 'xy','xz','yz'.")
        points.append((x, y, z))
    return points


def _line_points(
    start: Tuple[float, float, float], end: Tuple[float, float, float], n: int = 20
) -> List[Tuple[float, float, float]]:
    """Generate evenly spaced points along a straight line in 3D."""
    sx, sy, sz = start
    ex, ey, ez = end
    points: List[Tuple[float, float, float]] = []
    for i in range(n + 1):
        t = i / n
        x = sx + (ex - sx) * t
        y = sy + (ey - sy) * t
        z = sz + (ez - sz) * t
        points.append((x, y, z))
    return points


def generate_face_wireframe(
    eye_distance: float = 0.4,
    eye_size: float = 0.08,
    nose_width: float = 0.1,
    nose_height: float = 0.3,
    jaw_width: float = 0.8,
    face_height: float = 1.0,
    face_depth: float = 0.3,
) -> Dict[str, List[Tuple[float, float, float]]]:
    """Construct the 3D control curves for a simple, stylised face.

    The returned dictionary maps semantic names (e.g. 'jaw', 'left_eye')
    to ordered lists of 3D points describing those curves.

    Parameters
    ----------
    eye_distance : float
        Distance between the centres of the eyes.
    eye_size : float
        Size of the eyes (uniform scaling of the eye ellipses).
    nose_width : float
        Width of the nose at its base.
    nose_height : float
        Height of the nose from the bridge down to the base.
    jaw_width : float
        Width of the jaw at its widest point (between the cheeks).
    face_height : float
        Overall height of the face (used to scale the vertical placement of features).
    face_depth : float
        Depth of the face.  Controls how far the nose and chin protrude.

    Returns
    -------
    dict
        Mapping from curve names to point sequences.
    """
    curves: Dict[str, List[Tuple[float, float, float]]] = {}

    # Define canonical vertical positions relative to face height
    # Vertical anchor points.  The eye position is a fixed proportion of the face
    # height.  The nose base is offset below the eye position by the
    # user‑specified nose height.  The mouth and chin are positioned further
    # down relative to the nose base.
    z_eye = 0.2 * face_height
    # nose_height defines the vertical distance between the eye level and the
    # base of the nose.  A larger value elongates the nose downward.  We
    # clamp the nose_height to sensible proportions relative to face height.
    max_nose_h = 0.5 * face_height
    nh = min(nose_height, max_nose_h)
    z_nose_base = z_eye - nh
    z_mouth = z_nose_base - 0.2 * face_height
    z_chin = z_mouth - 0.25 * face_height

    # Depth positions
    nose_tip_y = face_depth  # nose protrudes towards viewer
    cheek_y = 0.1 * face_depth  # cheeks have slight depth
    chin_y = 0.05 * face_depth

    # Jaw: a half ellipse in the XZ plane connecting cheeks around the chin
    jaw_height = abs(z_nose_base - z_chin)
    jaw_radius_x = jaw_width / 2.0
    jaw_radius_z = jaw_height / 2.0
    jaw_center_z = (z_nose_base + z_chin) / 2.0
    jaw_points = _ellipse_points(
        center=(0.0, chin_y, jaw_center_z),
        radii=(jaw_radius_x, jaw_radius_z),
        axis="xz",
        n=50,
    )
    # Only take the lower half (from right cheek, down to chin, up to left cheek)
    half_jaw_points = jaw_points[: len(jaw_points) // 2 + 1]
    curves["jaw"] = half_jaw_points

    # Nose bridge: vertical line up the centre of the face
    nose_top_z = z_eye + 0.1 * face_height
    nose_bridge = _line_points(
        (0.0, 0.0, nose_top_z), (0.0, 0.0, z_nose_base), n=20
    )
    curves["nose_bridge"] = nose_bridge

    # Nose base: horizontal line with slight depth at the tip
    half_width = nose_width / 2.0
    nose_left = (-half_width, 0.0, z_nose_base)
    nose_right = (half_width, 0.0, z_nose_base)
    curves["nose_base"] = _line_points(nose_left, nose_right, n=20)

    # Nose tip: vertical line from base to tip
    curves["nose_tip"] = _line_points(
        (0.0, 0.0, z_nose_base), (0.0, nose_tip_y, z_nose_base), n=10
    )

    # Eyes: ellipses in the XZ plane centred at ±eye_distance/2
    eye_radius_x = eye_size
    eye_radius_z = eye_size * 0.6
    left_eye_center = (-eye_distance / 2.0, 0.0, z_eye)
    right_eye_center = (eye_distance / 2.0, 0.0, z_eye)
    curves["left_eye"] = _ellipse_points(
        center=left_eye_center,
        radii=(eye_radius_x, eye_radius_z),
        axis="xz",
        n=32,
    )
    curves["right_eye"] = _ellipse_points(
        center=right_eye_center,
        radii=(eye_radius_x, eye_radius_z),
        axis="xz",
        n=32,
    )

    # Mouth: small ellipse for lips
    mouth_radius_x = jaw_width * 0.4
    mouth_radius_z = 0.05 * face_height
    curves["mouth"] = _ellipse_points(
        center=(0.0, 0.0, z_mouth),
        radii=(mouth_radius_x, mouth_radius_z),
        axis="xz",
        n=40,
    )

    return curves