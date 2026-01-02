"""Functions for rotating and projecting 3D points to 2D.

We use a very simple pin‑hole perspective camera model.  The camera is
located along the positive Y axis looking back towards the origin.  By
adjusting the yaw and pitch the model is rotated prior to projection.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Tuple


def rotate_points(
    points: Iterable[Tuple[float, float, float]],
    yaw_deg: float = 0.0,
    pitch_deg: float = 0.0,
) -> List[Tuple[float, float, float]]:
    """Rotate a sequence of 3D points about the Z (yaw) and X (pitch) axes.

    Parameters
    ----------
    points : iterable of 3‑tuples
        The points to be rotated.
    yaw_deg : float
        Rotation about the Z axis in degrees.  Positive yaw rotates the
        face to the viewer's right (i.e. turning the head to its left).
    pitch_deg : float
        Rotation about the X axis in degrees.  Positive pitch looks down.

    Returns
    -------
    list of tuple(float, float, float)
        The rotated points.
    """
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    cos_pitch = math.cos(pitch)
    sin_pitch = math.sin(pitch)
    rotated: List[Tuple[float, float, float]] = []
    for (x, y, z) in points:
        # Yaw rotation around Z axis (affects x and y)
        x1 = x * cos_yaw - y * sin_yaw
        y1 = x * sin_yaw + y * cos_yaw
        z1 = z
        # Pitch rotation around X axis (affects y and z)
        y2 = y1 * cos_pitch - z1 * sin_pitch
        z2 = y1 * sin_pitch + z1 * cos_pitch
        rotated.append((x1, y2, z2))
    return rotated


def project_points(
    points: Iterable[Tuple[float, float, float]],
    camera_distance: float = 2.5,
    fov: float = 1.0,
) -> List[Tuple[float, float]]:
    """Project 3D points to 2D using a simple perspective model.

    The camera is positioned on the positive Y axis at a distance
    `camera_distance` from the origin and looks towards the origin.  The
    projection assumes the centre of the image corresponds to the origin.

    Parameters
    ----------
    points : iterable of 3‑tuples
        Points in 3D space to project.
    camera_distance : float
        The distance of the camera from the origin along the Y axis.  Larger
        values zoom out.
    fov : float
        A scaling factor applied to the projected coordinates.  Larger
        values zoom in.

    Returns
    -------
    list of tuple(float, float)
        The 2D projected points in normalised coordinates centred at (0, 0).
    """
    projected: List[Tuple[float, float]] = []
    for (x, y, z) in points:
        # Compute the distance from the camera along the viewing direction
        # The camera is at (0, camera_distance, 0)
        relative_y = camera_distance - y
        # Avoid division by zero by adding a small epsilon
        if relative_y == 0:
            relative_y = 1e-6
        # Perspective divide
        px = (x / relative_y) * fov
        pz = (z / relative_y) * fov
        projected.append((px, pz))
    return projected


def project_curves(
    curves: Dict[str, List[Tuple[float, float, float]]],
    yaw_deg: float = 0.0,
    pitch_deg: float = 0.0,
    camera_distance: float = 2.5,
    fov: float = 1.0,
) -> Dict[str, List[Tuple[float, float]]]:
    """Rotate and project an entire dictionary of curves.

    This helper applies the same rotation and projection to every curve in
    the provided dictionary.
    """
    out: Dict[str, List[Tuple[float, float]]] = {}
    for name, pts in curves.items():
        rotated = rotate_points(pts, yaw_deg=yaw_deg, pitch_deg=pitch_deg)
        projected = project_points(rotated, camera_distance=camera_distance, fov=fov)
        out[name] = projected
    return out