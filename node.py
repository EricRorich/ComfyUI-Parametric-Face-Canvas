"""ComfyUI node definition for the parametric face canvas.

This module defines a custom ComfyUI node that renders a parametric
3D facial structure to a 2D image.  The face is represented as a
collection of points and straight line segments (edges) augmented
with circular outlines for the eyes.  Two distinct topologies are
supported via a gender selector: one approximating the canonical
Marquardt "Repose Frontal" mask for male features, and another
derived from a female variant.  Sliders allow deformation of the
base topology along semantic axes (eye distance, jaw width, etc.),
and camera parameters control yaw, pitch, distance and field of
view.  A single "reset all" toggle restores defaults for the
selected gender and recentres the view.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch

from . import projection
from . import renderer
from .topology import male, female


class ParametricFaceCanvas:
    """Custom node that renders a parametric face to an image.

    The node exposes a set of sliders that deform an underlying
    facial topology defined by a fixed set of 3D points and edges.
    Adjusting the sliders scales or translates specific groups of
    points to change apparent proportions.  A camera model with yaw
    and pitch rotations and perspective projection is applied, and
    the resulting 2D points are drawn as straight lines with round
    eyes on a white canvas.  The gender selector chooses which
    topology and default parameter set to load.
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict[str, object]]:
        """Describe the inputs for ComfyUI.

        Inputs are grouped into a single `required` section for the
        sliders and a minimal `optional` section containing only the
        `reset_all` button.  Width and height control the canvas
        resolution; they do not scale the facial geometry.
        """
        return {
            "required": {
                "gender": (["male", "female"],),
                # Facial proportion controls
                "eye_distance": ("FLOAT", {"default": 0.30, "min": 0.10, "max": 1.00, "step": 0.01}),
                "eye_size": ("FLOAT", {"default": 0.12, "min": 0.02, "max": 0.30, "step": 0.01}),
                "nose_width": ("FLOAT", {"default": 0.10, "min": 0.02, "max": 0.40, "step": 0.01}),
                "nose_height": ("FLOAT", {"default": 0.30, "min": 0.05, "max": 0.80, "step": 0.01}),
                "jaw_width": ("FLOAT", {"default": 0.80, "min": 0.30, "max": 1.80, "step": 0.01}),
                "face_height": ("FLOAT", {"default": 1.00, "min": 0.50, "max": 2.00, "step": 0.01}),
                "face_depth": ("FLOAT", {"default": 0.30, "min": 0.00, "max": 1.50, "step": 0.01}),
                # Camera controls
                "yaw": ("FLOAT", {"default": 0.0, "min": -90.0, "max": 90.0, "step": 0.5}),
                "pitch": ("FLOAT", {"default": 0.0, "min": -60.0, "max": 60.0, "step": 0.5}),
                "camera_distance": ("FLOAT", {"default": 2.5, "min": 0.5, "max": 10.0, "step": 0.1}),
                "fov": ("FLOAT", {"default": 1.0, "min": 0.2, "max": 5.0, "step": 0.1}),
                # Output canvas size and line thickness
                "image_width": ("INT", {"default": 1024, "min": 64, "max": 4096}),
                "image_height": ("INT", {"default": 1024, "min": 64, "max": 4096}),
                "line_thickness": ("INT", {"default": 4, "min": 1, "max": 20}),
            },
            "optional": {
                # Single toggle to restore defaults for the chosen gender
                "reset_all": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    CATEGORY = "CS Custom Nodes/Face"
    FUNCTION = "execute"

    # Preset parameter values for each gender.
    MALE_DEFAULTS: Dict[str, float] = {
        "eye_distance": 0.30,
        "eye_size": 0.12,
        "nose_width": 0.10,
        "nose_height": 0.30,
        "jaw_width": 0.80,
        "face_height": 1.00,
        "face_depth": 0.30,
        "yaw": 0.0,
        "pitch": 0.0,
        "camera_distance": 2.5,
        "fov": 1.0,
    }

    FEMALE_DEFAULTS: Dict[str, float] = {
        "eye_distance": 0.28,
        "eye_size": 0.13,
        "nose_width": 0.09,
        "nose_height": 0.28,
        "jaw_width": 0.75,
        "face_height": 1.00,
        "face_depth": 0.25,
        "yaw": 0.0,
        "pitch": 0.0,
        "camera_distance": 2.5,
        "fov": 1.0,
    }

    def execute(
        self,
        gender: str,
        eye_distance: float,
        eye_size: float,
        nose_width: float,
        nose_height: float,
        jaw_width: float,
        face_height: float,
        face_depth: float,
        yaw: float,
        pitch: float,
        camera_distance: float,
        fov: float,
        image_width: int,
        image_height: int,
        line_thickness: int,
        reset_all: bool = False,
    ) -> Tuple[torch.Tensor]:
        """Generate and return the rendered face image.

        This method performs the following high level steps:

        1. Load the appropriate base topology (male or female).
        2. If `reset_all` is toggled, apply the default parameters for
           the selected gender.
        3. Copy the base points so that deformations do not affect the
           originals.
        4. Apply semantic deformations (eye distance, jaw width, etc.).
        5. Rotate and project the 3D points to 2D using the camera
           parameters.
        6. Render the resulting graph using straight lines for edges
           and circles for eyes.
        7. Convert the PIL image to a ComfyUI‑compatible BHWC tensor.
        """
        # Select topology and default parameters by gender
        topo = male if gender == "male" else female
        defaults = self.MALE_DEFAULTS if gender == "male" else self.FEMALE_DEFAULTS

        # Apply reset: override the input parameters with defaults
        if reset_all:
            eye_distance = defaults["eye_distance"]
            eye_size = defaults["eye_size"]
            nose_width = defaults["nose_width"]
            nose_height = defaults["nose_height"]
            jaw_width = defaults["jaw_width"]
            face_height = defaults["face_height"]
            face_depth = defaults["face_depth"]
            yaw = defaults["yaw"]
            pitch = defaults["pitch"]
            camera_distance = defaults["camera_distance"]
            fov = defaults["fov"]

        # Copy base point positions so modifications are local
        points = {k: list(v) for k, v in topo.POINTS.items()}

        # ----- Deformation logic -----
        # Overall face height (z scaling) and depth (y scaling)
        for k, (x, y, z) in topo.POINTS.items():
            points[k][2] = z * face_height
            points[k][1] = y * face_depth

        # Eye distance: adjust centres along x axis (left negative, right positive)
        if "eye_c_l" in points and "eye_c_r" in points:
            points["eye_c_l"][0] = -abs(eye_distance)
            points["eye_c_r"][0] = abs(eye_distance)

        # Nose width: adjust nostril x positions symmetrically
        for nk in ["nostril_l", "nostril_r"]:
            if nk in points:
                points[nk][0] = np.sign(topo.POINTS[nk][0]) * abs(nose_width)

        # Nose height: adjust vertical (z) positions of mid and base relative to nose_top
        if {"nose_top", "nose_mid", "nose_base"}.issubset(points):
            top_z = points["nose_top"][2]
            points["nose_mid"][2] = top_z - (nose_height * 0.6)
            points["nose_base"][2] = top_z - (nose_height * 1.0)

        # Jaw width: scale x positions of lateral jaw points
        jaw_keys = [
            "temple_l", "cheek_l", "jaw_l",
            "temple_r", "cheek_r", "jaw_r",
        ]
        for jk in jaw_keys:
            if jk in points:
                base_x = topo.POINTS[jk][0]
                # The default jaw width in the topology corresponds to 0.80 for male;
                # scale relative to that baseline
                scale_factor = jaw_width / 0.80
                points[jk][0] = np.sign(base_x) * abs(base_x) * scale_factor

        # ----- Projection -----
        # Convert points to tuples for projection
        pts3 = {k: tuple(v) for k, v in points.items()}
        pts2: Dict[str, Tuple[float, float]] = {}
        for k, p in pts3.items():
            # Apply rotation then projection
            rp = projection.rotate_points([p], yaw_deg=yaw, pitch_deg=pitch)[0]
            pp = projection.project_points([rp], camera_distance=camera_distance, fov=fov)[0]
            pts2[k] = pp  # (x, z) after perspective

        # Compute eye radii scaling relative to default sizes
        eye_defs = {}
        for side in ["left", "right"]:
            base = topo.EYES[side]
            # scale radius proportionally to input eye_size (baseline 0.12)
            scale = eye_size / 0.12
            eye_defs[side] = {
                "center_key": base["center_key"],
                "radius": base["radius"] * scale,
            }

        # Render the graph onto a canvas
        img = renderer.render_graph(
            points_2d=pts2,
            edges=topo.EDGES,
            eyes=eye_defs,
            image_size=(image_width, image_height),
            line_thickness=line_thickness,
        )

        # Convert to BHWC tensor for ComfyUI
        arr = renderer.pil_to_comfy_image(img)
        tensor = torch.from_numpy(arr)
        return (tensor,)