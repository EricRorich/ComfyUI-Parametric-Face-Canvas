"""ComfyUI node definition for the parametric face canvas."""

from __future__ import annotations

from typing import Dict, Tuple

import torch

from . import face_model
from . import projection
from . import renderer


class ParametricFaceCanvas:
    """
    A custom ComfyUI node that generates a 3D wireframe face and renders it to
    a 2D image.

    The node exposes sliders for facial proportions and camera orientation.
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict[str, object]]:
        """Define the node inputs for ComfyUI.

        In addition to the facial proportions and camera parameters, this
        definition introduces a ``gender`` selector (male/female), a handful
        of reset buttons, and symmetry toggles for certain measurements.
        Reset buttons allow the user to restore defaults for individual
        parameters or all parameters at once.  Symmetry toggles control
        whether a distance parameter is interpreted as a full span
        (symmetrical) or as a half‑distance from the midline (asymmetrical).
        """
        return {
            "required": {
                # Gender selector (drop‑down)
                "gender": (["male", "female"],),
                "eye_distance": ("FLOAT", {"default": 0.4, "min": 0.1, "max": 1.0, "step": 0.01}),
                "eye_size": ("FLOAT", {"default": 0.08, "min": 0.02, "max": 0.3, "step": 0.01}),
                "nose_width": ("FLOAT", {"default": 0.1, "min": 0.05, "max": 0.3, "step": 0.01}),
                "nose_height": ("FLOAT", {"default": 0.3, "min": 0.1, "max": 0.6, "step": 0.01}),
                "jaw_width": ("FLOAT", {"default": 0.8, "min": 0.4, "max": 1.5, "step": 0.01}),
                "face_height": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.05}),
                "face_depth": ("FLOAT", {"default": 0.3, "min": 0.1, "max": 1.0, "step": 0.05}),
                "yaw": ("FLOAT", {"default": 0.0, "min": -45.0, "max": 45.0, "step": 0.5}),
                "pitch": ("FLOAT", {"default": 0.0, "min": -45.0, "max": 45.0, "step": 0.5}),
                "camera_distance": ("FLOAT", {"default": 2.5, "min": 1.0, "max": 5.0, "step": 0.1}),
                "fov": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 3.0, "step": 0.1}),
                "image_width": ("INT", {"default": 512, "min": 64, "max": 2048}),
                "image_height": ("INT", {"default": 512, "min": 64, "max": 2048}),
                "line_thickness": ("INT", {"default": 2, "min": 1, "max": 10}),
            },
            "optional": {
                # Global reset button – restore all parameters to defaults for the selected gender
                "reset_all": ("BOOLEAN", {"default": False}),
                # Individual reset buttons – restore a single parameter to its default
                "reset_eye_distance": ("BOOLEAN", {"default": False}),
                "reset_eye_size": ("BOOLEAN", {"default": False}),
                "reset_nose_width": ("BOOLEAN", {"default": False}),
                "reset_nose_height": ("BOOLEAN", {"default": False}),
                "reset_jaw_width": ("BOOLEAN", {"default": False}),
                "reset_face_height": ("BOOLEAN", {"default": False}),
                "reset_face_depth": ("BOOLEAN", {"default": False}),
                "reset_yaw": ("BOOLEAN", {"default": False}),
                "reset_pitch": ("BOOLEAN", {"default": False}),
                "reset_camera_distance": ("BOOLEAN", {"default": False}),
                "reset_fov": ("BOOLEAN", {"default": False}),
                "reset_image_size": ("BOOLEAN", {"default": False}),
                "reset_line_thickness": ("BOOLEAN", {"default": False}),
                # Symmetry toggles – interpret values as full distances (True) or half distances (False)
                "eye_distance_sym": ("BOOLEAN", {"default": True}),
                "nose_width_sym": ("BOOLEAN", {"default": True}),
                "jaw_width_sym": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    CATEGORY = "CS Custom Nodes/Face"
    FUNCTION = "execute"

    # Default facial and camera parameters for male and female presets.
    MALE_DEFAULTS: Dict[str, object] = {
        "eye_distance": 0.4,
        "eye_size": 0.08,
        "nose_width": 0.12,
        "nose_height": 0.35,
        "jaw_width": 0.9,
        "face_height": 1.0,
        "face_depth": 0.35,
        "yaw": 0.0,
        "pitch": 0.0,
        "camera_distance": 2.5,
        "fov": 1.0,
        "image_width": 512,
        "image_height": 512,
        "line_thickness": 2,
    }

    FEMALE_DEFAULTS: Dict[str, object] = {
        "eye_distance": 0.36,
        "eye_size": 0.1,
        "nose_width": 0.09,
        "nose_height": 0.32,
        "jaw_width": 0.75,
        "face_height": 1.0,
        "face_depth": 0.3,
        "yaw": 0.0,
        "pitch": 0.0,
        "camera_distance": 2.5,
        "fov": 1.0,
        "image_width": 512,
        "image_height": 512,
        "line_thickness": 2,
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
        reset_eye_distance: bool = False,
        reset_eye_size: bool = False,
        reset_nose_width: bool = False,
        reset_nose_height: bool = False,
        reset_jaw_width: bool = False,
        reset_face_height: bool = False,
        reset_face_depth: bool = False,
        reset_yaw: bool = False,
        reset_pitch: bool = False,
        reset_camera_distance: bool = False,
        reset_fov: bool = False,
        reset_image_size: bool = False,
        reset_line_thickness: bool = False,
        eye_distance_sym: bool = True,
        nose_width_sym: bool = True,
        jaw_width_sym: bool = True,
    ) -> Tuple[torch.Tensor]:
        """Generate the face wireframe image with optional resets and symmetry toggles."""
        # Select defaults based on gender
        defaults = self.MALE_DEFAULTS if gender == "male" else self.FEMALE_DEFAULTS
        # Assemble current parameters into a dictionary
        params: Dict[str, object] = {
            "eye_distance": eye_distance,
            "eye_size": eye_size,
            "nose_width": nose_width,
            "nose_height": nose_height,
            "jaw_width": jaw_width,
            "face_height": face_height,
            "face_depth": face_depth,
            "yaw": yaw,
            "pitch": pitch,
            "camera_distance": camera_distance,
            "fov": fov,
            "image_width": image_width,
            "image_height": image_height,
            "line_thickness": line_thickness,
        }
        # Apply global reset
        if reset_all:
            params.update(defaults)
        else:
            # Handle individual resets
            individual_flags = {
                "eye_distance": reset_eye_distance,
                "eye_size": reset_eye_size,
                "nose_width": reset_nose_width,
                "nose_height": reset_nose_height,
                "jaw_width": reset_jaw_width,
                "face_height": reset_face_height,
                "face_depth": reset_face_depth,
                "yaw": reset_yaw,
                "pitch": reset_pitch,
                "camera_distance": reset_camera_distance,
                "fov": reset_fov,
                "image_width": reset_image_size,
                "image_height": reset_image_size,
                "line_thickness": reset_line_thickness,
            }
            for key, flag in individual_flags.items():
                if flag:
                    params[key] = defaults[key]
        # Symmetry toggles: treat distances as half distances when asymmetrical
        if not eye_distance_sym:
            params["eye_distance"] = float(params["eye_distance"]) * 2.0
        if not nose_width_sym:
            params["nose_width"] = float(params["nose_width"]) * 2.0
        if not jaw_width_sym:
            params["jaw_width"] = float(params["jaw_width"]) * 2.0
        # Generate the face curves
        curves = face_model.generate_face_wireframe(
            eye_distance=params["eye_distance"],
            eye_size=params["eye_size"],
            nose_width=params["nose_width"],
            nose_height=params["nose_height"],
            jaw_width=params["jaw_width"],
            face_height=params["face_height"],
            face_depth=params["face_depth"],
        )
        # Rotate and project curves
        projected = projection.project_curves(
            curves,
            yaw_deg=params["yaw"],
            pitch_deg=params["pitch"],
            camera_distance=params["camera_distance"],
            fov=params["fov"],
        )
        # Render to image
        img = renderer.render_wireframe(
            projected,
            image_size=(int(params["image_width"]), int(params["image_height"])),
            line_thickness=int(params["line_thickness"]),
        )
        # Convert to BHWC tensor normalised to [0,1]
        import numpy as np  # local import to avoid unnecessary dependency when unused
        arr = torch.tensor(np.array(img), dtype=torch.float32) / 255.0
        tensor = arr.unsqueeze(0)
        return (tensor,)