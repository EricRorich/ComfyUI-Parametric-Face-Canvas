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

        ComfyUI inspects this dictionary at runtime to build the UI.  Values
        can be floats, ints, booleans, etc., and ranges may be provided to
        constrain slider widgets.
        """
        return {
            "required": {
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
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    CATEGORY = "CS Custom Nodes/Face"
    FUNCTION = "execute"

    def execute(
        self,
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
    ) -> Tuple[torch.Tensor]:
        """Generate the face wireframe image.

        Parameters are described in the INPUT_TYPES definition.  The
        execution method must return a tuple of tensors corresponding to
        RETURN_TYPES.
        """
        # Generate the 3D control curves for the face
        curves = face_model.generate_face_wireframe(
            eye_distance=eye_distance,
            eye_size=eye_size,
            nose_width=nose_width,
            nose_height=nose_height,
            jaw_width=jaw_width,
            face_height=face_height,
            face_depth=face_depth,
        )
        # Rotate and project the curves into 2D
        projected = projection.project_curves(
            curves,
            yaw_deg=yaw,
            pitch_deg=pitch,
            camera_distance=camera_distance,
            fov=fov,
        )
        # Render to a PIL image
        img = renderer.render_wireframe(
            projected,
            image_size=(image_width, image_height),
            line_thickness=line_thickness,
        )
        # Convert to tensor in CHW format and wrap in a batch dimension
        chw = renderer.image_to_tensor(img)
        tensor = torch.from_numpy(chw).unsqueeze(0)
        return (tensor,)