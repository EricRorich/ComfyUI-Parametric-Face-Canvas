"""Approximate male golden ratio facial topology.

This file defines a coarse set of landmark points and connecting
segments inspired by the Marquardt beauty mask.  The coordinates are
specified in a normalised 3D space where the X‑axis corresponds to
horizontal left/right, the Y‑axis corresponds to depth (front/back),
and the Z‑axis corresponds to vertical up/down.  Values have been
chosen to roughly reflect the proportions of the male mask with the
origin at the centre of the face.  The model is symmetrical about the
X‑axis; modifying ``eye_distance``, ``jaw_width`` or other sliders will
scale these base positions accordingly.
"""

# Point definitions: name → (x, y, z)
# Note: y (depth) is initialised to 0 for all points.  The face_depth
# slider in the node will scale the y component uniformly.
POINTS = {
    # Outer head contour
    "top": (0.0, 0.0, 0.95),
    "temple_l": (-0.65, 0.0, 0.75),
    "cheek_l": (-0.78, 0.0, 0.10),
    "jaw_l": (-0.55, 0.0, -0.55),
    "chin": (0.0, 0.0, -0.78),
    "jaw_r": (0.55, 0.0, -0.55),
    "cheek_r": (0.78, 0.0, 0.10),
    "temple_r": (0.65, 0.0, 0.75),

    # Nose ridge and base
    "nose_top": (0.0, 0.0, 0.55),
    "nose_mid": (0.0, 0.0, 0.25),
    "nose_base": (0.0, 0.0, 0.05),
    "nostril_l": (-0.12, 0.0, 0.02),
    "nostril_r": (0.12, 0.0, 0.02),

    # Mouth
    "mouth_l": (-0.28, 0.0, -0.25),
    "mouth_r": (0.28, 0.0, -0.25),
    "mouth_top": (0.0, 0.0, -0.20),
    "mouth_bot": (0.0, 0.0, -0.33),

    # Brows
    "brow_l": (-0.35, 0.0, 0.45),
    "brow_r": (0.35, 0.0, 0.45),

    # Eye centres – eyes are drawn separately as perfect circles
    "eye_c_l": (-0.28, 0.0, 0.35),
    "eye_c_r": (0.28, 0.0, 0.35),
}

# Edge definitions: (start_point, end_point)
EDGES = [
    # Head outline
    ("top", "temple_l"), ("temple_l", "cheek_l"), ("cheek_l", "jaw_l"),
    ("jaw_l", "chin"), ("chin", "jaw_r"), ("jaw_r", "cheek_r"),
    ("cheek_r", "temple_r"), ("temple_r", "top"),
    
    # Nose
    ("nose_top", "nose_mid"), ("nose_mid", "nose_base"),
    ("nostril_l", "nose_base"), ("nostril_r", "nose_base"),

    # Mouth
    ("mouth_l", "mouth_top"), ("mouth_top", "mouth_r"),
    ("mouth_l", "mouth_bot"), ("mouth_bot", "mouth_r"),

    # Brows
    ("brow_l", "nose_top"), ("brow_r", "nose_top"),
]

# Eye definitions: key → {center_key: str, radius: float}
# Radii are chosen relative to the base model such that eye_size=0.12
EYES = {
    "left":  {"center_key": "eye_c_l", "radius": 0.11},
    "right": {"center_key": "eye_c_r", "radius": 0.11},
}