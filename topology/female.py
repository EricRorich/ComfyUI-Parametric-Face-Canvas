"""Approximate female golden ratio facial topology.

This topology is similar to the male version but with slightly
different proportions that reflect softer features: a narrower jaw,
smaller nose width, and slightly larger eyes.  Coordinates are
specified in a normalised 3D space as described in ``topology.male``.
You can replace these defaults with more accurate coordinates
extracted from a high resolution mask if you wish.
"""

POINTS = {
    "top": (0.0, 0.0, 0.98),
    "temple_l": (-0.60, 0.0, 0.78),
    "cheek_l": (-0.72, 0.0, 0.12),
    "jaw_l": (-0.50, 0.0, -0.55),
    "chin": (0.0, 0.0, -0.75),
    "jaw_r": (0.50, 0.0, -0.55),
    "cheek_r": (0.72, 0.0, 0.12),
    "temple_r": (0.60, 0.0, 0.78),

    "nose_top": (0.0, 0.0, 0.58),
    "nose_mid": (0.0, 0.0, 0.28),
    "nose_base": (0.0, 0.0, 0.06),
    "nostril_l": (-0.10, 0.0, 0.03),
    "nostril_r": (0.10, 0.0, 0.03),

    "mouth_l": (-0.26, 0.0, -0.23),
    "mouth_r": (0.26, 0.0, -0.23),
    "mouth_top": (0.0, 0.0, -0.19),
    "mouth_bot": (0.0, 0.0, -0.31),

    "brow_l": (-0.33, 0.0, 0.47),
    "brow_r": (0.33, 0.0, 0.47),

    "eye_c_l": (-0.27, 0.0, 0.36),
    "eye_c_r": (0.27, 0.0, 0.36),
}

EDGES = [
    ("top", "temple_l"), ("temple_l", "cheek_l"), ("cheek_l", "jaw_l"),
    ("jaw_l", "chin"), ("chin", "jaw_r"), ("jaw_r", "cheek_r"),
    ("cheek_r", "temple_r"), ("temple_r", "top"),

    ("nose_top", "nose_mid"), ("nose_mid", "nose_base"),
    ("nostril_l", "nose_base"), ("nostril_r", "nose_base"),

    ("mouth_l", "mouth_top"), ("mouth_top", "mouth_r"),
    ("mouth_l", "mouth_bot"), ("mouth_bot", "mouth_r"),

    ("brow_l", "nose_top"), ("brow_r", "nose_top"),
]

EYES = {
    "left":  {"center_key": "eye_c_l", "radius": 0.12},
    "right": {"center_key": "eye_c_r", "radius": 0.12},
}