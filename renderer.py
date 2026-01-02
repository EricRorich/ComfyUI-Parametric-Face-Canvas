"""Rendering utilities for drawing straight‑line graphs on a canvas.

This module provides functions to render a graph defined by 2D points and
edges, along with circular eye outlines, onto a PIL Image.  The
renderer automatically fits the facial structure into the canvas
dimensions and maintains aspect ratio regardless of image size.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple
from PIL import Image, ImageDraw
import numpy as np


def render_graph(
    points_2d: Dict[str, Tuple[float, float]],
    edges: Iterable[Tuple[str, str]],
    eyes: Dict[str, Dict[str, object]],
    image_size: Tuple[int, int] = (512, 512),
    line_thickness: int = 2,
    fit_margin: float = 0.85,
) -> Image.Image:
    """Render a face graph onto a white canvas.

    Parameters
    ----------
    points_2d : dict
        Mapping of point names to 2D coordinates (x, z) after projection.
    edges : iterable of tuples
        Pairs of point names that should be connected by straight lines.
    eyes : dict
        Mapping with keys ``"left"`` and ``"right"``.  Each value is a
        dictionary with keys ``center_key`` (the name of the point used
        as the eye centre) and ``radius`` (the eye radius in the same
        coordinate system as ``points_2d``).
    image_size : tuple(int, int)
        Output canvas size as (width, height).
    line_thickness : int
        Thickness of the rendered lines in pixels.
    fit_margin : float
        Fraction of the canvas used to fit the face.  The rendered
        structure will occupy up to ``fit_margin`` of the smaller
        dimension of the image, leaving equal margins around.

    Returns
    -------
    PIL.Image.Image
        A RGB image with the rendered face graph, black lines on white
        background.
    """
    w, h = image_size
    img = Image.new("RGB", (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Determine bounding box of projected face
    xs = [p[0] for p in points_2d.values()]
    ys = [p[1] for p in points_2d.values()]
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    bw = max(1e-6, (maxx - minx))
    bh = max(1e-6, (maxy - miny))
    # Compute uniform scale to fit into the canvas
    scale = fit_margin * min(w / bw, h / bh)
    # Face centre
    cx = (minx + maxx) * 0.5
    cy = (miny + maxy) * 0.5

    def to_px(x: float, y: float) -> Tuple[int, int]:
        px = (x - cx) * scale + (w * 0.5)
        py = (h * 0.5) - (y - cy) * scale
        return (int(px), int(py))

    # Draw straight edges
    for a, b in edges:
        if a not in points_2d or b not in points_2d:
            continue
        p1 = to_px(*points_2d[a])
        p2 = to_px(*points_2d[b])
        draw.line([p1, p2], fill=(0, 0, 0), width=line_thickness)

    # Draw eyes as perfect circles/ellipses
    for eye_data in eyes.values():
        ck = eye_data.get("center_key")
        radius = float(eye_data.get("radius", 0.0))
        if ck not in points_2d or radius <= 0:
            continue
        cx_, cy_ = points_2d[ck]
        cpx, cpy = to_px(cx_, cy_)
        r_scaled = radius * scale
        box = [cpx - r_scaled, cpy - r_scaled, cpx + r_scaled, cpy + r_scaled]
        draw.ellipse(box, outline=(0, 0, 0), width=line_thickness)

    return img


def pil_to_comfy_image(img: Image.Image) -> np.ndarray:
    """Convert a PIL Image to a ComfyUI‑compatible tensor.

    ComfyUI expects images in a BHWC format with float32 values in
    [0,1].  This helper returns such an array with a batch dimension
    added (batch size = 1).
    """
    arr = np.array(img).astype(np.float32) / 255.0
    return arr[None, ...]