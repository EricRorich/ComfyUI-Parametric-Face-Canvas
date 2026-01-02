"""Rendering utilities for drawing 2D wireframes to an image canvas."""

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

from PIL import Image, ImageDraw
import numpy as np


def render_wireframe(
    curves_2d: Dict[str, Iterable[Tuple[float, float]]],
    image_size: Tuple[int, int] = (512, 512),
    line_thickness: int = 2,
    line_color: Tuple[int, int, int] = (255, 255, 255),
    background_color: Tuple[int, int, int] = (0, 0, 0),
) -> Image.Image:
    """Render a set of 2D curves onto an image.

    Parameters
    ----------
    curves_2d : dict
        Mapping from curve names to sequences of 2D points.  The points
        should be centred around (0, 0) in normalised space.
    image_size : tuple(int, int)
        Size of the output image as (width, height).
    line_thickness : int
        Thickness of the drawn lines in pixels.
    line_color : tuple(int, int, int)
        RGB colour used for the wireframe.
    background_color : tuple(int, int, int)
        RGB colour used for the background.

    Returns
    -------
    PIL.Image.Image
        The rendered image.
    """
    width, height = image_size
    img = Image.new("RGB", (width, height), color=background_color)
    draw = ImageDraw.Draw(img)

    for pts in curves_2d.values():
        if not pts:
            continue
        # Transform normalised coords [-1,1] into pixel positions
        # X: -1 -> left, +1 -> right; Y (vertical): +1 -> top, -1 -> bottom
        screen_pts: List[Tuple[int, int]] = []
        for (x, y) in pts:
            # Scale to half the canvas and translate to centre
            sx = int(x * (width / 2.0) + width / 2.0)
            sy = int(height / 2.0 - y * (height / 2.0))
            screen_pts.append((sx, sy))
        # Draw polyline
        if len(screen_pts) > 1:
            draw.line(screen_pts, fill=line_color, width=line_thickness, joint="curve")
        else:
            # Single point – draw a small dot
            x0, y0 = screen_pts[0]
            draw.ellipse(
                [x0 - line_thickness, y0 - line_thickness, x0 + line_thickness, y0 + line_thickness],
                fill=line_color,
            )
    return img


def image_to_tensor(img: Image.Image) -> np.ndarray:
    """Convert a PIL image to a PyTorch‑style tensor (CHW) in the [0,1] range.

    ComfyUI expects tensors shaped as (B, C, H, W).  This helper returns
    the CHW array; the caller can add a batch dimension.
    """
    arr = np.array(img, dtype=np.float32) / 255.0
    # Convert HWC to CHW
    chw = np.transpose(arr, (2, 0, 1))
    return chw