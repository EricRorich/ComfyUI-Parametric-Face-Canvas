"""Topology definitions for the parametric face canvas.

This package contains pre‑defined point and edge data for the male and
female golden ratio masks.  Each module exposes three attributes:

  * ``POINTS`` – a mapping of point names to 3D coordinates (x, y, z)
  * ``EDGES``  – a list of tuples defining connections between point names
  * ``EYES``   – a mapping of ``"left"`` and ``"right"`` to dictionaries
    specifying ``center_key`` (the point name for the eye centre) and
    ``radius`` (the eye radius in the model's normalised space)

The default coordinates provided here approximate the general shape of
male and female faces and are intended as a starting point.  They can
be replaced with more precise values if desired.
"""

from . import male, female

__all__ = ["male", "female"]