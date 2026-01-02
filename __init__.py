"""Entry point for the Parametric Face Canvas custom node.

This file exposes the node class via the mappings expected by ComfyUI.

When ComfyUI scans the `custom_nodes` folder, it imports
`NODE_CLASS_MAPPINGS` and `NODE_DISPLAY_NAME_MAPPINGS` from each module.
"""

from .node import ParametricFaceCanvas

NODE_CLASS_MAPPINGS = {
    # The key is the internal node identifier and the value is the class.
    "Parametric Face Canvas": ParametricFaceCanvas,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    # Human‑readable name shown in the UI for this node.
    "ParametricFaceCanvas": "Parametric Face Canvas",
}