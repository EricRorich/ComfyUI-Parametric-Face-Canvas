# ComfyUI Parametric Face Canvas

This repository contains a custom node for [ComfyUI](https://github.com/comfyanonymous/ComfyUI) that generates a parametric 3D face wireframe and renders it as a 2D image.  The node exposes intuitive parameters for adjusting facial proportions (eye distance, nose width, jaw width, face height and depth) as well as camera orientation (yaw and pitch).  The resulting image can be used as conditioning input for AI pipelines such as ControlNet, pose/structure guidance or any workflow where a clean, configurable facial structure is useful.

## Features

- **Lightweight parametric model** defined as simple analytic curves – no heavy 3D engine or external models are required.
- **True 3D representation**: adjust camera yaw and pitch to view the face from different angles.
- **Adjustable proportions**: eye distance, eye size, nose width, jaw width, face height and depth can all be tuned at runtime via node sliders.
- **Customisable output**: configure image resolution and line thickness.
- **Minimal dependencies** beyond the packages shipped with ComfyUI (`numpy`, `Pillow` and `torch`).

## Installation

1. Clone this repository into your `ComfyUI/custom_nodes` directory.  From a terminal in the `ComfyUI/custom_nodes` folder run:

   ```bash
   git clone https://github.com/EricRorich/ComfyUI-Parametric-Face-Canvas.git
   ```

2. Install the Python dependencies (if they are not already available in your environment):

   ```bash
   pip install numpy pillow torch
   ```

3. Restart ComfyUI.  The new node will appear under the **CS Custom Nodes/Face** category.

## Usage

Add the **Parametric Face Canvas** node to your ComfyUI workflow.  Adjust the sliders for facial proportions and camera orientation.  The node outputs a single image containing the 2D rendering of the parametric face.  This output can be fed directly into downstream nodes such as **ControlNet**, compositing nodes or any other module that expects an image tensor.

Each parameter has sensible defaults, but you are free to explore creative variations.  For example, you can increase the jaw width for a more square jawline or decrease the eye distance for a narrower face.  The **Yaw** and **Pitch** sliders allow you to rotate the face left/right and up/down respectively, giving you both front and side views.

## Contributing

Contributions are welcome!  Please open issues to report bugs or suggest features.  Pull requests with improvements or additional facial parameters are appreciated.

## License

This project is licensed under the MIT License.  See the [LICENSE](LICENSE) file for full text.