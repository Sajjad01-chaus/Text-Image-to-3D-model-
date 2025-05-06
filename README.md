# Text-Image-to-3D-model

## Description

This project is a simple prototype application built for an assignment. It accepts either a single-object image file or a short text prompt as input and generates a basic 3D mesh model (in `.obj`) using the Shap-E generative model. It also includes features for basic image preprocessing, mesh cleanup, and visualization.

## Features

* Accepts text prompts (e.g., "a red chair").
* Accepts image file paths (`.png`, `.jpg`, `.jpeg`, `.bmp`, `.webp`).
* Optional background removal for image inputs using OpenCV GrabCut.
* Generates 3D models using OpenAI's Shap-E library (`text300M` for text, `image300M` for images).
* Saves output meshes as `.obj` files.
* Optional mesh cleaning (duplicate vertices, degenerate faces, hole filling, normal fixing) using Trimesh.
* Optional mesh simplification for large meshes (using Trimesh quadric decimation).
* Optional interactive 3D visualization of the generated model using matplotlib.

## Requirements

* Python 3.8+ recommended
* PyTorch
* Other dependencies as listed in `requirements.txt`. Key libraries include:
    * `shap-e` (installed via git)
    * `trimesh`
    * `numpy`
    * `Pillow`
    * `opencv-python`
    * `plotly` (for visualization)
    * `argparse` (standard library)

## Installation & Setup

1.  **Clone/Download:** Obtain the project files and place them in a directory (e.g., `ASSIGNMENT_PROJECT`).
2.  **Navigate:** Open a terminal or command prompt and navigate into the project directory:
    ```bash
    cd path/to/ASSIGNMENT_PROJECT
    ```
3.  **Create Virtual Environment:** It's highly recommended to use a virtual environment:
    ```bash
    python -m venv venv
    # or: python3 -m venv venv
    ```
4.  **Activate Environment:**
    * Windows: `.\venv\Scripts\activate`
    * macOS/Linux: `source venv/bin/activate`
5.  **Install Dependencies:** Install all required libraries:
    ```bash
    pip install -r requirements.txt
    ```
6.  **Model Download Note:** The first time you run a generation using Shap-E, the necessary pre-trained models (which are several gigabytes) will be downloaded automatically to a cache directory (usually `~/.cache/shap-e`). This requires an internet connection and sufficient disk space.

## Usage

Run the main script from the command line within the activated virtual environment.

```bash
python main.py <input> [options]
