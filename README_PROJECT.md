# IGI 2 MEF Viewer & Library

A high-performance, specification-driven viewer and parsing library for **IGI 2: Covert Strike** binary `.mef` models.

## Features
- **Fast Parsing**: Optimized binary parsing for Rigid, Bone, Lightmap, and Shadow models.
- **Accurate Rendering**: OpenGL-based 3D viewport with part-level coloring and wireframe support.
- **Visual Overlays**:
    - 🦴 Skeleton / Bone hierarchy
    - ✦ Magic Vertices (attachment points)
    - 🔺 Collision Meshes
    - 🚪 Portals
    - ✨ Glow Sprites
- **Shadow Model Support**: Full support for `SEMS`/`XTVS`/`CAFS` shadow geometry.

## Project Structure
- `src/igi2mef`: The core parsing library (pip-installable).
- `src/mef_viewer`: The PyQt5 + ModernGL desktop application.
- `docs/`: Format specifications and documentations.

## Installation

### For Users (Standalone)
Download the latest portable EXE from the [Releases](https://github.com/user/igi2-mef-viewer/releases) page. No Python installation required.

### For Developers
1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the viewer:
   ```bash
   python src/mef_viewer/main.py
   ```
4. Install the library for use in your own projects:
   ```bash
   pip install .
   ```

## License
MIT License. See `LICENSE` for details.
