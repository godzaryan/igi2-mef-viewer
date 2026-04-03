# IGI 2 MEF Viewer (Standalone & Portable) [v1.4.0]

An OpenGL-based 3D model viewer for **IGI 2: Covert Strike** binary `.mef` files.

## Features
- **High-Fidelity Rendering**: Supports Rigid, Bone, Lightmap, and Shadow models.
- **Detailed Overlays**: Toggle bones, magic vertices, collision meshes, portals, and glow sprites.
- **Embedded Documentation**: Built-in "MEF Guide" for learning the binary format.
- **Cross-Platform Readiness**: Portable EXE distribution for Windows.

## Installation

### For Users (Fast)
Download the latest `IGI2_MEF_Viewer.exe` from the [Releases](https://github.com/godzaryan/igi2-mef-viewer/releases) page. No Python required.

### For Developers
1. Clone this repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the viewer:
   ```bash
   python src/mef_viewer/main.py
   ```

## Creating a Portable EXE
Run the provided build script:
```bash
python build_exe.py
```
The result will be in the `dist/` folder.

## Library Support
This viewer uses the [igi2mef](https://github.com/godzaryan/igi2-mef-library) core library for all parsing tasks.

## License
MIT License.
