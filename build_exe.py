import PyInstaller.__main__
import sys
import os
from pathlib import Path

def build():
    print("Building IGI 2 MEF Viewer standalone EXE...")
    
    # Paths
    root = Path(__file__).parent.absolute()
    main_script = root / "src" / "mef_viewer" / "main.py"
    guide_file  = root / "src" / "mef_viewer" / "guidemef.md"
    
    # PyInstaller arguments
    args = [
        str(main_script),
        "--onefile",
        "--windowed",
        "--name=IGI2_MEF_Viewer",
        f"--paths={str(root / 'src')}",
        f"--add-data={str(guide_file)};.",
        "--clean",
        "--noconfirm",
    ]
    
    # Optimization: exclude unnecessary modules to reduce size
    args += ["--exclude-module=tkinter", "--exclude-module=unittest", "--exclude-module=pydoc"]

    # Run
    try:
        PyInstaller.__main__.run(args)
        print("\nSUCCESS: EXE created in the 'dist' folder.")
    except Exception as e:
        print(f"\nERROR: Build failed: {e}")

if __name__ == "__main__":
    build()
