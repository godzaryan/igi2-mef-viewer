"""
mef_viewer.py — IGI 2 MEF Viewer (Official PyPI Edition)
Final distribution version with fixed GL backend integration.
"""
from __future__ import annotations
import sys, os, traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

def resource_path(relative_path: str) -> Path:
    if hasattr(sys, '_MEIPASS'):
        return Path(sys._MEIPASS) / relative_path
    return Path(__file__).parent / relative_path

# ── Dependency checks ─────────────────────────────────────────────────────────
try:
    from PyQt5.QtWidgets import *
    from PyQt5.QtCore import *
    from PyQt5.QtGui import *
except ImportError:
    print("CRITICAL: PyQt5 missing. Please run: pip install PyQt5")
    sys.exit(1)

try:
    import moderngl
except ImportError:
    print("CRITICAL: moderngl missing. Please run: pip install moderngl")
    sys.exit(1)

try:
    from igi2mef import parse_mef, quick_validate, MefModel
except ImportError:
    print("CRITICAL: igi2mef library not found. Please run: pip install igi2mef")
    sys.exit(1)

from gl_backend import (
    OrbitCamera, GpuCache, build_grid, build_bone_overlay,
    build_points_overlay, build_lines_overlay,
    gl_bytes, VERT, FRAG, FLAT_VERT, FLAT_FRAG
)

# ── Styles ────────────────────────────────────────────────────────────────────
STYLE = """
QMainWindow,QWidget{background:#0d0d1a;color:#dde6f0;font-family:'Segoe UI';font-size:12px;}
QSplitter::handle{background:#1c1c38;}
QListWidget{background:#080814;border:none;outline:none;}
QListWidget::item{border-bottom:1px solid #14142a;padding:4px;}
QListWidget::item:selected{background:#162840;color:#60c8ff;}
QLineEdit{background:#0a0a18;border:1px solid #252548;border-radius:4px;padding:5px;color:#dde6f0;}
QTabWidget::pane{border:1px solid #1c1c38;background:#0d0d1a;}
QTabBar::tab{background:#0a0a18;color:#607080;padding:6px 14px;border-right:1px solid #1c1c38;}
QTabBar::tab:selected{background:#162840;color:#60c8ff;}
QToolBar{background:#080814;border-bottom:1px solid #1c1c38;padding:2px;}
QToolButton{background:transparent;color:#8090a8;border:none;padding:4px 8px;}
QToolButton:hover{background:#141430;color:#dde6f0;}
QToolButton:checked{background:#162840;color:#60c8ff;}
QStatusBar{background:#080814;color:#506070;border-top:1px solid #1c1c38;font-size:11px;}
QTextEdit{background:#080814;border:none;color:#c0ccd8;font-family:Consolas;font-size:11px;}
"""

class _ParseWorker(QThread):
    done = pyqtSignal(object)
    def __init__(self, paths: List[Path], cache: Dict):
        super().__init__(); self._paths, self._cache = paths, cache
    def run(self):
        try:
            for p in self._paths:
                if str(p) not in self._cache:
                    self._cache[str(p)] = parse_mef(p)
            self.done.emit(self._cache)
        except Exception:
            traceback.print_exc()

class MefViewer(QMainWindow):
    def __init__(self, initial_folder: str = None):
        super().__init__(); self.setWindowTitle("IGI 2 MEF Viewer (Official v1.1.2)")
        self.resize(1400, 900); self.setStyleSheet(STYLE)
        self.cache: Dict[str, MefModel] = {}
        self.current_path: Optional[Path] = None
        self.worker: Optional[_ParseWorker] = None
        self._init_ui(); self._init_gl_params()
        if initial_folder: self._scan_folder(Path(initial_folder))

    def _init_ui(self):
        self.toolbar = self.addToolBar("Main")
        self.btn_open = QAction("📁 Open Folder", self); self.btn_open.triggered.connect(self._on_open)
        self.toolbar.addAction(self.btn_open); self.toolbar.addSeparator()
        self.splitter = QSplitter(Qt.Horizontal); self.setCentralWidget(self.splitter)
        self.left_panel = QWidget(); self.left_layout = QVBoxLayout(self.left_panel)
        self.search_bar = QLineEdit(); self.search_bar.setPlaceholderText("Filter models..."); self.search_bar.textChanged.connect(self._on_filter)
        self.file_list = QListWidget(); self.file_list.itemSelectionChanged.connect(self._on_selection_changed)
        self.left_layout.addWidget(self.search_bar); self.left_layout.addWidget(self.file_list)
        self.splitter.addWidget(self.left_panel)
        self.tabs = QTabWidget(); self.splitter.addWidget(self.tabs)
        self.gl_widget = QOpenGLWidget(); self.gl_widget.setMinimumWidth(800)
        self.gl_widget.initializeGL = self._gl_init
        self.gl_widget.resizeGL = self._gl_resize
        self.gl_widget.paintGL = self._gl_paint
        self.tabs.addTab(self.gl_widget, "3D Viewport")
        self.doc_view = QTextEdit(); self.doc_view.setReadOnly(True); self.tabs.addTab(self.doc_view, "MEF Guide")
        self._load_doc()
        self.prop_view = QTextEdit(); self.prop_view.setReadOnly(True); self.tabs.addTab(self.prop_view, "Properties")
        self.status = self.statusBar(); self.status.showMessage("Ready"); self.splitter.setStretchFactor(1, 4)

    def _init_gl_params(self):
        self.gl_ctx = None; self.gpu_cache = None; self.cam = OrbitCamera(); self.cam.dist = 5.0
        self.show_wire = False; self.show_bones = True; self.show_magic = True
        self.show_glow = True
        self._add_toggle("Wires", "show_wire"); self._add_toggle("Bones", "show_bones", True)
        self._add_toggle("Magic", "show_magic", True); self._add_toggle("Glow", "show_glow", True)

    def _add_toggle(self, text, attr, default=False):
        act = QAction(text, self, checkable=True); act.setChecked(default)
        act.triggered.connect(lambda: (setattr(self, attr, act.isChecked()), self.gl_widget.update()))
        self.toolbar.addAction(act)

    def _load_doc(self):
        p = resource_path("guidemef.md")
        if p.exists(): self.doc_view.setMarkdown(p.read_text(errors="replace"))

    def _on_open(self):
        d = QFileDialog.getExistingDirectory(self, "Open Folder")
        if d:
            self._scan_folder(Path(d))

    def _scan_folder(self, path: Path):
        self.file_list.clear(); self.cache = {}
        for f in sorted(list(path.glob("*.mef")), key=lambda x: x.name.lower()): self.file_list.addItem(f.name)
        self.current_folder = path; self.status.showMessage(f"Loaded {self.file_list.count()} models")

    def _on_filter(self, text):
        for i in range(self.file_list.count()):
            it = self.file_list.item(i); it.setHidden(text.lower() not in it.text().lower())

    def _on_selection_changed(self):
        it = self.file_list.currentItem()
        if not it:
            return
        p = self.current_folder / it.text()
        self.current_path = p
        if str(p) in self.cache:
            self._display_model(self.cache[str(p)])
        else:
            self.worker = _ParseWorker([p], self.cache)
            self.worker.done.connect(self._on_parse_done)
            self.worker.start()

    def _on_parse_done(self):
        if self.current_path and str(self.current_path) in self.cache:
            self._display_model(self.cache[str(self.current_path)])

    def _display_model(self, m: MefModel):
        if not m.valid:
            self.status.showMessage(f"Error: {m.error}")
            return
        self.prop_view.setPlainText("\n".join([f"{k}: {v}" for k,v in {
            "Model": m.name, "Type": m.model_type_name, "Size": m.file_size_human,
            "Verts": m.total_vertices, "Tris": m.total_triangles, "Bones": len(m.bones)
        }.items()]))
        self.gl_widget.update()

    # ── OpenGL ────────────────────────────────────────────────────────────────
    def _gl_init(self):
        try:
            print("INFO: Initializing ModernGL Context...")
            self.gl_ctx = moderngl.create_context()
            print(f"INFO: OpenGL Context created (Vendor: {self.gl_ctx.info.get('GL_VENDOR')})")
            self.prog = self.gl_ctx.program(vertex_shader=VERT, fragment_shader=FRAG)
            self.flat_prog = self.gl_ctx.program(vertex_shader=FLAT_VERT, fragment_shader=FLAT_FRAG)
            self.gpu_cache = GpuCache(self.gl_ctx, self.prog)
            self.grid_vao, self.grid_n = build_grid(self.gl_ctx, self.flat_prog)
            print("INFO: Shader programs and Grid VAO successfully initialized.")
        except Exception as e:
            msg = f"ERROR: ModernGL Initialization failed:\n{traceback.format_exc()}"
            print(msg)
            QMessageBox.critical(self, "OpenGL Error", msg)

    def _gl_resize(self, w, h):
        if self.gl_ctx:
            self.gl_ctx.viewport = (0, 0, w, h)

    def _gl_paint(self):
        if not self.gl_ctx:
            return
        try:
            self.gl_ctx.clear(0.05, 0.05, 0.1)
            self.gl_ctx.enable(moderngl.DEPTH_TEST)
            m, p = self.cam.matrices(self.gl_widget.width(), self.gl_widget.height())
            mvp = p @ m
            
            self.flat_prog["u_mvp"].write(gl_bytes(mvp))
            self.flat_prog["u_color"].value = (0.2, 0.2, 0.4, 1.0)
            self.grid_vao.render(moderngl.LINES, vertices=self.grid_n)
            
            mdl = self.cache.get(str(self.current_path))
            if not mdl or not mdl.valid:
                return
            
            self.prog["u_mvp"].write(gl_bytes(mvp))
            self.prog["u_model"].write(gl_bytes(np.eye(4, dtype="f4")))
            self.prog["u_cam"].value = tuple(self.cam.eye())
            
            gpu = self.gpu_cache.get(mdl)
            if gpu:
                gpu.draw(self.prog)
            
            # Overlays
            self.gl_ctx.disable(moderngl.DEPTH_TEST)
            self.flat_prog["u_mvp"].write(gl_bytes(mvp))
            if self.show_bones and mdl.bones:
                res = build_bone_overlay(self.gl_ctx, self.flat_prog, mdl)
                if res:
                    lv, ln, jv, jn = res
                    self.flat_prog["u_color"].value = (1, 1, 1, 1)
                    lv.render(moderngl.LINES, vertices=ln)
                    self.flat_prog["u_color"].value = (1, 0, 0, 1)
                    jv.render(moderngl.POINTS, vertices=jn)
            if self.show_magic and mdl.magic_vertices:
                res = build_points_overlay(self.gl_ctx, self.flat_prog, [v.position for v in mdl.magic_vertices])
                if res:
                    v, n = res
                    self.flat_prog["u_color"].value = (1, 1, 0, 1)
                    self.gl_ctx.point_size = 5.0
                    v.render(moderngl.POINTS, vertices=n)
            if self.show_glow and mdl.glow_sprites:
                res = build_points_overlay(self.gl_ctx, self.flat_prog, [v.position for v in mdl.glow_sprites])
                if res:
                    v, n = res
                    self.flat_prog["u_color"].value = (1, 0.5, 0, 1)
                    self.gl_ctx.point_size = 8.0
                    v.render(moderngl.POINTS, vertices=n)
        except Exception:
            print(f"ERROR: GL Paint failed:\n{traceback.format_exc()}")

# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = QApplication(sys.argv)
    try:
        win = MefViewer()
        win.show()
        sys.exit(app.exec_())
    except Exception:
        print(f"CRITICAL: Application crashed:\n{traceback.format_exc()}")
