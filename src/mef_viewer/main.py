"""
mef_viewer.py — IGI 2 MEF Viewer (Official PyPI Edition)
Imports igi2mef from pip.
"""
from __future__ import annotations
import sys, os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

def resource_path(relative_path: str) -> Path:
    """ Get absolute path to resource, works for dev and for PyInstaller """
    if hasattr(sys, '_MEIPASS'):
        return Path(sys._MEIPASS) / relative_path
    return Path(__file__).parent / relative_path

# ── Dependency checks ─────────────────────────────────────────────────────────
try:
    from PyQt5.QtWidgets import *
    from PyQt5.QtCore import *
    from PyQt5.QtGui import *
except ImportError:
    print("ERROR: PyQt5 missing."); sys.exit(1)
try:
    import moderngl
except ImportError:
    print("ERROR: moderngl missing."); sys.exit(1)

# ── Import from Official PyPI Package ───────────────────────────────────────
try:
    from igi2mef import parse_mef, quick_validate, MefModel
except ImportError:
    print("ERROR: igi2mef library not found. Run: pip install igi2mef"); sys.exit(1)

# ── GL core ───────────────────────────────────────────────────────────────────
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

# ── Workers ───────────────────────────────────────────────────────────────────
class _ParseWorker(QThread):
    done = pyqtSignal(object)
    def __init__(self, paths: List[Path], cache: Dict):
        super().__init__(); self._paths, self._cache = paths, cache
    def run(self):
        for p in self._paths:
            if str(p) not in self._cache:
                self._cache[str(p)] = parse_mef(p)
        self.done.emit(self._cache)

# ── Main Application ──────────────────────────────────────────────────────────
class MefViewer(QMainWindow):
    def __init__(self, initial_folder: str = None):
        super().__init__(); self.setWindowTitle("IGI 2 MEF Viewer (Official Distribution)")
        self.resize(1400, 900); self.setStyleSheet(STYLE)
        
        self.cache: Dict[str, MefModel] = {}
        self.current_path: Optional[Path] = None
        self.worker: Optional[_ParseWorker] = None
        
        self._init_ui(); self._init_gl_params()
        if initial_folder: self._scan_folder(Path(initial_folder))

    def _init_ui(self):
        # Toolbar
        self.toolbar = self.addToolBar("Main")
        self.btn_open = QAction("📁 Open Folder", self); self.btn_open.triggered.connect(self._on_open)
        self.toolbar.addAction(self.btn_open)
        self.toolbar.addSeparator()
        
        # Splitter Layout
        self.splitter = QSplitter(Qt.Horizontal); self.setCentralWidget(self.splitter)
        
        # Left: File Browser
        self.left_panel = QWidget(); self.left_layout = QVBoxLayout(self.left_panel)
        self.search_bar = QLineEdit(); self.search_bar.setPlaceholderText("Filter models..."); self.search_bar.textChanged.connect(self._on_filter)
        self.file_list = QListWidget(); self.file_list.itemSelectionChanged.connect(self._on_selection_changed)
        self.left_layout.addWidget(self.search_bar); self.left_layout.addWidget(self.file_list)
        self.splitter.addWidget(self.left_panel)
        
        # Middle / Right: Tabs
        self.tabs = QTabWidget(); self.splitter.addWidget(self.tabs)
        
        # 3D Tab
        self.gl_widget = QOpenGLWidget(); self.gl_widget.setMinimumWidth(800)
        self.gl_widget.initializeGL = self._gl_init
        self.gl_widget.resizeGL = self._gl_resize
        self.gl_widget.paintGL = self._gl_paint
        self.tabs.addTab(self.gl_widget, "3D Viewport")
        
        # Guide Tab
        self.doc_view = QTextEdit(); self.doc_view.setReadOnly(True)
        self.tabs.addTab(self.doc_view, "MEF Guide")
        self._load_doc()
        
        # Properties Tab
        self.prop_view = QTextEdit(); self.prop_view.setReadOnly(True)
        self.tabs.addTab(self.prop_view, "Model Attributes")
        
        # Status Bar
        self.status = self.statusBar(); self.status.showMessage("Ready")
        self.splitter.setStretchFactor(1, 4)

    def _init_gl_params(self):
        self.gl_ctx = None; self.gpu_cache = None; self.cam = OrbitCamera(); self.cam.dist = 5.0
        self.show_wire = False; self.show_bones = True; self.show_magic = True
        self.show_portals = False; self.show_coll = False; self.show_glow = True
        
        # Overlay toggles in toolbar
        self._add_toggle("Wires", "show_wire"); self._add_toggle("Bones", "show_bones", True)
        self._add_toggle("Magic", "show_magic", True); self._add_toggle("Portals", "show_portals")
        self._add_toggle("Coll", "show_coll"); self._add_toggle("Glow", "show_glow", True)

    def _add_toggle(self, text, attr, default=False):
        act = QAction(text, self, checkable=True); act.setChecked(default)
        act.triggered.connect(lambda: (setattr(self, attr, act.isChecked()), self.gl_widget.update()))
        self.toolbar.addAction(act)

    def _load_doc(self):
        p = resource_path("guidemef.md")
        if p.exists(): self.doc_view.setMarkdown(p.read_text(errors="replace"))

    def _on_open(self):
        d = QFileDialog.getExistingDirectory(self, "Open Folder with MEF Files")
        if d: self._scan_folder(Path(d))

    def _scan_folder(self, path: Path):
        self.file_list.clear(); self.cache = {}
        files = sorted(list(path.glob("*.mef")), key=lambda x: x.name.lower())
        for f in files: self.file_list.addItem(f.name)
        self.current_folder = path; self.status.showMessage(f"Loaded {len(files)} models from {path.name}")

    def _on_filter(self, text):
        for i in range(self.file_list.count()):
            it = self.file_list.item(i); it.setHidden(text.lower() not in it.text().lower())

    def _on_selection_changed(self):
        it = self.file_list.currentItem()
        if not it: return
        p = self.current_folder / it.text(); self.current_path = p
        if str(p) in self.cache: self._display_model(self.cache[str(p)])
        else:
            self.status.showMessage(f"Parsing {p.name}...")
            self.worker = _ParseWorker([p], self.cache); self.worker.done.connect(self._on_parse_done); self.worker.start()

    def _on_parse_done(self):
        if self.current_path and str(self.current_path) in self.cache:
            self._display_model(self.cache[str(self.current_path)])

    def _display_model(self, model: MefModel):
        if not model.valid: self.status.showMessage(f"Error: {model.error}"); return
        self.status.showMessage(f"{model.name} | Type {model.model_type} | {model.total_vertices} Verts")
        self._update_props(model); self.gl_widget.update()

    def _update_props(self, m: MefModel):
        lines = [f"File: {m.name}", f"Size: {m.file_size_human}", f"Type: {m.model_type_name}",
                 f"HSEM Rev: {m.hsem_version}", f"Vertices: {m.total_vertices}", f"Triangles: {m.total_triangles}",
                 f"Parts/Meshes: {len(m.parts)}", f"Bones: {len(m.bones)}", f"Magic Verts: {len(m.magic_vertices)}",
                 f"Bounds Min: {m.bounds_min}", f"Bounds Max: {m.bounds_max}"]
        self.prop_view.setPlainText("\n".join(lines))

    # ── OpenGL Pipeline ────────────────────────────────────────────────────────
    def _gl_init(self):
        self.gl_ctx = moderngl.create_context(); self.gpu_cache = GpuCache(self.gl_ctx)
        self.prog = self.gl_ctx.program(vertex_shader=VERT, fragment_shader=FRAG)
        self.flat_prog = self.gl_ctx.program(vertex_shader=FLAT_VERT, fragment_shader=FLAT_FRAG)
        self.grid_vao = build_grid(self.gl_ctx, self.flat_prog)

    def _gl_resize(self, w, h): self.gl_ctx.viewport = (0, 0, w, h)

    def _gl_paint(self):
        self.gl_ctx.clear(0.05, 0.05, 0.1, 1.0); self.gl_ctx.enable(moderngl.DEPTH_TEST)
        m, p = self.cam.matrix(self.gl_widget.width(), self.gl_widget.height()), self.cam.proj(self.gl_widget.width(), self.gl_widget.height())
        self.flat_prog["m_proj"].write(p); self.flat_prog["m_view"].write(m); self.flat_prog["u_color"].value = (0.2, 0.2, 0.4)
        self.grid_vao.render(moderngl.LINES)
        
        it = self.file_list.currentItem()
        if not (it and str(self.current_folder / it.text()) in self.cache): return
        mdl = self.cache[str(self.current_folder / it.text())]
        
        # Render Parts
        self.prog["m_proj"].write(p); self.prog["m_view"].write(m)
        self.prog["u_wire"].value = self.show_wire
        for p_idx, part in enumerate(mdl.parts):
            vao = self.gpu_cache.get_part(part)
            self.prog["u_color"].value = ((p_idx*0.2)%1.0, (p_idx*0.5)%1.0, (p_idx*0.7)%1.0)
            vao.render(moderngl.TRIANGLES)
            
        # Overlays
        self.gl_ctx.disable(moderngl.DEPTH_TEST)
        if self.show_bones and mdl.bones:
            build_bone_overlay(self.gl_ctx, self.flat_prog, mdl.bones).render(moderngl.LINES)
        if self.show_magic and mdl.magic_vertices:
            pts = [v.position for v in mdl.magic_vertices]
            build_points_overlay(self.gl_ctx, self.flat_prog, pts, (1,1,0)).render(moderngl.POINTS)
        if self.show_glow and mdl.glow_sprites:
            pts = [v.position for v in mdl.glow_sprites]
            build_points_overlay(self.gl_ctx, self.flat_prog, pts, (1,0.5,0), 8.0).render(moderngl.POINTS)

# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = QApplication(sys.argv); win = MefViewer(); win.show(); sys.exit(app.exec_())
