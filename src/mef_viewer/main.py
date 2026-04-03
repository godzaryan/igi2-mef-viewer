"""
mef_viewer.py — IGI 2 MEF Viewer  (Specification-Driven Edition)
Entry point. Run: python mef_viewer.py [folder]

Requires: PyQt5  moderngl  numpy
Install:  pip install PyQt5 moderngl numpy
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
    print("ERROR: PyQt5 not installed.\n  pip install PyQt5"); sys.exit(1)
try:
    import moderngl
except ImportError:
    print("ERROR: moderngl not installed.\n  pip install moderngl"); sys.exit(1)

# ── igi2mef library ───────────────────────────────────────────────────────────
# In production, this should be installed via pip. 
# For development/local-run, we add the src dir.
_root = Path(__file__).parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

try:
    from igi2mef import parse_mef, quick_validate, MefModel
except ImportError:
    print("ERROR: igi2mef library not found. Install it or check src structure.")
    sys.exit(1)

# ── GL backend ────────────────────────────────────────────────────────────────
from gl_backend import (
    OrbitCamera, GpuCache, build_grid, build_bone_overlay,
    build_points_overlay, build_lines_overlay,
    gl_bytes, VERT, FRAG, FLAT_VERT, FLAT_FRAG,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE    = Path(__file__).parent
_MEFINFO = _HERE.parent.parent / "MEFInfo"
_DOC_MD  = _MEFINFO / "MEF_text.md"

# ── Dark stylesheet ───────────────────────────────────────────────────────────
STYLE = """
QMainWindow,QWidget{background:#0d0d1a;color:#dde6f0;
    font-family:'Segoe UI',Arial,sans-serif;font-size:12px;}
QSplitter::handle{background:#1c1c38;width:2px;height:2px;}
QListWidget{background:#080814;border:none;outline:none;}
QListWidget::item{border-bottom:1px solid #14142a;}
QListWidget::item:selected{background:#162840;}
QListWidget::item:hover:!selected{background:#111128;}
QLineEdit{background:#0a0a18;border:1px solid #252548;border-radius:4px;
    padding:5px 10px;color:#dde6f0;}
QLineEdit:focus{border-color:#0080cc;}
QLabel{color:#dde6f0;}
QLabel#sec{color:#00aaff;font-weight:bold;font-size:10px;letter-spacing:1.5px;}
QLabel#val{color:#90b8d8;}
QTabWidget::pane{border:1px solid #1c1c38;background:#0d0d1a;}
QTabBar::tab{background:#0a0a18;color:#607080;padding:6px 14px;border:none;
    border-right:1px solid #1c1c38;}
QTabBar::tab:selected{background:#162840;color:#60c8ff;}
QTableWidget{background:#080814;border:none;gridline-color:#14142a;
    selection-background-color:#162840;}
QHeaderView::section{background:#0a0a18;color:#607080;padding:4px 8px;
    border:none;border-bottom:1px solid #1c1c38;}
QToolBar{background:#080814;border-bottom:1px solid #1c1c38;
    spacing:2px;padding:2px 4px;}
QToolButton{background:transparent;color:#8090a8;border:none;
    border-radius:3px;padding:4px 10px;font-size:12px;}
QToolButton:hover{background:#141430;color:#dde6f0;}
QToolButton:checked{background:#162840;color:#60c8ff;}
QStatusBar{background:#080814;color:#506070;
    border-top:1px solid #1c1c38;font-size:11px;}
QStatusBar::item{border:none;}
QScrollBar:vertical{background:#0d0d1a;width:7px;margin:0;}
QScrollBar::handle:vertical{background:#252548;border-radius:3px;min-height:24px;}
QScrollBar::add-line:vertical,QScrollBar::sub-line:vertical{height:0;}
QSplitter{background:#0d0d1a;}
QTreeWidget{background:#080814;border:none;outline:none;}
QTreeWidget::item{padding:2px;}
QTreeWidget::item:selected{background:#162840;}
QTextEdit{background:#080814;border:none;color:#c0ccd8;font-family:Consolas,'Courier New',monospace;font-size:11px;}
"""

# ── Parse worker ──────────────────────────────────────────────────────────────
class _ParseWorker(QThread):
    done = pyqtSignal(object)

    def __init__(self, paths: List[Path], cache: Dict):
        super().__init__()
        self._paths = paths
        self._cache = cache
        self._cancelled = False

    def cancel(self): self._cancelled = True

    def run(self):
        try:
            for p in self._paths:
                if self._cancelled: return
                key = str(p)
                if key not in self._cache:
                    self._cache[key] = parse_mef(p)
                self.done.emit(self._cache[key])
        except Exception:
            import traceback; traceback.print_exc()

# ── Thumbnail delegate ────────────────────────────────────────────────────────
class _ThumbDelegate(QStyledItemDelegate):
    TW, TH, H, PAD = 96, 72, 90, 6

    def sizeHint(self, opt, idx): return QSize(opt.rect.width(), self.H)

    def paint(self, p: QPainter, opt, idx):
        d = idx.data(Qt.UserRole)
        if not d: return
        r = opt.rect
        sel = bool(opt.state & QStyle.State_Selected)

        p.fillRect(r, QColor("#162840") if sel else
                      QColor("#111128") if bool(opt.state & QStyle.State_MouseOver)
                      else QColor("#080814"))

        tx, ty = r.x() + self.PAD, r.y() + (r.height() - self.TH) // 2
        pm: Optional[QPixmap] = d.get("pixmap")
        if pm:
            p.drawPixmap(tx, ty, self.TW, self.TH, pm)
        else:
            p.fillRect(tx, ty, self.TW, self.TH, QColor("#0e0e24"))
            p.setPen(QColor("#252548"))
            p.drawRect(tx, ty, self.TW - 1, self.TH - 1)
            p.setPen(QColor("#303060"))
            p.drawText(tx, ty, self.TW, self.TH, Qt.AlignCenter, "…")

        p.setPen(QColor("#60c8ff" if sel else "#1e1e40"))
        p.drawRect(tx, ty, self.TW - 1, self.TH - 1)

        bx   = tx + self.TW + self.PAD + 2
        avail = r.right() - bx - self.PAD
        p.setPen(QColor("#dde6f0" if sel else "#c0ccd8"))
        font = QFont("Segoe UI", 8, QFont.Bold)
        p.setFont(font)
        name = QFontMetrics(font).elidedText(d.get("name", ""), Qt.ElideMiddle, avail)
        p.drawText(bx, r.y() + 18, name)
        p.setPen(QColor("#607080"))
        p.setFont(QFont("Segoe UI", 7))
        p.drawText(bx, r.y() + 34, d.get("info", ""))

# ── Overlay flags ─────────────────────────────────────────────────────────────
class OverlayFlags:
    def __init__(self):
        self.bones      = False
        self.magic_verts= False
        self.collision  = False
        self.portals    = False
        self.glow       = False

# ── Cached overlays for one model ─────────────────────────────────────────────
class ModelOverlays:
    def __init__(self):
        self.bone_lines  = None   # (vao, n)
        self.bone_joints = None   # (vao, n)
        self.magic_pts   = None   # (vao, n)
        self.coll_wire   = None   # list of (vao, n)
        self.portal_wire = None   # (vao, n)
        self.glow_pts    = None   # (vao, n)

# ── Viewport ──────────────────────────────────────────────────────────────────
class MEFViewport(QOpenGLWidget):
    thumbnail_ready = pyqtSignal(str, QPixmap)
    fps_changed     = pyqtSignal(float)

    THUMB_SZ = 128

    def __init__(self, parent=None):
        fmt = QSurfaceFormat()
        fmt.setVersion(3, 3)
        fmt.setProfile(QSurfaceFormat.CoreProfile)
        fmt.setDepthBufferSize(24)
        fmt.setSamples(4)
        super().__init__(parent)
        self.setFormat(fmt)

        self.cam             = OrbitCamera()
        self.current_model: Optional[MefModel] = None
        self.overlays        = OverlayFlags()

        self._ctx        = None
        self._prog       = None
        self._flat_prog  = None
        self._grid_vao   = None
        self._grid_n     = 0
        self._gpu_cache: Optional[GpuCache] = None
        self._thumb_fbo  = None

        # Per-model overlay cache
        self._overlay_cache: Dict[str, ModelOverlays] = {}

        self._thumb_queue: List[MefModel] = []
        self._thumb_cache: Dict[str, QPixmap] = {}

        self._drag_btn  = None
        self._drag_last = QPoint()
        self.wireframe  = False
        self.show_grid  = True

        self._frame_timer = QElapsedTimer()
        self._frame_timer.start()

    def initializeGL(self):
        self._ctx = moderngl.create_context()
        self._ctx.enable(moderngl.DEPTH_TEST)
        self._ctx.enable(moderngl.CULL_FACE)

        self._prog      = self._ctx.program(vertex_shader=VERT,      fragment_shader=FRAG)
        self._flat_prog = self._ctx.program(vertex_shader=FLAT_VERT, fragment_shader=FLAT_FRAG)
        self._grid_vao, self._grid_n = build_grid(self._ctx, self._flat_prog)
        self._gpu_cache = GpuCache(self._ctx, self._prog)

        sz  = self.THUMB_SZ
        tex = self._ctx.texture((sz, sz), 4)
        dep = self._ctx.depth_texture((sz, sz))
        self._thumb_fbo = self._ctx.framebuffer(
            color_attachments=[tex], depth_attachment=dep)

    def resizeGL(self, w, h):
        if self._ctx:
            self._ctx.viewport = (0, 0, w, h)

    def paintGL(self):
        w, h = self.width(), self.height()
        qt_fbo = self._ctx.detect_framebuffer(self.defaultFramebufferObject())
        qt_fbo.use()
        self._ctx.viewport = (0, 0, w, h)
        self._ctx.clear(0.04, 0.04, 0.10, 1.0)

        if self.current_model:
            mvp, eye = self.cam.matrices(w, h)
            self._prog["u_mvp"].write(gl_bytes(mvp))
            self._prog["u_model"].write(gl_bytes(np.eye(4, dtype="f4")))
            self._prog["u_cam"].value = tuple(map(float, eye))

            self._ctx.wireframe = self.wireframe
            gpu = self._gpu_cache.get(self.current_model)
            if gpu: gpu.draw(self._prog)
            self._ctx.wireframe = False

            if self.show_grid:
                self._flat_prog["u_mvp"].write(gl_bytes(mvp))
                self._flat_prog["u_color"].value = (0.15, 0.15, 0.28, 1.0)
                self._grid_vao.render(moderngl.LINES)

            self._draw_overlays(mvp)

        elapsed = self._frame_timer.restart()
        if elapsed > 0:
            self.fps_changed.emit(1000.0 / elapsed)

        self._process_one_thumb(qt_fbo)

    def _draw_overlays(self, mvp: np.ndarray):
        if not self.current_model:
            return
        key = str(self.current_model.path)
        if key not in self._overlay_cache:
            self._build_overlays(self.current_model)
        ov = self._overlay_cache.get(key)
        if not ov:
            return

        fp = self._flat_prog
        fp["u_mvp"].write(gl_bytes(mvp))

        # ── Bones ──────────────────────────────────────────────────────────────
        if self.overlays.bones:
            self._ctx.disable(moderngl.DEPTH_TEST)
            if ov.bone_lines:
                fp["u_color"].value = (1.0, 0.85, 0.2, 1.0)   # gold
                ov.bone_lines[0].render(moderngl.LINES, vertices=ov.bone_lines[1])
            if ov.bone_joints:
                fp["u_color"].value = (1.0, 0.4, 0.1, 1.0)    # orange
                self._ctx.point_size = 8.0
                ov.bone_joints[0].render(moderngl.POINTS, vertices=ov.bone_joints[1])
            self._ctx.enable(moderngl.DEPTH_TEST)

        # ── Magic Vertices ────────────────────────────────────────────────────
        if self.overlays.magic_verts and ov.magic_pts:
            fp["u_color"].value = (0.0, 1.0, 0.9, 1.0)        # cyan
            self._ctx.point_size = 7.0
            ov.magic_pts[0].render(moderngl.POINTS, vertices=ov.magic_pts[1])

        # ── Collision Mesh ───────────────────────────────────────────────────
        if self.overlays.collision and ov.coll_wire:
            fp["u_color"].value = (1.0, 0.2, 0.2, 0.8)        # red
            for vao, n in ov.coll_wire:
                vao.render(moderngl.LINES, vertices=n)

        # ── Portals ──────────────────────────────────────────────────────────
        if self.overlays.portals and ov.portal_wire:
            fp["u_color"].value = (1.0, 1.0, 0.2, 0.9)        # yellow
            ov.portal_wire[0].render(moderngl.LINES, vertices=ov.portal_wire[1])

        # ── Glow Sprites ─────────────────────────────────────────────────────
        if self.overlays.glow and ov.glow_pts:
            fp["u_color"].value = (1.0, 0.7, 0.1, 1.0)        # warm orange
            self._ctx.point_size = 10.0
            ov.glow_pts[0].render(moderngl.POINTS, vertices=ov.glow_pts[1])

    def _build_overlays(self, model: MefModel):
        """Build GL overlay VAOs for a model and cache them."""
        key = str(model.path)
        ov  = ModelOverlays()

        # Bones
        bone_result = build_bone_overlay(self._ctx, self._flat_prog, model)
        if bone_result:
            lines_vao, lines_n, joints_vao, joints_n = bone_result
            ov.bone_lines  = (lines_vao,  lines_n)
            ov.bone_joints = (joints_vao, joints_n)

        # Magic Vertices
        if model.magic_vertices:
            positions = [mv.position for mv in model.magic_vertices]
            ov.magic_pts = build_points_overlay(self._ctx, self._flat_prog, positions)

        # Collision Mesh
        ov.coll_wire = []
        for cm in model.collision:
            result = build_lines_overlay(self._ctx, self._flat_prog, cm.vertices, cm.faces)
            if result:
                ov.coll_wire.append(result)

        # Portals
        if model.portals:
            portal = model.portals[0]
            result = build_lines_overlay(self._ctx, self._flat_prog,
                                          portal.vertices, portal.faces)
            if result:
                ov.portal_wire = result

        # Glow Sprites
        if model.glow_sprites:
            positions = [gs.position for gs in model.glow_sprites]
            ov.glow_pts = build_points_overlay(self._ctx, self._flat_prog, positions)

        self._overlay_cache[key] = ov

    # ── Thumbnail ─────────────────────────────────────────────────────────────
    def queue_thumbnail(self, model: MefModel):
        key = str(model.path)
        if key not in self._thumb_cache and model not in self._thumb_queue:
            self._thumb_queue.append(model)
        self.update()

    def _process_one_thumb(self, screen_fbo):
        if not self._thumb_queue or not self._thumb_fbo:
            return
        model = self._thumb_queue.pop(0)
        key   = str(model.path)
        if key in self._thumb_cache:
            return

        sz = self.THUMB_SZ
        self._thumb_fbo.use()
        self._ctx.viewport = (0, 0, sz, sz)
        self._ctx.clear(0.05, 0.06, 0.13, 1.0)

        cam = OrbitCamera()
        cam.fit(model.center, model.radius)
        mvp, eye = cam.matrices(sz, sz)
        self._prog["u_mvp"].write(gl_bytes(mvp))
        self._prog["u_model"].write(gl_bytes(np.eye(4, dtype="f4")))
        self._prog["u_cam"].value = tuple(map(float, eye))

        gpu = self._gpu_cache.get(model)
        if gpu:
            self._ctx.wireframe = False
            gpu.draw(self._prog)

        raw = self._thumb_fbo.read(components=4, dtype="f1", clamp=True)
        img = QImage(bytes(raw), sz, sz, sz * 4, QImage.Format_RGBA8888).mirrored(False, True)
        pm  = QPixmap.fromImage(img).scaled(96, 72, Qt.KeepAspectRatio, Qt.SmoothTransformation)

        self._thumb_cache[key] = pm
        self.thumbnail_ready.emit(key, pm)

        screen_fbo.use()
        self._ctx.viewport = (0, 0, self.width(), self.height())
        if self._thumb_queue:
            self.update()

    # ── Mouse / wheel ─────────────────────────────────────────────────────────
    def mousePressEvent(self, e):
        self._drag_btn  = e.button()
        self._drag_last = e.pos()

    def mouseMoveEvent(self, e):
        d = e.pos() - self._drag_last
        self._drag_last = e.pos()
        if self._drag_btn == Qt.LeftButton:
            self.cam.orbit(d.x(), d.y())
        elif self._drag_btn == Qt.RightButton:
            self.cam.pan(d.x(), d.y())
        self.update()

    def mouseReleaseEvent(self, e): self._drag_btn = None

    def mouseDoubleClickEvent(self, e):
        if self.current_model:
            self.cam.fit(self.current_model.center, self.current_model.radius)
            self.update()

    def wheelEvent(self, e):
        self.cam.zoom(1 if e.angleDelta().y() > 0 else -1)
        self.update()

    def load_model(self, model: MefModel):
        self.current_model = model
        self.cam.fit(model.center, model.radius)
        self.update()

# ── File list panel ───────────────────────────────────────────────────────────
class FileListPanel(QWidget):
    model_selected = pyqtSignal(object)

    def __init__(self, viewport: MEFViewport, parent=None):
        super().__init__(parent)
        self._vp      = viewport
        self._cache:  Dict[str, MefModel] = {}
        self._worker: Optional[_ParseWorker] = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        hdr = QWidget(); hdr.setFixedHeight(38)
        hdr.setStyleSheet("background:#060612;border-bottom:1px solid #1c1c38;")
        hl  = QHBoxLayout(hdr); hl.setContentsMargins(10, 0, 10, 0)
        lbl = QLabel("MODELS"); lbl.setObjectName("sec")
        self._count = QLabel("0")
        self._count.setStyleSheet("color:#405060;font-size:11px;")
        hl.addWidget(lbl); hl.addStretch(); hl.addWidget(self._count)
        layout.addWidget(hdr)

        self._search = QLineEdit()
        self._search.setPlaceholderText("🔍  Filter models…")
        self._search.setStyleSheet("margin:6px 6px 4px 6px;")
        self._search.textChanged.connect(self._filter)
        layout.addWidget(self._search)

        self._list = QListWidget()
        self._list.setItemDelegate(_ThumbDelegate())
        self._list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._list.setSpacing(0)
        self._list.itemClicked.connect(self._on_click)
        layout.addWidget(self._list)

        viewport.thumbnail_ready.connect(self._on_thumb)

    def load_folder(self, folder: Path):
        if self._worker:
            self._worker.cancel(); self._worker.wait()
        self._list.clear()
        mef_files = sorted(folder.glob("*.mef"), key=lambda p: p.name.lower())
        valid = [p for p in mef_files if quick_validate(p)]
        self._count.setText(str(len(valid)))

        for path in valid:
            key   = str(path)
            pm    = self._vp._thumb_cache.get(key)
            model = self._cache.get(key)
            info  = "parsing…"
            if model:
                info = (f"Vtx {model.total_vertices:,}  Tri {model.total_triangles:,}  Parts {model.total_parts}"
                        if model.valid else f"⚠ {model.error}")
            d = {"path": key, "name": path.name, "pixmap": pm, "info": info, "model": model}
            item = QListWidgetItem()
            item.setData(Qt.UserRole, d)
            item.setSizeHint(QSize(0, _ThumbDelegate.H))
            self._list.addItem(item)

        self._worker = _ParseWorker(valid, self._cache)
        self._worker.done.connect(self._on_parsed)
        self._worker.start()

    def _on_parsed(self, model: MefModel):
        for i in range(self._list.count()):
            item = self._list.item(i)
            d = item.data(Qt.UserRole)
            if d["name"] == model.name:
                d["model"] = model
                d["info"]  = (f"Vtx {model.total_vertices:,}  Tri {model.total_triangles:,}  Parts {model.total_parts}"
                              if model.valid else f"⚠ {model.error}")
                if model.valid:
                    self._vp.queue_thumbnail(model)
                item.setData(Qt.UserRole, d)
                self._list.viewport().update()
                break

    def _on_thumb(self, path_str: str, pm: QPixmap):
        filename = Path(path_str).name
        for i in range(self._list.count()):
            item = self._list.item(i)
            d = item.data(Qt.UserRole)
            if d["name"] == filename:
                d["pixmap"] = pm
                item.setData(Qt.UserRole, d)
                self._list.viewport().update()
                break

    def _filter(self, text: str):
        lo = text.lower()
        for i in range(self._list.count()):
            item = self._list.item(i)
            d    = item.data(Qt.UserRole)
            item.setHidden(bool(lo) and lo not in d["name"].lower())

    def _on_click(self, item: QListWidgetItem):
        d = item.data(Qt.UserRole)
        if d and d.get("model") and d["model"].valid:
            self.model_selected.emit(d["model"])

# ── Info panel ────────────────────────────────────────────────────────────────
class InfoPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        hdr = QWidget(); hdr.setFixedHeight(38)
        hdr.setStyleSheet("background:#060612;border-bottom:1px solid #1c1c38;")
        hl  = QHBoxLayout(hdr); hl.setContentsMargins(10, 0, 10, 0)
        lbl = QLabel("MODEL INFO"); lbl.setObjectName("sec")
        hl.addWidget(lbl)
        layout.addWidget(hdr)

        tabs = QTabWidget()
        layout.addWidget(tabs)

        # ── Overview ──────────────────────────────────────────────────────────
        ow = QWidget(); tabs.addTab(ow, "Overview")
        fl = QFormLayout(ow)
        fl.setContentsMargins(12, 14, 12, 14); fl.setSpacing(8)
        fl.setLabelAlignment(Qt.AlignRight)

        self._ov: Dict[str, QLabel] = {}
        for label, key in [
            ("Name",          "name"), ("File Size",    "size"),
            ("Model Type",    "type"), ("HSEM Version", "hsem"),
            ("Parts",         "parts"),("Vertices",     "verts"),
            ("Triangles",     "tris"), ("Bones",        "bones"),
            ("Magic Verts",   "mverts"),("Portals",     "portals"),
            ("Collision",     "coll"), ("Width",        "dx"),
            ("Height",        "dy"),   ("Depth",        "dz"),
        ]:
            v = QLabel("—"); v.setObjectName("val"); v.setWordWrap(True)
            self._ov[key] = v
            fl.addRow(QLabel(label + ":"), v)

        # ── Parts ────────────────────────────────────────────────────────────
        self._parts_tbl = QTableWidget(0, 4)
        self._parts_tbl.setHorizontalHeaderLabels(["Part", "Vertices", "Triangles", "Origin (X Y Z)"])
        self._parts_tbl.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
        self._parts_tbl.setEditTriggers(QTableWidget.NoEditTriggers)
        self._parts_tbl.setSelectionBehavior(QTableWidget.SelectRows)
        tabs.addTab(self._parts_tbl, "Parts")

        # ── Bones ────────────────────────────────────────────────────────────
        self._bone_tree = QTreeWidget()
        self._bone_tree.setHeaderLabels(["Bone", "ID", "Parent", "World Pos"])
        self._bone_tree.setColumnWidth(0, 120)
        self._bone_tree.setColumnWidth(1, 30)
        self._bone_tree.setColumnWidth(2, 30)
        tabs.addTab(self._bone_tree, "Bones 🦴")

        # ── Chunks ───────────────────────────────────────────────────────────
        self._chunk_tbl = QTableWidget(0, 3)
        self._chunk_tbl.setHorizontalHeaderLabels(["Tag", "Offset", "Size"])
        self._chunk_tbl.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self._chunk_tbl.setEditTriggers(QTableWidget.NoEditTriggers)
        tabs.addTab(self._chunk_tbl, "Chunks")

        # ── Documentation ────────────────────────────────────────────────────
        self._doc_text = QTextEdit()
        self._doc_text.setReadOnly(True)
        self._load_doc()
        tabs.addTab(self._doc_text, "MEF Guide 📖")

    def _load_doc(self):
        # Use our resource_path helper for bundle compatibility
        doc_path = resource_path("guidemef.md")
        if doc_path.exists():
            try:
                text = doc_path.read_text(encoding="utf-8", errors="replace")
                self._doc_text.setPlainText(text)
            except Exception as e:
                self._doc_text.setPlainText(f"[Error loading guide: {e}]")
        else:
            self._doc_text.setPlainText(
                "MEF Guide not found.\n"
                f"Expected: {doc_path}\n\n"
                "Please ensure guidemef.md is in the application folder."
            )

    def show_model(self, m: MefModel):
        ov = self._ov
        e  = m.extents
        ov["name"].setText(m.name)
        ov["size"].setText(m.file_size_human)
        ov["type"].setText(f"{m.model_type_name}  (Type {m.model_type})")
        ov["hsem"].setText(f"v{m.hsem_version:.3f}")
        ov["parts"].setText(str(m.total_parts))
        ov["verts"].setText(f"{m.total_vertices:,}")
        ov["tris"].setText(f"{m.total_triangles:,}")
        ov["bones"].setText(str(len(m.bones)))
        ov["mverts"].setText(str(len(m.magic_vertices)))
        ov["portals"].setText(str(len(m.portals)))
        ov["coll"].setText(str(len(m.collision)))
        ov["dx"].setText(f"{e[0]:.4f} u")
        ov["dy"].setText(f"{e[1]:.4f} u")
        ov["dz"].setText(f"{e[2]:.4f} u")

        # Parts table
        t = self._parts_tbl
        t.setRowCount(len(m.parts))
        for row, part in enumerate(m.parts):
            t.setItem(row, 0, QTableWidgetItem(str(part.index)))
            t.setItem(row, 1, QTableWidgetItem(f"{part.vertex_count:,}"))
            t.setItem(row, 2, QTableWidgetItem(f"{part.triangle_count:,}"))
            px, py, pz = part.position
            t.setItem(row, 3, QTableWidgetItem(f"({px:.3f}, {py:.3f}, {pz:.3f})"))

        # Bone tree
        self._bone_tree.clear()
        bone_map = {b.bone_id: b for b in m.bones}
        items: Dict[int, QTreeWidgetItem] = {}

        def make_item(bone):
            wx, wy, wz = bone.world_offset
            ti = QTreeWidgetItem([
                bone.name,
                str(bone.bone_id),
                str(bone.parent_id) if bone.parent_id >= 0 else "root",
                f"({wx:.3f}, {wy:.3f}, {wz:.3f})"
            ])
            ti.setForeground(0, QColor("#60c8ff"))
            return ti

        # Build tree hierarchy
        roots = []
        for bone in sorted(m.bones, key=lambda b: b.bone_id):
            ti = make_item(bone)
            items[bone.bone_id] = ti
            if bone.parent_id < 0 or bone.parent_id not in bone_map:
                roots.append(ti)

        for bone in sorted(m.bones, key=lambda b: b.bone_id):
            if bone.parent_id >= 0 and bone.parent_id in bone_map:
                parent_item = items.get(bone.parent_id)
                child_item  = items.get(bone.bone_id)
                if parent_item and child_item:
                    parent_item.addChild(child_item)

        for root in roots:
            self._bone_tree.addTopLevelItem(root)
        self._bone_tree.expandAll()

        # Chunks table
        c = self._chunk_tbl
        c.setRowCount(len(m.chunks))
        for row, chunk in enumerate(m.chunks):
            c.setItem(row, 0, QTableWidgetItem(chunk.tag))
            c.setItem(row, 1, QTableWidgetItem(f"0x{chunk.offset:08X}"))
            c.setItem(row, 2, QTableWidgetItem(f"{chunk.size:,} B"))

# ── Main Window ───────────────────────────────────────────────────────────────
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("IGI 2 MEF Viewer  —  Specification-Driven Edition")
        self.resize(1500, 880)
        self.setMinimumSize(960, 600)

        # ── Toolbar ──────────────────────────────────────────────────────────
        tb = self.addToolBar("Main")
        tb.setMovable(False); tb.setFloatable(False)

        self._act_folder = tb.addAction("📂  Open Folder", self._open_folder)
        self._act_reload = tb.addAction("↺  Reload",       self._reload)
        tb.addSeparator()

        self._act_wire = QAction("⬡  Wireframe", self, checkable=True)
        self._act_wire.toggled.connect(self._toggle_wireframe)
        tb.addAction(self._act_wire)

        self._act_grid = QAction("⊞  Grid", self, checkable=True, checked=True)
        self._act_grid.toggled.connect(self._toggle_grid)
        tb.addAction(self._act_grid)

        tb.addSeparator()

        # Overlay toggles
        self._act_bones = QAction("🦴 Bones", self, checkable=True)
        self._act_bones.setToolTip("Toggle skeleton bone overlay")
        self._act_bones.toggled.connect(self._toggle_bones)
        tb.addAction(self._act_bones)

        self._act_mverts = QAction("✦ Magic Verts", self, checkable=True)
        self._act_mverts.setToolTip("Toggle magic vertex overlay (XTVM)")
        self._act_mverts.toggled.connect(self._toggle_mverts)
        tb.addAction(self._act_mverts)

        self._act_collision = QAction("🔺 Collision", self, checkable=True)
        self._act_collision.setToolTip("Toggle collision mesh wireframe (HSMC)")
        self._act_collision.toggled.connect(self._toggle_collision)
        tb.addAction(self._act_collision)

        self._act_portals = QAction("🚪 Portals", self, checkable=True)
        self._act_portals.setToolTip("Toggle portal wireframe (TROP)")
        self._act_portals.toggled.connect(self._toggle_portals)
        tb.addAction(self._act_portals)

        self._act_glow = QAction("✨ Glow", self, checkable=True)
        self._act_glow.setToolTip("Toggle glow sprite positions (WOLG)")
        self._act_glow.toggled.connect(self._toggle_glow)
        tb.addAction(self._act_glow)

        tb.addSeparator()
        self._act_fit   = tb.addAction("⊙  Fit View",   self._fit_view)
        self._act_reset = tb.addAction("⌂  Reset Cam",  self._reset_cam)

        for act in tb.actions():
            w = tb.widgetForAction(act)
            if w: w.setCursor(Qt.PointingHandCursor)

        # ── Central splitter ──────────────────────────────────────────────────
        splitter = QSplitter(Qt.Horizontal)
        self.setCentralWidget(splitter)

        self._vp = MEFViewport()
        self._vp.fps_changed.connect(self._on_fps)

        self._file_panel = FileListPanel(self._vp)
        self._file_panel.setFixedWidth(270)
        self._file_panel.model_selected.connect(self._on_model_selected)

        self._info_panel = InfoPanel()
        self._info_panel.setFixedWidth(320)

        splitter.addWidget(self._file_panel)
        splitter.addWidget(self._vp)
        splitter.addWidget(self._info_panel)
        splitter.setStretchFactor(1, 1)

        # ── Status bar ────────────────────────────────────────────────────────
        self._sb_folder = QLabel("No folder loaded")
        self._sb_model  = QLabel("")
        self._sb_fps    = QLabel("0 fps")
        for w in [self._sb_folder, self._sb_model, self._sb_fps]:
            self.statusBar().addPermanentWidget(w)
        self.statusBar().addPermanentWidget(QLabel(), 1)

        self._current_folder: Optional[Path] = None

        # Keyboard shortcuts
        QShortcut(QKeySequence("W"), self, self._act_wire.toggle)
        QShortcut(QKeySequence("G"), self, self._act_grid.toggle)
        QShortcut(QKeySequence("B"), self, self._act_bones.toggle)
        QShortcut(QKeySequence("M"), self, self._act_mverts.toggle)
        QShortcut(QKeySequence("C"), self, self._act_collision.toggle)
        QShortcut(QKeySequence("P"), self, self._act_portals.toggle)
        QShortcut(QKeySequence("F"), self, self._fit_view)
        QShortcut(QKeySequence("R"), self, self._reset_cam)
        QShortcut(QKeySequence("Ctrl+O"), self, self._open_folder)

    # ── Toolbar actions ───────────────────────────────────────────────────────
    def _open_folder(self):
        d = QFileDialog.getExistingDirectory(
            self, "Open MEF Folder",
            str(self._current_folder or Path.home()))
        if d: self._load_folder(Path(d))

    def _load_folder(self, folder: Path):
        self._current_folder = folder
        self._sb_folder.setText(f"  {folder}  ")
        self._file_panel.load_folder(folder)

    def _reload(self):
        if self._current_folder:
            if self._vp._gpu_cache:
                self._vp._gpu_cache._cache.clear()
            self._vp._overlay_cache.clear()
            self._load_folder(self._current_folder)

    def _toggle_wireframe(self, on: bool):
        self._vp.wireframe = on; self._vp.update()

    def _toggle_grid(self, on: bool):
        self._vp.show_grid = on; self._vp.update()

    def _toggle_bones(self, on: bool):
        self._vp.overlays.bones = on; self._vp.update()

    def _toggle_mverts(self, on: bool):
        self._vp.overlays.magic_verts = on; self._vp.update()

    def _toggle_collision(self, on: bool):
        self._vp.overlays.collision = on; self._vp.update()

    def _toggle_portals(self, on: bool):
        self._vp.overlays.portals = on; self._vp.update()

    def _toggle_glow(self, on: bool):
        self._vp.overlays.glow = on; self._vp.update()

    def _fit_view(self):
        if self._vp.current_model:
            m = self._vp.current_model
            self._vp.cam.fit(m.center, m.radius)
            self._vp.update()

    def _reset_cam(self):
        self._vp.cam = OrbitCamera()
        if self._vp.current_model:
            m = self._vp.current_model
            self._vp.cam.fit(m.center, m.radius)
        self._vp.update()

    def _on_model_selected(self, model: MefModel):
        self._vp.load_model(model)
        self._info_panel.show_model(model)
        prefix = ""
        if model.bones:        prefix += f"🦴{len(model.bones)} "
        if model.magic_vertices: prefix += f"✦{len(model.magic_vertices)} "
        if model.collision:    prefix += f"🔺{len(model.collision)} "
        self._sb_model.setText(
            f"  {model.name}  ·  {model.total_vertices:,} verts  ·  "
            f"{model.total_triangles:,} tris  {prefix} ")

    def _on_fps(self, fps: float):
        self._sb_fps.setText(f"  {fps:.0f} fps  ")

# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    fmt = QSurfaceFormat()
    fmt.setVersion(3, 3)
    fmt.setProfile(QSurfaceFormat.CoreProfile)
    fmt.setDepthBufferSize(24)
    fmt.setSamples(4)
    QSurfaceFormat.setDefaultFormat(fmt)

    app = QApplication(sys.argv)
    app.setApplicationName("IGI 2 MEF Viewer")
    app.setOrganizationName("Antigravity Toolchain")
    app.setStyleSheet(STYLE)

    win = MainWindow()
    win.show()

    if len(sys.argv) > 1:
        p = Path(sys.argv[1])
        if p.is_dir():
            QTimer.singleShot(100, lambda: win._load_folder(p))
        elif p.is_file() and p.suffix.lower() == ".mef":
            QTimer.singleShot(100, lambda: win._load_folder(p.parent))

    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
