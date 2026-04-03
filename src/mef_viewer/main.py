"""
mef_viewer.py — IGI 2 MEF Viewer  (Premium Edition)
Entry point. 

Imports and logic synchronized with the "perfectly working" static version,
now backed by the official igi2mef library.
"""

from __future__ import annotations
import sys, os, traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ── PyInstaller Resource Path ────────────────────────────────────────────────
def resource_path(relative_path: str) -> str:
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# ── Dependency checks ─────────────────────────────────────────────────────────
_here = Path(__file__).parent
if str(_here.parent) not in sys.path:
    sys.path.insert(0, str(_here.parent))

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

# ── Official Library ─────────────────────────────────────────────────────────
try:
    from igi2mef import parse_mef, quick_validate, MefModel, MefDebugParams
except ImportError:
    print(f"ERROR: igi2mef library not found:\n{traceback.format_exc()}")
    sys.exit(1)

# ── GL core ───────────────────────────────────────────────────────────────────
# We use the synchronized gl_backend.py in the same folder
try:
    from .gl_backend import (
        OrbitCamera, GpuCache, build_grid, build_bone_overlay,
        build_points_overlay, build_lines_overlay,
        gl_bytes, VERT, FRAG, FLAT_VERT, FLAT_FRAG,
    )
except ImportError:
    from gl_backend import (
        OrbitCamera, GpuCache, build_grid, build_bone_overlay,
        build_points_overlay, build_lines_overlay,
        gl_bytes, VERT, FRAG, FLAT_VERT, FLAT_FRAG,
    )

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
    def __init__(self, paths: List[Path], cache: Dict, debug_params: MefDebugParams):
        super().__init__()
        self._paths = paths
        self._cache = cache
        self._debug = debug_params
        self._cancelled = False
    def cancel(self): self._cancelled = True
    def run(self):
        try:
            for p in self._paths:
                if self._cancelled: return
                key = str(p)
                if key not in self._cache:
                    self._cache[key] = parse_mef(p, debug=self._debug)
                self.done.emit(self._cache[key])
        except Exception:
            traceback.print_exc()

# ── Thumbnail delegate ────────────────────────────────────────────────────────
class _ThumbDelegate(QStyledItemDelegate):
    TW, TH, H, PAD = 96, 72, 90, 6
    def sizeHint(self, opt, idx): return QSize(opt.rect.width(), self.H)
    def paint(self, p: QPainter, opt, idx):
        d = idx.data(Qt.UserRole)
        if not d: return
        r = opt.rect
        sel = bool(opt.state & QStyle.State_Selected)
        p.fillRect(r, QColor("#162840") if sel else QColor("#111128") if bool(opt.state & QStyle.State_MouseOver) else QColor("#080814"))
        tx, ty = r.x() + self.PAD, r.y() + (r.height() - self.TH) // 2
        pm: Optional[QPixmap] = d.get("pixmap")
        if pm: p.drawPixmap(tx, ty, self.TW, self.TH, pm)
        else:
            p.fillRect(tx, ty, self.TW, self.TH, QColor("#0e0e24"))
            p.setPen(QColor("#252548")); p.drawRect(tx, ty, self.TW - 1, self.TH - 1)
            p.setPen(QColor("#303060")); p.drawText(tx, ty, self.TW, self.TH, Qt.AlignCenter, "…")
        p.setPen(QColor("#60c8ff" if sel else "#1e1e40")); p.drawRect(tx, ty, self.TW - 1, self.TH - 1)
        bx = tx + self.TW + self.PAD + 2; avail = r.right() - bx - self.PAD
        p.setPen(QColor("#dde6f0" if sel else "#c0ccd8"))
        font = QFont("Segoe UI", 8, QFont.Bold); p.setFont(font)
        name = QFontMetrics(font).elidedText(d.get("name", ""), Qt.ElideMiddle, avail)
        p.drawText(bx, r.y() + 18, name)
        p.setPen(QColor("#607080")); p.setFont(QFont("Segoe UI", 7))
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
        self.cam = OrbitCamera(); self.current_model: Optional[MefModel] = None; self.overlays = OverlayFlags()
        self._ctx = None; self._prog = None; self._flat_prog = None; self._grid_vao = None; self._grid_n = 0; self._gpu_cache = None; self._thumb_fbo = None
        self._overlay_cache: Dict[str, ModelOverlays] = {}
        self._thumb_queue: List[MefModel] = []; self._thumb_cache: Dict[str, QPixmap] = {}
        self._drag_btn = None; self._drag_last = QPoint(); self.wireframe = False; self.show_grid = True
        self._frame_timer = QElapsedTimer(); self._frame_timer.start()
        self.debug_params = MefDebugParams()
        self.scales = [0.01, 0.001, 1/40.96, 1/4096, 1.0]
        self.swizzles = ["XZY", "XYZ", "YXZ", "YZX", "ZXY", "ZYX"]
        self.bone_modes = ["REL", "ABS", "ROOT"]
        
        # Lock to USER confirmed perfect configuration:
        self.idx_vscale = 0; self.idx_bscale = 0; self.idx_swizzle = 0; self.idx_bmode = 0
        self.debug_params.v_scale = self.scales[self.idx_vscale]
        self.debug_params.b_scale = self.scales[self.idx_bscale]
        self.debug_params.swizzle_mode = self.swizzles[self.idx_swizzle]
        self.debug_params.bone_mode = self.bone_modes[self.idx_bmode]
        
        self.setFocusPolicy(Qt.StrongFocus)

    def initializeGL(self):
        self._ctx = moderngl.create_context(); self._ctx.enable(moderngl.DEPTH_TEST); self._ctx.enable(moderngl.CULL_FACE)
        self._prog = self._ctx.program(vertex_shader=VERT, fragment_shader=FRAG)
        self._flat_prog = self._ctx.program(vertex_shader=FLAT_VERT, fragment_shader=FLAT_FRAG)
        self._grid_vao, self._grid_n = build_grid(self._ctx, self._flat_prog)
        self._gpu_cache = GpuCache(self._ctx, self._prog)
        sz = self.THUMB_SZ
        tex = self._ctx.texture((sz, sz), 4); dep = self._ctx.depth_texture((sz, sz))
        self._thumb_fbo = self._ctx.framebuffer(color_attachments=[tex], depth_attachment=dep)

    def resizeGL(self, w, h):
        if self._ctx: self._ctx.viewport = (0, 0, w, h)

    def paintGL(self):
        w, h = self.width(), self.height()
        qt_fbo = self._ctx.detect_framebuffer(self.defaultFramebufferObject())
        qt_fbo.use(); self._ctx.viewport = (0, 0, w, h); self._ctx.clear(0.04, 0.04, 0.10, 1.0)
        if self.current_model:
            mvp, eye = self.cam.matrices(w, h)
            self._prog["u_mvp"].write(gl_bytes(mvp)); self._prog["u_model"].write(gl_bytes(np.eye(4, dtype="f4")))
            self._prog["u_cam"].value = tuple(map(float, eye))
            self._ctx.wireframe = self.wireframe
            gpu = self._gpu_cache.get(self.current_model)
            if gpu: gpu.draw(self._prog)
            self._ctx.wireframe = False
            if self.show_grid:
                self._flat_prog["u_mvp"].write(gl_bytes(mvp)); self._flat_prog["u_color"].value = (0.15, 0.15, 0.28, 1.0)
                self._grid_vao.render(moderngl.LINES)
            self._draw_overlays(mvp)
        self._draw_debug_hud()
        elapsed = self._frame_timer.restart()
        if elapsed > 0: self.fps_changed.emit(1000.0 / elapsed)
        self._process_one_thumb(qt_fbo)

    def _draw_overlays(self, mvp: np.ndarray):
        if not self.current_model: return
        key = str(self.current_model.path)
        if key not in self._overlay_cache: self._build_overlays(self.current_model)
        ov = self._overlay_cache.get(key)
        if not ov: return
        fp = self._flat_prog; fp["u_mvp"].write(gl_bytes(mvp))
        if self.overlays.bones:
            self._ctx.disable(moderngl.DEPTH_TEST)
            if ov.bone_lines:
                fp["u_color"].value = (1.0, 0.85, 0.2, 1.0); ov.bone_lines[0].render(moderngl.LINES, vertices=ov.bone_lines[1])
            if ov.bone_joints:
                fp["u_color"].value = (1.0, 0.4, 0.1, 1.0); self._ctx.point_size = 8.0; ov.bone_joints[0].render(moderngl.POINTS, vertices=ov.bone_joints[1])
            self._ctx.enable(moderngl.DEPTH_TEST)
        if self.overlays.magic_verts and ov.magic_pts:
            fp["u_color"].value = (0.0, 1.0, 0.9, 1.0); self._ctx.point_size = 7.0; ov.magic_pts[0].render(moderngl.POINTS, vertices=ov.magic_pts[1])
        if self.overlays.collision and ov.coll_wire:
            fp["u_color"].value = (1.0, 0.2, 0.2, 0.8)
            for vao, n in ov.coll_wire: vao.render(moderngl.LINES, vertices=n)
        if self.overlays.portals and ov.portal_wire:
            fp["u_color"].value = (1.0, 1.0, 0.2, 0.9); ov.portal_wire[0].render(moderngl.LINES, vertices=ov.portal_wire[1])
        if self.overlays.glow and ov.glow_pts:
            fp["u_color"].value = (1.0, 0.7, 0.1, 1.0); self._ctx.point_size = 10.0; ov.glow_pts[0].render(moderngl.POINTS, vertices=ov.glow_pts[1])

    def _build_overlays(self, model: MefModel):
        key, ov = str(model.path), ModelOverlays()
        res = build_bone_overlay(self._ctx, self._flat_prog, model)
        if res: ov.bone_lines, ov.bone_joints = (res[0], res[1]), (res[2], res[3])
        if model.magic_vertices: ov.magic_pts = build_points_overlay(self._ctx, self._flat_prog, [mv.position for mv in model.magic_vertices])
        ov.coll_wire = []
        for cm in model.collision:
            r = build_lines_overlay(self._ctx, self._flat_prog, cm.vertices, cm.faces)
            if r: ov.coll_wire.append(r)
        if model.portals:
            r = build_lines_overlay(self._ctx, self._flat_prog, model.portals[0].vertices, model.portals[0].faces)
            if r: ov.portal_wire = r
        if model.glow_sprites: ov.glow_pts = build_points_overlay(self._ctx, self._flat_prog, [gs.position for gs in model.glow_sprites])
        self._overlay_cache[key] = ov

    def queue_thumbnail(self, model: MefModel):
        key = str(model.path)
        if key not in self._thumb_cache and model not in self._thumb_queue: self._thumb_queue.append(model)
        self.update()

    def _process_one_thumb(self, screen_fbo):
        if not self._thumb_queue or not self._thumb_fbo: return
        model = self._thumb_queue.pop(0); key = str(model.path)
        if key in self._thumb_cache: return
        sz = self.THUMB_SZ; self._thumb_fbo.use(); self._ctx.viewport = (0, 0, sz, sz); self._ctx.clear(0.05, 0.06, 0.13, 1.0)
        c = OrbitCamera(); c.fit(model.center, model.radius); mvp, eye = c.matrices(sz, sz)
        self._prog["u_mvp"].write(gl_bytes(mvp)); self._prog["u_model"].write(gl_bytes(np.eye(4, dtype="f4"))); self._prog["u_cam"].value = tuple(map(float, eye))
        gpu = self._gpu_cache.get(model)
        if gpu: self._ctx.wireframe = False; gpu.draw(self._prog)
        raw = self._thumb_fbo.read(components=4, dtype="f1", clamp=True)
        img = QImage(bytes(raw), sz, sz, sz * 4, QImage.Format_RGBA8888).mirrored(False, True)
        pm = QPixmap.fromImage(img).scaled(96, 72, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self._thumb_cache[key] = pm; self.thumbnail_ready.emit(key, pm)
        screen_fbo.use(); self._ctx.viewport = (0, 0, self.width(), self.height())
        if self._thumb_queue: self.update()

    def mousePressEvent(self, e): self._drag_btn, self._drag_last = e.button(), e.pos()
    def mouseMoveEvent(self, e):
        d = e.pos() - self._drag_last; self._drag_last = e.pos()
        if self._drag_btn == Qt.LeftButton: self.cam.orbit(d.x(), d.y()); self.update()
        elif self._drag_btn == Qt.RightButton: self.cam.pan(d.x(), d.y()); self.update()
    def mouseReleaseEvent(self, e): self._drag_btn = None
    def mouseDoubleClickEvent(self, e):
        if self.current_model: self.cam.fit(self.current_model.center, self.current_model.radius); self.update()
    def wheelEvent(self, e): self.cam.zoom(1 if e.angleDelta().y() > 0 else -1); self.update()
    def load_model(self, model: MefModel): self.current_model = model; self.cam.fit(model.center, model.radius); self.update()

    def _draw_debug_hud(self):
        p = QPainter(self); p.setRenderHint(QPainter.Antialiasing); p.fillRect(10, 10, 260, 230, QColor(0, 0, 0, 180))
        p.setPen(QColor("#00ff66")); p.setFont(QFont("Consolas", 9, QFont.Bold)); p.drawText(20, 30, "✅ VERIFIED CONFIGURATION")
        p.setFont(QFont("Consolas", 8))
        def _line(y, label, val, shortcut):
            p.setPen(QColor("#8090a8")); p.drawText(20, y, f"{shortcut} {label}:")
            p.setPen(QColor("#ffffff")); p.drawText(140, y, f"{val}")
        d = self.debug_params
        _line(55,  "Vert Scale", f"{d.v_scale:.6f}", "[F1]"); _line(75,  "Bone Scale", f"{d.b_scale:.6f}", "[F2]")
        _line(95,  "Swizzle",    f"{d.swizzle_mode}", "[F3]"); _line(115, "Bone Mode",   f"{d.bone_mode}",    "[F4]")
        _line(135, "ID Bias",     f"{d.bone_id_bias}",  "[F5]"); _line(165, "Flip X",      "ON" if d.flip_x else "off", "[F6]")
        _line(185, "Flip Y",      "ON" if d.flip_y else "off", "[F7]"); _line(205, "Flip Z",      "ON" if d.flip_z else "off", "[F8]")
        p.setPen(QColor("#ffaa00")); p.drawText(20, 230, "Double-click search path to re-apply ↺"); p.end()

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_F1: self.idx_vscale = (self.idx_vscale + 1) % len(self.scales); self.debug_params.v_scale = self.scales[self.idx_vscale]
        elif e.key() == Qt.Key_F2: self.idx_bscale = (self.idx_bscale + 1) % len(self.scales); self.debug_params.b_scale = self.scales[self.idx_bscale]
        elif e.key() == Qt.Key_F3: self.idx_swizzle = (self.idx_swizzle + 1) % len(self.swizzles); self.debug_params.swizzle_mode = self.swizzles[self.idx_swizzle]
        elif e.key() == Qt.Key_F4: self.idx_bmode = (self.idx_bmode + 1) % len(self.bone_modes); self.debug_params.bone_mode = self.bone_modes[self.idx_bmode]
        elif e.key() == Qt.Key_F5: self.debug_params.bone_id_bias = (self.debug_params.bone_id_bias + 2) % 3 - 1
        elif e.key() == Qt.Key_F6: self.debug_params.flip_x = not self.debug_params.flip_x
        elif e.key() == Qt.Key_F7: self.debug_params.flip_y = not self.debug_params.flip_y
        elif e.key() == Qt.Key_F8: self.debug_params.flip_z = not self.debug_params.flip_z
        else: super().keyPressEvent(e); return
        self._refresh_model(); self.update()

    def _refresh_model(self):
        if not self.current_model: return
        path = Path(self.current_model.path)
        if self._gpu_cache: self._gpu_cache.invalidate(str(path))
        self._overlay_cache.pop(str(path), None)
        self.current_model = parse_mef(path, debug=self.debug_params)

# ── Panels ────────────────────────────────────────────────────────────────────
class FileListPanel(QWidget):
    model_selected = pyqtSignal(object)
    def __init__(self, viewport: MEFViewport, parent=None):
        super().__init__(parent); self._vp = viewport; self._cache = {}; self._worker = None
        layout = QVBoxLayout(self); layout.setContentsMargins(0, 0, 0, 0); layout.setSpacing(0)
        hdr = QWidget(); hdr.setFixedHeight(38); hdr.setStyleSheet("background:#060612;border-bottom:1px solid #1c1c38;")
        hl = QHBoxLayout(hdr); hl.setContentsMargins(10, 0, 10, 0)
        lbl = QLabel("MODELS"); lbl.setObjectName("sec"); self._count = QLabel("0")
        hl.addWidget(lbl); hl.addStretch(); hl.addWidget(self._count); layout.addWidget(hdr)
        self._search = QLineEdit(); self._search.setPlaceholderText("🔍  Filter models…"); self._search.textChanged.connect(self._filter); layout.addWidget(self._search)
        self._list = QListWidget(); self._list.setItemDelegate(_ThumbDelegate()); self._list.itemClicked.connect(self._on_click); layout.addWidget(self._list)
        viewport.thumbnail_ready.connect(self._on_thumb)
    def load_folder(self, folder: Path):
        self._list.clear(); mef_files = sorted(folder.glob("*.mef"), key=lambda p: p.name.lower())
        valid = [p for p in mef_files if quick_validate(p)]; self._count.setText(str(len(valid)))
        for path in valid:
            d = {"path": str(path), "name": path.name, "pixmap": self._vp._thumb_cache.get(str(path)), "info": "parsing…", "model": None}
            item = QListWidgetItem(); item.setData(Qt.UserRole, d); item.setSizeHint(QSize(0, _ThumbDelegate.H)); self._list.addItem(item)
        if self._worker: self._worker.cancel(); self._worker.wait()
        self._worker = _ParseWorker(valid, self._cache, self._vp.debug_params); self._worker.done.connect(self._on_parsed); self._worker.start()
    def _on_parsed(self, model: MefModel):
        for i in range(self._list.count()):
            item = self._list.item(i); d = item.data(Qt.UserRole)
            if d["name"] == model.name:
                d["model"] = model; d["info"] = f"Vtx {model.total_vertices:,} Tri {model.total_triangles:,}" if model.valid else f"⚠ {model.error}"
                if model.valid: self._vp.queue_thumbnail(model)
                item.setData(Qt.UserRole, d); self._list.viewport().update(); break
    def _on_thumb(self, path_str: str, pm: QPixmap):
        for i in range(self._list.count()):
            item = self._list.item(i); d = item.data(Qt.UserRole)
            if d["path"] == path_str: d["pixmap"] = pm; item.setData(Qt.UserRole, d); self._list.viewport().update(); break
    def _filter(self, text: str):
        lo = text.lower(); 
        for i in range(self._list.count()): item = self._list.item(i); item.setHidden(lo != "" and lo not in item.data(Qt.UserRole)["name"].lower())
    def _on_click(self, item: QListWidgetItem):
        d = item.data(Qt.UserRole)
        if d and d["model"] and d["model"].valid: self.model_selected.emit(d["model"])

class InfoPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent); layout = QVBoxLayout(self); layout.setContentsMargins(0, 0, 0, 0); layout.setSpacing(0)
        hdr = QWidget(); hdr.setFixedHeight(38); hdr.setStyleSheet("background:#060612;border-bottom:1px solid #1c1c38;")
        hl = QHBoxLayout(hdr); hl.setContentsMargins(10, 0, 10, 0); hl.addWidget(QLabel("MODEL INFO", objectName="sec")); layout.addWidget(hdr)
        tabs = QTabWidget(); layout.addWidget(tabs)
        ow = QWidget(); tabs.addTab(ow, "Overview"); fl = QFormLayout(ow); fl.setContentsMargins(12, 14, 12, 14); fl.setSpacing(8); self._ov = {}
        for l, k in [("Name","name"),("Size","size"),("Type","type"),("HSEM","hsem"),("Parts","parts"),("Verts","verts"),("Tris","tris"),("Bones","bones"),("Ext-X","dx"),("Ext-Y","dy"),("Ext-Z","dz")]:
            v = QLabel("—", objectName="val"); v.setWordWrap(True); self._ov[k] = v; fl.addRow(QLabel(l+":"), v)
        self._parts_tbl = QTableWidget(0, 4); self._parts_tbl.setHorizontalHeaderLabels(["Part","Verts","Tris","Position"]); tabs.addTab(self._parts_tbl, "Parts")
        self._bone_tree = QTreeWidget(); self._bone_tree.setHeaderLabels(["Bone","ID","Parent","World Offset"]); tabs.addTab(self._bone_tree, "Bones")
        self._doc = QTextEdit(); self._doc.setReadOnly(True); tabs.addTab(self._doc, "Spec 📖")
        d_p = resource_path("guidemef.md")
        if os.path.exists(d_p): self._doc.setPlainText(open(d_p, encoding="utf-8").read())
    def show_model(self, m: MefModel):
        self._ov["name"].setText(m.name); self._ov["size"].setText(m.file_size_human); self._ov["type"].setText(m.model_type_name)
        self._ov["hsem"].setText(f"v{m.hsem_version}"); self._ov["parts"].setText(str(len(m.parts))); self._ov["verts"].setText(f"{m.total_vertices:,}")
        self._ov["tris"].setText(f"{m.total_triangles:,}"); self._ov["bones"].setText(str(len(m.bones)))
        e = m.extents; self._ov["dx"].setText(f"{e[0]:.3f}"); self._ov["dy"].setText(f"{e[1]:.3f}"); self._ov["dz"].setText(f"{e[2]:.3f}")
        self._parts_tbl.setRowCount(len(m.parts)); 
        for i, p in enumerate(m.parts):
            self._parts_tbl.setItem(i, 0, QTableWidgetItem(str(p.index))); self._parts_tbl.setItem(i, 1, QTableWidgetItem(str(p.vertex_count)))
            self._parts_tbl.setItem(i, 2, QTableWidgetItem(str(p.triangle_count))); self._parts_tbl.setItem(i, 3, QTableWidgetItem(str(p.position)))
        self._bone_tree.clear(); items = {}
        for b in sorted(m.bones, key=lambda x: x.bone_id):
            ti = QTreeWidgetItem([b.name, str(b.bone_id), str(b.parent_id), str(b.world_offset)]); items[b.bone_id] = ti
            if b.parent_id == -1: self._bone_tree.addTopLevelItem(ti)
            elif b.parent_id in items: items[b.parent_id].addChild(ti)
        self._bone_tree.expandAll()

# ── Main Window ───────────────────────────────────────────────────────────────
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__(); self.setWindowTitle("IGI 2 MEF Viewer (Premium Edition)"); self.resize(1500, 880)
        tb = self.addToolBar("Main"); tb.setMovable(False)
        tb.addAction("📂 Open", self._open); tb.addAction("↺ Reload", self._reload); tb.addSeparator()
        self._act_wire = QAction("⬡ Wire", self, checkable=True); self._act_wire.toggled.connect(lambda on: setattr(self._vp, "wireframe", on) or self._vp.update()); tb.addAction(self._act_wire)
        self._act_grid = QAction("⊞ Grid", self, checkable=True, checked=True); self._act_grid.toggled.connect(lambda on: setattr(self._vp, "show_grid", on) or self._vp.update()); tb.addAction(self._act_grid)
        tb.addSeparator()
        def _toggle_ov(flag): setattr(self._vp.overlays, flag, not getattr(self._vp.overlays, flag)); self._vp.update()
        tb.addAction("🦴 Bones", lambda: _toggle_ov("bones")); tb.addAction("✦ Magic", lambda: _toggle_ov("magic_verts"))
        tb.addAction("🔺 Coll", lambda: _toggle_ov("collision")); tb.addAction("🚪 Portals", lambda: _toggle_ov("portals")); tb.addAction("✨ Glow", lambda: _toggle_ov("glow"))
        tb.addSeparator(); tb.addAction("⊙ Fit", lambda: self._vp.cam.fit(self._vp.current_model.center, self._vp.current_model.radius) or self._vp.update() if self._vp.current_model else None)
        splitter = QSplitter(Qt.Horizontal); self.setCentralWidget(splitter)
        self._vp = MEFViewport(); self._file_panel = FileListPanel(self._vp); self._info_panel = InfoPanel()
        self._file_panel.model_selected.connect(lambda m: self._vp.load_model(m) or self._info_panel.show_model(m))
        splitter.addWidget(self._file_panel); splitter.addWidget(self._vp); splitter.addWidget(self._info_panel)
        splitter.setStretchFactor(1, 1); self._file_panel.setFixedWidth(270); self._info_panel.setFixedWidth(320)
        self._sb_status = QLabel("Ready"); self.statusBar().addPermanentWidget(self._sb_status)
    def _open(self):
        d = QFileDialog.getExistingDirectory(self, "Select MEF Folder"); 
        if d: self._file_panel.load_folder(Path(d)); self._sb_status.setText(d)
    def _reload(self):
        if self._vp.current_model: self._file_panel.load_folder(Path(self._vp.current_model.path).parent)

if __name__ == "__main__":
    app = QApplication(sys.argv); app.setStyleSheet(STYLE); win = MainWindow(); win.show(); sys.exit(app.exec_())
