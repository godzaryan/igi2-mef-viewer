"""
gl_core.py — OpenGL infrastructure for the IGI 2 MEF Viewer
Camera, GPU model wrapper, shaders, math helpers, overlay builders.
"""
from __future__ import annotations
import math
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import numpy as np
import moderngl
# ── Import Standalone Static Parser ─────────────────────────────────────────
import sys
from pathlib import Path
_here = Path(__file__).parent
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))
try:
    from igi2mef import MefModel
except ImportError:
    from mef_parser_static import MefModel # Fallback for local dev if not installed

# ── GLSL Shaders ─────────────────────────────────────────────────────────────
VERT = """
#version 330 core
layout(location=0) in vec3 in_pos;
layout(location=1) in vec3 in_norm;
uniform mat4 u_mvp;
uniform mat4 u_model;
out vec3 v_pos;
out vec3 v_norm;
void main() {
    vec4 wp = u_model * vec4(in_pos, 1.0);
    v_pos  = wp.xyz;
    v_norm = mat3(transpose(inverse(u_model))) * in_norm;
    gl_Position = u_mvp * vec4(in_pos, 1.0);
}
"""
FRAG = """
#version 330 core
in vec3 v_pos;
in vec3 v_norm;
uniform vec3 u_color;
uniform vec3 u_cam;
out vec4 f_color;
void main() {
    vec3 n  = normalize(v_norm) * (gl_FrontFacing ? 1.0 : -1.0);
    vec3 l1 = normalize(vec3(1.0, 2.0, 1.5));
    vec3 l2 = normalize(vec3(-1.0, -0.5, 0.5));
    vec3 h  = normalize(l1 + normalize(u_cam - v_pos));
    float d = max(dot(n,l1),0.0)*0.65 + max(dot(n,l2),0.0)*0.20 + 0.25;
    float s = pow(max(dot(n,h), 0.0), 32.0) * 0.25;
    f_color = vec4(u_color * (d + s), 1.0);
}
"""
FLAT_VERT = """
#version 330 core
layout(location=0) in vec3 in_pos;
uniform mat4 u_mvp;
void main() { gl_Position = u_mvp * vec4(in_pos, 1.0); }
"""
FLAT_FRAG = """
#version 330 core
uniform vec4 u_color;
out vec4 f_color;
void main() { f_color = u_color; }
"""

# ── Part colour palette ───────────────────────────────────────────────────────
PART_COLORS: List[Tuple[float,float,float]] = [
    (0.68,0.77,0.90), (0.90,0.75,0.60), (0.60,0.87,0.72),
    (0.87,0.65,0.75), (0.65,0.80,0.95), (0.95,0.85,0.60),
    (0.65,0.95,0.85), (0.80,0.65,0.95),
]

# ── Matrix helpers ────────────────────────────────────────────────────────────
def gl_bytes(m: np.ndarray) -> bytes:
    """Row-major numpy → column-major bytes for OpenGL."""
    return np.ascontiguousarray(m.T, dtype="f4").tobytes()

def perspective(fov: float, aspect: float, near: float, far: float) -> np.ndarray:
    f  = 1.0 / math.tan(math.radians(fov) * 0.5)
    nf = 1.0 / (near - far)
    return np.array([
        [f/aspect, 0,  0,                 0            ],
        [0,        f,  0,                 0            ],
        [0,        0,  (far+near)*nf,     2*far*near*nf],
        [0,        0, -1,                 0            ],
    ], dtype="f4")

def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    f = target - eye;   f /= (np.linalg.norm(f) or 1)
    r = np.cross(f, up); r /= (np.linalg.norm(r) or 1)
    u = np.cross(r, f)
    return np.array([
        [ r[0],  r[1],  r[2], -float(np.dot(r, eye))],
        [ u[0],  u[1],  u[2], -float(np.dot(u, eye))],
        [-f[0], -f[1], -f[2],  float(np.dot(f, eye))],
        [ 0,     0,     0,     1                     ],
    ], dtype="f4")

# ── Orbit camera ─────────────────────────────────────────────────────────────
class OrbitCamera:
    def __init__(self):
        self.az     = 45.0
        self.el     = 25.0
        self.dist   = 5.0
        self.fov    = 45.0
        self.target = np.zeros(3, "f4")

    def eye(self) -> np.ndarray:
        az = math.radians(self.az)
        el = math.radians(self.el)
        return self.target + np.array([
            self.dist * math.cos(el) * math.sin(az),
            self.dist * math.sin(el),
            self.dist * math.cos(el) * math.cos(az),
        ], "f4")

    def matrices(self, w: int, h: int) -> Tuple[np.ndarray, np.ndarray]:
        eye = self.eye()
        view = look_at(eye, self.target, np.array([0,1,0], "f4"))
        proj = perspective(self.fov, max(w,1)/max(h,1), 0.001, 5000.0)
        return proj @ view, eye

    def orbit(self, dx: float, dy: float):
        self.az += dx * 0.4
        self.el  = float(np.clip(self.el - dy * 0.4, -89, 89))

    def pan(self, dx: float, dy: float):
        az = math.radians(self.az)
        right = np.array([ math.cos(az), 0.0, -math.sin(az)], "f4")
        up    = np.array([0., 1., 0.], "f4")
        spd   = self.dist * 0.0012
        self.target -= right * dx * spd
        self.target += up    * dy * spd

    def zoom(self, direction: int):
        self.dist = max(0.001, self.dist * (0.88 if direction > 0 else 1.12))

    def fit(self, center, radius: float):
        self.target = np.array(center, "f4")
        self.dist   = max(radius * 2.8, 0.1)
        self.az, self.el = 45.0, 25.0

# ── GPU model (VBO + IBO + VAO per model) ────────────────────────────────────
class GpuModel:
    """Uploads one MefModel's geometry to GPU memory."""
    def __init__(self, ctx: moderngl.Context,
                 prog: moderngl.Program, model: MefModel):
        self.part_ranges: List[Tuple[int,int]] = []

        v_data:   List[float] = []
        idx_data: List[int]   = []
        vbase = 0

        for part in model.parts:
            start = len(idx_data)
            for i in range(part.vertex_count):
                x, y, z    = part.vertices[i]
                nx, ny, nz = part.normals[i]
                v_data.extend([x, y, z, nx, ny, nz])
            for i0, i1, i2 in part.faces:
                idx_data.extend([i0 + vbase, i1 + vbase, i2 + vbase])
            self.part_ranges.append((start, len(part.faces) * 3))
            vbase += part.vertex_count

        self.vbo = ctx.buffer(np.array(v_data,   dtype="f4").tobytes())
        self.ibo = ctx.buffer(np.array(idx_data, dtype="u4").tobytes())
        self.vao = ctx.vertex_array(
            prog,
            [(self.vbo, "3f 3f", "in_pos", "in_norm")],
            self.ibo,
        )

    def draw(self, prog: moderngl.Program):
        for i, (first, count) in enumerate(self.part_ranges):
            prog["u_color"].value = PART_COLORS[i % len(PART_COLORS)]
            self.vao.render(moderngl.TRIANGLES, vertices=count, first=first)

    def release(self):
        self.vao.release(); self.vbo.release(); self.ibo.release()


# ── LRU GPU cache ─────────────────────────────────────────────────────────────
class GpuCache:
    MAX = 15

    def __init__(self, ctx: moderngl.Context, prog: moderngl.Program):
        self._ctx  = ctx
        self._prog = prog
        self._cache: OrderedDict[str, GpuModel] = OrderedDict()

    def get(self, model: MefModel) -> Optional[GpuModel]:
        key = str(model.path)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        if not model.valid or not model.parts:
            return None
        gpu = GpuModel(self._ctx, self._prog, model)
        self._cache[key] = gpu
        if len(self._cache) > self.MAX:
            _, old = self._cache.popitem(last=False)
            try: old.release()
            except Exception: pass
        return gpu

    def invalidate(self, path_str: str):
        if path_str in self._cache:
            try: self._cache[path_str].release()
            except Exception: pass
            del self._cache[path_str]


# ── Grid geometry ──────────────────────────────────────────────────────────────
def build_grid(ctx: moderngl.Context,
               prog: moderngl.Program,
               half: int = 10) -> Tuple[moderngl.VertexArray, int]:
    lines: List[float] = []
    for i in range(-half, half + 1):
        lines += [float(i), 0.0, float(-half),
                  float(i), 0.0, float( half)]
        lines += [float(-half), 0.0, float(i),
                  float( half), 0.0, float(i)]
    buf = ctx.buffer(np.array(lines, "f4").tobytes())
    vao = ctx.vertex_array(prog, [(buf, "3f", "in_pos")])
    return vao, len(lines) // 3


# ── Overlay geometry helpers ──────────────────────────────────────────────────

def build_bone_overlay(ctx: moderngl.Context,
                       prog: moderngl.Program,
                       model: MefModel) -> Optional[Tuple[moderngl.VertexArray, int,
                                                           moderngl.VertexArray, int]]:
    """
    Build two VAOs for skeleton overlay:
      - Lines VAO: parent→child bone connections
      - Points VAO: joint positions (rendered as GL_POINTS)

    Returns (lines_vao, lines_n, points_vao, points_n) or None if no bones.
    """
    if not model.bones:
        return None

    bone_map = {b.bone_id: b for b in model.bones}
    line_pts: List[float] = []
    joint_pts: List[float] = []

    for bone in model.bones:
        wx, wy, wz = bone.world_offset
        joint_pts.extend([wx, wy, wz])
        if bone.parent_id >= 0 and bone.parent_id in bone_map:
            px, py, pz = bone_map[bone.parent_id].world_offset
            line_pts.extend([px, py, pz, wx, wy, wz])

    lines_n  = len(line_pts)  // 3
    joints_n = len(joint_pts) // 3

    if lines_n == 0 and joints_n == 0:
        return None

    line_buf  = ctx.buffer(np.array(line_pts  or [0,0,0,0,0,0], "f4").tobytes())
    joint_buf = ctx.buffer(np.array(joint_pts or [0,0,0],        "f4").tobytes())
    lines_vao  = ctx.vertex_array(prog, [(line_buf,  "3f", "in_pos")])
    joints_vao = ctx.vertex_array(prog, [(joint_buf, "3f", "in_pos")])

    return lines_vao, lines_n, joints_vao, joints_n


def build_points_overlay(ctx: moderngl.Context,
                         prog: moderngl.Program,
                         positions: List[Tuple[float,float,float]]) -> Optional[Tuple[moderngl.VertexArray, int]]:
    """Build a points VAO from a list of (x,y,z) positions."""
    if not positions:
        return None
    pts: List[float] = []
    for x, y, z in positions:
        pts.extend([x, y, z])
    buf = ctx.buffer(np.array(pts, "f4").tobytes())
    vao = ctx.vertex_array(prog, [(buf, "3f", "in_pos")])
    return vao, len(positions)


def build_lines_overlay(ctx: moderngl.Context,
                        prog: moderngl.Program,
                        vertices: List[Tuple[float,float,float]],
                        faces:    List[Tuple[int,int,int]]) -> Optional[Tuple[moderngl.VertexArray, int]]:
    """Build a wireframe lines VAO from triangle faces."""
    if not vertices or not faces:
        return None
    line_pts: List[float] = []
    for a, b, c in faces:
        if a < len(vertices) and b < len(vertices) and c < len(vertices):
            ax, ay, az = vertices[a]
            bx, by, bz = vertices[b]
            cx, cy, cz = vertices[c]
            line_pts.extend([ax,ay,az, bx,by,bz,
                              bx,by,bz, cx,cy,cz,
                              cx,cy,cz, ax,ay,az])
    if not line_pts:
        return None
    buf = ctx.buffer(np.array(line_pts, "f4").tobytes())
    vao = ctx.vertex_array(prog, [(buf, "3f", "in_pos")])
    return vao, len(line_pts) // 3
