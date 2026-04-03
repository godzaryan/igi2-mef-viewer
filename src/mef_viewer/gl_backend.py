"""
gl_backend.py — OpenGL infrastructure for the IGI 2 MEF Viewer
Camera, GPU model wrapper, shaders, math helpers, overlay builders.
"""

from __future__ import annotations
import math
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import numpy as np
import moderngl
from igi2mef import MefModel

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
    f = target - eye; f /= (np.linalg.norm(f) or 1)
    r = np.cross(f, up); r /= (np.linalg.norm(r) or 1)
    u = np.cross(r, f)
    return np.array([
        [ r[0], r[1], r[2], -float(np.dot(r, eye))],
        [ u[0], u[1], u[2], -float(np.dot(u, eye))],
        [-f[0],-f[1],-f[2], float(np.dot(f, eye))],
        [ 0,    0,    0,    1                     ],
    ], dtype="f4")

# ── Orbit camera ─────────────────────────────────────────────────────────────
class OrbitCamera:
    def __init__(self):
        self.az, self.el, self.dist, self.fov = 45.0, 25.0, 5.0, 45.0
        self.target = np.zeros(3, "f4")
    def eye(self) -> np.ndarray:
        az, el = math.radians(self.az), math.radians(self.el)
        return self.target + np.array([
            self.dist * math.cos(el) * math.sin(az),
            self.dist * math.sin(el),
            self.dist * math.cos(el) * math.cos(az),
        ], "f4")
    def matrices(self, w: int, h: int) -> Tuple[np.ndarray, np.ndarray]:
        eye = self.eye()
        v = look_at(eye, self.target, np.array([0,1,0], "f4"))
        p = perspective(self.fov, max(w,1)/max(h,1), 0.001, 5000.0)
        return p @ v, eye
    def orbit(self, dx: float, dy: float):
        self.az += dx * 0.4; self.el = float(np.clip(self.el - dy * 0.4, -89, 89))
    def pan(self, dx: float, dy: float):
        az = math.radians(self.az); r = np.array([math.cos(az),0.,-math.sin(az)],"f4")
        u = np.array([0.,1.,0.],"f4"); s = self.dist * 0.0012
        self.target -= r * dx * s; self.target += u * dy * s
    def zoom(self, dir: int): self.dist = max(0.001, self.dist * (0.88 if dir > 0 else 1.12))
    def fit(self, center, radius: float):
        self.target, self.dist = np.array(center, "f4"), max(radius * 2.8, 0.1)
        self.az, self.el = 45.0, 25.0

# ── GPU model ────────────────────────────────────────────────────────────────
class GpuModel:
    def __init__(self, ctx: moderngl.Context, prog: moderngl.Program, model: MefModel):
        self.part_ranges: List[Tuple[int,int]] = []
        v_data, idx_data, vbase = [], [], 0
        for part in model.parts:
            start = len(idx_data)
            for i in range(part.vertex_count):
                v_data.extend([*part.vertices[i], *part.normals[i]])
            for f in part.faces: idx_data.extend([f[0]+vbase, f[1]+vbase, f[2]+vbase])
            self.part_ranges.append((start, len(part.faces)*3)); vbase += part.vertex_count
        self.vbo = ctx.buffer(np.array(v_data, "f4").tobytes())
        self.ibo = ctx.buffer(np.array(idx_data, "u4").tobytes())
        self.vao = ctx.vertex_array(prog, [(self.vbo, "3f 3f", "in_pos", "in_norm")], self.ibo)
    def draw(self, prog: moderngl.Program):
        for i, (first, cnt) in enumerate(self.part_ranges):
            prog["u_color"].value = PART_COLORS[i % len(PART_COLORS)]
            self.vao.render(moderngl.TRIANGLES, vertices=cnt, first=first)
    def release(self): self.vao.release(); self.vbo.release(); self.ibo.release()

class GpuCache:
    MAX = 15
    def __init__(self, ctx: moderngl.Context, prog: moderngl.Program):
        self._ctx, self._prog, self._cache = ctx, prog, OrderedDict()
    def get(self, model: MefModel) -> Optional[GpuModel]:
        key = str(model.path)
        if key in self._cache: self._cache.move_to_end(key); return self._cache[key]
        if not model.valid or not model.parts: return None
        gpu = GpuModel(self._ctx, self._prog, model); self._cache[key] = gpu
        if len(self._cache) > self.MAX:
            _, old = self._cache.popitem(last=False)
            try: old.release()
            except: pass
        return gpu
    def invalidate(self, path_str: str):
        if path_str in self._cache:
            try: self._cache[path_str].release()
            except: pass
            del self._cache[path_str]

# ── Builders ──────────────────────────────────────────────────────────────────
def build_grid(ctx, prog, half=10):
    lines = []
    for i in range(-half, half + 1):
        lines += [float(i),0,float(-half), float(i),0,float(half)]
        lines += [float(-half),0,float(i), float(half),0,float(i)]
    buf = ctx.buffer(np.array(lines, "f4").tobytes())
    return ctx.vertex_array(prog, [(buf, "3f", "in_pos")]), len(lines)//3

def build_bone_overlay(ctx, prog, model):
    if not model.bones: return None
    bmap = {b.bone_id: b for b in model.bones}; lp, jp = [], []
    for b in model.bones:
        wx, wy, wz = b.world_offset; jp.extend([wx,wy,wz])
        if b.parent_id >= 0 and b.parent_id in bmap:
            px, py, pz = bmap[b.parent_id].world_offset; lp.extend([px,py,pz, wx,wy,wz])
    if not lp and not jp: return None
    lb = ctx.buffer(np.array(lp or [0,0,0,0,0,0], "f4").tobytes())
    jb = ctx.buffer(np.array(jp or [0,0,0], "f4").tobytes())
    return ctx.vertex_array(prog, [(lb, "3f", "in_pos")]), len(lp)//3, ctx.vertex_array(prog, [(jb,"3f","in_pos")]), len(jp)//3

def build_points_overlay(ctx, prog, positions):
    if not positions: return None
    pts = []
    for p in positions: pts.extend(p)
    buf = ctx.buffer(np.array(pts, "f4").tobytes())
    return ctx.vertex_array(prog, [(buf, "3f", "in_pos")]), len(positions)

def build_lines_overlay(ctx, prog, vertices, faces):
    if not vertices or not faces: return None
    lp = []
    for a,b,c in faces:
        if max(a,b,c) < len(vertices):
            lp.extend([*vertices[a], *vertices[b], *vertices[b], *vertices[c], *vertices[c], *vertices[a]])
    if not lp: return None
    buf = ctx.buffer(np.array(lp, "f4").tobytes())
    return ctx.vertex_array(prog, [(buf, "3f", "in_pos")]), len(lp)//3
