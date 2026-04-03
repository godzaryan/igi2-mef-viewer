
import os
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

"""
igi2mef.exceptions
~~~~~~~~~~~~~~~~~~
All exceptions raised by the igi2mef library.
"""


class MefError(Exception):
    """Base class for all igi2mef errors."""


class MefParseError(MefError):
    """
    Raised when a MEF file cannot be parsed.

    Attributes
    ----------
    path : str
        The file path that caused the error (if available).
    reason : str
        Short human-readable description of what went wrong.
    """

    def __init__(self, reason: str, path: str = ""):
        self.reason = reason
        self.path = path
        msg = f"{reason}"
        if path:
            msg = f"[{path}] {reason}"
        super().__init__(msg)


class MefValidationError(MefError):
    """
    Raised when a file exists but does not pass the ILFF magic check.
    """
"""
igi2mef._constants
~~~~~~~~~~~~~~~~~~
All binary-format constants for the IGI 2 MEF model format.
Verified from io_scene_igi2_mef.py — zero assumptions made.
"""

# File magic
MAGIC_ILFF: bytes = b"ILFF"

# Scale: IGI world units → normalised viewer units
SCALE: float = 0.01
INV_SCALE: float = 100.0

# Per-model-type vertex stride (bytes)
XTRV_STRIDE: dict = {0: 32, 1: 40, 2: 44, 3: 28}

# Byte offset of the position vector inside one XTRV vertex record
XTRV_POS_OFF: dict = {0: 0, 1: 0, 2: 0, 3: 0}

# Byte offset of the normal vector inside one XTRV vertex record
XTRV_NORM_OFF: dict = {0: 12, 1: 12, 2: 20, 3: 20}

# Byte offset of the primary UV pair inside one XTRV vertex record
XTRV_UV1_OFF: dict = {0: 24, 1: 24, 2: 32, 3: 12}

# Per-model-type DNER (part-descriptor) record stride (bytes)
DNER_STRIDE: dict = {0: 32, 1: 32, 2: 32, 3: 28}

# Human-readable names for each model type
MODEL_TYPE_NAMES: dict = {
    0: "Standard (Rigid)",
    1: "Extended (Bone/Dynamic)",
    2: "Extended UV2 (Lightmap)",
    3: "Compact (Shadow)",
}

# Shadow format strides (bytes)
SEMS_STRIDE: int = 28
XTVS_STRIDE: int = 12
CAFS_STRIDE: int = 28
EGDE_STRIDE: int = 8

# All chunk four-character codes we know about
KNOWN_CHUNK_TAGS: list = [
    # --- Core Render ---
    b"HSEM",  # Mesh header / model metadata
    b"D3DR",  # Render descriptor (mesh count etc.)
    b"DNER",  # Part descriptors (position, index ranges)
    b"XTRV",  # Vertex pool (pos + normal + UV)
    b"ECAF",  # Face index list (triangle indices)
    b"PMTL",  # Render Mesh Lightmaps
    b"XTXM",  # Texture map references (optional)
    b"TCST",  # Texture coordinate sets (optional)
    # --- Skeleton / Bones ---
    b"REIH",  # Bone Hierarchy (parent index table)
    b"MANB",  # Bone Names (fixed 16-char strings)
    b"HPRM",  # Bone Parameters (local translation + rotation)
    # --- Magic Vertices ---
    b"XTVM",  # Magic Vertex List (attachment points)
    b"ATTA",  # Attachment definitions
    # --- Portals ---
    b"TROP",  # Portal Definition
    b"XVTP",  # Portal Vertices
    b"CFTP",  # Portal Faces
    # --- Collision Mesh ---
    b"HSMC",  # Collision Mesh Header
    b"XTVC",  # Collision Mesh Vertices
    b"ECFC",  # Collision Mesh Faces
    b"TAMC",  # Collision Mesh Materials
    b"HPSC",  # Collision Mesh Spheres
    # --- Glow Sprites ---
    b"WOLG",  # Glow Sprite List
    # --- Shadow Chunks ---
    b"SEMS",  # Shadow Element Mesh Structs
    b"XTVS",  # Shadow Vertex Pool
    b"CAFS",  # Shadow Faces
    b"EGDE",  # Shadow Edges
    # --- Other ---
    b"LLUN",  # Null / terminator chunk (optional)
    b"XTRN",  # External reference (optional)
    b"NHSA",  # Animation skeleton hint (optional)
]

# Byte offset in D3DR payload where mesh_count lives (type-dependent)
D3DR_MESH_COUNT_OFFSET: dict = {
    0: 8, 1: 8, 2: 8, 3: 12,
}
"""
igi2mef.models
~~~~~~~~~~~~~~
Data classes that represent a parsed IGI 2 MEF model.

These are plain Python objects — no rendering, no I/O.
All coordinates are returned in **Y-up** space
(X = right, Y = up, Z = toward viewer) at viewer scale
(IGI world units × 0.01).
"""





# ---------------------------------------------------------------------------
# Debug Experiments Matrix
# ---------------------------------------------------------------------------

@dataclass
class MefDebugParams:
    """Dynamic parameters for coordinate and scale experiments."""
    v_scale:      float = 0.01      # Multiplier for raw vertex floats
    b_scale:      float = 0.01      # Multiplier for raw bone floats
    swizzle_mode: str   = "XZY"     # Axis mapping (e.g. "XYZ", "XZY", "YXZ", etc.)
    bone_mode:    str   = "REL"     # "REL" (to parent), "ABS" (ignore hierarchy), "ROOT" (to bone 0)
    bone_id_bias: int   = 0         # Offset applied to b_id (0, -1, 1)
    flip_x:       bool  = False
    flip_y:       bool  = False
    flip_z:       bool  = False

    def swizzle(self, x: float, y: float, z: float, scale: float) -> Tuple[float, float, float]:
        """Apply dynamic axis mapping and scaling."""
        tx, ty, tz = x, y, z
        if self.flip_x: tx = -tx
        if self.flip_y: ty = -ty
        if self.flip_z: tz = -tz
        
        mode = self.swizzle_mode.upper()
        # Initial: IGI (X,Y,Z) where Z is UP.
        # Desired: OpenGL (X,Y,Z) where Y is UP.
        if mode == "XZY": # Standard IGI->GL (X->X, Z->Y, Y->Z)
            return tx * scale, tz * scale, ty * scale
        elif mode == "XYZ": # No mapping
            return tx * scale, ty * scale, tz * scale
        elif mode == "YXZ": # Swap X/Y
            return ty * scale, tx * scale, tz * scale
        elif mode == "ZXY": 
            return tz * scale, tx * scale, ty * scale
        elif mode == "YZX":
            return ty * scale, tz * scale, tx * scale
        elif mode == "ZYX":
            return tz * scale, ty * scale, tx * scale
        return tx * scale, ty * scale, tz * scale


# ---------------------------------------------------------------------------
# Skeletal Bone Hierarchy (Dynamic Models)
# ---------------------------------------------------------------------------

@dataclass
class MefBone:
    """A single bone node from a Dynamic Model's HPRM/REIH/MANB chunks."""
    bone_id:      int
    name:         str
    parent_id:    int   # -1 = root
    local_offset: Tuple[float, float, float]
    world_offset: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    children:     List[int] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Magic Vertices (Attachment / Effect Points)
# ---------------------------------------------------------------------------

@dataclass
class MagicVertex:
    """
    An XTVM magic vertex — a named attachment or effect-spawning point.

    Attributes
    ----------
    index : int
        Zero-based index into the XTVM chunk.
    position : tuple of 3 floats
        World-space position (Y-up, viewer scale).
    normal : tuple of 3 floats
        Surface normal at the vertex (Y-up).
    name : str
        Optional name decoded from the payload, blank if unavailable.
    """
    index:    int
    position: Tuple[float, float, float]
    normal:   Tuple[float, float, float] = (0.0, 1.0, 0.0)
    name:     str = ""


# ---------------------------------------------------------------------------
# Portals
# ---------------------------------------------------------------------------

@dataclass
class Portal:
    """
    One portal polygon from the TROP/XVTP/CFTP chunks.

    Portals define visibility zones used by the IGI 2 engine for
    occlusion culling.
    """
    index:    int
    vertices: List[Tuple[float, float, float]] = field(default_factory=list)
    faces:    List[Tuple[int, int, int]]       = field(default_factory=list)


# ---------------------------------------------------------------------------
# Collision Mesh
# ---------------------------------------------------------------------------

@dataclass
class CollisionMesh:
    """
    A collision geometry set parsed from XTVC + ECFC chunks.

    IGI 2 stores two independent collision mesh sets per model
    (type-0 and type-1).
    """
    mesh_type: int   # 0 or 1
    vertices:  List[Tuple[float, float, float]] = field(default_factory=list)
    faces:     List[Tuple[int, int, int]]       = field(default_factory=list)
    spheres:   List[Tuple[float, float, float, float]] = field(default_factory=list)  # (cx,cy,cz,r)


# ---------------------------------------------------------------------------
# Glow Sprites
# ---------------------------------------------------------------------------

@dataclass
class GlowSprite:
    """
    A glow/lens-flare sprite from the WOLG chunk.
    """
    index:    int
    position: Tuple[float, float, float]
    radius:   float = 1.0
    color:    Tuple[float, float, float] = (1.0, 1.0, 0.5)


# ---------------------------------------------------------------------------
# Chunk metadata
# ---------------------------------------------------------------------------

@dataclass
class ChunkInfo:
    """
    Describes one four-CC chunk found inside the ILFF container.

    Attributes
    ----------
    tag : str
        Four-character chunk identifier (e.g. ``"XTRV"``).
    offset : int
        Byte offset of the chunk header from the start of the file.
    size : int
        Payload size in bytes (not including the 16-byte header).
    """
    tag:    str
    offset: int
    size:   int

    def __repr__(self) -> str:
        return f"ChunkInfo(tag={self.tag!r}, offset=0x{self.offset:08X}, size={self.size:,})"


# ---------------------------------------------------------------------------
# A single mesh part (sub-object) inside a model
# ---------------------------------------------------------------------------

@dataclass
class MefPart:
    """
    One mesh partition inside a MEF model.

    IGI 2 splits a model into several *parts*, each with its own
    world-space origin and a sub-set of the global vertex pool.
    """
    index:    int
    position: Tuple[float, float, float]
    vertices: List[Tuple[float, float, float]]
    normals:  List[Tuple[float, float, float]]
    uvs:      List[Tuple[float, float]]
    faces:    List[Tuple[int, int, int]]

    @property
    def vertex_count(self) -> int:
        return len(self.vertices)

    @property
    def triangle_count(self) -> int:
        return len(self.faces)

    @property
    def bounds(self) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        if not self.vertices:
            return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
        xs = [v[0] for v in self.vertices]
        ys = [v[1] for v in self.vertices]
        zs = [v[2] for v in self.vertices]
        return (min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs))

    def __repr__(self) -> str:
        return f"MefPart(index={self.index}, verts={self.vertex_count}, tris={self.triangle_count})"


# ---------------------------------------------------------------------------
# The top-level model
# ---------------------------------------------------------------------------

@dataclass
class MefModel:
    """A fully-parsed IGI 2 binary MEF model."""

    path:      Path
    valid:     bool  = False
    error:     str   = ""

    model_type:    int   = 0
    hsem_version:  float = 0.0
    hsem_game_ver: int   = 0

    parts:          List[MefPart]      = field(default_factory=list)
    chunks:         List[ChunkInfo]    = field(default_factory=list)
    bones:          List[MefBone]      = field(default_factory=list)
    magic_vertices: List[MagicVertex]  = field(default_factory=list)
    portals:        List[Portal]       = field(default_factory=list)
    collision:      List[CollisionMesh]= field(default_factory=list)
    glow_sprites:   List[GlowSprite]   = field(default_factory=list)

    file_size:       int = 0
    total_vertices:  int = 0
    total_triangles: int = 0

    bounds_min: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    bounds_max: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def stem(self) -> str:
        return self.path.stem

    @property
    def total_parts(self) -> int:
        return len(self.parts)

    @property
    def model_type_name(self) -> str:
        return MODEL_TYPE_NAMES.get(self.model_type, f"Unknown (type {self.model_type})")

    @property
    def center(self) -> Tuple[float, float, float]:
        return (
            (self.bounds_min[0] + self.bounds_max[0]) * 0.5,
            (self.bounds_min[1] + self.bounds_max[1]) * 0.5,
            (self.bounds_min[2] + self.bounds_max[2]) * 0.5,
        )

    @property
    def extents(self) -> Tuple[float, float, float]:
        return (
            self.bounds_max[0] - self.bounds_min[0],
            self.bounds_max[1] - self.bounds_min[1],
            self.bounds_max[2] - self.bounds_min[2],
        )

    @property
    def radius(self) -> float:
        e = self.extents
        return max(e[0], e[1], e[2]) * 0.5 or 1.0

    @property
    def file_size_human(self) -> str:
        s = self.file_size
        if s < 1024:       return f"{s} B"
        if s < 1024 * 1024: return f"{s / 1024:.1f} KB"
        return f"{s / 1024 / 1024:.2f} MB"

    def get_chunk(self, tag: str) -> Optional[ChunkInfo]:
        for c in self.chunks:
            if c.tag == tag:
                return c
        return None

    def __repr__(self) -> str:
        status = "valid" if self.valid else f"invalid: {self.error}"
        return (f"MefModel({self.name!r}, parts={self.total_parts}, "
                f"verts={self.total_vertices}, tris={self.total_triangles}, "
                f"bones={len(self.bones)}, magic_verts={len(self.magic_vertices)}, "
                f"{status})")
"""
igi2mef.parser
~~~~~~~~~~~~~~
Core parsing logic for IGI 2 binary MEF files.
"""





# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _read_chunk(data: bytes, tag: bytes) -> Tuple[Optional[bytes], int]:
    """Locate the first occurrence of *tag* and return its payload."""
    idx = data.find(tag)
    if idx == -1:
        return None, -1
    if idx + 16 > len(data):
        return None, idx
    size  = struct.unpack_from("<I", data, idx + 4)[0]
    start = idx + 16
    end   = start + size
    return data[start: min(end, len(data))], idx


def _read_all_chunks(data: bytes, tag: bytes) -> List[bytes]:
    """Return payloads for ALL occurrences of *tag* (e.g. repeated XTVC)."""
    results = []
    pos = 0
    while True:
        idx = data.find(tag, pos)
        if idx == -1:
            break
        if idx + 16 > len(data):
            break
        size  = struct.unpack_from("<I", data, idx + 4)[0]
        start = idx + 16
        end   = start + size
        results.append(data[start: min(end, len(data))])
        pos = idx + 1
    return results


def _scan_all_chunks(data: bytes) -> List[ChunkInfo]:
    """Build a sorted list of every recognised ILFF chunk in *data*."""
    chunks: List[ChunkInfo] = []
    seen_tags = set()
    for tag in KNOWN_CHUNK_TAGS:
        pos = 0
        while True:
            idx = data.find(tag, pos)
            if idx == -1:
                break
            if idx + 8 <= len(data):
                size = struct.unpack_from("<I", data, idx + 4)[0]
                chunks.append(ChunkInfo(tag=tag.decode("latin-1"),
                                        offset=idx, size=size))
            pos = idx + 1
    chunks.sort(key=lambda c: c.offset)
    return chunks


def _swizzle(x: float, y: float, z: float) -> Tuple[float, float, float]:
    """IGI (X,Y,Z) → Y-up OpenGL (X, Y, Z) mapping.
    Wait, in IGI (3dsmax), Z is UP. In OpenGL Y is UP. 
    So Y_gl = Z_igi, Z_gl = Y_igi."""
    return x * SCALE, z * SCALE, y * SCALE


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def quick_validate(path) -> bool:
    """Return ``True`` if *path* looks like an IGI 2 binary MEF file."""
    try:
        with open(path, "rb") as fh:
            return fh.read(4) == MAGIC_ILFF
    except OSError:
        return False


def parse_mef(path, raise_on_error: bool = False, debug: Optional[MefDebugParams] = None) -> MefModel:
    """Fully parse an IGI 2 binary MEF file and return a :class:`~igi2mef.MefModel`."""
    path  = Path(path).resolve()
    model = MefModel(path=path)
    if debug is None:
        debug = MefDebugParams()

    def _fail(reason: str) -> MefModel:
        model.error = reason
        if raise_on_error:
            raise MefParseError(reason, str(path))
        return model

    try:
        model.file_size = os.path.getsize(path)
        with open(path, "rb") as fh:
            data = fh.read()
    except OSError as exc:
        return _fail(f"Cannot read file: {exc}")

    # ── Magic ──────────────────────────────────────────────────────────────────
    if len(data) < 20 or data[:4] != MAGIC_ILFF:
        return _fail("Not a valid IGI 2 MEF file (ILFF magic not found)")

    # ── Chunk inventory ────────────────────────────────────────────────────────
    model.chunks = _scan_all_chunks(data)

    # ── HSEM — model header ───────────────────────────────────────────────────
    hsem_data, _ = _read_chunk(data, b"HSEM")
    if hsem_data is None:
        # Check for shadow model (SEMS chunk exists)
        sems_data, _ = _read_chunk(data, b"SEMS")
        if sems_data:
            model.model_type = 3 # Shadow
            _parse_shadow_model(data, model, debug)
            _parse_skeleton(data, model, debug)
            return model
        return _fail("Missing HSEM chunk")

    model_type = 0
    if len(hsem_data) >= 4:
        model.hsem_version  = struct.unpack_from("<f", hsem_data, 0)[0]
    if len(hsem_data) >= 8:
        model.hsem_game_ver = struct.unpack_from("<I", hsem_data, 4)[0]
    if len(hsem_data) >= 36:
        raw_type   = struct.unpack_from("<I", hsem_data, 32)[0]
        model_type = raw_type if raw_type in XTRV_STRIDE else 0

    model.model_type = model_type
    stride      = XTRV_STRIDE[model_type]
    pos_off     = XTRV_POS_OFF[model_type]
    norm_off    = XTRV_NORM_OFF[model_type]
    uv_off      = XTRV_UV1_OFF[model_type]
    dner_stride = DNER_STRIDE[model_type]
    mc_off      = D3DR_MESH_COUNT_OFFSET[model_type]

    # ── D3DR — render descriptor ───────────────────────────────────────────────
    d3dr_data, _ = _read_chunk(data, b"D3DR")
    if d3dr_data is None:
        return _fail("Missing D3DR chunk")
    if len(d3dr_data) < mc_off + 4:
        return _fail("D3DR chunk is too small to contain a mesh count")

    mesh_count = struct.unpack_from("<I", d3dr_data, mc_off)[0]
    if mesh_count == 0 or mesh_count > 65535:
        return _fail(f"Implausible mesh count: {mesh_count}")

    # ── DNER — part descriptors ────────────────────────────────────────────────
    dner_data, _ = _read_chunk(data, b"DNER")
    if dner_data is None:
        return _fail("Missing DNER chunk")

    parts_meta = []
    for i in range(mesh_count):
        base = i * dner_stride
        if base + 20 > len(dner_data):
            break
        px, py, pz = struct.unpack_from("<fff", dner_data, base + 4)
        idx_start, tri_count = struct.unpack_from("<HH", dner_data, base + 16)
        parts_meta.append({"pos": (px, py, pz),
                            "idx_start": idx_start,
                            "tri_count": tri_count})

    # ── XTRV — vertex pool ────────────────────────────────────────────────────
    xtrv_data, _ = _read_chunk(data, b"XTRV")
    if xtrv_data is None:
        return _fail("Missing XTRV chunk")

    pool_size = len(xtrv_data) // stride
    v_raw: List[Tuple] = []
    n_raw: List[Tuple] = []
    u_raw: List[Tuple] = []

    for i in range(pool_size):
        b = i * stride
        vx, vy, vz = struct.unpack_from("<fff", xtrv_data, b + pos_off)
        nx, ny, nz = (struct.unpack_from("<fff", xtrv_data, b + norm_off)
                      if b + norm_off + 12 <= len(xtrv_data) else (0.0, 0.0, 1.0))
        uv = (struct.unpack_from("<ff", xtrv_data, b + uv_off)
              if b + uv_off + 8 <= len(xtrv_data) else (0.0, 0.0))
        bone_id = 0
        if stride >= 40 and b + 39 < len(xtrv_data):
            # offset 32 = weight (float), offset 36 = vertex_id (uint16), offset 38 = bone_id (uint16)
            bone_id = struct.unpack_from("<H", xtrv_data, b + 38)[0]
            
        v_raw.append((vx, vy, vz, bone_id))
        n_raw.append((nx, ny, nz))
        u_raw.append(uv)

    # ── ECAF — face index list ─────────────────────────────────────────────────
    ecaf_data, _ = _read_chunk(data, b"ECAF")
    if ecaf_data is None:
        return _fail("Missing ECAF chunk")

    # ── Skeleton (REIH + MANB) ────────────────────────────────────────────────
    _parse_skeleton(data, model, debug)

    # ── Magic Vertices (XTVM) ─────────────────────────────────────────────────
    _parse_magic_vertices(data, model, debug)

    # ── Portals (TROP + XVTP + CFTP) ─────────────────────────────────────────
    _parse_portals(data, model, debug)

    # ── Collision Mesh (HSMC + XTVC + ECFC + HPSC) ──────────────────────────
    _parse_collision(data, model, debug)

    # ── Glow Sprites (WOLG) ───────────────────────────────────────────────────
    _parse_glow_sprites(data, model, debug)

    # ── Build per-part geometry ────────────────────────────────────────────────
    all_viewer_verts: List[Tuple] = []

    for i, meta in enumerate(parts_meta):
        ox, oy, oz = meta["pos"]
        byte_start  = meta["idx_start"] * 2
        tri_count   = meta["tri_count"]

        used: set = set()
        for t in range(tri_count):
            b = byte_start + t * 6
            if b + 6 > len(ecaf_data):
                break
            i0, i1, i2 = struct.unpack_from("<HHH", ecaf_data, b)
            used.add(i0); used.add(i1); used.add(i2)

        if not used:
            continue

        unique  = sorted(used)
        idx_map = {old: new for new, old in enumerate(unique)}

        local_verts:   List[Tuple] = []
        local_normals: List[Tuple] = []
        local_uvs:     List[Tuple] = []

        for gi in unique:
            if gi >= len(v_raw):
                continue
            vx, vy, vz, b_id = v_raw[gi]
            nx, ny, nz = n_raw[gi]
            u, v = u_raw[gi]
            
            # Apply DEBUG SCALE and SWIZZLE to vertex
            dx, dy, dz = debug.swizzle(vx, vy, vz, debug.v_scale)
            
            # Apply DEBUG BONE BIAS
            eff_bid = b_id + debug.bone_id_bias
            
            # For Bone Models (model_type == 1), vertices are relative!
            if model.model_type == 1 and model.bones and 0 <= eff_bid < len(model.bones):
                bx, by, bz = model.bones[eff_bid].world_offset
                dx += bx
                dy += by
                dz += bz
                
            local_verts.append((dx, dy, dz))
            local_normals.append(debug.swizzle(nx, ny, nz, 1.0))
            local_uvs.append((u, 1.0 - v))

        local_faces: List[Tuple] = []
        for t in range(tri_count):
            b = byte_start + t * 6
            if b + 6 > len(ecaf_data):
                break
            i0, i1, i2 = struct.unpack_from("<HHH", ecaf_data, b)
            if i0 in idx_map and i1 in idx_map and i2 in idx_map:
                local_faces.append((idx_map[i0], idx_map[i1], idx_map[i2]))

        if not local_verts or not local_faces:
            continue

        model.parts.append(MefPart(
            index    = i,
            position = debug.swizzle(ox, oy, oz, debug.b_scale),
            vertices = local_verts,
            normals  = local_normals,
            uvs      = local_uvs,
            faces    = local_faces,
        ))
        all_viewer_verts.extend(local_verts)

    # ── Aggregate stats & bounds ───────────────────────────────────────────────
    model.total_vertices  = sum(p.vertex_count  for p in model.parts)
    model.total_triangles = sum(p.triangle_count for p in model.parts)

    if all_viewer_verts:
        xs = [v[0] for v in all_viewer_verts]
        ys = [v[1] for v in all_viewer_verts]
        zs = [v[2] for v in all_viewer_verts]
        model.bounds_min = (min(xs), min(ys), min(zs))
        model.bounds_max = (max(xs), max(ys), max(zs))

    model.valid = True
    return model


# ---------------------------------------------------------------------------
# Shadow Model parser
# ---------------------------------------------------------------------------

def _parse_shadow_model(data: bytes, model: MefModel, debug: MefDebugParams) -> None:
    """Parse SEMS, XTVS, CAFS chunks for shadow model geometry."""
    sems_data, _ = _read_chunk(data, b"SEMS")
    xtvs_data, _ = _read_chunk(data, b"XTVS")
    cafs_data, _ = _read_chunk(data, b"CAFS")
    
    if not sems_data or not xtvs_data or not cafs_data:
        return

    num_meshes     = len(sems_data) // SEMS_STRIDE
    num_pool_verts = len(xtvs_data) // XTVS_STRIDE
    v_raw = [ debug.swizzle(*struct.unpack_from("<fff", xtvs_data, i * XTVS_STRIDE), debug.v_scale)
              for i in range(num_pool_verts) ]
        
    all_viewer_verts = []
    for i in range(num_meshes):
        off = i * SEMS_STRIDE
        if off + SEMS_STRIDE > len(sems_data):
            break
            
        (v_start, f_start, e_start, 
         v_cnt_raw, f_cnt_raw, _, b_id) = struct.unpack_from("<IIIIIII", sems_data, off)
        
        # Robust counts (swap if necessary)
        if v_start + v_cnt_raw > len(v_raw) and v_start + f_cnt_raw <= len(v_raw):
             v_count, f_count = f_cnt_raw, v_cnt_raw
        else:
             v_count, f_count = v_cnt_raw, f_cnt_raw

        if v_start >= len(v_raw):
            continue
            
        local_v = v_raw[v_start : v_start + v_count]
        local_n = [(0.0, 1.0, 0.0)] * len(local_v)
        local_f = []
        
        for f in range(f_count):
            foff = (f_start + f) * CAFS_STRIDE
            if foff + CAFS_STRIDE > len(cafs_data):
                break
            i0, i1, i2 = struct.unpack_from("<III", cafs_data, foff)
            local_f.append((i0 - v_start, i1 - v_start, i2 - v_start))
        
        if local_v and local_f:
            model.parts.append(MefPart(
                index    = i,
                position = (0.0, 0.0, 0.0),
                vertices = local_v,
                normals  = local_n,
                uvs      = [(0.0, 0.0)] * len(local_v),
                faces    = local_f,
            ))
            all_viewer_verts.extend(local_v)
            
    model.total_vertices  = sum(p.vertex_count for p in model.parts)
    model.total_triangles = sum(p.triangle_count for p in model.parts)
    if all_viewer_verts:
        xs = [v[0] for v in all_viewer_verts]
        ys = [v[1] for v in all_viewer_verts]
        zs = [v[2] for v in all_viewer_verts]
        model.bounds_min = (min(xs), min(ys), min(zs))
        model.bounds_max = (max(xs), max(ys), max(zs))
    model.valid = True


# ---------------------------------------------------------------------------
# Skeleton parser
# ---------------------------------------------------------------------------

def _parse_skeleton(data: bytes, model: MefModel, debug: MefDebugParams) -> None:
    """Parse REIH + MANB chunks into model.bones."""
    reih_data, _ = _read_chunk(data, b"REIH")
    manb_data, _ = _read_chunk(data, b"MANB")

    if not reih_data or not manb_data:
        return

    num_bones = len(manb_data) // 16
    child_counts = list(reih_data[:num_bones])
    
    # 1. Deduce the BFS mapping tree
    import collections
    parent_ids = [-1] * num_bones
    if num_bones > 0:
        q = collections.deque([0])
        next_id = 1
        while q and next_id < num_bones:
            curr = q.popleft()
            count = child_counts[curr]
            for _ in range(count):
                if next_id < num_bones:
                    parent_ids[next_id] = curr
                    q.append(next_id)
                    next_id += 1

    # 2. Advance pointer past counts and padding
    pos = num_bones
    if pos % 4 != 0:
        pos += 4 - (pos % 4)

    raw_bones: List[MefBone] = []
    for i in range(num_bones):
        p_id = parent_ids[i]

        # Name from MANB (16 bytes per string)
        name = f"bone_{i:02d}"
        if len(manb_data) >= (i + 1) * 16:
            raw = manb_data[i * 16: (i + 1) * 16]
            name = raw.split(b'\x00')[0].decode('ascii', errors='replace').strip() or name

        # Position from REIH
        if pos + 12 <= len(reih_data):
            hx, hy, hz = struct.unpack_from("<fff", reih_data, pos)
            pos += 12
        else:
            hx, hy, hz = 0.0, 0.0, 0.0

        bone = MefBone(
            bone_id      = i,
            name         = name,
            parent_id    = p_id,
            local_offset = debug.swizzle(hx, hy, hz, debug.b_scale),
        )
        raw_bones.append(bone)

    # Build children lists
    bone_map = {b.bone_id: b for b in raw_bones}
    for b in raw_bones:
        if b.parent_id >= 0 and b.parent_id in bone_map:
            bone_map[b.parent_id].children.append(b.bone_id)

    # Accumulate world offsets (DFS, cycle-safe)
    world_cache: Dict[int, Tuple] = {}

    def get_world(bid: int, stack: set = None) -> Tuple:
        if stack is None: stack = set()
        if bid in world_cache: return world_cache[bid]
        if bid not in bone_map or bid in stack: return (0.0, 0.0, 0.0)
        stack.add(bid)
        b = bone_map[bid]
        lx, ly, lz = b.local_offset
        
        # DEBUG EXPERIMENT: BONE_MODE
        if debug.bone_mode == "ABS":
            # Treat REIH pivots as absolute world-space
            world_cache[bid] = (lx, ly, lz)
        elif debug.bone_mode == "ROOT":
            # Relative only to the master root (bone 0)
            if bid == 0:
                world_cache[bid] = (lx, ly, lz)
            else:
                rx, ry, rz = get_world(0, stack)
                world_cache[bid] = (lx + rx, ly + ry, lz + rz)
        else: # "REL" (Standard)
            if b.parent_id < 0 or b.parent_id not in bone_map:
                world_cache[bid] = (lx, ly, lz)
            else:
                px, py, pz = get_world(b.parent_id, stack)
                world_cache[bid] = (lx + px, ly + py, lz + pz)
        
        return world_cache[bid]

    for b in raw_bones:
        b.world_offset = get_world(b.bone_id)
        model.bones.append(b)


# ---------------------------------------------------------------------------
# Magic Vertices parser
# ---------------------------------------------------------------------------

def _parse_magic_vertices(data: bytes, model: MefModel, debug: MefDebugParams) -> None:
    """Parse XTVM chunk into model.magic_vertices."""
    xtvm_data, _ = _read_chunk(data, b"XTVM")
    if not xtvm_data or len(xtvm_data) < 4:
        return

    # XTVM: 4-byte count, then 16-byte records: [pos xyz float*3, 4 bytes padding/ID]
    count = struct.unpack_from("<I", xtvm_data, 0)[0]
    stride = 16
    for i in range(count):
        off = 4 + i * stride
        if off + stride > len(xtvm_data):
            break
        vx, vy, vz = struct.unpack_from("<fff", xtvm_data, off)
        model.magic_vertices.append(MagicVertex(
            index    = i,
            position = debug.swizzle(vx, vy, vz, debug.v_scale),
            normal   = debug.swizzle(0.0, 1.0, 0.0, 1.0),
        ))


# ---------------------------------------------------------------------------
# Portals parser
# ---------------------------------------------------------------------------

def _parse_portals(data: bytes, model: MefModel, debug: MefDebugParams) -> None:
    """Parse TROP + XVTP + CFTP chunks into model.portals."""
    trop_data, _ = _read_chunk(data, b"TROP")
    xvtp_data, _ = _read_chunk(data, b"XVTP")
    cftp_data, _ = _read_chunk(data, b"CFTP")

    if not trop_data or not xvtp_data:
        return

    # TROP: 4-byte portal count
    if len(trop_data) < 4:
        return
    portal_count = struct.unpack_from("<I", trop_data, 0)[0]

    # XVTP: portal vertices (xyz floats, 12 bytes each)
    # CFTP: portal face indices (3 × uint16, 6 bytes each)
    n_verts = len(xvtp_data) // 12
    verts: List[Tuple] = []
    for i in range(n_verts):
        vx, vy, vz = struct.unpack_from("<fff", xvtp_data, i * 12)
        verts.append(debug.swizzle(vx, vy, vz, debug.v_scale))

    n_faces = len(cftp_data) // 6 if cftp_data else 0
    faces: List[Tuple] = []
    for i in range(n_faces):
        a, b, c = struct.unpack_from("<HHH", cftp_data, i * 6)
        faces.append((a, b, c))

    # For now, bundle all portal verts/faces into one portal per count
    # (detailed per-portal subdivision needs TROP record parsing)
    portal = Portal(index=0, vertices=verts, faces=faces)
    model.portals.append(portal)


# ---------------------------------------------------------------------------
# Collision Mesh parser
# ---------------------------------------------------------------------------

def _parse_collision(data: bytes, model: MefModel, debug: MefDebugParams) -> None:
    """Parse HSMC + XTVC + ECFC + HPSC into model.collision."""
    hsmc_data, _ = _read_chunk(data, b"HSMC")
    if not hsmc_data or len(hsmc_data) < 32:
        return

    # HSMC layout: two sets of (n_face, n_vertex, n_material, n_sph, 4×zero)
    n_face0, n_vert0, n_mat0, n_sph0 = struct.unpack_from("<IIII", hsmc_data, 0)
    n_face1, n_vert1, n_mat1, n_sph1 = struct.unpack_from("<IIII", hsmc_data, 16)

    # XTVC appears twice: type-0 then type-1
    xtvc_all  = _read_all_chunks(data, b"XTVC")
    ecfc_all  = _read_all_chunks(data, b"ECFC")
    hpsc_all  = _read_all_chunks(data, b"HPSC")

    for mesh_idx, (n_v, n_f, n_s) in enumerate([(n_vert0, n_face0, n_sph0),
                                                   (n_vert1, n_face1, n_sph1)]):
        verts: List[Tuple] = []
        faces: List[Tuple] = []
        spheres: List[Tuple] = []

        # Parse collision vertices (20 bytes each: x,y,z,_,_)
        if mesh_idx < len(xtvc_all):
            vd = xtvc_all[mesh_idx]
            vstride = 20
            for i in range(len(vd) // vstride):
                off = i * vstride
                vx, vy, vz = struct.unpack_from("<fff", vd, off)
                
                # Apply DEBUG SCALE and SWIZZLE
                dx, dy, dz = debug.swizzle(vx, vy, vz, debug.v_scale)
                
                # BONE MODELS: extract bone_id from Byte 12 (uint16)
                if model.model_type == 1 and model.bones:
                    bc_id = struct.unpack_from("<H", vd, off + 12)[0]
                    eff_bc = bc_id + debug.bone_id_bias
                    if 0 <= eff_bc < len(model.bones):
                        bx, by, bz = model.bones[eff_bc].world_offset
                        dx += bx
                        dy += by
                        dz += bz
                
                verts.append((dx, dy, dz))

        # Parse collision faces (12 bytes each: a,b,c, _, _, _  all uint16)
        if mesh_idx < len(ecfc_all):
            fd = ecfc_all[mesh_idx]
            fstride = 12
            for i in range(len(fd) // fstride):
                a, b, c = struct.unpack_from("<HHH", fd, i * fstride)
                faces.append((a, b, c))

        # Parse collision spheres (16 bytes each: cx,cy,cz,radius)
        if mesh_idx < len(hpsc_all):
            sd = hpsc_all[mesh_idx]
            sstride = 16
            for i in range(len(sd) // sstride):
                cx, cy, cz, r = struct.unpack_from("<ffff", sd, i * sstride)
                sx, sy, sz = debug.swizzle(cx, cy, cz, debug.v_scale)
                
                # BONE MODELS: spheres are often relative to root (bone 0)
                if model.model_type == 1 and model.bones:
                    bx, by, bz = model.bones[0].world_offset
                    sx, sy, sz = sx + bx, sy + by, sz + bz
                    
                spheres.append((sx, sy, sz, r * debug.v_scale))

        if verts or spheres:
            model.collision.append(CollisionMesh(
                mesh_type = mesh_idx,
                vertices  = verts,
                faces     = faces,
                spheres   = spheres,
            ))


# ---------------------------------------------------------------------------
# Glow Sprites parser
# ---------------------------------------------------------------------------

def _parse_glow_sprites(data: bytes, model: MefModel, debug: MefDebugParams) -> None:
    """Parse WOLG chunk into model.glow_sprites."""
    wolg_data, _ = _read_chunk(data, b"WOLG")
    if not wolg_data or len(wolg_data) < 4:
        return

    # WOLG: 4-byte count, then variable records
    count = struct.unpack_from("<I", wolg_data, 0)[0]
    stride = 32
    for i in range(count):
        off = 4 + i * stride
        if off + stride > len(wolg_data):
            break
        gx, gy, gz = struct.unpack_from("<fff", wolg_data, off)
        model.glow_sprites.append(GlowSprite(
            index    = i,
            position = debug.swizzle(gx, gy, gz, debug.v_scale),
            radius   = 1.0 * debug.v_scale, # Just a default size
        ))
