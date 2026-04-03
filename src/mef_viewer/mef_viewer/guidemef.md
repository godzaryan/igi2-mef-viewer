# The Master-Class Guide to IGI 2 MEF Files (v2.0)

This document is the definitive, byte-level specification for the **MEF** (Mesh-Exported-File) binary format used by **IGI 2: Covert Strike**. It contains everything you need to know to build your own parser or manually read a model in a hex editor.

---

## 1. The Binary Foundation: ILFF & Chunks

MEF files use a "Chunk-based" container called **ILFF** (Inner Loop File Format). This allows the game engine to skip data it doesn't understand.

### The File Header (First 8 Bytes)
Every file starts with a fixed signature:
1. **Signature (4 bytes)**: `ILFF`
2. **Total File Size (4 bytes)**: Big-endian total length of the file minus 8 bytes.

### The Chunk Structure (16 Bytes Header)
Inside the `ILFF` container, data is split into **Chunks**. Each chunk has a standard 16-byte header:
- **Tag (4 bytes)**: The identity of the data (e.g., `DNER`, `ECAF`).
- **Data Size (4 bytes)**: How many bytes of content follow the header.
- **Alignment (4 bytes)**: Fixed at `0x00000004` (all chunks start on 4-byte boundaries).
- **Offset (4 bytes)**: Usually `0x00000000` (reserved).

**Rule**: To find the next chunk, read the `Data Size`, add 16 (for the header), and jump that many bytes from the start of the current tag.

---

## 2. Core Model Definition: `HSEM`

The **HSEM** (Header-Setup-Experimental-Mesh) chunk defines the model "Mode" and its physical boundaries.

| Byte Offset | Data Type | Name | Purpose |
| :--- | :--- | :--- | :--- |
| **0** | `uint32` | **Model Type** | 0=Rigid, 1=Bone, 2=Lightmap, 3=Shadow. |
| **4** | `float32` | **Version** | Version of the format (e.g., 1.1). |
| **8** | `float32[3]`| **Bound Min** | Minimum X, Y, Z coordinates (AABB). |
| **20** | `float32[3]`| **Bound Max** | Maximum X, Y, Z coordinates (AABB). |
| **32** | `float32` | **Radius** | Bounding sphere radius from origin (0,0,0). |

---

## 3. The Geometry Pipeline: Vertices and Faces

Rendered geometry is split across multiple chunks that must be read in sequence and linked together.

### `XTRV` (Vertex Pool) ─ Stride: 32 Bytes
This chunk is a flat list of vertices. Each entry is **exactly 32 bytes**.
- **Byte 0-11**: `Position (X, Y, Z)` (3 floats).
- **Byte 12-23**: `Normal (NX, NY, NZ)` (3 floats). (NOTE: In **Lightmap** models, this is replaced by the 2nd UV set).
- **Byte 24-31**: `Texture UV (U, V)` (2 floats).

### `ECAF` (Face Index Pool) ─ Stride: 6 Bytes
This chunk contains triangles. Each face is 3 integers.
- **Byte 0-1**: `Vertex Index 1` (uint16).
- **Byte 2-3**: `Vertex Index 2` (uint16).
- **Byte 4-5**: `Vertex Index 3` (uint16).
- **Important**: These are indices into the `XTRV` pool.

### `DNER` (Mesh Partitioning) ─ Stride: 28-36 Bytes
The `DNER` chunk defines the "Parts" (meshes) of the model.
- **Part Origin**: 3 floats defining the part's pivot.
- **Start Face**: Index where this part begins in `ECAF`.
- **Face Count**: Number of triangles in this part.
- **Start Vertex**: Index where this part begins in `XTRV`.
- **Vertex Count**: Number of vertices effectively used by this part.

---

## 4. Skeletal Animation: `REIH`, `MANB`, `XTRW`

If the model is a **Bone Model** (Model Type 1), it contains a skeleton.

- **`MANB`**: List of bone names. Each name is a **16-byte fixed-width string**.
- **`REIH`**: Hierarchy and Pivots.
    - **Step 1**: Read `N` bytes (where N is bone count). Each byte is the `num_children` for that bone.
    - **Step 2**: Read `N * 12` bytes. Each 12-byte block is the `Pivot Position (X, Y, Z)` for that bone.
- **`XTRW`**: Vertex Weights. Links `XTRV` vertices to `MANB` bones.
    - **Stride**: 12 Bytes (`VertexID`, `BoneID`, `Weight`).

---

## 5. Shadow Pipeline: `SEMS`, `XTVS`, `CAFS`

Shadow models are optimized versions designed for the lighting engine. They ignore the standard geometry pipeline and use these three tags instead:

| Chunk | Equivalent | Purpose |
| :--- | :--- | :--- |
| **`SEMS`** | `DNER` | Defines which vertices/faces belong to which mesh. |
| **`XTVS`** | `XTRV` | Light vertex pool (Only `X, Y, Z` floats, stride 12). |
| **`CAFS`** | `ECAF` | Face pool with normals (indices + normal vector). |

---

## 6. Mathematical Keys (The Secret Sauce)

### The "Great Swizzle" (Coordinate Swap)
IGI 2 stores coordinates in a specific format that differs from most engines:
- **Internal**: X, Z, -Y
- **Standard (Unity/Blender)**: X, Y, Z
- **Manual Step**: When you read a point `(px, py, pz)`, map it as follows:
    - `Result X = px`
    - `Result Y = pz`
    - `Result Z = -py`

### The Scale Constant
You will often see the number **40.96** in IGI 2 data. This is the global scaling factor between "Local Mesh Units" and "Game World Units". If your model looks 40 times too small/big, check this factor!

---

## 7. Tutorial: Building Your Own MEF Viewer

Follow these steps to parse a model from scratch in your language of choice (Python, C++, JS):

### Stage 1: The Scanner
1. Read the first 4 bytes. If `ILFF` or `HSEM`, proceed.
2. Loop through the file. Read 4 bytes as a String (`Tag`), then 4 bytes as an Integer (`Size`).
3. Store the file offset of each Tag in a Map (`Map<Tag, Offset>`).
4. Skip `Size` bytes and repeat until you reach the end of the file.

### Stage 2: Mesh Reconstruction
1. Look up the `DNER` data.
2. For each Material/Part defined in `DNER`:
    - Go to the `XTRV` pool at the specified `Start Vertex`.
    - Read `Vertex Count` entries (32 bytes each).
    - Go to the `ECAF` pool at the specified `Start Face`.
    - Read `Face Count` entries (6 bytes each).
3. **Important**: The face indices in `ECAF` are relative to the *whole* vertex pool, so subtract the `Start Vertex` from the index to get local Part indices.

### Stage 3: Rendering
1. Bind your reconstructed vertex and face arrays to your Graphics API.
2. Apply the **Swizzle** (`x, z, -y`).
3. Set the winding order to **Counter-Clockwise**.
4. The model should now appear perfectly centered at `(0,0,0)`.
