# Metaral Engine

Metaral Engine is a voxel-based world simulation and rendering engine focused on **plausible, emergent interactions between materials at planetary scale**.

It’s the tech stack behind the game **Metaral: Neventinue**, where the first simulated planet (*Neventinue*) lives. Players can build freely (a-la Minecraft), but the real star is the **world itself**: lava cools into rock, water vaporizes into steam, materials react with each other, and the planet behaves like a dense, rule-driven sandbox.

---

## Build note (Linux/Wayland)

If SDL3 fails to configure with “could not find X11 or Wayland development libraries”, the smallest Wayland set on Debian/Ubuntu is:

```bash
sudo apt install libwayland-dev libxkbcommon-dev libdrm-dev libgbm-dev libegl1-mesa-dev
```

Vulkan targets are enabled by default; install the headers/loader plus `glslc` (Shaderc) for shader compilation:

```bash
sudo apt install libvulkan-dev shaderc
```
```>***

---

## High-Level Module Breakdown

The repository is organized around a clear separation of concerns between **core types**, **world representation**, **simulation**, and **rendering**, plus tooling and platform glue.

Proposed top-level layout:

```text
metaral-engine/
  include/metaral/...
  src/
    core/
    world/
    sim/
    render/
    platform/
    io/
    tools/
  examples/
  cmake/
  CMakeLists.txt
  README.md
````

---

### `core/` – Foundations, Math & Coordinate Spaces

Low-level building blocks that everything else leans on:

* **Math & geometry**

  * Vector/matrix types (or wrappers over glm)
  * Basic transforms, interpolation, noise, and random utilities
* **Coordinate space definitions**

  * Type-safe structs or aliases for the different spaces:

    * `LocalVoxelCoord`   (u8/u16 per axis, inside a chunk)
    * `ChunkCoord`        (int32 per axis)
    * `WorldVoxelCoord`   (int32 per axis, global voxel index)
    * `PlanetPosition`    (float3 in meters, origin at planet center)
    * `RenderPosition`    (float3 in meters, camera-relative)
* **Config & diagnostics**

  * Engine-wide config (`voxel_size_m`, `chunk_size`, `planet_radius_m`, etc.)
  * Logging, assertions, profiling hooks
* **Shared data structures**

  * Grids, small fixed-size arrays, pools used by `world/` and `sim/`
  * Generic grids / sparse structures shared across modules

> Think: no game logic here. Just the “standard library” of Metaral Engine, with coordinate spaces explicit and hard to mix up—you should never quietly add a `ChunkCoord` to a `PlanetPosition` without an obvious conversion.

Example (conceptual API):

```cpp
namespace metaral::core {

struct LocalVoxelCoord  { uint8_t x, y, z; };   // [0, chunk_size)
struct ChunkCoord       { int32_t x, y, z; };   // chunk index in the grid
struct WorldVoxelCoord  { int32_t x, y, z; };   // global voxel index
struct PlanetPosition   { float x, y, z; };     // meters, origin at planet center

struct CoordinateConfig {
    float   voxel_size_m;      // e.g. 0.10 m per voxel
    int32_t chunk_size;        // e.g. 32 or 64 voxels per side
    float   planet_radius_m;   // base radius of the planet
    // derived fields like planet radius in voxels can be cached here
};

} // namespace metaral::core
```

---

### `world/` – Planetary Voxel Topology, Data & Coordinate Transforms

Everything about **how the planet is stored, chunked, and how you move between spaces**.

#### Coordinate Spaces Overview

Metaral uses a simple, hierarchical set of spaces:

1. **Local space (`LocalVoxelCoord`)**

    * Integer coordinates inside a chunk: `[0, chunk_size)` per axis.
    * Origin at one corner of the chunk.
    * Cheap to store; used heavily in tight loops.

2. **Chunk space (`ChunkCoord`)**

    * Integer chunk indices in a 3D grid.
    * `(0,0,0)` is an arbitrary reference chunk near the planet’s center.
    * Each chunk covers `chunk_size³` voxels.

3. **World voxel space (`WorldVoxelCoord`)**

    * Integer index for every voxel in the world grid.
    * Derived from chunk + local coordinates.
    * Conceptually infinite, but practically bounded by the planet and LOD scheme.

4. **Planetary space (`PlanetPosition`)**

    * Continuous 3D position in **meters**, origin at planet center.
    * Used for physics-ish math, simulation, and raymarching.
    * From here you can easily derive radius, surface normal, and height above/below the reference radius.

5. **Render / camera space (`RenderPosition`)**

    * Planetary position transformed into camera-relative space for rendering.
    * `render_pos = view_matrix * planet_pos4`.

#### Core Transform Relationships

1. **Local → World voxel**

```cpp
WorldVoxelCoord world_voxel(ChunkCoord C, LocalVoxelCoord L, int chunk_size) {
    return {
        C.x * chunk_size + L.x,
        C.y * chunk_size + L.y,
        C.z * chunk_size + L.z
    };
}
```

This is purely integer math: “which voxel in the global grid is this?”

2. **World voxel → Planetary**

We define:

* `voxel_size_m`  – how many meters each voxel edge represents.
* `planet_center_offset_voxels` – where the planet center sits in world-voxel space, so that the sphere is roughly centered in the grid.

```cpp
PlanetPosition voxel_center_to_planet(
    WorldVoxelCoord V,
    const CoordinateConfig& cfg,
    const WorldVoxelCoord& planet_center_offset_voxels)
{
    // Offset so that (0,0,0) in planetary space is the planet center.
    int32_t vx = V.x - planet_center_offset_voxels.x;
    int32_t vy = V.y - planet_center_offset_voxels.y;
    int32_t vz = V.z - planet_center_offset_voxels.z;

    // Convert from voxel index to meters.
    float sx = static_cast<float>(vx) * cfg.voxel_size_m;
    float sy = static_cast<float>(vy) * cfg.voxel_size_m;
    float sz = static_cast<float>(vz) * cfg.voxel_size_m;

    return { sx, sy, sz };
}
```

Think of it as:

> **World voxel → shift so the planet center is at the origin → scale by meters-per-voxel.**

3. **Planetary → World voxel (for lookups)**

```cpp
WorldVoxelCoord planet_to_world_voxel(
    PlanetPosition P,
    const CoordinateConfig& cfg,
    const WorldVoxelCoord& planet_center_offset_voxels)
{
    float inv = 1.0f / cfg.voxel_size_m;
    int32_t vx = static_cast<int32_t>(std::floor(P.x * inv));
    int32_t vy = static_cast<int32_t>(std::floor(P.y * inv));
    int32_t vz = static_cast<int32_t>(std::floor(P.z * inv));

    return {
        vx + planet_center_offset_voxels.x,
        vy + planet_center_offset_voxels.y,
        vz + planet_center_offset_voxels.z
    };
}
```

Lets you determine which voxel(s) a raymarch step hits or “what material lives at this planetary position?”

4. **Planetary height & surface normal**

```cpp
float r      = std::sqrt(P.x*P.x + P.y*P.y + P.z*P.z);
float height = r - cfg.planet_radius_m;
PlanetPosition normal = { P.x / r, P.y / r, P.z / r };
```

* `height > 0` → above nominal surface
* `height < 0` → inside the crust/mantle
* `normal` is the “up” direction for gravity, erosion, camera controls, particles, etc.

#### Data Layout, Chunking & Fields

* **World topology**

  * Planet-scale coordinate systems (spherical world, local chunk coords, etc.)
  * Conversions between “planet surface position” and voxel indices
* **Chunking & storage**

  * Chunk/brick representations (e.g. 32³ or 64³ blocks)
  * Streaming in/out of chunks based on camera and simulation interest
* **Voxel & field representation**

  * Base voxel/material IDs
  * Optional TSDF/SDF/occupancy fields per chunk for smooth raymarching
  * Metadata for temperature, pressure, etc. (if stored alongside voxels)

Example classes:

* `metaral::world::World`
* `metaral::world::Chunk`
* `metaral::world::PlanetTopology`

#### How This Plays With Chunks & Simulation

* `world/` owns **where data lives** (chunks + voxel indices) and the **integer transforms** between local/chunk/world-voxel.
* It also exposes helpers to step into `PlanetPosition` and back.
* `sim/` and `render/` almost never care about `ChunkCoord` or `LocalVoxelCoord` directly:

  * `sim/` consumes “sample at this planetary position” or iterates over ranges of `WorldVoxelCoord`.
  * `render/` raymarches in planetary space, then calls back into `world/`/`sim/` to sample fields.

---

### `sim/` – Material, Thermal & Environmental Simulation

This is the **“plausible physics” heart** of Metaral Engine.

* **Material system**

  * Material definitions: density, phases, melting/boiling “pseudo” thresholds
  * Interaction rules: lava + water → steam + basalt; gas condensation; etc.
* **State fields**

  * Temperature / energy grids
  * Phase state per voxel (solid/liquid/gas/plasma/other custom)
  * Optional velocity fields for liquids/gases (if doing fluid-ish behavior)
* **Simulation steps**

  * Discrete timesteps that update material states based on neighbor interactions, energy transfer, and simple diffusion/convection/advection schemes
  * Hooks for game-level events (player digging, explosions, etc.)

It leans heavily on the transforms defined in `core/` + `world/` to walk chunks efficiently in integer space, while reasoning about gravity wells, surface normals, and heights in planetary space.

Example classes:

* `metaral::sim::MaterialSystem`
* `metaral::sim::ThermalSystem`
* `metaral::sim::WorldSimulator`
* `metaral::sim::FluidSim` / `metaral::sim::DensityGrid` — SPH-inspired fluid scaffolding that will migrate to Vulkan compute; current CPU path keeps the API stable.

> High-level idea: **`world/` owns the data layout**, **`sim/` owns how that data evolves over time**.

---

### `render/` – Raymarching & Visual Representation

All visual output lives here, with a focus on **raymarching over fields derived from `world/` + `sim/`**.

* **View & camera**

  * Camera representation, frustum, projection, view transforms
  * Planetary position → camera space (`RenderPosition`)
* **Field extraction**

  * Building render-time scalar fields (distance/occupancy) from voxel data
  * Level-of-detail and sampling strategies
* **Raymarcher**

  * CPU/GPU raymarching core operating directly in planetary space, sampling the underlying materials/fields
  * Material-based shading (lava glow, steam translucency, etc.)
  * Normal estimation from fields for lighting
* **Rendering backend integration**

  * Vulkan pipelines, descriptor sets, buffers
  * Minimal abstraction so the engine can be embedded in different frontends

Example classes:

* `metaral::render::RaymarchRenderer`
* `metaral::render::Camera`
* `metaral::render::FrameGraph`

---

### `platform/` – Windowing, Input & OS Abstraction

The thin layer that **talks to the OS** and feeds the engine:

* **Window & context**

  * SDL3 or similar for window creation, input, and Vulkan surface setup
* **Timing**

  * High-precision clock, fixed/variable timestep support
* **Input**

  * Keyboard/mouse/gamepad events normalized into an engine-friendly format

Example:

* `metaral::platform::Window`
* `metaral::platform::InputState`

> The idea is that `platform/` is swappable: desktop test harness now, maybe something else later.

---

### `io/` – Persistence, Assets & Debug Data

Structured in/out for both gameplay data and dev tooling:

* **World persistence**

  * Save/load of voxel chunks, simulation fields, and material definitions
* **Config & assets**

  * Engine config (JSON/TOML/etc.)
  * Material definition file readers
* **Debug dumps**

  * Binary or textual dumps of bricks/chunks for inspection
  * Hooks for external tools (e.g. “brick probe” utilities)

Example:

* `metaral::io::WorldSerializer`
* `metaral::io::MaterialDatabaseLoader`

---

### `tools/` – Standalone Utilities & Inspection

Small binaries linked against `metaral-engine` to introspect and debug:

* **Brick/Chunk probe tools**

  * Sample TSDF/occupancy/temperature in a given brick
  * Output human-readable or image/volume data for visualization
* **Benchmarking tools**

  * Performance tests for raymarching, chunk streaming, and simulation steps
* **Content tools**

  * (Later) simple editors or generators for material sets, presets, etc.

Example targets:

* `tools/debug_dump_brick`
* `tools/bench_raymarch`

---

### `examples/` – Minimal Embedding & Playground Apps

Small, focused programs that show **how to embed Metaral Engine**:

* `examples/vulkan_viewer`

  * Create a window, load a world, render with the Vulkan analytic sphere raymarcher
* `examples/sim_sandbox`

  * Simple UI to spawn materials (lava, water, etc.) and watch them interact
* `examples/planet_flythrough`

  * Fly around Neventinue’s surface to test streaming + rendering

These examples serve as both **developer documentation** and **experimental sandboxes** while the main game (`Metaral: Neventinue`) evolves in its own repo.

---
