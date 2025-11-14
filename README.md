# Metaral Engine

Metaral Engine is a voxel-based world simulation and rendering engine focused on **plausible, emergent interactions between materials at planetary scale**.

It’s the tech stack behind the game **Metaral: Neventinue**, where the first simulated planet (*Neventinue*) lives. Players can build freely (a-la Minecraft), but the real star is the **world itself**: lava cools into rock, water vaporizes into steam, materials react with each other, and the planet behaves like a dense, rule-driven sandbox.

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

### `core/` – Foundations & Shared Infrastructure

Low-level building blocks that everything else leans on:

* **Math & geometry**

  * Vector/matrix types (or wrappers over glm)
  * Transform utilities (local ↔ world ↔ planetary coordinates)
  * Noise, interpolation, random utilities
* **Config & diagnostics**

  * Engine-wide configuration structs
  * Logging, assertions, profiling hooks
* **Data structures**

  * Small allocators/pools used by chunk storage and simulation grids
  * Generic grids / sparse structures shared by `world/` and `sim/`

> Think: no game logic here. Just the “standard library” of Metaral Engine.

---

### `world/` – Planetary Voxel Topology & Data

Everything about **how the planet is stored and addressed**:

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

  * Discrete timesteps that update material states based on:

    * Neighbor interactions
    * Energy transfer, cooling/heating
    * Simple diffusion/convection/advection schemes
  * Hooks for game-level events (player digging, explosions, etc.)

Example classes:

* `metaral::sim::MaterialSystem`
* `metaral::sim::ThermalSystem`
* `metaral::sim::WorldSimulator`

> High-level idea: **`world/` owns the data layout**, **`sim/` owns how that data evolves over time**.

---

### `render/` – Raymarching & Visual Representation

All visual output lives here, with a focus on **raymarching over fields derived from `world/` + `sim/`**.

* **View & camera**

  * Camera representation, frustum, projection, view transforms
* **Field extraction**

  * Building render-time scalar fields (distance/occupancy) from voxel data
  * Level-of-detail and sampling strategies
* **Raymarcher**

  * CPU/GPU raymarching core
  * Material-based shading (lava glow, steam translucency, etc.)
  * Normal estimation from fields for lighting
* **Rendering backend integration**

  * Vulkan pipelines, descriptor sets, buffers
  * Minimal abstraction so the engine can be embedded in different frontends

Example classes:

* `metaral::render::RaymarchRenderer`
* `metaral::render::Camera`
* `metaral::render::FrameGraph` (if you go that route)

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
  * Material definition files readers
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

* `examples/minimal_viewer`

  * Create a window, load a world, render with the raymarcher
* `examples/sim_sandbox`

  * Simple UI to spawn materials (lava, water, etc.) and watch them interact
* `examples/planet_flythrough`

  * Fly around Neventinue’s surface to test streaming + rendering

These examples serve as both **developer documentation** and **experimental sandboxes** while the main game (`Metaral: Neventinue`) evolves in its own repo.

---


