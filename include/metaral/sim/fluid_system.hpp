// Thin orchestration layer that owns a FluidSim instance plus a few helpers
// for spawning and exporting density to the renderer. This keeps higher-level
// code from dealing with FluidSim internals directly.

#pragma once

#include "metaral/core/coords.hpp"
#include "metaral/sim/density_grid.hpp"
#include "metaral/sim/sph_fluid.hpp"
#include "metaral/world/world.hpp"

#include <cstddef>
#include <vector>

namespace metaral::sim {

class FluidSystem {
public:
    FluidSystem(const core::CoordinateConfig& coords, const SphParams& params, world::MaterialId water_material = 2);

    void set_params(const SphParams& params) { sim_.set_params(params); }
    const SphParams& params() const noexcept { return sim_.params(); }

    // Spawn particles inside a sphere with random jitter. Useful for tests and
    // simple waterfalls.
    void spawn_sphere(const core::PlanetPosition& center,
                      float radius_m,
                      std::size_t count);

    // Spawn one particle per voxel whose material matches water_material_id_
    // within the provided chunk bounds. This is intentionally naive; callers
    // should limit the region.
    void spawn_from_world(const world::World& world,
                          const core::ChunkCoord& min_chunk,
                          const core::ChunkCoord& max_chunk);

    void clear_particles() { sim_.clear(); }

    void step(float dt, const ICollisionField* collision = nullptr) { sim_.step(dt, collision); }

    const FluidSim& sim() const noexcept { return sim_; }
    FluidSim& sim_mut() noexcept { return sim_; }

    // Builds a coarse density grid near a region of interest, forwarding to the
    // helper in density_grid.cpp.
    void build_density_grid(const core::PlanetPosition& min_p,
                            const core::PlanetPosition& max_p,
                            std::uint32_t max_dim,
                            DensityGrid& out) const;

    world::MaterialId water_material_id() const noexcept { return water_material_id_; }

private:
    core::CoordinateConfig coords_;
    FluidSim sim_;
    world::MaterialId water_material_id_;
};

} // namespace metaral::sim
