#pragma once

#include "metaral/core/coords.hpp"
#include "metaral/world/world.hpp"

#include <cstdint>
#include <vector>

namespace metaral::render {

struct SdfGrid {
    std::vector<float> values;
    std::vector<std::uint16_t> materials; // MaterialId; 0 = empty
    std::uint32_t dim = 0;       // dim^3 samples
    float voxel_size = 0.0f;     // meters between samples
    float half_extent = 0.0f;    // grid spans [-half_extent, +half_extent] in each axis
    float planet_radius = 0.0f;  // for out-of-bounds fallback
};

void build_sdf_grid_from_world(const world::World& world,
                               const core::CoordinateConfig& cfg,
                               SdfGrid& out);

float sample_sdf(const SdfGrid& grid,
                 const core::PlanetPosition& pos);

float raymarch_sdf(const SdfGrid& grid,
                   const core::PlanetPosition& ray_origin,
                   const core::PlanetPosition& ray_dir,
                   float max_dist,
                   float surf_epsilon,
                   int   max_steps,
                   core::PlanetPosition* out_hit_pos = nullptr);

bool raycast_sdf(const SdfGrid& grid,
                 const core::PlanetPosition& ray_origin,
                 const core::PlanetPosition& ray_dir,
                 float max_dist,
                 float surf_epsilon,
                 int   max_steps,
                 core::PlanetPosition& out_hit_pos);

} // namespace metaral::render
