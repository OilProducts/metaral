#pragma once

#include "metaral/core/coords.hpp"
#include "metaral/world/world.hpp"

#include <cstdint>
#include <vector>

namespace metaral::render {

// Default fraction of the SDF voxel size used as an iso-surface offset when
// rendering or raycasting. This controls how "rounded" corners appear.
constexpr float kDefaultSdfIsoFraction = .25f;

struct SdfGrid {
    std::vector<float> values;
    std::vector<std::uint16_t> materials; // MaterialId; 0 = empty
    std::vector<std::uint8_t> occupancy;  // 1 = solid, 0 = empty; matches values/materials layout
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
                   core::PlanetPosition* out_hit_pos = nullptr,
                   float iso_offset = 0.0f);

bool raycast_sdf(const SdfGrid& grid,
                 const core::PlanetPosition& ray_origin,
                 const core::PlanetPosition& ray_dir,
                 float max_dist,
                 float surf_epsilon,
                 int   max_steps,
                 core::PlanetPosition& out_hit_pos,
                 float iso_offset = 0.0f);

void update_sdf_region_from_world(const world::World& world,
                                  const core::CoordinateConfig& cfg,
                                  const core::PlanetPosition& min_p,
                                  const core::PlanetPosition& max_p,
                                  SdfGrid& out,
                                  float* sdf_gpu = nullptr,
                                  std::uint32_t* mat_gpu = nullptr);

} // namespace metaral::render
