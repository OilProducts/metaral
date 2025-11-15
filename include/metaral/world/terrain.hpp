#pragma once

#include "metaral/core/coords.hpp"
#include "metaral/world/world.hpp"

namespace metaral::world::terrain {

// Analytic signed-distance field for the noisy sphere terrain.
// Negative inside the terrain, positive outside.
float terrain_signed_distance(const core::PlanetPosition& pos,
                              const core::CoordinateConfig& cfg) noexcept;

// Simple planet generator that writes voxel materials into a World using the
// analytic terrain field.
void generate_planet(World& world,
                     int chunk_radius,
                     const core::CoordinateConfig& cfg,
                     MaterialId solid_material = 1,
                     MaterialId empty_material = 0);

} // namespace metaral::world::terrain

