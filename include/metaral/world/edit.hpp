#pragma once

#include "metaral/core/coords.hpp"
#include "metaral/world/world.hpp"

namespace metaral::world {

enum class EditMode {
    Dig,   // remove solid -> air
    Fill,  // add solid
    Paint, // change material but keep solid/empty
};

struct EditStats {
    std::size_t voxels_touched = 0;
    std::size_t voxels_changed = 0;
};

void apply_spherical_brush(World& world,
                           const core::CoordinateConfig& cfg,
                           const core::PlanetPosition& center,
                           float radius_m,
                           EditMode mode,
                           MaterialId material,
                           EditStats* out_stats = nullptr);

} // namespace metaral::world

