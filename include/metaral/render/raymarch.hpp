#pragma once

#include "metaral/core/coords.hpp"
#include "metaral/render/camera.hpp"
#include "metaral/world/world.hpp"

#include <cstdint>
#include <vector>

namespace metaral::render {

struct RaymarchSettings {
    float max_distance_m = 1000.0f;
    float step_size_m = 0.5f;
};

struct RayResult {
    bool hit = false;
    float distance = 0.0f;
    float height = 0.0f;
};

RayResult march_ray(const world::World& world,
                    const core::PlanetPosition& origin,
                    const core::PlanetPosition& direction,
                    const RaymarchSettings& settings);

} // namespace metaral::render
