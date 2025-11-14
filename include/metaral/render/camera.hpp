#pragma once

#include "metaral/core/coords.hpp"

#include <numbers>

namespace metaral::render {

struct Camera {
    core::PlanetPosition position{};
    core::PlanetPosition forward{};
    core::PlanetPosition up{};
    float fov_y_radians = std::numbers::pi_v<float> / 3.0f; // 60 degrees
};

struct OrbitParameters {
    float altitude_m = 0.0f;        // height above surface
    float latitude_radians = 0.0f;  // [-pi/2, pi/2]
    float longitude_radians = 0.0f; // [-pi, pi]
};

Camera make_orbit_camera(const core::CoordinateConfig& cfg,
                         const OrbitParameters& orbit,
                         float yaw_offset_radians = 0.0f,
                         float pitch_offset_radians = 0.0f);

} // namespace metaral::render
