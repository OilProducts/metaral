#pragma once

#include "metaral/core/coords.hpp"

#include <cmath>

namespace metaral::render {

struct Camera {
    core::PlanetPosition position{};
    core::PlanetPosition forward{};
    core::PlanetPosition up{};
    float fov_y_radians = 60.0f * static_cast<float>(M_PI) / 180.0f;
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
