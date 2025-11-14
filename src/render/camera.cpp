#include "metaral/render/camera.hpp"

#include <cmath>

namespace metaral::render {

namespace {
core::PlanetPosition lat_long_to_cartesian(float radius, float latitude, float longitude) {
    const float cos_lat = std::cos(latitude);
    return {
        radius * cos_lat * std::cos(longitude),
        radius * std::sin(latitude),
        radius * cos_lat * std::sin(longitude),
    };
}

core::PlanetPosition normalized(const core::PlanetPosition& v) {
    const float len = metaral::core::length(v);
    if (len < 1e-6f) {
        return {0.0f, 1.0f, 0.0f};
    }
    const float inv = 1.0f / len;
    return {v.x * inv, v.y * inv, v.z * inv};
}

core::PlanetPosition cross(const core::PlanetPosition& a, const core::PlanetPosition& b) {
    return {
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x,
    };
}

core::PlanetPosition rotate_about_axis(const core::PlanetPosition& v,
                                       const core::PlanetPosition& axis,
                                       float angle) {
    // Rodrigues' rotation formula
    const float cos_a = std::cos(angle);
    const float sin_a = std::sin(angle);

    const core::PlanetPosition term1{v.x * cos_a, v.y * cos_a, v.z * cos_a};
    const core::PlanetPosition term2{
        axis.y * v.z - axis.z * v.y,
        axis.z * v.x - axis.x * v.z,
        axis.x * v.y - axis.y * v.x,
    };
    const core::PlanetPosition term3{
        axis.x * (axis.x * v.x + axis.y * v.y + axis.z * v.z) * (1.0f - cos_a),
        axis.y * (axis.x * v.x + axis.y * v.y + axis.z * v.z) * (1.0f - cos_a),
        axis.z * (axis.x * v.x + axis.y * v.y + axis.z * v.z) * (1.0f - cos_a),
    };

    return {
        term1.x + term2.x * sin_a + term3.x,
        term1.y + term2.y * sin_a + term3.y,
        term1.z + term2.z * sin_a + term3.z,
    };
}
}

Camera make_orbit_camera(const core::CoordinateConfig& cfg,
                         const OrbitParameters& orbit,
                         float yaw_offset_radians,
                         float pitch_offset_radians)
{
    const float target_radius = cfg.planet_radius_m;
    const float camera_radius = target_radius + orbit.altitude_m;

    const core::PlanetPosition target = lat_long_to_cartesian(target_radius, orbit.latitude_radians, orbit.longitude_radians);
    const core::PlanetPosition camera_pos = lat_long_to_cartesian(camera_radius, orbit.latitude_radians, orbit.longitude_radians);

    core::PlanetPosition forward = normalized({target.x - camera_pos.x, target.y - camera_pos.y, target.z - camera_pos.z});
    core::PlanetPosition up = normalized(target);

    const core::PlanetPosition right = normalized(cross(forward, up));
    up = normalized(cross(right, forward));

    core::PlanetPosition adjusted_forward = forward;
    if (yaw_offset_radians != 0.0f) {
        adjusted_forward = rotate_about_axis(adjusted_forward, up, yaw_offset_radians);
    }
    if (pitch_offset_radians != 0.0f) {
        adjusted_forward = rotate_about_axis(adjusted_forward, right, pitch_offset_radians);
    }
    adjusted_forward = normalized(adjusted_forward);

    Camera camera;
    camera.position = camera_pos;
    camera.forward = adjusted_forward;
    camera.up = up;
    return camera;
}

} // namespace metaral::render
