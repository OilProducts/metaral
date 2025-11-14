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

float dot(const core::PlanetPosition& a, const core::PlanetPosition& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

core::PlanetPosition cross(const core::PlanetPosition& a, const core::PlanetPosition& b) {
    return {
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x,
    };
}

core::PlanetPosition rotate_about_axis(const core::PlanetPosition& v,
                                       const core::PlanetPosition& axis_normalized,
                                       float angle) {
    // Rodrigues' rotation formula (axis assumed normalized)
    const float cos_a = std::cos(angle);
    const float sin_a = std::sin(angle);

    const core::PlanetPosition term1{v.x * cos_a, v.y * cos_a, v.z * cos_a};
    const core::PlanetPosition term2 = cross(axis_normalized, v);
    const float d = dot(axis_normalized, v);
    const core::PlanetPosition term3{
        axis_normalized.x * d * (1.0f - cos_a),
        axis_normalized.y * d * (1.0f - cos_a),
        axis_normalized.z * d * (1.0f - cos_a),
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

    const core::PlanetPosition target =
        lat_long_to_cartesian(target_radius,
                              orbit.latitude_radians,
                              orbit.longitude_radians);
    const core::PlanetPosition camera_pos =
        lat_long_to_cartesian(camera_radius,
                              orbit.latitude_radians,
                              orbit.longitude_radians);

    // Radial "world up" (out from planet center)
    core::PlanetPosition world_up = normalized(target);

    // Base forward: camera -> target (will be ~ -world_up)
    core::PlanetPosition forward = normalized(core::PlanetPosition{
        target.x - camera_pos.x,
        target.y - camera_pos.y,
        target.z - camera_pos.z,
    });

    // Build a tangent frame using a global up to break degeneracy
    core::PlanetPosition global_up{0.0f, 1.0f, 0.0f};
    if (std::abs(dot(world_up, global_up)) > 0.99f) {
        // Near the poles, pick a different reference
        global_up = {1.0f, 0.0f, 0.0f};
    }

    core::PlanetPosition right = normalized(cross(global_up, world_up));
    core::PlanetPosition up    = normalized(cross(world_up, right));

    // Apply yaw then pitch, updating basis each time
    if (yaw_offset_radians != 0.0f) {
        forward = normalized(rotate_about_axis(forward, up, yaw_offset_radians));
        right   = normalized(cross(forward, up));
    }

    if (pitch_offset_radians != 0.0f) {
        forward = normalized(rotate_about_axis(forward, right, pitch_offset_radians));
        up      = normalized(cross(right, forward));
    }

    Camera camera;
    camera.position = camera_pos;
    camera.forward  = forward;
    camera.up       = up;
    // fov_y_radians kept as default
    return camera;
}


} // namespace metaral::render
