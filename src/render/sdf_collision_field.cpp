#include "metaral/render/sdf_collision_field.hpp"

namespace metaral::render {

namespace {

inline core::PlanetPosition normalize(const core::PlanetPosition& v) {
    const float len = metaral::core::length(v);
    if (len < 1e-6f) {
        return {0.0f, 1.0f, 0.0f};
    }
    const float inv = 1.0f / len;
    return {v.x * inv, v.y * inv, v.z * inv};
}

} // namespace

float SdfCollisionField::signed_distance(const core::PlanetPosition& p) const {
    if (!grid_ || grid_->dim == 0) {
        // Positive distance if grid is absent; keeps particles free-falling.
        return 1e9f;
    }
    return sample_sdf(*grid_, p);
}

core::PlanetPosition SdfCollisionField::surface_normal(const core::PlanetPosition& p) const {
    if (!grid_ || grid_->dim == 0) {
        return {0.0f, 1.0f, 0.0f};
    }

    constexpr float h = 0.05f; // small offset in meters
    const float dx = sample_sdf(*grid_, {p.x + h, p.y, p.z}) - sample_sdf(*grid_, {p.x - h, p.y, p.z});
    const float dy = sample_sdf(*grid_, {p.x, p.y + h, p.z}) - sample_sdf(*grid_, {p.x, p.y - h, p.z});
    const float dz = sample_sdf(*grid_, {p.x, p.y, p.z + h}) - sample_sdf(*grid_, {p.x, p.y, p.z - h});

    return normalize({dx, dy, dz});
}

} // namespace metaral::render
