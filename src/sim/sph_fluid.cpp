#include "metaral/sim/sph_fluid.hpp"

#include "metaral/core/coords.hpp"

#include <algorithm>

namespace metaral::sim {

namespace {

inline float dot3(const core::PlanetPosition& a, const core::PlanetPosition& b) noexcept {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline core::PlanetPosition operator+(const core::PlanetPosition& a,
                                      const core::PlanetPosition& b) noexcept {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

inline core::PlanetPosition operator-(const core::PlanetPosition& a,
                                      const core::PlanetPosition& b) noexcept {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

inline core::PlanetPosition operator*(const core::PlanetPosition& a, float s) noexcept {
    return {a.x * s, a.y * s, a.z * s};
}

inline core::PlanetPosition operator*(float s, const core::PlanetPosition& a) noexcept {
    return a * s;
}

} // namespace

FluidSim::FluidSim(const core::CoordinateConfig& coords, const SphParams& params)
    : coords_(coords), params_(params) {}

void FluidSim::spawn_particles(std::span<const FluidParticle> particles) {
    particles_.insert(particles_.end(), particles.begin(), particles.end());
}

void FluidSim::clear() {
    particles_.clear();
}

void FluidSim::integrate(float dt) {
    for (auto& p : particles_) {
        // Pull toward/away from the planet center so collisions match the spherical boundary.
        const core::PlanetPosition radial_up = core::surface_normal(p.position);
        p.velocity = p.velocity + radial_up * (params_.gravity * dt);
        p.position = p.position + p.velocity * dt;
    }
}

void FluidSim::resolve_collisions(const ICollisionField* collision) {
    const float damp = params_.collision_damping;

    for (auto& p : particles_) {
        float penetration = 0.0f;
        core::PlanetPosition normal{};

        if (collision) {
            const float sdf = collision->signed_distance(p.position);
            if (sdf < 0.0f) {
                penetration = -sdf;
                normal = collision->surface_normal(p.position);
            }
        } else {
            // Fallback to spherical planet boundary.
            const float r = core::length(p.position);
            const float sdf = r - coords_.planet_radius_m;
            if (sdf < 0.0f) {
                penetration = -sdf;
                normal = core::surface_normal(p.position);
            }
        }

        if (penetration > 0.0f) {
            // Push out of the surface.
            p.position = p.position + normal * penetration;
            // Reflect and damp velocity along the normal.
            const float vn = dot3(p.velocity, normal);
            if (vn < 0.0f) {
                p.velocity = p.velocity - (1.0f + (1.0f - damp)) * vn * normal;
                p.velocity = p.velocity * damp;
            }
        }
    }
}

void FluidSim::step(float dt, const ICollisionField* collision) {
    if (dt <= 0.0f || particles_.empty()) {
        return;
    }

    integrate(dt);
    resolve_collisions(collision);
}

} // namespace metaral::sim
