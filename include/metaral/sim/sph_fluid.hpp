// Inspired by Seb Lague's GPU Fluid-Sim pipeline.
// This is an initial scaffold for a SPH-like particle fluid running on CPU/GPU.
// The current implementation is CPU-only and keeps interfaces narrow so a
// Vulkan compute backend can be swapped in later without touching callers.

#pragma once

#include "metaral/core/coords.hpp"

#include <cstdint>
#include <functional>
#include <span>
#include <vector>

namespace metaral::sim {

struct SphParams {
    float gravity = -10.0f;            // m/s^2, applied along +Y up axis
    float smoothing_radius = 0.2f;     // kernel radius (meters)
    float target_density = 630.0f;     // rest density (kg/m^3)
    float pressure_multiplier = 288.0f;
    float near_pressure_multiplier = 2.15f;
    float viscosity_strength = 0.0f;   // 0 disables viscosity
    float collision_damping = 0.95f;   // velocity multiplier on collision
};

struct FluidParticle {
    core::PlanetPosition position{}; // meters, planet space
    core::PlanetPosition velocity{}; // m/s
};

// Optional collision query used during stepping. Kept as a minimal interface to
// avoid a dependency on the render module (SDF sampling). Callers can supply a
// lambda that samples their own SDF volume.
struct ICollisionField {
    virtual ~ICollisionField() = default;
    // Signed distance in meters; negative is inside solid.
    virtual float signed_distance(const core::PlanetPosition& p) const = 0;
    // Unit surface normal for the solid at position p (only used when inside).
    virtual core::PlanetPosition surface_normal(const core::PlanetPosition& p) const = 0;
};

class FluidSim {
public:
    FluidSim(const core::CoordinateConfig& coords, const SphParams& params);

    void set_params(const SphParams& params) { params_ = params; }
    const SphParams& params() const noexcept { return params_; }

    // Own the particles internally; in the future this can be replaced with GPU
    // buffers. Positions are expected to be in planet space (meters).
    void spawn_particles(std::span<const FluidParticle> particles);
    void clear();

    // Advance simulation by dt seconds. If a collision field is provided,
    // simple collision response is applied; otherwise the planet radius from
    // CoordinateConfig acts as a spherical collider.
    void step(float dt, const ICollisionField* collision = nullptr);

    const std::vector<FluidParticle>& particles() const noexcept { return particles_; }
    std::vector<FluidParticle>& particles_mut() noexcept { return particles_; }

private:
    void integrate(float dt);
    void resolve_collisions(const ICollisionField* collision);

    core::CoordinateConfig coords_;
    SphParams params_;
    std::vector<FluidParticle> particles_;
};

} // namespace metaral::sim
