// Simple CPU density splat used as a placeholder until the GPU path is wired.
// Produces a coarse scalar field that can later be consumed by the renderer.

#pragma once

#include "metaral/core/coords.hpp"
#include "metaral/sim/sph_fluid.hpp"

#include <cstdint>
#include <vector>

namespace metaral::sim {

struct DensityGrid {
    std::vector<float> values;          // dim_x * dim_y * dim_z scalars
    core::PlanetPosition min_corner{};  // meters, planet space
    float cell_size = 0.0f;             // meters per cell
    std::uint32_t dim_x = 0;
    std::uint32_t dim_y = 0;
    std::uint32_t dim_z = 0;

    bool empty() const noexcept { return values.empty() || dim_x == 0 || dim_y == 0 || dim_z == 0; }
};

// Build a uniform grid covering [min_p, max_p] (inclusive) with at most
// max_dim cells on the largest axis. For now we just splat a unit value per
// particle into its cell; later this can turn into a proper SPH kernel splat.
void splat_simple_density(const FluidSim& sim,
                          const core::PlanetPosition& min_p,
                          const core::PlanetPosition& max_p,
                          std::uint32_t max_dim,
                          DensityGrid& out);

} // namespace metaral::sim
