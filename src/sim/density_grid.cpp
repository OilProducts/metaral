#include "metaral/sim/density_grid.hpp"

#include <algorithm>
#include <cmath>

namespace metaral::sim {

namespace {

inline std::size_t linear_index(std::uint32_t dim_x,
                                std::uint32_t dim_y,
                                std::uint32_t x,
                                std::uint32_t y,
                                std::uint32_t z) noexcept
{
    return (static_cast<std::size_t>(z) * dim_y + static_cast<std::size_t>(y)) *
               dim_x +
           static_cast<std::size_t>(x);
}

} // namespace

void splat_simple_density(const FluidSim& sim,
                          const core::PlanetPosition& min_p,
                          const core::PlanetPosition& max_p,
                          std::uint32_t max_dim,
                          DensityGrid& out)
{
    const float dx = max_p.x - min_p.x;
    const float dy = max_p.y - min_p.y;
    const float dz = max_p.z - min_p.z;
    const float max_extent = std::max({dx, dy, dz, 1e-3f});

    const float cell_size = max_extent / static_cast<float>(std::max<std::uint32_t>(1, max_dim));
    const auto to_dim = [cell_size](float extent) -> std::uint32_t {
        return static_cast<std::uint32_t>(std::ceil(extent / cell_size)) + 1u;
    };

    const std::uint32_t dim_x = std::max<std::uint32_t>(1, to_dim(dx));
    const std::uint32_t dim_y = std::max<std::uint32_t>(1, to_dim(dy));
    const std::uint32_t dim_z = std::max<std::uint32_t>(1, to_dim(dz));

    out.min_corner = min_p;
    out.cell_size = cell_size;
    out.dim_x = dim_x;
    out.dim_y = dim_y;
    out.dim_z = dim_z;
    out.values.assign(static_cast<std::size_t>(dim_x) * dim_y * dim_z, 0.0f);

    for (const auto& p : sim.particles()) {
        const float lx = (p.position.x - min_p.x) / cell_size;
        const float ly = (p.position.y - min_p.y) / cell_size;
        const float lz = (p.position.z - min_p.z) / cell_size;

        const int ix = static_cast<int>(std::floor(lx));
        const int iy = static_cast<int>(std::floor(ly));
        const int iz = static_cast<int>(std::floor(lz));

        if (ix < 0 || iy < 0 || iz < 0 ||
            ix >= static_cast<int>(dim_x) ||
            iy >= static_cast<int>(dim_y) ||
            iz >= static_cast<int>(dim_z)) {
            continue;
        }

        const std::size_t idx =
            linear_index(dim_x, dim_y,
                         static_cast<std::uint32_t>(ix),
                         static_cast<std::uint32_t>(iy),
                         static_cast<std::uint32_t>(iz));
        out.values[idx] += 1.0f;
    }
}

} // namespace metaral::sim
