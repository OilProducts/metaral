#include "metaral/render/sdf_grid.hpp"

#include "metaral/world/chunk.hpp"

#include <cmath>

namespace metaral::render {

namespace {

constexpr float kSdfVoxelSizeMeters = 1.0f;
constexpr float kSdfMarginVoxels = 4.0f;

inline std::size_t linear_index(std::uint32_t dim,
                                std::uint32_t x,
                                std::uint32_t y,
                                std::uint32_t z) noexcept
{
    return (static_cast<std::size_t>(z) * dim +
            static_cast<std::size_t>(y)) *
               dim +
           static_cast<std::size_t>(x);
}

} // namespace

void build_sdf_grid_from_world(const world::World& world,
                               const core::CoordinateConfig& cfg,
                               SdfGrid& out)
{
    const float voxel_size = kSdfVoxelSizeMeters;

    // Expand the SDF grid so that the sampled region comfortably covers the
    // planet plus a small margin. The actual surface detail comes from the
    // voxel world, not from analytic noise here.
    const float half_extent =
        cfg.planet_radius_m + kSdfMarginVoxels * voxel_size;

    const float diameter = half_extent * 2.0f;
    const std::uint32_t dim =
        static_cast<std::uint32_t>(std::ceil(diameter / voxel_size));

    const std::size_t cell_count =
        static_cast<std::size_t>(dim) * dim * dim;

    out.dim = dim;
    out.voxel_size = voxel_size;
    out.half_extent = half_extent;
    out.planet_radius = cfg.planet_radius_m;
    out.values.assign(cell_count, 0.0f);
    out.materials.assign(cell_count, 0);

    const float grid_min = -half_extent;

    for (std::uint32_t z_index = 0; z_index < dim; ++z_index) {
        for (std::uint32_t y_index = 0; y_index < dim; ++y_index) {
            for (std::uint32_t x_index = 0; x_index < dim; ++x_index) {
                const float x = grid_min + (static_cast<float>(x_index) + 0.5f) * voxel_size;
                const float y = grid_min + (static_cast<float>(y_index) + 0.5f) * voxel_size;
                const float z = grid_min + (static_cast<float>(z_index) + 0.5f) * voxel_size;

                const core::PlanetPosition pos{x, y, z};
                const core::WorldVoxelCoord world_voxel =
                    core::to_world_voxel(pos, cfg);

                const world::Voxel* voxel = world.find_voxel(world_voxel);
                const bool solid = voxel && voxel->material != 0;

                // Very simple binary "SDF": just encode inside/outside with a
                // fixed magnitude proportional to the world voxel size.
                const float distance =
                    solid ? -0.5f * cfg.voxel_size_m : 0.5f * cfg.voxel_size_m;

                const std::size_t idx =
                    linear_index(dim, x_index, y_index, z_index);

                out.values[idx] = distance;

                std::uint16_t material_id = 0;
                if (solid && voxel) {
                    material_id = static_cast<std::uint16_t>(voxel->material);
                }
                out.materials[idx] = material_id;
            }
        }
    }
}

float sample_sdf(const SdfGrid& grid,
                 const core::PlanetPosition& pos)
{
    const float half_extent = grid.half_extent;
    const float voxel_size = grid.voxel_size;
    const std::uint32_t dim = grid.dim;

    if (dim == 0 || voxel_size <= 0.0f) {
        return metaral::core::length(pos) - grid.planet_radius;
    }

    const float dim_f = static_cast<float>(dim);

    // Map position into grid coordinates, matching the GLSL sample_sdf mapping.
    const float cx = (pos.x + half_extent) / voxel_size - 0.5f;
    const float cy = (pos.y + half_extent) / voxel_size - 0.5f;
    const float cz = (pos.z + half_extent) / voxel_size - 0.5f;

    const float bx = std::floor(cx);
    const float by = std::floor(cy);
    const float bz = std::floor(cz);

    const float fx = cx - bx;
    const float fy = cy - by;
    const float fz = cz - bz;

    const int i0x = static_cast<int>(bx);
    const int i0y = static_cast<int>(by);
    const int i0z = static_cast<int>(bz);
    const int i1x = i0x + 1;
    const int i1y = i0y + 1;
    const int i1z = i0z + 1;

    const int dim_i = static_cast<int>(dim_f);

    if (i0x < 0 || i0y < 0 || i0z < 0 ||
        i1x >= dim_i || i1y >= dim_i || i1z >= dim_i) {
        return metaral::core::length(pos) - grid.planet_radius;
    }

    const auto idx000 = linear_index(dim,
                                     static_cast<std::uint32_t>(i0x),
                                     static_cast<std::uint32_t>(i0y),
                                     static_cast<std::uint32_t>(i0z));
    const auto idx100 = linear_index(dim,
                                     static_cast<std::uint32_t>(i1x),
                                     static_cast<std::uint32_t>(i0y),
                                     static_cast<std::uint32_t>(i0z));
    const auto idx010 = linear_index(dim,
                                     static_cast<std::uint32_t>(i0x),
                                     static_cast<std::uint32_t>(i1y),
                                     static_cast<std::uint32_t>(i0z));
    const auto idx110 = linear_index(dim,
                                     static_cast<std::uint32_t>(i1x),
                                     static_cast<std::uint32_t>(i1y),
                                     static_cast<std::uint32_t>(i0z));
    const auto idx001 = linear_index(dim,
                                     static_cast<std::uint32_t>(i0x),
                                     static_cast<std::uint32_t>(i0y),
                                     static_cast<std::uint32_t>(i1z));
    const auto idx101 = linear_index(dim,
                                     static_cast<std::uint32_t>(i1x),
                                     static_cast<std::uint32_t>(i0y),
                                     static_cast<std::uint32_t>(i1z));
    const auto idx011 = linear_index(dim,
                                     static_cast<std::uint32_t>(i0x),
                                     static_cast<std::uint32_t>(i1y),
                                     static_cast<std::uint32_t>(i1z));
    const auto idx111 = linear_index(dim,
                                     static_cast<std::uint32_t>(i1x),
                                     static_cast<std::uint32_t>(i1y),
                                     static_cast<std::uint32_t>(i1z));

    const float v000 = grid.values[idx000];
    const float v100 = grid.values[idx100];
    const float v010 = grid.values[idx010];
    const float v110 = grid.values[idx110];
    const float v001 = grid.values[idx001];
    const float v101 = grid.values[idx101];
    const float v011 = grid.values[idx011];
    const float v111 = grid.values[idx111];

    const float vx00 = v000 + (v100 - v000) * fx;
    const float vx10 = v010 + (v110 - v010) * fx;
    const float vx01 = v001 + (v101 - v001) * fx;
    const float vx11 = v011 + (v111 - v011) * fx;

    const float vy0 = vx00 + (vx10 - vx00) * fy;
    const float vy1 = vx01 + (vx11 - vx01) * fy;

    return vy0 + (vy1 - vy0) * fz;
}

float raymarch_sdf(const SdfGrid& grid,
                   const core::PlanetPosition& ray_origin,
                   const core::PlanetPosition& ray_dir,
                   float max_dist,
                   float surf_epsilon,
                   int   max_steps,
                   core::PlanetPosition* out_hit_pos)
{
    float t = 0.0f;

    for (int i = 0; i < max_steps; ++i) {
        core::PlanetPosition p{
            ray_origin.x + ray_dir.x * t,
            ray_origin.y + ray_dir.y * t,
            ray_origin.z + ray_dir.z * t,
        };

        const float d = sample_sdf(grid, p);
        if (d < surf_epsilon) {
            if (out_hit_pos) {
                *out_hit_pos = p;
            }
            return t;
        }

        constexpr float kMinStep = 0.01f;
        const float step = std::max(d, kMinStep);
        t += step;
        if (t > max_dist) {
            break;
        }
    }

    return max_dist;
}

bool raycast_sdf(const SdfGrid& grid,
                 const core::PlanetPosition& ray_origin,
                 const core::PlanetPosition& ray_dir,
                 float max_dist,
                 float surf_epsilon,
                 int   max_steps,
                 core::PlanetPosition& out_hit_pos)
{
    core::PlanetPosition hit{};
    const float t = raymarch_sdf(grid,
                                 ray_origin,
                                 ray_dir,
                                 max_dist,
                                 surf_epsilon,
                                 max_steps,
                                 &hit);
    if (t >= max_dist) {
        return false;
    }
    out_hit_pos = hit;
    return true;
}

void update_sdf_region_from_world(const world::World& world,
                                  const core::CoordinateConfig& cfg,
                                  const core::PlanetPosition& min_p,
                                  const core::PlanetPosition& max_p,
                                  SdfGrid& out,
                                  float* sdf_gpu,
                                  std::uint32_t* mat_gpu)
{
    const float voxel_size = out.voxel_size;
    const float half_extent = out.half_extent;
    const std::uint32_t dim = out.dim;

    if (dim == 0 || voxel_size <= 0.0f) {
        return;
    }

    const float grid_min = -half_extent;

    auto clamp_i = [dim](int i) {
        if (i < 0) {
            return 0;
        }
        const int max_i = static_cast<int>(dim) - 1;
        if (i > max_i) {
            return max_i;
        }
        return i;
    };

    auto to_index = [&](float p) {
        const float coord = (p + half_extent) / voxel_size - 0.5f;
        return static_cast<int>(std::floor(coord));
    };

    const int ix_min = clamp_i(to_index(min_p.x));
    const int iy_min = clamp_i(to_index(min_p.y));
    const int iz_min = clamp_i(to_index(min_p.z));
    const int ix_max = clamp_i(to_index(max_p.x));
    const int iy_max = clamp_i(to_index(max_p.y));
    const int iz_max = clamp_i(to_index(max_p.z));

    for (int iz = iz_min; iz <= iz_max; ++iz) {
        for (int iy = iy_min; iy <= iy_max; ++iy) {
            for (int ix = ix_min; ix <= ix_max; ++ix) {
                const float x = grid_min + (static_cast<float>(ix) + 0.5f) * voxel_size;
                const float y = grid_min + (static_cast<float>(iy) + 0.5f) * voxel_size;
                const float z = grid_min + (static_cast<float>(iz) + 0.5f) * voxel_size;

                const core::PlanetPosition pos{x, y, z};
                const core::WorldVoxelCoord world_voxel =
                    core::to_world_voxel(pos, cfg);

                const world::Voxel* voxel = world.find_voxel(world_voxel);
                const bool solid = voxel && voxel->material != 0;

                const float distance =
                    solid ? -0.5f * cfg.voxel_size_m : 0.5f * cfg.voxel_size_m;

                const std::size_t idx = linear_index(
                    dim,
                    static_cast<std::uint32_t>(ix),
                    static_cast<std::uint32_t>(iy),
                    static_cast<std::uint32_t>(iz));

                out.values[idx] = distance;

                std::uint16_t material_id = 0;
                if (solid && voxel) {
                    material_id = static_cast<std::uint16_t>(voxel->material);
                }
                out.materials[idx] = material_id;

                if (sdf_gpu) {
                    sdf_gpu[idx] = distance;
                }
                if (mat_gpu) {
                    mat_gpu[idx] = static_cast<std::uint32_t>(material_id);
                }
            }
        }
    }
}

} // namespace metaral::render
