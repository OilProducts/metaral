#include "metaral/render/sdf_grid.hpp"

#include "metaral/world/chunk.hpp"

#include <cmath>
#include <limits>
#include <queue>

namespace metaral::render {

namespace {

constexpr float kSdfVoxelSizeMeters = .5f;
constexpr float kSdfMarginVoxels = 40.0f;
// Width of the band (in cells) around the surface within which we smooth
// the signed-distance field. Keeps far-field distances unchanged.
constexpr float kSdfSmoothBandCells = 2.0f;

// Chamfer mask costs for 3D 26-neighborhood: face, edge, corner.
constexpr float kChamferFaceCost   = 1.0f;
constexpr float kChamferEdgeCost   = 1.41421356237f; // sqrt(2)
constexpr float kChamferCornerCost = 1.73205080757f; // sqrt(3)

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

inline void decode_linear_index(std::uint32_t dim,
                                std::uint32_t index,
                                std::uint32_t& x,
                                std::uint32_t& y,
                                std::uint32_t& z) noexcept
{
    const std::uint32_t slice = dim * dim;
    z = index / slice;
    const std::uint32_t rem = index % slice;
    y = rem / dim;
    x = rem % dim;
}

struct DistanceNode {
    float distance;
    std::uint32_t index;
};

struct DistanceCompare {
    bool operator()(const DistanceNode& a, const DistanceNode& b) const noexcept {
        // Min-heap on distance
        return a.distance > b.distance;
    }
};

struct NeighborOffset {
    int dx;
    int dy;
    int dz;
    float cost;
};

// Precomputed neighbor offsets (26-connected neighborhood) and their costs.
static constexpr NeighborOffset kNeighbors[] = {
    // Faces
    { 1,  0,  0, kChamferFaceCost},
    {-1,  0,  0, kChamferFaceCost},
    { 0,  1,  0, kChamferFaceCost},
    { 0, -1,  0, kChamferFaceCost},
    { 0,  0,  1, kChamferFaceCost},
    { 0,  0, -1, kChamferFaceCost},
    // Edges
    { 1,  1,  0, kChamferEdgeCost},
    { 1, -1,  0, kChamferEdgeCost},
    {-1,  1,  0, kChamferEdgeCost},
    {-1, -1,  0, kChamferEdgeCost},
    { 1,  0,  1, kChamferEdgeCost},
    { 1,  0, -1, kChamferEdgeCost},
    {-1,  0,  1, kChamferEdgeCost},
    {-1,  0, -1, kChamferEdgeCost},
    { 0,  1,  1, kChamferEdgeCost},
    { 0,  1, -1, kChamferEdgeCost},
    { 0, -1,  1, kChamferEdgeCost},
    { 0, -1, -1, kChamferEdgeCost},
    // Corners
    { 1,  1,  1, kChamferCornerCost},
    { 1,  1, -1, kChamferCornerCost},
    { 1, -1,  1, kChamferCornerCost},
    { 1, -1, -1, kChamferCornerCost},
    {-1,  1,  1, kChamferCornerCost},
    {-1,  1, -1, kChamferCornerCost},
    {-1, -1,  1, kChamferCornerCost},
    {-1, -1, -1, kChamferCornerCost},
};

// Run a chamfer-distance Dijkstra over a full dim^3 volume using the
// occupancy buffer (1=solid,0=empty). Produces unsigned distances in "cell"
// units in out_dist; sign is applied later based on occupancy.
void compute_chamfer_sdf_full(const std::vector<std::uint8_t>& occupancy,
                              std::uint32_t dim,
                              std::vector<float>& out_dist)
{
    const std::size_t cell_count = static_cast<std::size_t>(dim) * dim * dim;
    out_dist.assign(cell_count, std::numeric_limits<float>::infinity());

    std::priority_queue<DistanceNode,
                        std::vector<DistanceNode>,
                        DistanceCompare> queue;

    // Seed boundary cells (where occupancy differs from any 6-connected neighbor)
    for (std::uint32_t z = 0; z < dim; ++z) {
        for (std::uint32_t y = 0; y < dim; ++y) {
            for (std::uint32_t x = 0; x < dim; ++x) {
                const std::size_t idx = linear_index(dim, x, y, z);
                const std::uint8_t center = occupancy[idx];

                bool is_boundary = false;
                if (x > 0) {
                    const std::size_t nidx = linear_index(dim, x - 1, y, z);
                    if (occupancy[nidx] != center) {
                        is_boundary = true;
                    }
                }
                if (!is_boundary && x + 1 < dim) {
                    const std::size_t nidx = linear_index(dim, x + 1, y, z);
                    if (occupancy[nidx] != center) {
                        is_boundary = true;
                    }
                }
                if (!is_boundary && y > 0) {
                    const std::size_t nidx = linear_index(dim, x, y - 1, z);
                    if (occupancy[nidx] != center) {
                        is_boundary = true;
                    }
                }
                if (!is_boundary && y + 1 < dim) {
                    const std::size_t nidx = linear_index(dim, x, y + 1, z);
                    if (occupancy[nidx] != center) {
                        is_boundary = true;
                    }
                }
                if (!is_boundary && z > 0) {
                    const std::size_t nidx = linear_index(dim, x, y, z - 1);
                    if (occupancy[nidx] != center) {
                        is_boundary = true;
                    }
                }
                if (!is_boundary && z + 1 < dim) {
                    const std::size_t nidx = linear_index(dim, x, y, z + 1);
                    if (occupancy[nidx] != center) {
                        is_boundary = true;
                    }
                }

                if (is_boundary) {
                    out_dist[idx] = 0.0f;
                    queue.push(DistanceNode{0.0f, static_cast<std::uint32_t>(idx)});
                }
            }
        }
    }

    if (queue.empty()) {
        // Degenerate case: entirely solid or entirely empty. Distances remain
        // infinite and will be handled by the caller.
        return;
    }

    while (!queue.empty()) {
        const DistanceNode node = queue.top();
        queue.pop();

        if (node.distance > out_dist[node.index]) {
            continue; // outdated entry
        }

        std::uint32_t x, y, z;
        decode_linear_index(dim, node.index, x, y, z);

        for (const NeighborOffset& n : kNeighbors) {
            const int nx = static_cast<int>(x) + n.dx;
            const int ny = static_cast<int>(y) + n.dy;
            const int nz = static_cast<int>(z) + n.dz;

            if (nx < 0 || ny < 0 || nz < 0 ||
                nx >= static_cast<int>(dim) ||
                ny >= static_cast<int>(dim) ||
                nz >= static_cast<int>(dim)) {
                continue;
            }

            const std::size_t nidx = linear_index(
                dim,
                static_cast<std::uint32_t>(nx),
                static_cast<std::uint32_t>(ny),
                static_cast<std::uint32_t>(nz));

            const float new_dist = node.distance + n.cost;
            if (new_dist < out_dist[nidx]) {
                out_dist[nidx] = new_dist;
                queue.push(DistanceNode{new_dist, static_cast<std::uint32_t>(nidx)});
            }
        }
    }
}

// Simple 3x3x3 box blur applied to the SDF values near the surface. This
// operates in-place on the grid's values, leaving far-field distances
// unchanged to avoid unnecessary work and to keep the analytic fallback
// behavior consistent.
void smooth_sdf_full(SdfGrid& grid)
{
    const std::uint32_t dim = grid.dim;
    if (dim == 0 || grid.voxel_size <= 0.0f) {
        return;
    }

    const float band_meters = kSdfSmoothBandCells * grid.voxel_size;
    if (band_meters <= 0.0f) {
        return;
    }

    const std::size_t cell_count =
        static_cast<std::size_t>(dim) * dim * dim;
    if (grid.values.size() != cell_count) {
        return;
    }

    const auto& src = grid.values;
    std::vector<float> dst(cell_count, 0.0f);

    const int dim_i = static_cast<int>(dim);

    for (std::uint32_t z = 0; z < dim; ++z) {
        for (std::uint32_t y = 0; y < dim; ++y) {
            for (std::uint32_t x = 0; x < dim; ++x) {
                const std::size_t idx_center =
                    linear_index(dim, x, y, z);

                const float center = src[idx_center];
                if (std::fabs(center) > band_meters) {
                    // Outside the smoothing band: keep the original distance.
                    dst[idx_center] = center;
                    continue;
                }

                float sum = 0.0f;
                int count = 0;

                for (int dz = -1; dz <= 1; ++dz) {
                    const int zz = static_cast<int>(z) + dz;
                    if (zz < 0 || zz >= dim_i) {
                        continue;
                    }
                    for (int dy = -1; dy <= 1; ++dy) {
                        const int yy = static_cast<int>(y) + dy;
                        if (yy < 0 || yy >= dim_i) {
                            continue;
                        }
                        for (int dx = -1; dx <= 1; ++dx) {
                            const int xx = static_cast<int>(x) + dx;
                            if (xx < 0 || xx >= dim_i) {
                                continue;
                            }

                            const std::size_t nidx = linear_index(
                                dim,
                                static_cast<std::uint32_t>(xx),
                                static_cast<std::uint32_t>(yy),
                                static_cast<std::uint32_t>(zz));
                            sum += src[nidx];
                            ++count;
                        }
                    }
                }

                if (count > 0) {
                    dst[idx_center] = sum / static_cast<float>(count);
                } else {
                    dst[idx_center] = center;
                }
            }
        }
    }

    grid.values.swap(dst);
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
    out.occupancy.assign(cell_count, 0);

    const float grid_min = -half_extent;

    // First pass: populate occupancy and materials from the voxel world.
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

                const std::size_t idx =
                    linear_index(dim, x_index, y_index, z_index);

                std::uint16_t material_id = 0;
                if (solid && voxel) {
                    material_id = static_cast<std::uint16_t>(voxel->material);
                }
                out.materials[idx] = material_id;
                out.occupancy[idx] = solid ? 1u : 0u;
            }
        }
    }

    // Second pass: compute unsigned distance in cell units with a chamfer
    // metric, then apply sign and convert to meters.
    std::vector<float> dist_cells;
    compute_chamfer_sdf_full(out.occupancy, dim, dist_cells);

    const float cell_to_meters = voxel_size;

    for (std::size_t idx = 0; idx < cell_count; ++idx) {
        const float d_cells = dist_cells[idx];
        if (!std::isfinite(d_cells)) {
            // No boundary found (degenerate case). Treat distance as zero so
            // that sample_sdf falls back to the analytic radial SDF outside
            // the grid.
            out.values[idx] = 0.0f;
            continue;
        }

        const float d_meters = d_cells * cell_to_meters;
        const bool solid = out.occupancy[idx] != 0;
        out.values[idx] = solid ? -d_meters : d_meters;
    }

    // Apply a small smoothing kernel near the surface to visually soften
    // voxel-scale artifacts in the raymarched geometry.
    smooth_sdf_full(out);
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
                   core::PlanetPosition* out_hit_pos,
                   float iso_offset)
{
    float t = 0.0f;

    for (int i = 0; i < max_steps; ++i) {
        core::PlanetPosition p{
            ray_origin.x + ray_dir.x * t,
            ray_origin.y + ray_dir.y * t,
            ray_origin.z + ray_dir.z * t,
        };

        const float d_raw = sample_sdf(grid, p);
        const float d = d_raw - iso_offset;
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
                 core::PlanetPosition& out_hit_pos,
                 float iso_offset)
{
    core::PlanetPosition hit{};
    const float t = raymarch_sdf(grid,
                                 ray_origin,
                                 ray_dir,
                                 max_dist,
                                 surf_epsilon,
                                 max_steps,
                                 &hit,
                                 iso_offset);
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

    if (ix_max < ix_min || iy_max < iy_min || iz_max < iz_min) {
        return;
    }

    const std::uint32_t region_dim_x =
        static_cast<std::uint32_t>(ix_max - ix_min + 1);
    const std::uint32_t region_dim_y =
        static_cast<std::uint32_t>(iy_max - iy_min + 1);
    const std::uint32_t region_dim_z =
        static_cast<std::uint32_t>(iz_max - iz_min + 1);

    const std::size_t region_cell_count =
        static_cast<std::size_t>(region_dim_x) *
        region_dim_y *
        region_dim_z;

    auto region_index = [region_dim_x, region_dim_y](std::uint32_t x,
                                                    std::uint32_t y,
                                                    std::uint32_t z) noexcept {
        return (static_cast<std::size_t>(z) * region_dim_y +
                static_cast<std::size_t>(y)) *
                   region_dim_x +
               static_cast<std::size_t>(x);
    };

    // Local occupancy and distance buffers for the region.
    std::vector<std::uint8_t> region_occupancy(region_cell_count, 0);
    std::vector<float> region_dist(region_cell_count,
                                   std::numeric_limits<float>::infinity());

    // First pass: resample world -> occupancy/materials for the region.
    for (std::uint32_t rz = 0; rz < region_dim_z; ++rz) {
        const int gz = iz_min + static_cast<int>(rz);
        for (std::uint32_t ry = 0; ry < region_dim_y; ++ry) {
            const int gy = iy_min + static_cast<int>(ry);
            for (std::uint32_t rx = 0; rx < region_dim_x; ++rx) {
                const int gx = ix_min + static_cast<int>(rx);

                const float x = grid_min + (static_cast<float>(gx) + 0.5f) * voxel_size;
                const float y = grid_min + (static_cast<float>(gy) + 0.5f) * voxel_size;
                const float z = grid_min + (static_cast<float>(gz) + 0.5f) * voxel_size;

                const core::PlanetPosition pos{x, y, z};
                const core::WorldVoxelCoord world_voxel =
                    core::to_world_voxel(pos, cfg);

                const world::Voxel* voxel = world.find_voxel(world_voxel);
                const bool solid = voxel && voxel->material != 0;

                const std::size_t global_idx =
                    linear_index(dim,
                                 static_cast<std::uint32_t>(gx),
                                 static_cast<std::uint32_t>(gy),
                                 static_cast<std::uint32_t>(gz));

                std::uint16_t material_id = 0;
                if (solid && voxel) {
                    material_id = static_cast<std::uint16_t>(voxel->material);
                }
                out.materials[global_idx] = material_id;
                out.occupancy[global_idx] = solid ? 1u : 0u;

                if (mat_gpu) {
                    mat_gpu[global_idx] = static_cast<std::uint32_t>(material_id);
                }

                const std::size_t local_idx =
                    region_index(rx, ry, rz);
                region_occupancy[local_idx] = solid ? 1u : 0u;
                out.occupancy[global_idx] = solid ? 1u : 0u;
            }
        }
    }

    // Seed distances inside the region:
    //  - boundary cells (occupancy transition) get distance 0
    //  - region border cells get their previous global SDF distance as a seed
    std::priority_queue<DistanceNode,
                        std::vector<DistanceNode>,
                        DistanceCompare> queue;

    for (std::uint32_t rz = 0; rz < region_dim_z; ++rz) {
        const int gz = iz_min + static_cast<int>(rz);
        for (std::uint32_t ry = 0; ry < region_dim_y; ++ry) {
            const int gy = iy_min + static_cast<int>(ry);
            for (std::uint32_t rx = 0; rx < region_dim_x; ++rx) {
                const int gx = ix_min + static_cast<int>(rx);

                const std::size_t local_idx =
                    region_index(rx, ry, rz);
                const std::uint8_t center = region_occupancy[local_idx];

                bool is_boundary = false;

                auto neighbor_solid = [&](int nx, int ny, int nz) {
                    if (nx < 0 || ny < 0 || nz < 0 ||
                        nx >= static_cast<int>(dim) ||
                        ny >= static_cast<int>(dim) ||
                        nz >= static_cast<int>(dim)) {
                        return std::uint8_t{0};
                    }
                    const std::size_t nidx = linear_index(
                        dim,
                        static_cast<std::uint32_t>(nx),
                        static_cast<std::uint32_t>(ny),
                        static_cast<std::uint32_t>(nz));
                    return out.occupancy[nidx];
                };

                if (gx > 0 && neighbor_solid(gx - 1, gy, gz) != center) {
                    is_boundary = true;
                }
                if (!is_boundary && gx + 1 < static_cast<int>(dim) &&
                    neighbor_solid(gx + 1, gy, gz) != center) {
                    is_boundary = true;
                }
                if (!is_boundary && gy > 0 &&
                    neighbor_solid(gx, gy - 1, gz) != center) {
                    is_boundary = true;
                }
                if (!is_boundary && gy + 1 < static_cast<int>(dim) &&
                    neighbor_solid(gx, gy + 1, gz) != center) {
                    is_boundary = true;
                }
                if (!is_boundary && gz > 0 &&
                    neighbor_solid(gx, gy, gz - 1) != center) {
                    is_boundary = true;
                }
                if (!is_boundary && gz + 1 < static_cast<int>(dim) &&
                    neighbor_solid(gx, gy, gz + 1) != center) {
                    is_boundary = true;
                }

                if (is_boundary) {
                    region_dist[local_idx] = 0.0f;
                    queue.push(DistanceNode{
                        0.0f,
                        static_cast<std::uint32_t>(local_idx)});
                }

                const bool on_region_border =
                    (gx == ix_min) || (gx == ix_max) ||
                    (gy == iy_min) || (gy == iy_max) ||
                    (gz == iz_min) || (gz == iz_max);

                if (on_region_border) {
                    const std::size_t global_idx =
                        linear_index(dim,
                                     static_cast<std::uint32_t>(gx),
                                     static_cast<std::uint32_t>(gy),
                                     static_cast<std::uint32_t>(gz));

                    const float prev_dist_m = std::fabs(out.values[global_idx]);
                    const float prev_dist_cells = prev_dist_m / voxel_size;
                    if (prev_dist_cells < region_dist[local_idx]) {
                        region_dist[local_idx] = prev_dist_cells;
                        queue.push(DistanceNode{
                            prev_dist_cells,
                            static_cast<std::uint32_t>(local_idx)});
                    }
                }
            }
        }
    }

    // Dijkstra within the region using the same chamfer neighborhood.
    while (!queue.empty()) {
        const DistanceNode node = queue.top();
        queue.pop();

        if (node.distance > region_dist[node.index]) {
            continue;
        }

        const std::uint32_t slice = region_dim_x * region_dim_y;
        const std::uint32_t rz = node.index / slice;
        const std::uint32_t rem = node.index % slice;
        const std::uint32_t ry = rem / region_dim_x;
        const std::uint32_t rx = rem % region_dim_x;

        for (const NeighborOffset& n : kNeighbors) {
            const int nrx = static_cast<int>(rx) + n.dx;
            const int nry = static_cast<int>(ry) + n.dy;
            const int nrz = static_cast<int>(rz) + n.dz;

            if (nrx < 0 || nry < 0 || nrz < 0 ||
                nrx >= static_cast<int>(region_dim_x) ||
                nry >= static_cast<int>(region_dim_y) ||
                nrz >= static_cast<int>(region_dim_z)) {
                continue;
            }

            const std::size_t n_local_idx =
                region_index(static_cast<std::uint32_t>(nrx),
                             static_cast<std::uint32_t>(nry),
                             static_cast<std::uint32_t>(nrz));

            const float new_dist = node.distance + n.cost;
            if (new_dist < region_dist[n_local_idx]) {
                region_dist[n_local_idx] = new_dist;
                queue.push(DistanceNode{
                    new_dist,
                    static_cast<std::uint32_t>(n_local_idx)});
            }
        }
    }

    const float cell_to_meters = voxel_size;

    // Write back signed distances (meters) to the global grid and GPU buffers.
    for (std::uint32_t rz = 0; rz < region_dim_z; ++rz) {
        const int gz = iz_min + static_cast<int>(rz);
        for (std::uint32_t ry = 0; ry < region_dim_y; ++ry) {
            const int gy = iy_min + static_cast<int>(ry);
            for (std::uint32_t rx = 0; rx < region_dim_x; ++rx) {
                const int gx = ix_min + static_cast<int>(rx);

                const std::size_t local_idx =
                    region_index(rx, ry, rz);

                const float d_cells = region_dist[local_idx];
                const float d_meters =
                    std::isfinite(d_cells) ? d_cells * cell_to_meters : 0.0f;

                const std::size_t global_idx =
                    linear_index(dim,
                                 static_cast<std::uint32_t>(gx),
                                 static_cast<std::uint32_t>(gy),
                                 static_cast<std::uint32_t>(gz));

                const bool solid = region_occupancy[local_idx] != 0;
                const float signed_distance =
                    solid ? -d_meters : d_meters;

                out.values[global_idx] = signed_distance;
                if (sdf_gpu) {
                    sdf_gpu[global_idx] = signed_distance;
                }
            }
        }
    }

    // Locally smooth the SDF in the updated region using the same 3x3x3 box
    // blur as the full-build pass, limited to a narrow band around the
    // surface for performance.
    const float band_meters = kSdfSmoothBandCells * voxel_size;
    if (band_meters > 0.0f) {
        const std::size_t region_cell_count =
            static_cast<std::size_t>(region_dim_x) *
            region_dim_y *
            region_dim_z;

        std::vector<float> region_smoothed(region_cell_count, 0.0f);
        const int dim_i = static_cast<int>(dim);

        for (std::uint32_t rz = 0; rz < region_dim_z; ++rz) {
            const int gz = iz_min + static_cast<int>(rz);
            for (std::uint32_t ry = 0; ry < region_dim_y; ++ry) {
                const int gy = iy_min + static_cast<int>(ry);
                for (std::uint32_t rx = 0; rx < region_dim_x; ++rx) {
                    const int gx = ix_min + static_cast<int>(rx);

                    const std::size_t local_idx =
                        region_index(rx, ry, rz);
                    const std::size_t global_idx =
                        linear_index(dim,
                                     static_cast<std::uint32_t>(gx),
                                     static_cast<std::uint32_t>(gy),
                                     static_cast<std::uint32_t>(gz));

                    const float center = out.values[global_idx];
                    if (std::fabs(center) > band_meters) {
                        region_smoothed[local_idx] = center;
                        continue;
                    }

                    float sum = 0.0f;
                    int count = 0;

                    for (int dz = -1; dz <= 1; ++dz) {
                        const int zz = gz + dz;
                        if (zz < 0 || zz >= dim_i) {
                            continue;
                        }
                        for (int dy = -1; dy <= 1; ++dy) {
                            const int yy = gy + dy;
                            if (yy < 0 || yy >= dim_i) {
                                continue;
                            }
                            for (int dx = -1; dx <= 1; ++dx) {
                                const int xx = gx + dx;
                                if (xx < 0 || xx >= dim_i) {
                                    continue;
                                }

                                const std::size_t n_global_idx =
                                    linear_index(dim,
                                                 static_cast<std::uint32_t>(xx),
                                                 static_cast<std::uint32_t>(yy),
                                                 static_cast<std::uint32_t>(zz));
                                sum += out.values[n_global_idx];
                                ++count;
                            }
                        }
                    }

                    if (count > 0) {
                        region_smoothed[local_idx] =
                            sum / static_cast<float>(count);
                    } else {
                        region_smoothed[local_idx] = center;
                    }
                }
            }
        }

        // Write back smoothed distances into the global grid and GPU buffer
        // for the region.
        for (std::uint32_t rz = 0; rz < region_dim_z; ++rz) {
            const int gz = iz_min + static_cast<int>(rz);
            for (std::uint32_t ry = 0; ry < region_dim_y; ++ry) {
                const int gy = iy_min + static_cast<int>(ry);
                for (std::uint32_t rx = 0; rx < region_dim_x; ++rx) {
                    const int gx = ix_min + static_cast<int>(rx);

                    const std::size_t local_idx =
                        region_index(rx, ry, rz);
                    const std::size_t global_idx =
                        linear_index(dim,
                                     static_cast<std::uint32_t>(gx),
                                     static_cast<std::uint32_t>(gy),
                                     static_cast<std::uint32_t>(gz));

                    const float smoothed = region_smoothed[local_idx];
                    out.values[global_idx] = smoothed;
                    if (sdf_gpu) {
                        sdf_gpu[global_idx] = smoothed;
                    }
                }
            }
        }
    }
}

} // namespace metaral::render
