#include "metaral/render/sdf_grid.hpp"

#include "metaral/world/chunk.hpp"
#include "metaral/core/task/task_pool.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <queue>
#include <latch>
#include <vector>

namespace metaral::render {

constexpr std::uint8_t kSdfOctreeNodeEmpty      = 1u << 0;
constexpr std::uint8_t kSdfOctreeNodeSolid      = 1u << 1;
constexpr std::uint8_t kSdfOctreeNodeHasSurface = 1u << 2;

namespace {

constexpr float kSdfVoxelSizeMeters = .5f;
constexpr float kSdfMarginVoxels = 40.0f;
// Width of the band (in cells) around the surface within which we smooth
// the signed-distance field. Keeps far-field distances unchanged.
constexpr float kSdfSmoothBandCells = 2.0f;

// Octree configuration. These values are intentionally conservative and can be
// tuned once the basic traversal is in place.
constexpr std::uint32_t kSdfOctreeLeafBlockDim = 4; // voxels per leaf edge
constexpr std::uint32_t kSdfOctreeMaxLevels    = 6; // not counting the dense grid

constexpr float kInf = std::numeric_limits<float>::infinity();

// Chamfer costs used by the incremental region updater.
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

inline float dot3(const core::PlanetPosition& a,
                  const core::PlanetPosition& b) noexcept
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline float length_squared(const core::PlanetPosition& v) noexcept
{
    return dot3(v, v);
}

inline std::uint32_t ceil_div(std::uint32_t a, std::uint32_t b) noexcept
{
    return (a + b - 1u) / b;
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

// 1D exact squared distance transform (Felzenszwalb/Huttenlocher).
void edt_1d_inplace(float* f, std::uint32_t n) {
    bool has_seed = false;
    for (std::uint32_t i = 0; i < n; ++i) {
        if (f[i] == 0.0f) { // seeds are zero
            has_seed = true;
            break;
        }
    }
    if (!has_seed) {
        return; // leave as large constant
    }

    struct Scratch {
        std::vector<int> v;
        std::vector<float> z;
        std::vector<float> g;
    };
    thread_local Scratch scratch;
    scratch.v.resize(n);
    scratch.z.resize(static_cast<std::size_t>(n) + 1);
    scratch.g.resize(n);

    std::vector<int>& v = scratch.v;
    std::vector<float>& z = scratch.z;
    std::vector<float>& g = scratch.g;

    int k = 0;
    v[0] = 0;
    z[0] = -kInf;
    z[1] = kInf;

    for (std::uint32_t q = 1; q < n; ++q) {
        float s = ((f[q] + static_cast<float>(q) * static_cast<float>(q)) -
                   (f[static_cast<std::uint32_t>(v[k])] +
                    static_cast<float>(v[k]) * static_cast<float>(v[k]))) /
                  (2.0f * static_cast<float>(q - v[k]));
        while (s <= z[k]) {
            --k;
            s = ((f[q] + static_cast<float>(q) * static_cast<float>(q)) -
                 (f[static_cast<std::uint32_t>(v[k])] +
                  static_cast<float>(v[k]) * static_cast<float>(v[k]))) /
                (2.0f * static_cast<float>(q - v[k]));
        }
        ++k;
        v[k] = static_cast<int>(q);
        z[k] = s;
        z[k + 1] = kInf;
    }

    k = 0;
    for (std::uint32_t q = 0; q < n; ++q) {
        while (z[k + 1] < static_cast<float>(q)) {
            ++k;
        }
        const float dx = static_cast<float>(q - v[k]);
        g[q] = dx * dx + f[static_cast<std::uint32_t>(v[k])];
    }

    for (std::uint32_t i = 0; i < n; ++i) {
        f[i] = g[i];
    }
}

// Exact unsigned squared distance to the nearest occupancy boundary voxel.
// Seeds are voxels that differ from any 6-connected neighbor.
void compute_edt_3d(const std::vector<std::uint8_t>& occupancy,
                    std::uint32_t dim,
                    std::vector<float>& out_dist_sq,
                    core::TaskPool& pool)
{
    pool.start(0); // ensure threads are running
    const std::size_t cell_count = static_cast<std::size_t>(dim) * dim * dim;
    const float max_sq_dist = static_cast<float>(dim) * static_cast<float>(dim) * 3.0f;
    out_dist_sq.assign(cell_count, max_sq_dist);

    auto boundary_index = [dim](std::uint32_t x, std::uint32_t y, std::uint32_t z) noexcept {
        return (static_cast<std::size_t>(z) * dim + static_cast<std::size_t>(y)) * dim +
               static_cast<std::size_t>(x);
    };

    // Seed boundaries.
    for (std::uint32_t z = 0; z < dim; ++z) {
        for (std::uint32_t y = 0; y < dim; ++y) {
            for (std::uint32_t x = 0; x < dim; ++x) {
                const std::size_t idx = boundary_index(x, y, z);
                const std::uint8_t center = occupancy[idx];

                bool is_boundary = false;
                if (x > 0) {
                    if (occupancy[idx - 1] != center) is_boundary = true;
                }
                if (!is_boundary && x + 1 < dim) {
                    if (occupancy[idx + 1] != center) is_boundary = true;
                }
                if (!is_boundary && y > 0) {
                    if (occupancy[idx - dim] != center) is_boundary = true;
                }
                if (!is_boundary && y + 1 < dim) {
                    if (occupancy[idx + dim] != center) is_boundary = true;
                }
                if (!is_boundary && z > 0) {
                    if (occupancy[idx - static_cast<std::size_t>(dim) * dim] != center) is_boundary = true;
                }
                if (!is_boundary && z + 1 < dim) {
                    if (occupancy[idx + static_cast<std::size_t>(dim) * dim] != center) is_boundary = true;
                }

                if (is_boundary) {
                    out_dist_sq[idx] = 0.0f;
                }
            }
        }
    }

    // Helper to parallelize a set of lines using the task pool.
    auto parallel_lines = [&](std::size_t line_count, auto&& func) {
        const std::size_t workers = pool.worker_count() == 0 ? 1u : pool.worker_count();
        std::atomic<std::size_t> next{0};
        std::latch latch(static_cast<std::ptrdiff_t>(workers));

        for (std::size_t w = 0; w < workers; ++w) {
            pool.submit([&](std::stop_token st) {
                while (!st.stop_requested()) {
                    const std::size_t line_idx = next.fetch_add(1, std::memory_order_relaxed);
                    if (line_idx >= line_count) {
                        break;
                    }
                    func(line_idx);
                }
                latch.count_down();
            });
        }

        latch.wait();
    };

    // Pass 1: X axis (contiguous lines).
    const std::size_t line_count_x = static_cast<std::size_t>(dim) * dim;
    parallel_lines(line_count_x, [&](std::size_t line_idx) {
        const std::uint32_t y = static_cast<std::uint32_t>(line_idx % dim);
        const std::uint32_t z = static_cast<std::uint32_t>(line_idx / dim);
        float* line = out_dist_sq.data() + boundary_index(0, y, z);
        edt_1d_inplace(line, dim);
    });
    // Pass 1: X axis (contiguous lines).

    // Pass 2: Y axis (requires stride copy).
    const std::size_t line_count_y = static_cast<std::size_t>(dim) * dim;
    parallel_lines(line_count_y, [&](std::size_t line_idx) {
        const std::uint32_t x = static_cast<std::uint32_t>(line_idx % dim);
        const std::uint32_t z = static_cast<std::uint32_t>(line_idx / dim);

        thread_local std::vector<float> tmp;
        tmp.resize(dim);

        const std::size_t base = static_cast<std::size_t>(z) * dim * dim + x;
        for (std::uint32_t y = 0; y < dim; ++y) {
            tmp[y] = out_dist_sq[base + static_cast<std::size_t>(y) * dim];
        }

        edt_1d_inplace(tmp.data(), dim);

        for (std::uint32_t y = 0; y < dim; ++y) {
            out_dist_sq[base + static_cast<std::size_t>(y) * dim] = tmp[y];
        }
    });
    // Pass 3: Z axis (requires stride copy).
    const std::size_t line_count_z = static_cast<std::size_t>(dim) * dim;
    parallel_lines(line_count_z, [&](std::size_t line_idx) {
        const std::uint32_t x = static_cast<std::uint32_t>(line_idx % dim);
        const std::uint32_t y = static_cast<std::uint32_t>(line_idx / dim);

        thread_local std::vector<float> tmp;
        tmp.resize(dim);

        for (std::uint32_t z = 0; z < dim; ++z) {
            const std::size_t idx = boundary_index(x, y, z);
            tmp[z] = out_dist_sq[idx];
        }

        edt_1d_inplace(tmp.data(), dim);

        for (std::uint32_t z = 0; z < dim; ++z) {
            const std::size_t idx = boundary_index(x, y, z);
            out_dist_sq[idx] = tmp[z];
        }
    });
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

// Build a conservative, multi-level octree over the existing SDF grid. The
// current implementation focuses on producing usable metadata (min distance
// bounds and empty/solid/mixed classification); traversal integration and
// incremental updates are layered on separately.
void build_sdf_octree_full(const SdfGrid& grid, SdfGrid& out_grid)
{
    const std::uint32_t dim = grid.dim;
    if (dim == 0 || grid.voxel_size <= 0.0f || grid.half_extent <= 0.0f) {
        out_grid.has_octree = false;
        out_grid.octree.clear();
        return;
    }

    const std::size_t cell_count =
        static_cast<std::size_t>(dim) * dim * dim;
    if (grid.values.size() != cell_count ||
        grid.occupancy.size() != cell_count) {
        out_grid.has_octree = false;
        out_grid.octree.clear();
        return;
    }

    // Compute per-level dimensions starting from a leaf grid whose cells each
    // cover kSdfOctreeLeafBlockDim^3 dense SDF samples.
    std::vector<std::uint32_t> level_dims;
    level_dims.reserve(kSdfOctreeMaxLevels);

    const std::uint32_t leaf_dim =
        ceil_div(dim, kSdfOctreeLeafBlockDim);
    if (leaf_dim == 0) {
        out_grid.has_octree = false;
        out_grid.octree.clear();
        return;
    }
    level_dims.push_back(leaf_dim);

    std::uint32_t current_dim = leaf_dim;
    while (current_dim > 1u &&
           level_dims.size() < kSdfOctreeMaxLevels) {
        current_dim = ceil_div(current_dim, 2u);
        level_dims.push_back(current_dim);
    }

    const std::uint32_t depth =
        static_cast<std::uint32_t>(level_dims.size());

    // Compute level offsets and total node count.
    std::vector<std::uint32_t> level_offsets(depth, 0);
    std::size_t total_nodes = 0;
    for (std::uint32_t level = 0; level < depth; ++level) {
        level_offsets[level] = static_cast<std::uint32_t>(total_nodes);
        const std::uint64_t d = level_dims[level];
        const std::uint64_t count = d * d * d;
        total_nodes += static_cast<std::size_t>(count);
    }

    if (total_nodes == 0) {
        out_grid.has_octree = false;
        out_grid.octree.clear();
        return;
    }

    std::vector<SdfOctreeNode> nodes(total_nodes);

    const float voxel_size = grid.voxel_size;
    const float grid_min = -grid.half_extent;

    // Build the leaf level directly from the dense SDF/occupancy.
    {
        const std::uint32_t leaf_dim_local = level_dims[0];
        const std::uint32_t leaf_level_offset = level_offsets[0];

        for (std::uint32_t z = 0; z < leaf_dim_local; ++z) {
            const std::uint32_t base_z = z * kSdfOctreeLeafBlockDim;
            const std::uint32_t max_z =
                std::min(base_z + kSdfOctreeLeafBlockDim, dim);

            for (std::uint32_t y = 0; y < leaf_dim_local; ++y) {
                const std::uint32_t base_y = y * kSdfOctreeLeafBlockDim;
                const std::uint32_t max_y =
                    std::min(base_y + kSdfOctreeLeafBlockDim, dim);

                for (std::uint32_t x = 0; x < leaf_dim_local; ++x) {
                    const std::uint32_t base_x = x * kSdfOctreeLeafBlockDim;
                    const std::uint32_t max_x =
                        std::min(base_x + kSdfOctreeLeafBlockDim, dim);

                    const std::size_t node_index =
                        leaf_level_offset +
                        linear_index(leaf_dim_local, x, y, z);
                    SdfOctreeNode& node = nodes[node_index];

                    bool any_solid = false;
                    bool any_empty = false;
                    float min_abs_dist = std::numeric_limits<float>::infinity();
                    float max_abs_dist = 0.0f;

                    for (std::uint32_t zz = base_z; zz < max_z; ++zz) {
                        for (std::uint32_t yy = base_y; yy < max_y; ++yy) {
                            for (std::uint32_t xx = base_x; xx < max_x; ++xx) {
                                const std::size_t cell_index =
                                    linear_index(dim, xx, yy, zz);

                                const std::uint8_t occ =
                                    grid.occupancy[cell_index];
                                if (occ != 0u) {
                                    any_solid = true;
                                } else {
                                    any_empty = true;
                                }

                                const float v = grid.values[cell_index];
                                if (std::isfinite(v)) {
                                    const float a = std::fabs(v);
                                    if (a < min_abs_dist) {
                                        min_abs_dist = a;
                                    }
                                    if (a > max_abs_dist) {
                                        max_abs_dist = a;
                                    }
                                }
                            }
                        }
                    }

                    // Compute world-space bounds for this node.
                    const float x_min = grid_min +
                        static_cast<float>(base_x) * voxel_size;
                    const float x_max = grid_min +
                        static_cast<float>(max_x) * voxel_size;
                    const float y_min = grid_min +
                        static_cast<float>(base_y) * voxel_size;
                    const float y_max = grid_min +
                        static_cast<float>(max_y) * voxel_size;
                    const float z_min = grid_min +
                        static_cast<float>(base_z) * voxel_size;
                    const float z_max = grid_min +
                        static_cast<float>(max_z) * voxel_size;

                    node.center.x = 0.5f * (x_min + x_max);
                    node.center.y = 0.5f * (y_min + y_max);
                    node.center.z = 0.5f * (z_min + z_max);
                    node.half_size = 0.5f * std::max(
                        std::max(x_max - x_min, y_max - y_min),
                        z_max - z_min);

                    node.min_distance = min_abs_dist;
                    node.max_distance = max_abs_dist;
                    node.first_child = std::numeric_limits<std::uint32_t>::max();

                    std::uint8_t occ_mask = 0;
                    if (any_empty) occ_mask |= kSdfOctreeNodeEmpty;
                    if (any_solid) occ_mask |= kSdfOctreeNodeSolid;
                    node.occupancy_mask = occ_mask;

                    std::uint8_t flags = 0;
                    if (any_empty && any_solid) {
                        flags |= kSdfOctreeNodeHasSurface;
                    }
                    node.flags = flags;
                    node.reserved = 0;
                }
            }
        }
    }

    // Build higher levels by aggregating child nodes in 2x2x2 groups.
    for (std::uint32_t level = 1; level < depth; ++level) {
        const std::uint32_t dim_here = level_dims[level];
        const std::uint32_t dim_child = level_dims[level - 1];
        const std::uint32_t offset_here = level_offsets[level];
        const std::uint32_t offset_child = level_offsets[level - 1];

        for (std::uint32_t z = 0; z < dim_here; ++z) {
            for (std::uint32_t y = 0; y < dim_here; ++y) {
                for (std::uint32_t x = 0; x < dim_here; ++x) {
                    const std::size_t parent_index =
                        offset_here + linear_index(dim_here, x, y, z);
                    SdfOctreeNode& parent = nodes[parent_index];

                    bool any_child = false;
                    bool any_solid = false;
                    bool any_empty = false;
                    bool any_surface = false;
                    float min_abs_dist = std::numeric_limits<float>::infinity();
                    float max_abs_dist = 0.0f;

                    core::PlanetPosition bounds_min{
                        std::numeric_limits<float>::infinity(),
                        std::numeric_limits<float>::infinity(),
                        std::numeric_limits<float>::infinity()};
                    core::PlanetPosition bounds_max{
                        -std::numeric_limits<float>::infinity(),
                        -std::numeric_limits<float>::infinity(),
                        -std::numeric_limits<float>::infinity()};

                    const std::uint32_t child_x0 = x * 2u;
                    const std::uint32_t child_y0 = y * 2u;
                    const std::uint32_t child_z0 = z * 2u;

                    for (std::uint32_t dz = 0; dz < 2u; ++dz) {
                        const std::uint32_t cz = child_z0 + dz;
                        if (cz >= dim_child) continue;
                        for (std::uint32_t dy = 0; dy < 2u; ++dy) {
                            const std::uint32_t cy = child_y0 + dy;
                            if (cy >= dim_child) continue;
                            for (std::uint32_t dx = 0; dx < 2u; ++dx) {
                                const std::uint32_t cx = child_x0 + dx;
                                if (cx >= dim_child) continue;

                                const std::size_t child_index =
                                    offset_child +
                                    linear_index(dim_child, cx, cy, cz);
                                SdfOctreeNode& child = nodes[child_index];

                                any_child = true;

                                const float child_min = child.min_distance;
                                const float child_max = child.max_distance;
                                if (child_min < min_abs_dist) {
                                    min_abs_dist = child_min;
                                }
                                if (child_max > max_abs_dist) {
                                    max_abs_dist = child_max;
                                }

                                if (child.occupancy_mask & kSdfOctreeNodeEmpty) {
                                    any_empty = true;
                                }
                                if (child.occupancy_mask & kSdfOctreeNodeSolid) {
                                    any_solid = true;
                                }
                                if (child.flags & kSdfOctreeNodeHasSurface) {
                                    any_surface = true;
                                }

                                const float hx = child.half_size;
                                const core::PlanetPosition cmin{
                                    child.center.x - hx,
                                    child.center.y - hx,
                                    child.center.z - hx};
                                const core::PlanetPosition cmax{
                                    child.center.x + hx,
                                    child.center.y + hx,
                                    child.center.z + hx};

                                bounds_min.x = std::min(bounds_min.x, cmin.x);
                                bounds_min.y = std::min(bounds_min.y, cmin.y);
                                bounds_min.z = std::min(bounds_min.z, cmin.z);
                                bounds_max.x = std::max(bounds_max.x, cmax.x);
                                bounds_max.y = std::max(bounds_max.y, cmax.y);
                                bounds_max.z = std::max(bounds_max.z, cmax.z);
                            }
                        }
                    }

                    if (!any_child) {
                        parent.center = core::PlanetPosition{0.0f, 0.0f, 0.0f};
                        parent.half_size = 0.0f;
                        parent.min_distance = std::numeric_limits<float>::infinity();
                        parent.max_distance = 0.0f;
                        parent.first_child = std::numeric_limits<std::uint32_t>::max();
                        parent.occupancy_mask = 0;
                        parent.flags = 0;
                        parent.reserved = 0;
                        continue;
                    }

                    parent.center.x = 0.5f * (bounds_min.x + bounds_max.x);
                    parent.center.y = 0.5f * (bounds_min.y + bounds_max.y);
                    parent.center.z = 0.5f * (bounds_min.z + bounds_max.z);
                    parent.half_size = 0.5f * std::max(
                        std::max(bounds_max.x - bounds_min.x,
                                 bounds_max.y - bounds_min.y),
                        bounds_max.z - bounds_min.z);

                    parent.min_distance = min_abs_dist;
                    parent.max_distance = max_abs_dist;
                    parent.first_child = offset_child +
                        linear_index(dim_child, child_x0, child_y0, child_z0);

                    std::uint8_t occ_mask = 0;
                    if (any_empty) occ_mask |= kSdfOctreeNodeEmpty;
                    if (any_solid) occ_mask |= kSdfOctreeNodeSolid;
                    parent.occupancy_mask = occ_mask;

                    std::uint8_t flags = 0;
                    if (any_surface || (any_empty && any_solid)) {
                        flags |= kSdfOctreeNodeHasSurface;
                    }
                    parent.flags = flags;
                    parent.reserved = 0;
                }
            }
        }
    }

    // Commit to the output grid.
    out_grid.octree.nodes = std::move(nodes);
    out_grid.octree.level_offsets = std::move(level_offsets);
    out_grid.octree.depth = depth;
    out_grid.has_octree = true;
}

} // namespace
void build_sdf_grid_from_world(const world::World& world,
                               const core::CoordinateConfig& cfg,
                               SdfGrid& out)
{
    const auto log_ms = [](const char* label, auto start, auto end) {
        const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::cout << "[sdf] " << label << " took " << ms << " ms\n";
    };

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
    auto& pool = core::global_task_pool();
    pool.start(0); // no-op if already started
    std::size_t worker_count = pool.worker_count();
    if (worker_count == 0) {
        worker_count = 1;
    }
    if (worker_count > dim) {
        worker_count = dim;
    }

    std::atomic<std::uint32_t> next_z{0};
    std::latch latch(static_cast<std::ptrdiff_t>(worker_count));

    std::cout << "[sdf] occupancy fill on " << worker_count
              << " worker(s)\n";

    const auto occ_start = std::chrono::steady_clock::now();

    for (std::size_t w = 0; w < worker_count; ++w) {
        pool.submit([&, voxel_size, grid_min](std::stop_token st) {
            while (!st.stop_requested()) {
                const std::uint32_t z_index = next_z.fetch_add(1, std::memory_order_relaxed);
                if (z_index >= dim) {
                    break;
                }

                const float z = grid_min + (static_cast<float>(z_index) + 0.5f) * voxel_size;

                for (std::uint32_t y_index = 0; y_index < dim; ++y_index) {
                    const float y = grid_min + (static_cast<float>(y_index) + 0.5f) * voxel_size;

                    for (std::uint32_t x_index = 0; x_index < dim; ++x_index) {
                        const float x = grid_min + (static_cast<float>(x_index) + 0.5f) * voxel_size;

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
            latch.count_down();
        });
    }

    latch.wait();
    log_ms("occupancy fill", occ_start, std::chrono::steady_clock::now());

    // Second pass: compute exact unsigned distance in cell units (squared),
    // then apply sign and convert to meters.
    const auto edt_start = std::chrono::steady_clock::now();
    std::vector<float> dist_sq_cells;
    compute_edt_3d(out.occupancy, dim, dist_sq_cells, pool);
    const float max_reasonable = static_cast<float>(dim) * static_cast<float>(dim) * 3.0f;
    for (float& v : dist_sq_cells) {
        if (v > max_reasonable || !std::isfinite(v)) {
            v = max_reasonable;
        }
        if (v < 0.0f) {
            v = 0.0f;
        }
    }
    log_ms("edt distance", edt_start, std::chrono::steady_clock::now());

    const float cell_to_meters = voxel_size;

    const auto sign_start = std::chrono::steady_clock::now();
    for (std::size_t idx = 0; idx < cell_count; ++idx) {
        const float d_cells_sq = dist_sq_cells[idx];
        if (!std::isfinite(d_cells_sq)) {
            // No boundary found (degenerate case). Treat distance as zero so
            // that sample_sdf falls back to the analytic radial SDF outside
            // the grid.
            out.values[idx] = 0.0f;
            continue;
        }

        const float d_cells = std::sqrt(d_cells_sq);
        const float d_meters = d_cells * cell_to_meters;
        const bool solid = out.occupancy[idx] != 0;
        out.values[idx] = solid ? -d_meters : d_meters;
    }
    log_ms("sign/apply distance", sign_start, std::chrono::steady_clock::now());

    // Apply a small smoothing kernel near the surface to visually soften
    // voxel-scale artifacts in the raymarched geometry.
    const auto smooth_start = std::chrono::steady_clock::now();
    smooth_sdf_full(out);
    log_ms("smoothing", smooth_start, std::chrono::steady_clock::now());

    // Build a conservative octree over the finished grid to accelerate
    // raymarching. If construction fails for any reason, the grid simply
    // reports has_octree == false and traversal falls back to the dense SDF.
    const auto octree_start = std::chrono::steady_clock::now();
    build_sdf_octree_full(out, out);
    log_ms("octree build", octree_start, std::chrono::steady_clock::now());
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

inline bool point_inside_octree_node(const SdfOctreeNode& node,
                                     const core::PlanetPosition& p) noexcept
{
    const float h = node.half_size;
    if (h <= 0.0f) {
        return false;
    }
    const float dx = p.x - node.center.x;
    const float dy = p.y - node.center.y;
    const float dz = p.z - node.center.z;
    return std::fabs(dx) <= h &&
           std::fabs(dy) <= h &&
           std::fabs(dz) <= h;
}

bool advance_ray_with_octree(const SdfGrid& grid,
                             const core::PlanetPosition& ray_origin,
                             const core::PlanetPosition& ray_dir,
                             float max_dist,
                             float& t)
{
    if (!grid.has_octree ||
        grid.octree.depth == 0 ||
        grid.octree.nodes.empty() ||
        grid.octree.level_offsets.size() < grid.octree.depth) {
        return false;
    }

    const float dir_len_sq = length_squared(ray_dir);
    if (dir_len_sq <= 0.0f) {
        return false;
    }

    const std::uint32_t root_index =
        grid.octree.level_offsets[grid.octree.depth - 1u];
    if (root_index >= grid.octree.nodes.size()) {
        return false;
    }

    const auto& nodes = grid.octree.nodes;

    core::PlanetPosition p{
        ray_origin.x + ray_dir.x * t,
        ray_origin.y + ray_dir.y * t,
        ray_origin.z + ray_dir.z * t,
    };

    if (!point_inside_octree_node(nodes[root_index], p)) {
        return false;
    }

    std::uint32_t current = root_index;

    // Descend towards the deepest node that contains the current point.
    for (;;) {
        const SdfOctreeNode& node = nodes[current];

        // If this node is not purely empty or has no children, stop.
        const bool has_solid =
            (node.occupancy_mask & kSdfOctreeNodeSolid) != 0;
        const bool has_empty =
            (node.occupancy_mask & kSdfOctreeNodeEmpty) != 0;
        if (!has_empty || node.first_child == std::numeric_limits<std::uint32_t>::max()) {
            break;
        }

        const std::uint32_t first_child = node.first_child;
        const std::uint32_t max_child = first_child + 8u;
        std::uint32_t next = std::numeric_limits<std::uint32_t>::max();

        for (std::uint32_t child = first_child;
             child < max_child && child < nodes.size();
             ++child) {
            if (point_inside_octree_node(nodes[child], p)) {
                next = child;
                break;
            }
        }

        if (next == std::numeric_limits<std::uint32_t>::max()) {
            break;
        }
        current = next;
    }

    const SdfOctreeNode& leaf = nodes[current];
    const bool leaf_is_pure_empty =
        (leaf.occupancy_mask == kSdfOctreeNodeEmpty);
    if (!leaf_is_pure_empty) {
        return false;
    }

    const float hx = leaf.half_size;
    if (hx <= 0.0f) {
        return false;
    }

    const float min_x = leaf.center.x - hx;
    const float max_x = leaf.center.x + hx;
    const float min_y = leaf.center.y - hx;
    const float max_y = leaf.center.y + hx;
    const float min_z = leaf.center.z - hx;
    const float max_z = leaf.center.z + hx;

    float tmin = -std::numeric_limits<float>::infinity();
    float tmax =  std::numeric_limits<float>::infinity();

    auto intersect_axis = [&](float ro, float rd, float mn, float mx) -> bool {
        if (std::fabs(rd) < 1e-6f) {
            return ro >= mn && ro <= mx;
        }
        const float inv = 1.0f / rd;
        float t1 = (mn - ro) * inv;
        float t2 = (mx - ro) * inv;
        if (t1 > t2) {
            const float tmp = t1;
            t1 = t2;
            t2 = tmp;
        }
        tmin = std::max(tmin, t1);
        tmax = std::min(tmax, t2);
        return tmax >= tmin;
    };

    if (!intersect_axis(ray_origin.x, ray_dir.x, min_x, max_x) ||
        !intersect_axis(ray_origin.y, ray_dir.y, min_y, max_y) ||
        !intersect_axis(ray_origin.z, ray_dir.z, min_z, max_z)) {
        return false;
    }

    // We know the current point lies inside this box, so t is within
    // [tmin, tmax]. We want to advance to the exit point.
    const float exit_t = tmax;
    if (exit_t <= t + 1e-4f) {
        return false;
    }

    const float new_t = std::min(exit_t, max_dist);
    if (new_t <= t) {
        return false;
    }

    t = new_t;
    return true;
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
    const float original_max_dist = max_dist;
    float t = 0.0f;

    const float boundary_radius = grid.half_extent;
    if (boundary_radius > 0.0f) {
        const float dir_len_sq = length_squared(ray_dir);
        if (dir_len_sq > 0.0f) {
            const float radius_sq = boundary_radius * boundary_radius;
            const float origin_len_sq = length_squared(ray_origin);
            const float c = origin_len_sq - radius_sq;
            const bool origin_outside = c > 0.0f;
            const float b = 2.0f * dot3(ray_origin, ray_dir);
            const float discriminant = b * b - 4.0f * dir_len_sq * c;

            if (discriminant < 0.0f) {
                if (origin_outside) {
                    return original_max_dist;
                }
            } else {
                const float sqrt_disc = std::sqrt(discriminant);
                const float inv_denom = 0.5f / dir_len_sq;
                float entry = (-b - sqrt_disc) * inv_denom;
                float exit = (-b + sqrt_disc) * inv_denom;
                if (entry > exit) {
                    const float tmp = entry;
                    entry = exit;
                    exit = tmp;
                }

                if (origin_outside) {
                    if (exit < 0.0f) {
                        return original_max_dist;
                    }
                    t = std::max(entry, 0.0f);
                    if (t > max_dist) {
                        return original_max_dist;
                    }
                }
            }
        }
    }

    float best_abs_d = std::numeric_limits<float>::infinity();
    float best_t = 0.0f;
    core::PlanetPosition best_pos{};

    for (int i = 0; i < max_steps; ++i) {
        if (grid.has_octree) {
            if (advance_ray_with_octree(grid,
                                        ray_origin,
                                        ray_dir,
                                        max_dist,
                                        t)) {
                if (t >= max_dist) {
                    break;
                }
                continue;
            }
        }

        core::PlanetPosition p{
            ray_origin.x + ray_dir.x * t,
            ray_origin.y + ray_dir.y * t,
            ray_origin.z + ray_dir.z * t,
        };

        const float d_raw = sample_sdf(grid, p);
        const float d = d_raw - iso_offset;

        const float abs_d = std::fabs(d);
        if (abs_d < best_abs_d) {
            best_abs_d = abs_d;
            best_pos = p;
            best_t = t;
        }

        if (d < surf_epsilon) {
            if (out_hit_pos) {
                *out_hit_pos = p;
            }
            return t;
        }

        constexpr float kMinStep = 0.01f;
        constexpr float kStepSafety = 0.8f;
        const float step = std::max(d * kStepSafety, kMinStep);
        t += step;
        if (t > max_dist) {
            break;
        }
    }

    // If we didn't register a proper hit but passed very close to the
    // surface, treat the closest approach as a hit. This helps avoid
    // "leaking" background pixels at grazing angles.
    const float near_miss_epsilon =
        std::max(surf_epsilon * 2.0f, 0.25f * grid.voxel_size);
    if (best_abs_d < near_miss_epsilon && best_t <= max_dist) {
        if (out_hit_pos) {
            *out_hit_pos = best_pos;
        }
        return best_t;
    }

    return original_max_dist;
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

    // The octree currently mirrors the full dense grid, so when a region is
    // updated we conservatively rebuild the hierarchy. This can be optimized
    // later to only touch affected nodes.
    build_sdf_octree_full(out, out);
}

} // namespace metaral::render
