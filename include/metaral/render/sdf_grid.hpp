#pragma once

#include "metaral/core/coords.hpp"
#include "metaral/world/world.hpp"

#include <cstdint>
#include <vector>
#include <limits>

namespace metaral::render {

// Default fraction of the SDF voxel size used as an iso-surface offset when
// rendering or raycasting. This controls how "rounded" corners appear.
constexpr float kDefaultSdfIsoFraction = .25f;

// Simple hierarchical acceleration structure over an SDF grid. Nodes are stored
// level-by-level in a single contiguous array so that GPU upload is
// straightforward. The layout is intentionally conservative and can be refined
// once traversal is in place.
struct SdfOctreeNode {
    core::PlanetPosition center;  // world-space center of this node
    float half_size = 0.0f;       // half-edge length in meters

    // Conservative distance bounds (in meters) for all samples covered by this
    // node. Builders should err on the side of under-estimating min_distance
    // (and over-estimating max_distance) so traversal never incorrectly skips a
    // region that might contain the surface.
    float min_distance = std::numeric_limits<float>::infinity();
    float max_distance = 0.0f;

    // Index of the first child node in the global node array, or
    // std::numeric_limits<std::uint32_t>::max() if this is a leaf.
    std::uint32_t first_child = std::numeric_limits<std::uint32_t>::max();

    // Bitmask describing the occupancy/material state of this node. The exact
    // interpretation will be refined as the builder and traversal are
    // implemented, but at minimum it distinguishes empty / full / mixed.
    std::uint8_t occupancy_mask = 0;
    std::uint8_t flags = 0; // e.g., bit 0 = contains surface
    std::uint16_t reserved = 0;
};

struct SdfOctree {
    // All nodes for all levels in a single array.
    std::vector<SdfOctreeNode> nodes;
    // Offsets into "nodes" for each level; level_offsets[L] gives the first
    // node index for level L, and level_offsets.back() == nodes.size().
    std::vector<std::uint32_t> level_offsets;
    std::uint32_t depth = 0;      // number of levels (root == 1)

    bool empty() const noexcept {
        return nodes.empty() || depth == 0;
    }

    void clear() {
        nodes.clear();
        level_offsets.clear();
        depth = 0;
    }
};

struct SdfGrid {
    std::vector<float> values;
    std::vector<std::uint16_t> materials; // MaterialId; 0 = empty
    std::vector<std::uint8_t> occupancy;  // 1 = solid, 0 = empty; matches values/materials layout
    std::uint32_t dim = 0;       // dim^3 samples
    float voxel_size = 0.0f;     // meters between samples
    float half_extent = 0.0f;    // grid spans [-half_extent, +half_extent] in each axis
    float planet_radius = 0.0f;  // for out-of-bounds fallback

    // Optional acceleration structure used to skip empty regions during
    // raymarching. When has_octree is false, traversal falls back to the dense
    // grid alone.
    bool has_octree = false;
    SdfOctree octree;
};

void build_sdf_grid_from_world(const world::World& world,
                               const core::CoordinateConfig& cfg,
                               SdfGrid& out);

float sample_sdf(const SdfGrid& grid,
                 const core::PlanetPosition& pos);

float raymarch_sdf(const SdfGrid& grid,
                   const core::PlanetPosition& ray_origin,
                   const core::PlanetPosition& ray_dir,
                   float max_dist,
                   float surf_epsilon,
                   int   max_steps,
                   core::PlanetPosition* out_hit_pos = nullptr,
                   float iso_offset = 0.0f);

bool raycast_sdf(const SdfGrid& grid,
                 const core::PlanetPosition& ray_origin,
                 const core::PlanetPosition& ray_dir,
                 float max_dist,
                 float surf_epsilon,
                 int   max_steps,
                 core::PlanetPosition& out_hit_pos,
                 float iso_offset = 0.0f);

void update_sdf_region_from_world(const world::World& world,
                                  const core::CoordinateConfig& cfg,
                                  const core::PlanetPosition& min_p,
                                  const core::PlanetPosition& max_p,
                                  SdfGrid& out,
                                  float* sdf_gpu = nullptr,
                                  std::uint32_t* mat_gpu = nullptr);

} // namespace metaral::render
