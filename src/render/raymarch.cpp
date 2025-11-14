#include "metaral/render/raymarch.hpp"

#include <cmath>
#include <limits>

namespace metaral::render {

namespace {

inline core::ChunkCoord chunk_coord_from_world(const world::World& world, const core::WorldVoxelCoord& coord) {
    const int cs = world.coords().chunk_size;
    return {
        static_cast<int32_t>(std::floor(static_cast<float>(coord.x) / cs)),
        static_cast<int32_t>(std::floor(static_cast<float>(coord.y) / cs)),
        static_cast<int32_t>(std::floor(static_cast<float>(coord.z) / cs)),
    };
}

inline core::LocalVoxelCoord local_coord_from_world(const world::World& world, const core::WorldVoxelCoord& coord) {
    const int cs = world.coords().chunk_size;
    const int lx = static_cast<int>((coord.x % cs + cs) % cs);
    const int ly = static_cast<int>((coord.y % cs + cs) % cs);
    const int lz = static_cast<int>((coord.z % cs + cs) % cs);
    return {static_cast<uint16_t>(lx), static_cast<uint16_t>(ly), static_cast<uint16_t>(lz)};
}

inline std::size_t voxel_index(const world::World& world, const core::LocalVoxelCoord& local) {
    const int cs = world.coords().chunk_size;
    return static_cast<std::size_t>(local.z) * cs * cs + static_cast<std::size_t>(local.y) * cs + static_cast<std::size_t>(local.x);
}

inline bool sample_world(const world::World& world, const core::WorldVoxelCoord& coord) {
    const core::ChunkCoord chunk_coord = chunk_coord_from_world(world, coord);
    const auto* chunk = world.find_chunk(chunk_coord);
    if (!chunk) {
        return false;
    }

    const core::LocalVoxelCoord local = local_coord_from_world(world, coord);
    const std::size_t index = voxel_index(world, local);
    return index < chunk->voxels.size() && chunk->voxels[index].material != 0;
}

inline core::PlanetPosition normalized(const core::PlanetPosition& v) {
    const float len = metaral::core::length(v);
    if (len < 1e-6f) {
        return {0.0f, 1.0f, 0.0f};
    }
    const float inv = 1.0f / len;
    return {v.x * inv, v.y * inv, v.z * inv};
}

} // namespace

RayResult march_ray(const world::World& world,
                    const core::PlanetPosition& origin,
                    const core::PlanetPosition& direction,
                    const RaymarchSettings& settings) {
    const core::PlanetPosition dir = normalized(direction);
    const float step = settings.step_size_m;
    const float max_dist = settings.max_distance_m;

    core::PlanetPosition pos = origin;
    float traveled = 0.0f;

    while (traveled <= max_dist) {
        const core::WorldVoxelCoord voxel = core::to_world_voxel(pos, world.coords());
        if (sample_world(world, voxel)) {
            return {true, traveled, core::height_above_surface(pos, world.coords())};
        }

        pos.x += dir.x * step;
        pos.y += dir.y * step;
        pos.z += dir.z * step;
        traveled += step;
    }

    return {false, max_dist, 0.0f};
}

} // namespace metaral::render
