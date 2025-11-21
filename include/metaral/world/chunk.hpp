#pragma once

#include "metaral/core/coords.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace metaral::world {

using MaterialId = uint16_t;

struct Voxel {
    MaterialId material = 0;
};

struct Chunk {
    core::ChunkCoord coord{};
    std::vector<Voxel> voxels;
};

// Independently generated chunk payload, prior to inserting into a World.
struct ChunkData {
    core::ChunkCoord coord{};
    std::vector<Voxel> voxels;
};

struct ChunkAndLocal {
    core::ChunkCoord chunk{};
    core::LocalVoxelCoord local{};
};

// x is fastest, then y, then z:
// idx = (z * cs + y) * cs + x
inline std::size_t voxel_index(const core::CoordinateConfig& cfg,
                               uint16_t x,
                               uint16_t y,
                               uint16_t z) noexcept
{
    const auto cs = static_cast<std::size_t>(cfg.chunk_size);
    const auto xs = static_cast<std::size_t>(x);
    const auto ys = static_cast<std::size_t>(y);
    const auto zs = static_cast<std::size_t>(z);
    return (zs * cs + ys) * cs + xs;
}

inline std::size_t voxel_index(const core::CoordinateConfig& cfg,
                               const core::LocalVoxelCoord& local) noexcept
{
    return voxel_index(cfg, local.x, local.y, local.z);
}

inline ChunkAndLocal split_world_voxel(const core::WorldVoxelCoord& world,
                                       const core::CoordinateConfig& cfg) noexcept
{
    ChunkAndLocal result{};

    const int64_t cs = static_cast<int64_t>(cfg.chunk_size);

    auto split_axis = [cs](int64_t w, int32_t& chunk, uint16_t& local) noexcept {
        int64_t q = w / cs;
        int64_t r = w % cs;
        if (r < 0) {
            r += cs;
            --q;
        }
        chunk = static_cast<int32_t>(q);
        local = static_cast<uint16_t>(r);
    };

    split_axis(world.x, result.chunk.x, result.local.x);
    split_axis(world.y, result.chunk.y, result.local.y);
    split_axis(world.z, result.chunk.z, result.local.z);

    return result;
}

} // namespace metaral::world
