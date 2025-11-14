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

} // namespace metaral::world
