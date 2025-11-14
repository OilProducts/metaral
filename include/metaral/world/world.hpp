#pragma once

#include "metaral/world/chunk.hpp"

#include <functional>
#include <unordered_map>

namespace metaral::world {

struct ChunkKeyHash {
    std::size_t operator()(const core::ChunkCoord& coord) const noexcept {
        std::size_t h1 = std::hash<int32_t>{}(coord.x);
        std::size_t h2 = std::hash<int32_t>{}(coord.y);
        std::size_t h3 = std::hash<int32_t>{}(coord.z);
        return h1 ^ (h2 << 1) ^ (h3 << 2);
    }
};

struct ChunkCoordEqual {
    bool operator()(const core::ChunkCoord& a, const core::ChunkCoord& b) const noexcept {
        return a.x == b.x && a.y == b.y && a.z == b.z;
    }
};

class World {
public:
    explicit World(const core::CoordinateConfig& config) noexcept;

    [[nodiscard]] const core::CoordinateConfig& coords() const noexcept { return config_; }

    Chunk& get_or_create_chunk(const core::ChunkCoord& coord);
    Chunk* find_chunk(const core::ChunkCoord& coord);
    const Chunk* find_chunk(const core::ChunkCoord& coord) const;

    std::unordered_map<core::ChunkCoord, Chunk, ChunkKeyHash, ChunkCoordEqual>& chunks() noexcept { return chunks_; }
    const std::unordered_map<core::ChunkCoord, Chunk, ChunkKeyHash, ChunkCoordEqual>& chunks() const noexcept { return chunks_; }

private:
    core::CoordinateConfig config_;
    std::unordered_map<core::ChunkCoord, Chunk, ChunkKeyHash, ChunkCoordEqual> chunks_;
};

void fill_sphere(World& world, int chunk_radius, MaterialId solid_material, MaterialId empty_material);

} // namespace metaral::world
