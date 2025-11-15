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
    Chunk& ensure_chunk_loaded(const core::ChunkCoord& coord);
    Chunk*       find_chunk(const core::ChunkCoord& coord);
    const Chunk* find_chunk(const core::ChunkCoord& coord) const;

    Voxel*       find_voxel(const core::WorldVoxelCoord& coord);
    const Voxel* find_voxel(const core::WorldVoxelCoord& coord) const;
    Voxel&       get_or_create_voxel(const core::WorldVoxelCoord& coord);

    template <typename Func>
    void for_each_chunk_in_region(const core::ChunkCoord& min_coord,
                                  const core::ChunkCoord& max_coord,
                                  Func&& f)
    {
        for (auto& [coord, chunk] : chunks_) {
            if (coord.x < min_coord.x || coord.x > max_coord.x ||
                coord.y < min_coord.y || coord.y > max_coord.y ||
                coord.z < min_coord.z || coord.z > max_coord.z) {
                continue;
            }
            f(coord, chunk);
        }
    }

    template <typename Func>
    void for_each_chunk_in_region(const core::ChunkCoord& min_coord,
                                  const core::ChunkCoord& max_coord,
                                  Func&& f) const
    {
        for (const auto& [coord, chunk] : chunks_) {
            if (coord.x < min_coord.x || coord.x > max_coord.x ||
                coord.y < min_coord.y || coord.y > max_coord.y ||
                coord.z < min_coord.z || coord.z > max_coord.z) {
                continue;
            }
            f(coord, chunk);
        }
    }

    std::unordered_map<core::ChunkCoord, Chunk, ChunkKeyHash, ChunkCoordEqual>& chunks() noexcept { return chunks_; }
    const std::unordered_map<core::ChunkCoord, Chunk, ChunkKeyHash, ChunkCoordEqual>& chunks() const noexcept { return chunks_; }

private:
    core::CoordinateConfig config_;
    std::unordered_map<core::ChunkCoord, Chunk, ChunkKeyHash, ChunkCoordEqual> chunks_;
};

void fill_sphere(World& world, int chunk_radius, MaterialId solid_material, MaterialId empty_material);

inline core::WorldVoxelCoord world_voxel_from_planet_position(
    const core::PlanetPosition& pos,
    const core::CoordinateConfig& cfg) noexcept
{
    return core::to_world_voxel(pos, cfg);
}

} // namespace metaral::world
