#include "metaral/world/world.hpp"
#include "metaral/core/coords.hpp"

#include <cassert>
#include <cmath>
#include <cstddef>
#include <utility>

namespace metaral::world {

World::World(const core::CoordinateConfig& config) noexcept
    : config_(config) {}

Chunk& World::get_or_create_chunk(const core::ChunkCoord& coord) {
    auto [it, inserted] = chunks_.try_emplace(coord);
    if (inserted) {
        it->second.coord = coord;

        const auto cs = static_cast<std::size_t>(config_.chunk_size);
        const auto voxel_count = cs * cs * cs;
        it->second.voxels.resize(voxel_count);
    }
    return it->second;
}

Chunk& World::ensure_chunk_loaded(const core::ChunkCoord& coord) {
    return get_or_create_chunk(coord);
}

Chunk* World::find_chunk(const core::ChunkCoord& coord) {
    auto it = chunks_.find(coord);
    if (it == chunks_.end()) {
        return nullptr;
    }
    return &it->second;
}

const Chunk* World::find_chunk(const core::ChunkCoord& coord) const {
    auto it = chunks_.find(coord);
    if (it == chunks_.end()) {
        return nullptr;
    }
    return &it->second;
}

Voxel* World::find_voxel(const core::WorldVoxelCoord& coord) {
    ChunkAndLocal split = split_world_voxel(coord, config_);
    Chunk* chunk = find_chunk(split.chunk);
    if (!chunk) {
        return nullptr;
    }
    const auto idx = voxel_index(config_, split.local);
    return &chunk->voxels[idx];
}

const Voxel* World::find_voxel(const core::WorldVoxelCoord& coord) const {
    ChunkAndLocal split = split_world_voxel(coord, config_);
    const Chunk* chunk = find_chunk(split.chunk);
    if (!chunk) {
        return nullptr;
    }
    const auto idx = voxel_index(config_, split.local);
    return &chunk->voxels[idx];
}

Voxel& World::get_or_create_voxel(const core::WorldVoxelCoord& coord) {
    ChunkAndLocal split = split_world_voxel(coord, config_);
    Chunk& chunk = get_or_create_chunk(split.chunk);
    const auto idx = voxel_index(config_, split.local);
    return chunk.voxels[idx];
}

void World::adopt_chunk(ChunkData&& chunk_data) {
    const auto cs = static_cast<std::size_t>(config_.chunk_size);
    const auto expected_voxels = cs * cs * cs;
    assert(chunk_data.voxels.size() == expected_voxels &&
           "ChunkData voxel count must match chunk_size^3");

    auto [it, inserted] = chunks_.try_emplace(chunk_data.coord);
    Chunk& chunk = it->second;
    chunk.coord = chunk_data.coord;
    chunk.voxels = std::move(chunk_data.voxels);
}

void fill_sphere(World& world,
                 int chunk_radius,
                 MaterialId solid_material,
                 MaterialId empty_material)
{
    const auto& cfg = world.coords();
    const auto cs   = static_cast<std::size_t>(cfg.chunk_size);

    // Iterate over a cube of chunks centered at the origin in chunk space
    for (int cx = -chunk_radius; cx <= chunk_radius; ++cx) {
        for (int cy = -chunk_radius; cy <= chunk_radius; ++cy) {
            for (int cz = -chunk_radius; cz <= chunk_radius; ++cz) {
                core::ChunkCoord chunk_coord{cx, cy, cz};
                Chunk& chunk = world.ensure_chunk_loaded(chunk_coord);

                // Fill each voxel in the chunk based on distance from planet center
                for (std::size_t z = 0; z < cs; ++z) {
                    for (std::size_t y = 0; y < cs; ++y) {
                        for (std::size_t x = 0; x < cs; ++x) {
                            core::LocalVoxelCoord local{
                                static_cast<uint16_t>(x),
                                static_cast<uint16_t>(y),
                                static_cast<uint16_t>(z),
                            };

                            const core::WorldVoxelCoord world_voxel =
                                core::to_world_voxel(chunk_coord, local, cfg);

                            const core::PlanetPosition position =
                                core::to_planet_position(world_voxel, cfg);

                            const float radius = core::length(position);
                            const MaterialId material =
                                radius < cfg.planet_radius_m
                                    ? solid_material
                                    : empty_material;

                            const auto idx = voxel_index(cfg, local);
                            chunk.voxels[idx].material = material;
                        }
                    }
                }
            }
        }
    }
}

} // namespace metaral::world
