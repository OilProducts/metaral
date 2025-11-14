#include "metaral/world/world.hpp"
#include "metaral/core/coords.hpp"

#include <cmath>
#include <cstddef>

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

namespace {

// x is fastest, then y, then z:
// idx = (z * cs + y) * cs + x
inline std::size_t voxel_index(const core::CoordinateConfig& cfg,
                               std::size_t x,
                               std::size_t y,
                               std::size_t z) noexcept
{
    const auto cs = static_cast<std::size_t>(cfg.chunk_size);
    return (z * cs + y) * cs + x;
}

} // namespace

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
                Chunk& chunk = world.get_or_create_chunk(chunk_coord);

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

                            const auto idx = voxel_index(cfg, x, y, z);
                            chunk.voxels[idx].material = material;
                        }
                    }
                }
            }
        }
    }
}

} // namespace metaral::world
