#include "metaral/world/terrain.hpp"

#include "metaral/world/chunk.hpp"

#include <cmath>

namespace metaral::world::terrain {

namespace {

constexpr float kTerrainNoiseFrequency = 0.18f;
constexpr float kTerrainNoiseAmplitude = 2.0f;

} // namespace

float terrain_signed_distance(const core::PlanetPosition& pos,
                              const core::CoordinateConfig& cfg) noexcept
{
    const float len = metaral::core::length(pos);
    const float wobble =
        std::sin(pos.x * kTerrainNoiseFrequency) *
        std::sin(pos.y * kTerrainNoiseFrequency) *
        std::sin(pos.z * kTerrainNoiseFrequency);
    return len - cfg.planet_radius_m + wobble * kTerrainNoiseAmplitude;
}

void generate_planet(World& world,
                     int chunk_radius,
                     const core::CoordinateConfig& cfg,
                     MaterialId solid_material,
                     MaterialId empty_material)
{
    const auto cs = static_cast<std::size_t>(cfg.chunk_size);

    for (int cx = -chunk_radius; cx <= chunk_radius; ++cx) {
        for (int cy = -chunk_radius; cy <= chunk_radius; ++cy) {
            for (int cz = -chunk_radius; cz <= chunk_radius; ++cz) {
                core::ChunkCoord chunk_coord{cx, cy, cz};
                Chunk& chunk = world.ensure_chunk_loaded(chunk_coord);

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
                            const core::PlanetPosition pos =
                                core::to_planet_position(world_voxel, cfg);

                            const float d = terrain_signed_distance(pos, cfg);
                            const MaterialId material =
                                d < 0.0f ? solid_material : empty_material;

                            const auto idx = voxel_index(cfg, local);
                            chunk.voxels[idx].material = material;
                        }
                    }
                }
            }
        }
    }
}

} // namespace metaral::world::terrain
