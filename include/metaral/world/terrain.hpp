#pragma once

#include "metaral/core/coords.hpp"
#include "metaral/world/world.hpp"
#include "metaral/world/chunk_inbox.hpp"

#include <cstddef>
#include <thread>

namespace metaral::world {
class IChunkProvider;

namespace terrain {

// Analytic signed-distance field for the noisy sphere terrain.
// Negative inside the terrain, positive outside.
float terrain_signed_distance(const core::PlanetPosition& pos,
                              const core::CoordinateConfig& cfg) noexcept;

// Generate a single chunk worth of voxel data for the analytic terrain.
ChunkData generate_chunk(const core::ChunkCoord& chunk_coord,
                         const core::CoordinateConfig& cfg,
                         MaterialId solid_material = 1,
                         MaterialId empty_material = 0);

// Generate a rectangular region of chunks [min_chunk, max_chunk] inclusive and
// insert them into the world. Useful for synchronous bootstrapping.
void generate_region(World& world,
                     const core::ChunkCoord& min_chunk,
                     const core::ChunkCoord& max_chunk,
                     const core::CoordinateConfig& cfg,
                     MaterialId solid_material = 1,
                     MaterialId empty_material = 0,
                     std::size_t worker_count = std::thread::hardware_concurrency(),
                     ChunkInbox* inbox = nullptr);

// Generate using a provider (cache + generator) for each chunk.
void generate_region_with_provider(World& world,
                                   IChunkProvider& provider,
                                   const core::ChunkCoord& min_chunk,
                                   const core::ChunkCoord& max_chunk,
                                   std::size_t worker_count = std::thread::hardware_concurrency(),
                                   ChunkInbox* inbox = nullptr);

// Simple planet generator that writes voxel materials into a World using the
// analytic terrain field.
void generate_planet(World& world,
                     int chunk_radius,
                     const core::CoordinateConfig& cfg,
                     MaterialId solid_material = 1,
                     MaterialId empty_material = 0);

} // namespace terrain
} // namespace metaral::world
