#include "metaral/world/terrain.hpp"

#include "metaral/world/chunk.hpp"
#include "metaral/core/task/task_pool.hpp"

#include <cmath>
#include <iostream>
#include <chrono>
#include <latch>
#include <vector>

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

ChunkData generate_chunk(const core::ChunkCoord& chunk_coord,
                         const core::CoordinateConfig& cfg,
                         MaterialId solid_material,
                         MaterialId empty_material)
{
    const auto cs = static_cast<std::size_t>(cfg.chunk_size);
    ChunkData chunk{};
    chunk.coord = chunk_coord;
    chunk.voxels.resize(cs * cs * cs);

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
                const MaterialId material = d < 0.0f ? solid_material : empty_material;

                const auto idx = voxel_index(cfg, local);
                chunk.voxels[idx].material = material;
            }
        }
    }

    return chunk;
}

void generate_region(World& world,
                     const core::ChunkCoord& min_chunk,
                     const core::ChunkCoord& max_chunk,
                     const core::CoordinateConfig& cfg,
                     MaterialId solid_material,
                     MaterialId empty_material,
                     std::size_t worker_count,
                     ChunkInbox* inbox)
{
    // Flatten the request space into a list we can index atomically.
    std::vector<core::ChunkCoord> coords;
    coords.reserve(static_cast<std::size_t>(max_chunk.x - min_chunk.x + 1) *
                   static_cast<std::size_t>(max_chunk.y - min_chunk.y + 1) *
                   static_cast<std::size_t>(max_chunk.z - min_chunk.z + 1));

    for (int cx = min_chunk.x; cx <= max_chunk.x; ++cx) {
        for (int cy = min_chunk.y; cy <= max_chunk.y; ++cy) {
            for (int cz = min_chunk.z; cz <= max_chunk.z; ++cz) {
                coords.push_back(core::ChunkCoord{cx, cy, cz});
            }
        }
    }

    const std::size_t job_count = coords.size();
    if (job_count == 0) {
        return;
    }

    auto& pool = core::global_task_pool();
    pool.start(worker_count); // no-op if already started
    worker_count = pool.worker_count();
    if (worker_count > job_count) {
        worker_count = job_count;
    }

    const bool use_inbox = inbox != nullptr;
    std::vector<ChunkData> results;
    if (!use_inbox) {
        results.resize(job_count);
    }

    std::cout << "[terrain] generate_region: " << job_count
              << " chunk(s) on " << worker_count << " worker(s)\n";

    const auto dispatch_start = std::chrono::steady_clock::now();

    std::latch done(static_cast<std::ptrdiff_t>(job_count));

    for (std::size_t idx = 0; idx < job_count; ++idx) {
        pool.submit([&, idx](std::stop_token) {
            const core::ChunkCoord coord = coords[idx];
            ChunkData chunk =
                generate_chunk(coord, cfg, solid_material, empty_material);
            if (use_inbox) {
                inbox->push(std::move(chunk));
            } else {
                results[idx] = std::move(chunk);
            }
            done.count_down();
        });
    }

    done.wait();
    const auto end = std::chrono::steady_clock::now();
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - dispatch_start).count();
    std::cout << "[terrain] region complete in " << ms << " ms\n";

    if (!use_inbox) {
        for (auto& chunk : results) {
            world.adopt_chunk(std::move(chunk));
        }
    }
}

void generate_planet(World& world,
                     int chunk_radius,
                     const core::CoordinateConfig& cfg,
                     MaterialId solid_material,
                     MaterialId empty_material)
{
    core::ChunkCoord min_chunk{-chunk_radius, -chunk_radius, -chunk_radius};
    core::ChunkCoord max_chunk{ chunk_radius,  chunk_radius,  chunk_radius};
    generate_region(world, min_chunk, max_chunk, cfg, solid_material, empty_material);
}

} // namespace metaral::world::terrain
