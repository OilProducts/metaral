#include "metaral/sim/fluid_system.hpp"

#include "metaral/world/chunk.hpp"

#include <random>

namespace metaral::sim {

namespace {

// Deterministic PRNG for repeatable spawn jitter.
inline float frand(std::mt19937& rng) {
    static std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    return dist(rng);
}

} // namespace

FluidSystem::FluidSystem(const core::CoordinateConfig& coords,
                         const SphParams& params,
                         world::MaterialId water_material)
    : coords_(coords)
    , sim_(coords, params)
    , water_material_id_(water_material)
{
}

void FluidSystem::spawn_sphere(const core::PlanetPosition& center,
                               float radius_m,
                               std::size_t count)
{
    if (radius_m <= 0.0f || count == 0) {
        return;
    }

    std::mt19937 rng(1337u);
    std::vector<FluidParticle> particles;
    particles.reserve(count);

    for (std::size_t i = 0; i < count; ++i) {
        // Pick a random point in the sphere using rejection sampling.
        core::PlanetPosition p{};
        do {
            const float rx = frand(rng) * 2.0f - 1.0f;
            const float ry = frand(rng) * 2.0f - 1.0f;
            const float rz = frand(rng) * 2.0f - 1.0f;
            const float len2 = rx*rx + ry*ry + rz*rz;
            if (len2 <= 1.0f) {
                p = {center.x + rx * radius_m,
                     center.y + ry * radius_m,
                     center.z + rz * radius_m};
                break;
            }
        } while (true);

        particles.push_back(FluidParticle{p, {0.0f, 0.0f, 0.0f}});
    }

    sim_.spawn_particles(particles);
}

void FluidSystem::spawn_from_world(const world::World& world,
                                   const core::ChunkCoord& min_chunk,
                                   const core::ChunkCoord& max_chunk)
{
    const auto& cfg = world.coords();
    const auto cs   = static_cast<std::size_t>(cfg.chunk_size);

    std::vector<FluidParticle> particles;

    for (int cz = min_chunk.z; cz <= max_chunk.z; ++cz) {
        for (int cy = min_chunk.y; cy <= max_chunk.y; ++cy) {
            for (int cx = min_chunk.x; cx <= max_chunk.x; ++cx) {
                core::ChunkCoord coord{cx, cy, cz};
                const world::Chunk* chunk = world.find_chunk(coord);
                if (!chunk) {
                    continue;
                }

                for (std::size_t z = 0; z < cs; ++z) {
                    for (std::size_t y = 0; y < cs; ++y) {
                        for (std::size_t x = 0; x < cs; ++x) {
                            const auto idx = world::voxel_index(cfg,
                                static_cast<uint16_t>(x),
                                static_cast<uint16_t>(y),
                                static_cast<uint16_t>(z));
                            const auto mat = chunk->voxels[idx].material;
                            if (mat != water_material_id_) {
                                continue;
                            }

                            const core::LocalVoxelCoord local{
                                static_cast<uint16_t>(x),
                                static_cast<uint16_t>(y),
                                static_cast<uint16_t>(z),
                            };
                            const core::WorldVoxelCoord wv =
                                core::to_world_voxel(coord, local, cfg);
                            const core::PlanetPosition pos =
                                core::to_planet_position(wv, cfg);

                            particles.push_back(FluidParticle{pos, {0.0f, 0.0f, 0.0f}});
                        }
                    }
                }
            }
        }
    }

    if (!particles.empty()) {
        sim_.spawn_particles(particles);
    }
}

void FluidSystem::build_density_grid(const core::PlanetPosition& min_p,
                                     const core::PlanetPosition& max_p,
                                     std::uint32_t max_dim,
                                     DensityGrid& out) const
{
    splat_simple_density(sim_, min_p, max_p, max_dim, out);
}

} // namespace metaral::sim
