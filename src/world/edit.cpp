#include "metaral/world/edit.hpp"

namespace metaral::world {

void apply_spherical_brush(World& world,
                           const core::CoordinateConfig& cfg,
                           const core::PlanetPosition& center,
                           float radius_m,
                           EditMode mode,
                           MaterialId material,
                           EditStats* out_stats)
{
    if (radius_m <= 0.0f) {
        return;
    }

    const float r = radius_m;

    core::PlanetPosition min_p{
        center.x - r,
        center.y - r,
        center.z - r,
    };
    core::PlanetPosition max_p{
        center.x + r,
        center.y + r,
        center.z + r,
    };

    const core::WorldVoxelCoord min_v = core::to_world_voxel(min_p, cfg);
    const core::WorldVoxelCoord max_v = core::to_world_voxel(max_p, cfg);

    EditStats local_stats{};

    for (int64_t vz = min_v.z; vz <= max_v.z; ++vz) {
        for (int64_t vy = min_v.y; vy <= max_v.y; ++vy) {
            for (int64_t vx = min_v.x; vx <= max_v.x; ++vx) {
                core::WorldVoxelCoord voxel_coord{vx, vy, vz};

                const core::PlanetPosition voxel_center =
                    core::to_planet_position(voxel_coord, cfg);

                const float dx = voxel_center.x - center.x;
                const float dy = voxel_center.y - center.y;
                const float dz = voxel_center.z - center.z;
                const float dist = std::sqrt(dx * dx + dy * dy + dz * dz);

                if (dist > radius_m) {
                    continue;
                }

                ++local_stats.voxels_touched;

                Voxel& voxel = world.get_or_create_voxel(voxel_coord);
                const MaterialId before = voxel.material;

                switch (mode) {
                case EditMode::Dig:
                    if (voxel.material != 0) {
                        voxel.material = 0;
                    }
                    break;
                case EditMode::Fill:
                    if (voxel.material == 0) {
                        voxel.material = material;
                    }
                    break;
                case EditMode::Paint:
                    if (voxel.material != 0) {
                        voxel.material = material;
                    }
                    break;
                }

                if (voxel.material != before) {
                    ++local_stats.voxels_changed;
                }
            }
        }
    }

    if (out_stats) {
        *out_stats = local_stats;
    }
}

} // namespace metaral::world

