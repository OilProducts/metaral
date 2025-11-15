#include "metaral/world/world.hpp"
#include "metaral/world/terrain.hpp"

#include <iostream>

int main() {
    metaral::core::CoordinateConfig cfg;
    cfg.voxel_size_m = 0.5f;
    cfg.chunk_size = 16;
    cfg.planet_radius_m = 20.0f;
    cfg.planet_center_offset_voxels = {0, 0, 0};

    metaral::world::World world(cfg);
    metaral::world::terrain::generate_planet(world, 2, cfg);

    std::size_t solid_count = 0;
    for (const auto& [coord, chunk] : world.chunks()) {
        for (const auto& voxel : chunk.voxels) {
            if (voxel.material == 1) {
                ++solid_count;
            }
        }
    }

    std::cout << "Generated " << world.chunks().size() << " chunks and "
              << solid_count << " solid voxels\n";
    return 0;
}
