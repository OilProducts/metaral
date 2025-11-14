#include "metaral/core/coords.hpp"

#include <iomanip>
#include <iostream>

using namespace metaral::core;

int main() {
    CoordinateConfig cfg;
    cfg.voxel_size_m = 0.25f;        // 25 cm voxels
    cfg.chunk_size = 32;
    cfg.planet_radius_m = 500.0f;    // 500 m radius
    cfg.planet_center_offset_voxels = {1000, 1000, 1000};

    ChunkCoord chunk{1, -2, 0};
    LocalVoxelCoord local{12, 5, 20};

    WorldVoxelCoord world = to_world_voxel(chunk, local, cfg);
    PlanetPosition pos = to_planet_position(world, cfg);
    WorldVoxelCoord roundtrip = to_world_voxel(pos, cfg);

    float radius = length(pos);
    float height = height_above_surface(pos, cfg);
    PlanetPosition normal = surface_normal(pos);

    std::cout << std::fixed << std::setprecision(3);

    std::cout << "Coordinate config:\n";
    std::cout << "  voxel_size_m: " << cfg.voxel_size_m << "\n";
    std::cout << "  chunk_size:   " << cfg.chunk_size << "\n";
    std::cout << "  planet_radius_m: " << cfg.planet_radius_m << "\n";
    std::cout << "  center_offset_voxels: (" << cfg.planet_center_offset_voxels.x << ", "
              << cfg.planet_center_offset_voxels.y << ", " << cfg.planet_center_offset_voxels.z << ")\n\n";

    std::cout << "Input chunk/local:\n";
    std::cout << "  chunk: (" << chunk.x << ", " << chunk.y << ", " << chunk.z << ")\n";
    std::cout << "  local: (" << local.x << ", " << local.y << ", " << local.z << ")\n\n";

    std::cout << "World voxel coord: (" << world.x << ", " << world.y << ", " << world.z << ")\n";
    std::cout << "Planet position (m): (" << pos.x << ", " << pos.y << ", " << pos.z << ")\n";
    std::cout << "Roundtrip world voxel: (" << roundtrip.x << ", " << roundtrip.y << ", " << roundtrip.z << ")\n\n";

    std::cout << "Radius: " << radius << " m\n";
    std::cout << "Height above surface: " << height << " m\n";
    std::cout << "Surface normal: (" << normal.x << ", " << normal.y << ", " << normal.z << ")\n";

    return 0;
}
