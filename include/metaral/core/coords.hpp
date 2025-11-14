#pragma once

#include <cmath>
#include <cstdint>

namespace metaral::core {

struct LocalVoxelCoord {
    uint16_t x = 0;
    uint16_t y = 0;
    uint16_t z = 0;
};

struct ChunkCoord {
    int32_t x = 0;
    int32_t y = 0;
    int32_t z = 0;
};

struct WorldVoxelCoord {
    int64_t x = 0;
    int64_t y = 0;
    int64_t z = 0;
};

struct PlanetPosition {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
};

struct CoordinateConfig {
    float voxel_size_m = 0.1f;
    int32_t chunk_size = 32;
    float planet_radius_m = 1000.0f;
    WorldVoxelCoord planet_center_offset_voxels{};
};

inline WorldVoxelCoord to_world_voxel(ChunkCoord chunk, LocalVoxelCoord local, const CoordinateConfig& cfg) noexcept {
    const int64_t cs = cfg.chunk_size;
    return {
        static_cast<int64_t>(chunk.x) * cs + local.x,
        static_cast<int64_t>(chunk.y) * cs + local.y,
        static_cast<int64_t>(chunk.z) * cs + local.z,
    };
}

inline PlanetPosition to_planet_position(const WorldVoxelCoord& voxel, const CoordinateConfig& cfg) noexcept {
    int64_t vx = voxel.x - cfg.planet_center_offset_voxels.x;
    int64_t vy = voxel.y - cfg.planet_center_offset_voxels.y;
    int64_t vz = voxel.z - cfg.planet_center_offset_voxels.z;

    const float scale = cfg.voxel_size_m;
    constexpr float half = 0.5f;
    return {
        (static_cast<float>(vx) + half) * scale,
        (static_cast<float>(vy) + half) * scale,
        (static_cast<float>(vz) + half) * scale,
    };
}

inline WorldVoxelCoord to_world_voxel(const PlanetPosition& pos, const CoordinateConfig& cfg) noexcept {
    const float inv = 1.0f / cfg.voxel_size_m;
    return {
        static_cast<int64_t>(std::floor(pos.x * inv)) + cfg.planet_center_offset_voxels.x,
        static_cast<int64_t>(std::floor(pos.y * inv)) + cfg.planet_center_offset_voxels.y,
        static_cast<int64_t>(std::floor(pos.z * inv)) + cfg.planet_center_offset_voxels.z,
    };
}

inline float length(const PlanetPosition& pos) noexcept {
    return std::sqrt(pos.x * pos.x + pos.y * pos.y + pos.z * pos.z);
}

inline float height_above_surface(const PlanetPosition& pos, const CoordinateConfig& cfg) noexcept {
    return length(pos) - cfg.planet_radius_m;
}

inline PlanetPosition surface_normal(const PlanetPosition& pos) noexcept {
    float len = length(pos);
    if (len < 1e-6f) {
        return {0.0f, 1.0f, 0.0f};
    }
    float inv = 1.0f / len;
    return {pos.x * inv, pos.y * inv, pos.z * inv};
}

} // namespace metaral::core
