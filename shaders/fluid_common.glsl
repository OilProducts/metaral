// Common definitions for fluid compute shaders.
struct Particle {
    vec4 position; // xyz = pos, w = unused
    vec4 velocity; // xyz = vel, w = unused
};

layout(std430, binding = 0) buffer PositionsIn {
    Particle particles[];
} positions_in;

layout(std430, binding = 1) buffer PositionsOut {
    Particle particles[];
} positions_out;

layout(std430, binding = 11) buffer BucketOffsets {
    uint offsets[];
};

layout(std430, binding = 12) buffer BucketCounts {
    uint counts[];
};

layout(std430, binding = 13) buffer SortedIndices {
    uint sortedIndices[];
};

layout(push_constant) uniform FluidPush {
    float deltaTime;
    float smoothingRadius;
    float targetDensity;
    float pressureMultiplier;
    float nearPressureMultiplier;
    float viscosityStrength;
    float gravity;
    uint  numParticles;
    uint  aux0;
    uint  aux1;
    uint  aux2;
} params;

layout(std140, binding = 9) uniform FluidParams {
    vec4 volumeOriginCell; // xyz = world-space min corner, w = cell size
    vec4 volumeDimIso;     // xyz = dimensions (float), w = iso threshold
    vec4 planetParams;     // x = planet radius, y = collision damping, z/w unused
    vec4 sdfParams;        // x = sdf dim, y = voxel size, z = half extent, w = planet radius fallback
    vec4 octreeParams;     // x = node count, y = root index, z = max depth, w = enabled flag (>0)
} uFluid;

layout(std430, binding = 10) readonly buffer SdfValues {
    float values[];
} uSdf;

// Convenience: clamp thread id to particle count and return -1 if out.
int particleIndex() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= params.numParticles) {
        return -1;
    }
    return int(idx);
}
