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

// Convenience: clamp thread id to particle count and return -1 if out.
int particleIndex() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= params.numParticles) {
        return -1;
    }
    return int(idx);
}
