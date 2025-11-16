#version 450

layout(location = 0) in vec2 vUV;
layout(location = 0) out vec4 outColor;

layout(push_constant) uniform CameraParams {
    vec3 camPos;      float planetRadius;
    vec3 forward;     float fovY;
    vec3 right;       float aspect;
    vec3 up;          float pad1;
    float gridDim;
    float gridVoxelSize;
    float gridHalfExtent;
    float isoFraction;
} uCamera;

layout(std430, set = 0, binding = 0) readonly buffer SdfGrid {
    float values[];
} uSdf;

layout(std430, set = 0, binding = 1) readonly buffer SdfMaterials {
    uint values[];
} uMaterials;

vec3 ray_direction(vec2 uv) {
    // uv in [0,1]; convert to NDC [-1,1]
    float ndcX = 2.0 * uv.x - 1.0;
    float ndcY = 1.0 - 2.0 * uv.y;

    float scale = tan(uCamera.fovY * 0.5);
    float x = ndcX * uCamera.aspect * scale;
    float y = ndcY * scale;

    vec3 dir = normalize(uCamera.forward + x * uCamera.right + y * uCamera.up);
    return dir;
}

float sample_sdf(vec3 p) {
    float halfExtent = uCamera.gridHalfExtent;
    float dim = uCamera.gridDim;
    float voxelSize = uCamera.gridVoxelSize;

    vec3 coord = (p + vec3(halfExtent)) / voxelSize - vec3(0.5);

    vec3 base = floor(coord);
    vec3 frac = coord - base;
    ivec3 i0 = ivec3(base);
    ivec3 i1 = i0 + ivec3(1);

    int gridDimInt = int(dim);

    if (any(lessThan(i0, ivec3(0))) ||
        any(greaterThanEqual(i1, ivec3(gridDimInt)))) {
        return length(p) - uCamera.planetRadius;
    }

    int idx000 = (i0.z * gridDimInt + i0.y) * gridDimInt + i0.x;
    int idx100 = idx000 + 1;
    int idx010 = idx000 + gridDimInt;
    int idx110 = idx010 + 1;
    int idx001 = idx000 + gridDimInt * gridDimInt;
    int idx101 = idx001 + 1;
    int idx011 = idx001 + gridDimInt;
    int idx111 = idx011 + 1;

    float v000 = uSdf.values[idx000];
    float v100 = uSdf.values[idx100];
    float v010 = uSdf.values[idx010];
    float v110 = uSdf.values[idx110];
    float v001 = uSdf.values[idx001];
    float v101 = uSdf.values[idx101];
    float v011 = uSdf.values[idx011];
    float v111 = uSdf.values[idx111];

    float vx00 = mix(v000, v100, frac.x);
    float vx10 = mix(v010, v110, frac.x);
    float vx01 = mix(v001, v101, frac.x);
    float vx11 = mix(v011, v111, frac.x);

    float vy0 = mix(vx00, vx10, frac.y);
    float vy1 = mix(vx01, vx11, frac.y);

    return mix(vy0, vy1, frac.z);
}

vec3 estimate_normal(vec3 p) {
    float eps = 0.5 * uCamera.gridVoxelSize;
    float dx = sample_sdf(p + vec3(eps, 0.0, 0.0)) - sample_sdf(p - vec3(eps, 0.0, 0.0));
    float dy = sample_sdf(p + vec3(0.0, eps, 0.0)) - sample_sdf(p - vec3(0.0, eps, 0.0));
    float dz = sample_sdf(p + vec3(0.0, 0.0, eps)) - sample_sdf(p - vec3(0.0, 0.0, eps));
    return normalize(vec3(dx, dy, dz));
}

bool march_sdf(vec3 ro, vec3 rd, out vec3 hitPos, out vec3 normal) {
    const int   MAX_STEPS = 128;
    const float MAX_DIST  = 500.0;
    const float SURF_EPS  = 0.01;
    const float MIN_STEP  = 0.01;
    float isoOffset = uCamera.isoFraction * uCamera.gridVoxelSize;

    float t = 0.0;
    for (int i = 0; i < MAX_STEPS; ++i) {
        vec3 p = ro + rd * t;
        float d_raw = sample_sdf(p);
        float d = d_raw - isoOffset;

        if (d < SURF_EPS) {
            hitPos = p;
            normal = estimate_normal(p);
            return true;
        }

        float step = max(d, MIN_STEP);
        t += step;
        if (t > MAX_DIST) {
            break;
        }
    }
    return false;
}

void main() {
    vec3 ro = uCamera.camPos;
    vec3 rd = ray_direction(vUV);

    vec3 p, n;
    if (!march_sdf(ro, rd, p, n)) {
        outColor = vec4(0.02, 0.04, 0.1, 1.0);
        return;
    }

    vec3 lightDir = normalize(vec3(0.3, 0.8, 0.4));
    float ndotl = max(dot(n, lightDir), 0.0);

    // Use height above the nominal radius (based on the
    // perturbed SDF) to add color variation.
    float height = sample_sdf(p);
    float hNorm = clamp(height * 0.15 + 0.5, 0.0, 1.0);

    vec3 lowColor  = vec3(0.15, 0.10, 0.05); // darker lowlands
    vec3 midColor  = vec3(0.35, 0.28, 0.14); // mid terrain
    vec3 highColor = vec3(0.80, 0.80, 0.80); // bright peaks

    vec3 baseColor = mix(lowColor, midColor, hNorm);
    baseColor = mix(baseColor, highColor, smoothstep(0.6, 1.0, hNorm));

    vec3 color = baseColor * (0.25 + 0.75 * ndotl);

    outColor = vec4(color, 1.0);
}
