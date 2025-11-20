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
    vec3 sunDirection; float sunIntensity;
    vec3 sunColor;     float pad2;
    float octreeRootIndex;
    float octreeMaxDepth;
    float octreeNodeCount;
    float octreeEnabled;
} uCamera;

layout(std430, set = 0, binding = 0) readonly buffer SdfGrid {
    float values[];
} uSdf;

layout(std430, set = 0, binding = 1) readonly buffer SdfMaterials {
    uint values[];
} uMaterials;

struct SdfOctreeNode {
    // center.xyz = node center in world space; center.w = half-size in meters.
    vec4 centerAndHalf;
    float minDistance;
    float maxDistance;
    uint firstChild;
    uint occupancyMask;
    uint flags;
    uint pad0;
    uint pad1;
    uint pad2;
};

layout(std430, set = 0, binding = 2) readonly buffer SdfOctreeNodes {
    SdfOctreeNode nodes[];
} uOctree;

const uint SDF_OCTREE_NODE_EMPTY      = 1u << 0;
const uint SDF_OCTREE_NODE_SOLID      = 1u << 1;
const uint SDF_OCTREE_NODE_HAS_SURFACE = 1u << 2;

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

bool ray_sphere_bounds(vec3 ro,
                       vec3 rd,
                       float radius,
                       out float tEnter,
                       out float tExit) {
    float a = dot(rd, rd);
    if (a <= 0.0) {
        tEnter = 0.0;
        tExit = 0.0;
        return false;
    }

    float b = 2.0 * dot(ro, rd);
    float c = dot(ro, ro) - radius * radius;
    float disc = b * b - 4.0 * a * c;
    if (disc < 0.0) {
        return false;
    }

    float s = sqrt(disc);
    float inv = 0.5 / a;
    tEnter = (-b - s) * inv;
    tExit  = (-b + s) * inv;
    if (tEnter > tExit) {
        float tmp = tEnter;
        tEnter = tExit;
        tExit = tmp;
    }
    return true;
}

bool point_inside_node(SdfOctreeNode node, vec3 p) {
    float h = node.centerAndHalf.w;
    if (h <= 0.0) {
        return false;
    }
    vec3 d = p - node.centerAndHalf.xyz;
    return abs(d.x) <= h && abs(d.y) <= h && abs(d.z) <= h;
}

bool intersect_box_axis(float ro, float rd, float mn, float mx,
                        inout float tmin, inout float tmax) {
    if (abs(rd) < 1e-6) {
        return ro >= mn && ro <= mx;
    }
    float inv = 1.0 / rd;
    float t1 = (mn - ro) * inv;
    float t2 = (mx - ro) * inv;
    if (t1 > t2) {
        float tmp = t1;
        t1 = t2;
        t2 = tmp;
    }
    tmin = max(tmin, t1);
    tmax = min(tmax, t2);
    return tmax >= tmin;
}

bool advance_with_octree(vec3 ro, vec3 rd, float maxDist, inout float t) {
    if (uCamera.octreeEnabled <= 0.5 ||
        uCamera.octreeNodeCount <= 0.0 ||
        uCamera.octreeRootIndex < 0.0) {
        return false;
    }

    uint nodeCount = uint(uCamera.octreeNodeCount);
    uint rootIndex = uint(uCamera.octreeRootIndex);
    if (nodeCount == 0u || rootIndex >= nodeCount) {
        return false;
    }

    float dirLenSq = dot(rd, rd);
    if (dirLenSq <= 0.0) {
        return false;
    }

    vec3 p = ro + rd * t;
    SdfOctreeNode root = uOctree.nodes[rootIndex];
    if (!point_inside_node(root, p)) {
        return false;
    }

    uint current = rootIndex;
    for (;;) {
        SdfOctreeNode node = uOctree.nodes[current];

        bool hasEmpty = (node.occupancyMask & SDF_OCTREE_NODE_EMPTY) != 0u;
        bool hasSolid = (node.occupancyMask & SDF_OCTREE_NODE_SOLID) != 0u;

        if (!hasEmpty || node.firstChild == 0xffffffffu) {
            break;
        }

        uint firstChild = node.firstChild;
        uint maxChild = firstChild + 8u;
        uint next = 0xffffffffu;

        for (uint child = firstChild;
             child < maxChild && child < nodeCount;
             ++child) {
            if (point_inside_node(uOctree.nodes[child], p)) {
                next = child;
                break;
            }
        }

        if (next == 0xffffffffu) {
            break;
        }
        current = next;
    }

    SdfOctreeNode leaf = uOctree.nodes[current];
    bool pureEmpty = (leaf.occupancyMask == SDF_OCTREE_NODE_EMPTY);
    if (!pureEmpty) {
        return false;
    }

    float h = leaf.centerAndHalf.w;
    if (h <= 0.0) {
        return false;
    }

    vec3 c = leaf.centerAndHalf.xyz;
    vec3 bmin = c - vec3(h);
    vec3 bmax = c + vec3(h);

    float tmin = -1e30;
    float tmax =  1e30;

    if (!intersect_box_axis(ro.x, rd.x, bmin.x, bmax.x, tmin, tmax) ||
        !intersect_box_axis(ro.y, rd.y, bmin.y, bmax.y, tmin, tmax) ||
        !intersect_box_axis(ro.z, rd.z, bmin.z, bmax.z, tmin, tmax)) {
        return false;
    }

    float exitT = tmax;
    if (exitT <= t + 1e-4) {
        return false;
    }

    float newT = min(exitT, maxDist);
    if (newT <= t) {
        return false;
    }

    t = newT;
    return true;
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

// Simple hemisphere ambient: blend between a ground color and a sky color
// based on the up component of the normal.
vec3 hemisphere_ambient(vec3 n) {
    float t = clamp(n.y * 0.5 + 0.5, 0.0, 1.0);
    vec3 skyColor    = vec3(0.25, 0.35, 0.55);
    vec3 groundColor = vec3(0.05, 0.03, 0.02);
    return mix(groundColor, skyColor, t);
}

// Cheap ambient occlusion approximation using a few SDF samples along the
// normal. Concave regions accumulate more occlusion.
float calc_ao(vec3 p, vec3 n) {
    float ao = 0.0;
    float sca = 1.0;
    // Step size in meters; proportional to voxel size.
    float baseStep = 0.75 * uCamera.gridVoxelSize;

    for (int i = 0; i < 4; ++i) {
        float h = baseStep * (1.0 + float(i));
        float d = sample_sdf(p + n * h);
        // If d is smaller than the sampling distance h, treat it as occlusion.
        float delta = max(0.0, h - d);
        ao += delta * sca;
        sca *= 0.7;
    }

    // Scale and invert to [0,1].
    return clamp(1.0 - 1.5 * ao, 0.0, 1.0);
}

bool march_sdf(vec3 ro, vec3 rd, out vec3 hitPos, out vec3 normal) {
    const int   MAX_STEPS = 192;
    const float SURF_EPS  = 0.01;
    const float MIN_STEP  = 0.01;
    const float STEP_SAFETY = 0.8;
    float isoOffset = uCamera.isoFraction * uCamera.gridVoxelSize;

    float boundaryRadius = uCamera.gridHalfExtent;
    float maxDist = (boundaryRadius > 0.0) ? boundaryRadius * 2.0 : 1e6;

    float t = 0.0;
    if (boundaryRadius > 0.0) {
        float tEnter, tExit;
        bool hitSphere = ray_sphere_bounds(ro, rd, boundaryRadius, tEnter, tExit);
        float radiusSq = boundaryRadius * boundaryRadius;
        float originSq = dot(ro, ro);
        bool outside = originSq > radiusSq;

        if (outside) {
            if (!hitSphere || tExit < 0.0) {
                return false;
            }
            t = max(tEnter, 0.0);
        } else if (!hitSphere) {
            // Inside the volume but ray never exits; no need to adjust t.
        }
    }

    float bestAbsD = 1e30;
    vec3  bestPos  = ro;
    float bestT    = 0.0;

    for (int i = 0; i < MAX_STEPS; ++i) {
        if (advance_with_octree(ro, rd, maxDist, t)) {
            if (t >= maxDist) {
                break;
            }
            continue;
        }

        vec3 p = ro + rd * t;
        float d_raw = sample_sdf(p);
        float d = d_raw - isoOffset;

        float absD = abs(d);
        if (absD < bestAbsD) {
            bestAbsD = absD;
            bestPos = p;
            bestT = t;
        }

        if (d < SURF_EPS) {
            hitPos = p;
            normal = estimate_normal(p);
            return true;
        }

        float step = max(d * STEP_SAFETY, MIN_STEP);
        t += step;
    }

    // Near-miss fallback: if we passed very close to the surface without
    // formally hitting it, treat the closest approach as a hit. This helps
    // avoid background pixels at grazing angles.
    float nearMissEps = max(SURF_EPS * 2.0, 0.25 * uCamera.gridVoxelSize);
    if (bestAbsD < nearMissEps) {
        hitPos = bestPos;
        normal = estimate_normal(bestPos);
        return true;
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

    vec3 lightDir = normalize(uCamera.sunDirection);
    vec3 viewDir  = normalize(uCamera.camPos - p);
    float ndotl   = max(dot(n, lightDir), 0.0);

    // Use height above the nominal radius (based on the
    // perturbed SDF) to add color variation.
    float height = sample_sdf(p);
    float hNorm = clamp(height * 0.15 + 0.5, 0.0, 1.0);

    vec3 lowColor  = vec3(0.15, 0.10, 0.05); // darker lowlands
    vec3 midColor  = vec3(0.35, 0.28, 0.14); // mid terrain
    vec3 highColor = vec3(0.80, 0.80, 0.80); // bright peaks

    vec3 baseColor = mix(lowColor, midColor, hNorm);
    baseColor = mix(baseColor, highColor, smoothstep(0.6, 1.0, hNorm));

    // Ambient term with simple AO.
    float ao     = calc_ao(p, n);
    vec3 ambient = hemisphere_ambient(n) * 0.4 * ao;

    // Lambertian diffuse.
    vec3 diffuse = baseColor * ndotl * uCamera.sunIntensity;

    // Blinn-Phong specular highlight.
    float spec = 0.0;
    if (ndotl > 0.0) {
        vec3 h = normalize(lightDir + viewDir);
        float ndoth = max(dot(n, h), 0.0);
        spec = pow(ndoth, 32.0);
    }
    vec3 specular = uCamera.sunColor * spec * 0.25 * uCamera.sunIntensity;

    vec3 color = ambient + diffuse * 0.9 * ao + specular;

    outColor = vec4(color, 1.0);
}
