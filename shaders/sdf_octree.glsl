// Shared SDF octree definitions for GLSL (compute/fragment).
// Consumers can override SDF_OCTREE_BINDING before including to set the buffer
// binding slot used for the nodes SSBO.
#ifndef SDF_OCTREE_BINDING
#define SDF_OCTREE_BINDING 2
#endif

struct SdfOctreeNode {
    // centerAndHalf.xyz = node center in world space; .w = half-size in meters.
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

layout(std430, binding = SDF_OCTREE_BINDING) readonly buffer SdfOctreeNodes {
    SdfOctreeNode nodes[];
} uOctreeNodes;

const uint SDF_OCTREE_NODE_EMPTY       = 1u << 0;
const uint SDF_OCTREE_NODE_SOLID       = 1u << 1;
const uint SDF_OCTREE_NODE_HAS_SURFACE = 1u << 2;

// Dense SDF sample helper that matches the CPU layout used for collision.
float sample_sdf_dense(vec3 p) {
    float dim = uFluid.sdfParams.x;
    float voxelSize = uFluid.sdfParams.y;
    float halfExtent = uFluid.sdfParams.z;
    if (dim < 1.0 || voxelSize <= 0.0 || halfExtent <= 0.0) {
        return length(p) - uFluid.planetParams.x;
    }

    vec3 grid = (p + vec3(halfExtent)) / voxelSize - vec3(0.5);
    vec3 c0 = floor(grid);
    vec3 f = clamp(grid - c0, vec3(0.0), vec3(1.0));
    ivec3 i0 = ivec3(clamp(c0, vec3(0.0), vec3(dim - 1.0)));
    ivec3 i1 = ivec3(clamp(c0 + vec3(1.0), vec3(0.0), vec3(dim - 1.0)));
    int d = int(dim);
    int idx000 = (i0.z * d + i0.y) * d + i0.x;
    int idx100 = (i0.z * d + i0.y) * d + i1.x;
    int idx010 = (i0.z * d + i1.y) * d + i0.x;
    int idx110 = (i0.z * d + i1.y) * d + i1.x;
    int idx001 = (i1.z * d + i0.y) * d + i0.x;
    int idx101 = (i1.z * d + i0.y) * d + i1.x;
    int idx011 = (i1.z * d + i1.y) * d + i0.x;
    int idx111 = (i1.z * d + i1.y) * d + i1.x;

    float c000 = uSdf.values[idx000];
    float c100 = uSdf.values[idx100];
    float c010 = uSdf.values[idx010];
    float c110 = uSdf.values[idx110];
    float c001 = uSdf.values[idx001];
    float c101 = uSdf.values[idx101];
    float c011 = uSdf.values[idx011];
    float c111 = uSdf.values[idx111];

    float c00 = mix(c000, c100, f.x);
    float c10 = mix(c010, c110, f.x);
    float c01 = mix(c001, c101, f.x);
    float c11 = mix(c011, c111, f.x);
    float c0z = mix(c00, c10, f.y);
    float c1z = mix(c01, c11, f.y);
    return mix(c0z, c1z, f.z);
}

bool point_inside_node(SdfOctreeNode node, vec3 p) {
    float h = node.centerAndHalf.w;
    if (h <= 0.0) {
        return false;
    }
    vec3 d = abs(p - node.centerAndHalf.xyz);
    return d.x <= h && d.y <= h && d.z <= h;
}

// Sample SDF using the octree occupancy where possible; fall back to dense grid.
float sample_sdf_octree(vec3 p) {
    if (uFluid.octreeParams.w <= 0.5) {
        return sample_sdf_dense(p);
    }

    uint nodeCount = uint(uFluid.octreeParams.x);
    uint rootIndex = uint(uFluid.octreeParams.y);
    if (nodeCount == 0u || rootIndex >= nodeCount) {
        return sample_sdf_dense(p);
    }

    SdfOctreeNode root = uOctreeNodes.nodes[rootIndex];
    if (!point_inside_node(root, p)) {
        return length(p) - uFluid.planetParams.x;
    }

    uint current = rootIndex;
    for (;;) {
        SdfOctreeNode node = uOctreeNodes.nodes[current];
        bool hasEmpty = (node.occupancyMask & SDF_OCTREE_NODE_EMPTY) != 0u;
        bool leaf = (node.firstChild == 0xffffffffu) || !hasEmpty;
        if (leaf) {
            // Pure empty: return a positive distance if available.
            if (node.occupancyMask == SDF_OCTREE_NODE_EMPTY) {
                if (node.minDistance > 0.0) {
                    return node.minDistance;
                }
                return sample_sdf_dense(p);
            }
            // Pure solid: return a conservative negative distance if available.
            if (node.occupancyMask == SDF_OCTREE_NODE_SOLID && node.maxDistance > 0.0) {
                return -node.maxDistance;
            }
            // Mixed / has surface: sample dense grid for accuracy.
            return sample_sdf_dense(p);
        }

        uint next = 0xffffffffu;
        uint firstChild = node.firstChild;
        uint maxChild = firstChild + 8u;
        for (uint child = firstChild; child < maxChild && child < nodeCount; ++child) {
            if (point_inside_node(uOctreeNodes.nodes[child], p)) {
                next = child;
                break;
            }
        }
        if (next == 0xffffffffu) {
            return sample_sdf_dense(p);
        }
        current = next;
    }
}

vec3 sample_sdf_normal_octree(vec3 p) {
    float eps = max(0.5, uFluid.sdfParams.y) * 0.5;
    float dx = sample_sdf_octree(p + vec3(eps, 0.0, 0.0)) - sample_sdf_octree(p - vec3(eps, 0.0, 0.0));
    float dy = sample_sdf_octree(p + vec3(0.0, eps, 0.0)) - sample_sdf_octree(p - vec3(0.0, eps, 0.0));
    float dz = sample_sdf_octree(p + vec3(0.0, 0.0, eps)) - sample_sdf_octree(p - vec3(0.0, 0.0, eps));
    vec3 n = vec3(dx, dy, dz);
    float len = length(n);
    if (len < 1e-5) {
        return vec3(0.0, 1.0, 0.0);
    }
    return n / len;
}
