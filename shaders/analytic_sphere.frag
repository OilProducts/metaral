#version 450

layout(location = 0) in vec2 vUV;
layout(location = 0) out vec4 outColor;

layout(push_constant) uniform CameraParams {
    vec3 camPos;      float planetRadius;
    vec3 forward;     float fovY;
    vec3 right;       float aspect;
    vec3 up;          float pad1;
} uCamera;

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

bool march_sphere(vec3 ro, vec3 rd, float radius, out vec3 hitPos, out vec3 normal) {
    const int   MAX_STEPS = 128;
    const float MAX_DIST  = 500.0;
    const float SURF_EPS  = 0.001;
    const float MIN_STEP  = 0.01;

    float t = 0.0;
    for (int i = 0; i < MAX_STEPS; ++i) {
        vec3 p = ro + rd * t;
        float d = length(p) - radius; // exact SDF for sphere at origin

        if (d < SURF_EPS) {
            hitPos = p;
            normal = normalize(p);
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
    if (!march_sphere(ro, rd, uCamera.planetRadius, p, n)) {
        outColor = vec4(0.02, 0.04, 0.1, 1.0);
        return;
    }

    vec3 lightDir = normalize(vec3(0.3, 0.8, 0.4));
    float ndotl = max(dot(n, lightDir), 0.0);

    vec3 baseColor = mix(vec3(0.2, 0.15, 0.05), vec3(0.4, 0.35, 0.1), n.y * 0.5 + 0.5);
    vec3 color = baseColor * (0.2 + 0.8 * ndotl);

    outColor = vec4(color, 1.0);
}
