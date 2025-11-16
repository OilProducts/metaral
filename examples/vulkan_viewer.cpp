#include "metaral/core/coords.hpp"
#include "metaral/platform/platform.hpp"
#include "metaral/render/camera.hpp"
#include "metaral/render/sdf_grid.hpp"
#include "metaral/render/vulkan_renderer.hpp"
#include "metaral/world/edit.hpp"
#include "metaral/world/terrain.hpp"
#include "metaral/world/world.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>

#ifdef METARAL_ENABLE_VULKAN

namespace {

using metaral::core::PlanetPosition;

struct Brush {
    float radius_m = 2.0f;
    float hardness = 1.0f;    // 0–1, for future falloff
    metaral::world::MaterialId material = 1; // default solid
};

struct DirtyRegion {
    PlanetPosition min_p{};
    PlanetPosition max_p{};
    bool valid = false;
};

PlanetPosition normalized(const PlanetPosition& v) {
    const float len = metaral::core::length(v);
    if (len < 1e-6f) {
        return {0.0f, 0.0f, 0.0f};
    }
    const float inv = 1.0f / len;
    return {v.x * inv, v.y * inv, v.z * inv};
}

PlanetPosition add(const PlanetPosition& a, const PlanetPosition& b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

PlanetPosition sub(const PlanetPosition& a, const PlanetPosition& b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

PlanetPosition scale(const PlanetPosition& v, float s) {
    return {v.x * s, v.y * s, v.z * s};
}

PlanetPosition cross(const PlanetPosition& a, const PlanetPosition& b) {
    return {
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x,
    };
}

float dot(const PlanetPosition& a, const PlanetPosition& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

struct CameraBasis {
    PlanetPosition forward{};
    PlanetPosition right{};
    PlanetPosition up{};
};

CameraBasis make_freefly_basis(float yaw_radians, float pitch_radians) {
    CameraBasis basis{};
    const float cos_pitch = std::cos(pitch_radians);
    basis.forward = normalized(PlanetPosition{
        cos_pitch * std::cos(yaw_radians),
        std::sin(pitch_radians),
        cos_pitch * std::sin(yaw_radians),
    });

    const PlanetPosition world_up{0.0f, 1.0f, 0.0f};
    basis.right = normalized(cross(basis.forward, world_up));
    if (metaral::core::length(basis.right) < 1e-6f) {
        basis.right = {1.0f, 0.0f, 0.0f};
    }

    basis.up = normalized(cross(basis.right, basis.forward));
    return basis;
}

PlanetPosition safe_radial_up(const PlanetPosition& pos) {
    PlanetPosition radial = normalized(pos);
    if (metaral::core::length(radial) < 1e-6f) {
        radial = {0.0f, 1.0f, 0.0f};
    }
    return radial;
}

PlanetPosition rotate_around_axis(const PlanetPosition& v,
                                  const PlanetPosition& axis,
                                  float angle) {
    PlanetPosition a = normalized(axis);
    const float c = std::cos(angle);
    const float s = std::sin(angle);

    PlanetPosition term1 = scale(v, c);
    PlanetPosition term2 = scale(cross(a, v), s);
    PlanetPosition term3 = scale(a, dot(a, v) * (1.0f - c));
    return add(add(term1, term2), term3);
}

PlanetPosition project_tangent(const PlanetPosition& v,
                               const PlanetPosition& up) {
    PlanetPosition tangent = sub(v, scale(up, dot(v, up)));
    float len = metaral::core::length(tangent);
    if (len < 1e-6f) {
        PlanetPosition fallback{0.0f, 0.0f, 1.0f};
        if (std::abs(dot(fallback, up)) > 0.95f) {
            fallback = {1.0f, 0.0f, 0.0f};
        }
        tangent = sub(fallback, scale(up, dot(fallback, up)));
        len = metaral::core::length(tangent);
        if (len < 1e-6f) {
            tangent = {1.0f, 0.0f, 0.0f};
            len = metaral::core::length(tangent);
        }
    }
    return scale(tangent, 1.0f / len);
}

float surface_signed_distance(const PlanetPosition& pos,
                              const metaral::render::SdfGrid& grid) {
    // Interpret the grid SDF as a true height-above-surface by raymarching
    // along the local "down" direction instead of using the raw (binary) SDF
    // sample directly. This keeps the walkabout grounding logic, which expects
    // a distance in meters, consistent with the rendered SDF.
    PlanetPosition up = safe_radial_up(pos);
    PlanetPosition down = scale(up, -1.0f);

    constexpr float kMaxDist     = 500.0f;
    constexpr float kSurfEpsilon = 0.01f;
    constexpr int   kMaxSteps    = 128;

    const float iso_offset =
        metaral::render::kDefaultSdfIsoFraction * grid.voxel_size;
    const float t = metaral::render::raymarch_sdf(grid,
                                                  pos,
                                                  down,
                                                  kMaxDist,
                                                  kSurfEpsilon,
                                                  kMaxSteps,
                                                  nullptr,
                                                  iso_offset);

    if (t >= kMaxDist) {
        // Fallback to the analytic radial SDF when the grid does not provide
        // a hit (e.g., far from the planet).
        return metaral::core::length(pos) - grid.planet_radius;
    }

    // Outside the surface this is approximately the height above ground in
    // meters along the radial direction, matching what the walkabout code
    // expects.
    return t;
}

enum class MovementMode {
    FreeFly,
    Walkabout,
};

class VulkanViewer final : public metaral::platform::IApp {
public:
    void on_init(const metaral::platform::AppInitContext& ctx) override;
    void on_frame(const metaral::platform::FrameContext& ctx) override;
    void on_shutdown() override;

private:
    metaral::core::CoordinateConfig coords_{};
    std::unique_ptr<metaral::world::World> world_;
    std::unique_ptr<metaral::render::VulkanRenderer> renderer_;
    metaral::render::Camera camera_{};
    int window_width_ = 0;
    int window_height_ = 0;
    MovementMode mode_ = MovementMode::FreeFly;
    metaral::world::EditMode current_mode_ = metaral::world::EditMode::Dig;
    Brush brush_{};
    DirtyRegion sdf_dirty_{};
    float yaw_freefly_ = 0.0f;
    float pitch_freefly_ = 0.0f;
    float walk_eye_height_m_ = 2.0f;
    float vertical_velocity_ = 0.0f;
};

void VulkanViewer::on_init(const metaral::platform::AppInitContext& ctx) {
    coords_.voxel_size_m = 0.5f;
    coords_.chunk_size = 32;
    coords_.planet_radius_m = 100.0f;
    coords_.planet_center_offset_voxels = {0, 0, 0};

    world_ = std::make_unique<metaral::world::World>(coords_);
    // Generate enough chunks to cover the planet radius (in meters) plus a
    // small margin, so that the voxel world fully contains the analytic
    // terrain surface that the SDF grid will approximate.
    const float meters_per_chunk =
        coords_.voxel_size_m * static_cast<float>(coords_.chunk_size);
    const float chunks_for_radius =
        coords_.planet_radius_m / meters_per_chunk;
    const int chunk_radius =
        static_cast<int>(std::ceil(chunks_for_radius)) + 1; // one-chunk margin
    metaral::world::terrain::generate_planet(*world_, chunk_radius, coords_);

    window_width_ = ctx.window_width;
    window_height_ = ctx.window_height;

    renderer_ = std::make_unique<metaral::render::VulkanRenderer>(ctx.vulkan,
                                                                   static_cast<std::uint32_t>(window_width_),
                                                                   static_cast<std::uint32_t>(window_height_));

    metaral::render::OrbitParameters orbit{};
    orbit.altitude_m = 80.0f;          // high enough to see most of the planet
    orbit.latitude_radians = 0.5f;
    orbit.longitude_radians = 0.8f;
    camera_ = metaral::render::make_orbit_camera(coords_, orbit);

    const PlanetPosition initial_forward = normalized(camera_.forward);
    yaw_freefly_ = std::atan2(initial_forward.z, initial_forward.x);
    pitch_freefly_ = std::asin(std::clamp(initial_forward.y, -1.0f, 1.0f));

    const CameraBasis basis = make_freefly_basis(yaw_freefly_, pitch_freefly_);
    camera_.forward = basis.forward;
    camera_.up = basis.up;
}

void VulkanViewer::on_frame(const metaral::platform::FrameContext& ctx) {
    if (ctx.input.key_escape || ctx.input.quit_requested) {
        ctx.request_quit();
        return;
    }

    if (!renderer_) {
        return;
    }

    const float dt = ctx.dt_seconds;
    const float free_speed = 40.0f;     // meters per second
    const float walk_speed = 10.0f;
    const float gravity_accel = -30.0f; // meters per second^2 toward planet
    const float jump_speed = 15.0f;     // meters per second upward impulse
    const float mouse_sensitivity = 0.0025f;
    const float max_free_pitch = 1.3f;  // ~75 degrees
    const float max_walk_pitch = 0.6f;  // ~35 degrees

    PlanetPosition radial_up = safe_radial_up(camera_.position);

    // Tool/brush input (no editing yet).
    if (ctx.input.key_1_pressed) {
        current_mode_ = metaral::world::EditMode::Dig;
        std::cout << "Edit mode: Dig\n";
    }
    if (ctx.input.key_2_pressed) {
        current_mode_ = metaral::world::EditMode::Fill;
        std::cout << "Edit mode: Fill\n";
    }
    if (ctx.input.key_3_pressed) {
        current_mode_ = metaral::world::EditMode::Paint;
        std::cout << "Edit mode: Paint\n";
    }

    constexpr float kMinBrushRadius = 0.5f;
    constexpr float kMaxBrushRadius = 20.0f;
    constexpr float kBrushStep = 0.5f;

    if (ctx.input.key_period_pressed) {
        brush_.radius_m = std::min(kMaxBrushRadius, brush_.radius_m + kBrushStep);
        std::cout << "Brush radius increased to " << brush_.radius_m << " m\n";
    }
    if (ctx.input.key_comma_pressed) {
        brush_.radius_m = std::max(kMinBrushRadius, brush_.radius_m - kBrushStep);
        std::cout << "Brush radius decreased to " << brush_.radius_m << " m\n";
    }

    // Cycle through a small set of demo material IDs: 1,2,3,...
    if (ctx.input.key_bracket_right_pressed) {
        if (brush_.material == 0) {
            brush_.material = 1;
        } else {
            brush_.material = static_cast<metaral::world::MaterialId>(brush_.material + 1);
        }
        std::cout << "Brush material increased to " << brush_.material << "\n";
    }
    if (ctx.input.key_bracket_left_pressed) {
        if (brush_.material > 1) {
            brush_.material = static_cast<metaral::world::MaterialId>(brush_.material - 1);
        }
        std::cout << "Brush material decreased to " << brush_.material << "\n";
    }

    // Tool fire: left mouse button. This computes the hit, applies the brush
    // to voxels, tracks a dirty SDF region, and logs debug info.
    if (ctx.input.mouse_left_button && renderer_) {
        const metaral::render::SdfGrid* grid = renderer_->sdf_grid();
        if (grid) {
            constexpr float kMaxDist = 1000.0f;
            constexpr float kSurfEpsilon = 0.05f;
            constexpr int   kMaxSteps = 128;

            PlanetPosition dir = normalized(camera_.forward);
            PlanetPosition hit_pos{};
            const float iso_offset =
                metaral::render::kDefaultSdfIsoFraction * grid->voxel_size;
            const bool hit = metaral::render::raycast_sdf(
                *grid,
                camera_.position,
                dir,
                kMaxDist,
                kSurfEpsilon,
                kMaxSteps,
                hit_pos,
                iso_offset);

            if (hit) {
                // Approximate surface normal by radial direction. This matches
                // the spherical planet assumption and is good enough for now.
                PlanetPosition normal = normalized(hit_pos);

                PlanetPosition brush_center = hit_pos;
                if (current_mode_ == metaral::world::EditMode::Dig) {
                    const float offset = 0.5f * brush_.radius_m;
                    brush_center = PlanetPosition{
                        hit_pos.x - normal.x * offset,
                        hit_pos.y - normal.y * offset,
                        hit_pos.z - normal.z * offset,
                    };
                }

                PlanetPosition min_p{
                    brush_center.x - brush_.radius_m,
                    brush_center.y - brush_.radius_m,
                    brush_center.z - brush_.radius_m,
                };
                PlanetPosition max_p{
                    brush_center.x + brush_.radius_m,
                    brush_center.y + brush_.radius_m,
                    brush_center.z + brush_.radius_m,
                };

                if (world_) {
                    metaral::world::EditStats stats{};
                    metaral::world::apply_spherical_brush(
                        *world_,
                        coords_,
                        brush_center,
                        brush_.radius_m,
                        current_mode_,
                        brush_.material,
                        &stats);

                    // Expand the accumulated dirty region in world space.
                    if (!sdf_dirty_.valid) {
                        sdf_dirty_.min_p = min_p;
                        sdf_dirty_.max_p = max_p;
                        sdf_dirty_.valid = true;
                    } else {
                        sdf_dirty_.min_p.x = std::min(sdf_dirty_.min_p.x, min_p.x);
                        sdf_dirty_.min_p.y = std::min(sdf_dirty_.min_p.y, min_p.y);
                        sdf_dirty_.min_p.z = std::min(sdf_dirty_.min_p.z, min_p.z);
                        sdf_dirty_.max_p.x = std::max(sdf_dirty_.max_p.x, max_p.x);
                        sdf_dirty_.max_p.y = std::max(sdf_dirty_.max_p.y, max_p.y);
                        sdf_dirty_.max_p.z = std::max(sdf_dirty_.max_p.z, max_p.z);
                    }

                    std::cout << "Brush applied: touched "
                              << stats.voxels_touched
                              << ", changed "
                              << stats.voxels_changed
                              << "\n";

                    // Mark the renderer's SDF grid as dirty in this region so
                    // it is updated to reflect the changed voxels.
                    renderer_->mark_sdf_dirty(min_p, max_p);
                }

                const metaral::core::WorldVoxelCoord min_v =
                    metaral::core::to_world_voxel(min_p, coords_);
                const metaral::core::WorldVoxelCoord max_v =
                    metaral::core::to_world_voxel(max_p, coords_);

                std::cout << "Tool fired in mode "
                          << (current_mode_ == metaral::world::EditMode::Dig ? "Dig" :
                              current_mode_ == metaral::world::EditMode::Fill ? "Fill" : "Paint")
                          << " at hit_pos=("
                          << hit_pos.x << ", " << hit_pos.y << ", " << hit_pos.z
                          << "), brush_center=("
                          << brush_center.x << ", " << brush_center.y << ", " << brush_center.z
                          << "), voxel_bounds=[("
                          << min_v.x << ", " << min_v.y << ", " << min_v.z << ") -> ("
                          << max_v.x << ", " << max_v.y << ", " << max_v.z << ")]\n";
            } else {
                std::cout << "Tool fired but raycast hit nothing\n";
            }
        } else {
            std::cout << "Tool fired but SDF grid not built yet\n";
        }
    }

    if (ctx.input.key_f_pressed && renderer_) {
        const metaral::render::SdfGrid* grid = renderer_->sdf_grid();
        if (grid) {
            // Debug raycast: shoot from camera along forward, map the hit
            // point back to a world voxel. To avoid hitting the outer "air"
            // voxel shell due to SDF interpolation (zero-crossing landing
            // between solid/empty voxel centers), we bias the hit point a
            // bit along the SDF normal into the solid region before
            // converting to a voxel coordinate.
            constexpr float kMaxDist = 500.0f;
            constexpr float kSurfEpsilon = 0.01f;
            constexpr int   kMaxSteps = 128;

            PlanetPosition hit_pos{};
            const float iso_offset =
                metaral::render::kDefaultSdfIsoFraction * grid->voxel_size;
            const bool hit = metaral::render::raycast_sdf(
                *grid,
                camera_.position,
                camera_.forward,
                kMaxDist,
                kSurfEpsilon,
                kMaxSteps,
                hit_pos,
                iso_offset);

            if (hit && world_) {
                // Estimate SDF normal at the hit; this mirrors the GLSL
                // estimate_normal() helper used in analytic_sphere.frag.
                const float eps = 0.5f * grid->voxel_size;
                const float dx =
                    metaral::render::sample_sdf(*grid, PlanetPosition{hit_pos.x + eps, hit_pos.y, hit_pos.z}) -
                    metaral::render::sample_sdf(*grid, PlanetPosition{hit_pos.x - eps, hit_pos.y, hit_pos.z});
                const float dy =
                    metaral::render::sample_sdf(*grid, PlanetPosition{hit_pos.x, hit_pos.y + eps, hit_pos.z}) -
                    metaral::render::sample_sdf(*grid, PlanetPosition{hit_pos.x, hit_pos.y - eps, hit_pos.z});
                const float dz =
                    metaral::render::sample_sdf(*grid, PlanetPosition{hit_pos.x, hit_pos.y, hit_pos.z + eps}) -
                    metaral::render::sample_sdf(*grid, PlanetPosition{hit_pos.x, hit_pos.y, hit_pos.z - eps});

                PlanetPosition normal_ws = normalized(PlanetPosition{dx, dy, dz});

                // Nudge slightly inside the surface along -normal so that
                // we land in the solid voxel instead of the outer empty one.
                const float bias = 0.5f * coords_.voxel_size_m;
                PlanetPosition biased_hit{
                    hit_pos.x - normal_ws.x * bias,
                    hit_pos.y - normal_ws.y * bias,
                    hit_pos.z - normal_ws.z * bias,
                };

                const auto world_voxel =
                    metaral::world::world_voxel_from_planet_position(biased_hit, coords_);
                const metaral::world::Voxel* voxel =
                    world_->find_voxel(world_voxel);

                const auto material =
                    voxel ? voxel->material : metaral::world::MaterialId{0};

                std::cout << "Ray hit voxel at ("
                          << world_voxel.x << ", "
                          << world_voxel.y << ", "
                          << world_voxel.z << "), material="
                          << material << "\n";
            } else {
                std::cout << "Ray hit nothing in SDF grid\n";
            }
        } else {
            std::cout << "SDF grid not built yet; no raycast\n";
        }
    }

    if (ctx.input.key_tab_pressed) {
        if (mode_ == MovementMode::FreeFly) {
            mode_ = MovementMode::Walkabout;
            vertical_velocity_ = 0.0f;
            const float target_radius = coords_.planet_radius_m + walk_eye_height_m_;
            camera_.position = scale(radial_up, target_radius);
            radial_up = safe_radial_up(camera_.position);

            PlanetPosition forward = camera_.forward;
            if (metaral::core::length(forward) < 1e-6f) {
                forward = {0.0f, 0.0f, 1.0f};
            }
            camera_.forward = project_tangent(forward, radial_up);
            camera_.up = radial_up;
        } else {
            mode_ = MovementMode::FreeFly;
            yaw_freefly_ = std::atan2(camera_.forward.z, camera_.forward.x);
            pitch_freefly_ = std::asin(std::clamp(camera_.forward.y, -1.0f, 1.0f));
            vertical_velocity_ = 0.0f;
        }
    }

    if (ctx.input.mouse_right_button) {
        const float yaw_delta = ctx.input.mouse_delta_x * mouse_sensitivity;
        const float pitch_delta = ctx.input.mouse_delta_y * mouse_sensitivity;
        if (mode_ == MovementMode::Walkabout) {
            PlanetPosition up = safe_radial_up(camera_.position);
            PlanetPosition forward = camera_.forward;
            if (metaral::core::length(forward) < 1e-6f) {
                forward = {0.0f, 0.0f, 1.0f};
            }

            const float current_pitch = std::asin(std::clamp(dot(forward, up), -1.0f, 1.0f));
            const float target_pitch = std::clamp(current_pitch - pitch_delta,
                                                  -max_walk_pitch,
                                                  max_walk_pitch);

            PlanetPosition forward_flat = project_tangent(forward, up);
            forward_flat = rotate_around_axis(forward_flat, up, -yaw_delta);

            PlanetPosition new_forward =
                normalized(add(scale(forward_flat, std::cos(target_pitch)),
                               scale(up, std::sin(target_pitch))));
            camera_.forward = new_forward;
            camera_.up = up;
        } else {
            yaw_freefly_   -= yaw_delta;
            pitch_freefly_ = std::clamp(pitch_freefly_ - pitch_delta, -max_free_pitch, max_free_pitch);
        }
    }

    PlanetPosition forward{};
    PlanetPosition right{};
    PlanetPosition camera_up{};
    if (mode_ == MovementMode::Walkabout) {
        camera_up = safe_radial_up(camera_.position);
        forward = camera_.forward;
        if (metaral::core::length(forward) < 1e-6f) {
            forward = {0.0f, 0.0f, 1.0f};
        }
        forward = normalized(forward);
        right = normalized(cross(forward, camera_up));
        if (metaral::core::length(right) < 1e-6f) {
            right = project_tangent({1.0f, 0.0f, 0.0f}, camera_up);
        }
        camera_.forward = forward;
        camera_.up = camera_up;
    } else {
        CameraBasis basis = make_freefly_basis(yaw_freefly_, pitch_freefly_);
        forward = basis.forward;
        right = basis.right;
        camera_up = basis.up;
        camera_.forward = forward;
        camera_.up = camera_up;
    }

    PlanetPosition move_forward = forward;
    PlanetPosition move_right = right;
    if (mode_ == MovementMode::Walkabout) {
        move_forward = project_tangent(move_forward, camera_up);
        move_right = project_tangent(move_right, camera_up);
    }

    PlanetPosition velocity{};
    if (ctx.input.key_w) {
        velocity = add(velocity, move_forward);
    }
    if (ctx.input.key_s) {
        velocity = sub(velocity, move_forward);
    }
    if (ctx.input.key_d) {
        velocity = add(velocity, move_right);
    }
    if (ctx.input.key_a) {
        velocity = sub(velocity, move_right);
    }
    if (mode_ == MovementMode::FreeFly) {
        if (ctx.input.key_space) {
            velocity = add(velocity, camera_up);
        }
        if (ctx.input.key_shift) {
            velocity = sub(velocity, camera_up);
        }
    }

    const float velocity_len = metaral::core::length(velocity);
    if (velocity_len > 1e-3f) {
        const PlanetPosition dir = scale(velocity, 1.0f / velocity_len);
        const float move_speed = (mode_ == MovementMode::Walkabout) ? walk_speed : free_speed;
        const PlanetPosition delta = scale(dir, move_speed * dt);
        camera_.position = add(camera_.position, delta);
    }

    if (mode_ == MovementMode::Walkabout) {
        radial_up = safe_radial_up(camera_.position);
        const float desired_height = walk_eye_height_m_;
        const metaral::render::SdfGrid* grid =
            renderer_ ? renderer_->sdf_grid() : nullptr;

        float surface_height = 0.0f;
        if (grid) {
            surface_height = surface_signed_distance(camera_.position, *grid);
        } else {
            surface_height =
                metaral::world::terrain::terrain_signed_distance(camera_.position, coords_);
        }
        const bool grounded = surface_height <= desired_height + 0.05f && vertical_velocity_ <= 0.0f;

        if (ctx.input.key_space && grounded) {
            vertical_velocity_ = jump_speed;
        } else if (ctx.input.key_shift && !grounded) {
            vertical_velocity_ -= jump_speed * 0.5f;
        }

        vertical_velocity_ += gravity_accel * dt;
        camera_.position = add(camera_.position, scale(radial_up, vertical_velocity_ * dt));

        if (grid) {
            surface_height = surface_signed_distance(camera_.position, *grid);
        } else {
            surface_height =
                metaral::world::terrain::terrain_signed_distance(camera_.position, coords_);
        }
        if (surface_height < desired_height) {
            const float correction = desired_height - surface_height;
            camera_.position = add(camera_.position, scale(radial_up, correction));
            vertical_velocity_ = 0.0f;
        }
    }

    if (ctx.input.window_resized || ctx.window_width != window_width_ || ctx.window_height != window_height_) {
        window_width_ = ctx.window_width;
        window_height_ = ctx.window_height;
        renderer_->resize(static_cast<std::uint32_t>(window_width_),
                          static_cast<std::uint32_t>(window_height_));
    }

    renderer_->draw_frame(camera_, *world_);
}

void VulkanViewer::on_shutdown() {
    if (renderer_) {
        renderer_->wait_idle();
        renderer_.reset();
    }
    world_.reset();
}

} // namespace

int main() {
    VulkanViewer app;
    metaral::platform::WindowConfig config;
    config.width = 1280;
    config.height = 720;
    config.enable_vulkan = true;
    config.title = "Metaral Vulkan Viewer";
    return metaral::platform::run_app(app, config);
}

#else // METARAL_ENABLE_VULKAN

#include <cstdio>

int main() {
    std::fprintf(stderr, "Vulkan is disabled in this build. Enable METARAL_ENABLE_VULKAN to run this viewer.\n");
    return 1;
}

#endif // METARAL_ENABLE_VULKAN
