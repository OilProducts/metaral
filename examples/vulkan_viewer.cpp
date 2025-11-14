#include "metaral/core/coords.hpp"
#include "metaral/platform/platform.hpp"
#include "metaral/render/camera.hpp"
#include "metaral/render/vulkan_renderer.hpp"
#include "metaral/world/world.hpp"

#include <algorithm>
#include <memory>

#ifdef METARAL_ENABLE_VULKAN

namespace {

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
    metaral::render::OrbitParameters orbit_{};
    float yaw_offset_ = 0.0f;
    float pitch_offset_ = 0.0f;
};

void VulkanViewer::on_init(const metaral::platform::AppInitContext& ctx) {
    coords_.voxel_size_m = 1.0f;
    coords_.chunk_size = 32;
    coords_.planet_radius_m = 100.0f;
    coords_.planet_center_offset_voxels = {0, 0, 0};

    world_ = std::make_unique<metaral::world::World>(coords_);
    metaral::world::fill_sphere(*world_, 4, /*solid*/ 1, /*empty*/ 0);

    window_width_ = ctx.window_width;
    window_height_ = ctx.window_height;

    renderer_ = std::make_unique<metaral::render::VulkanRenderer>(ctx.vulkan,
                                                                   static_cast<std::uint32_t>(window_width_),
                                                                   static_cast<std::uint32_t>(window_height_));

    orbit_.altitude_m = 80.0f;          // high enough to see most of the planet
    orbit_.latitude_radians = 0.5f;
    orbit_.longitude_radians = 0.8f;
    camera_ = metaral::render::make_orbit_camera(coords_, orbit_);
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
    const float orbit_speed = 0.4f;      // radians per second
    const float altitude_speed = 40.0f;  // meters per second
    const float min_altitude = 5.0f;
    const float max_lat = 1.4f;          // ~80 degrees
    const float mouse_sensitivity = 0.0025f;

    if (ctx.input.key_w) {
        orbit_.latitude_radians += orbit_speed * dt;
    }
    if (ctx.input.key_s) {
        orbit_.latitude_radians -= orbit_speed * dt;
    }
    if (ctx.input.key_a) {
        orbit_.longitude_radians -= orbit_speed * dt;
    }
    if (ctx.input.key_d) {
        orbit_.longitude_radians += orbit_speed * dt;
    }

    orbit_.latitude_radians =
        std::clamp(orbit_.latitude_radians, -max_lat, max_lat);

    if (ctx.input.key_space) {
        orbit_.altitude_m += altitude_speed * dt;
    }
    if (ctx.input.key_shift) {
        orbit_.altitude_m = std::max(min_altitude, orbit_.altitude_m - altitude_speed * dt);
    }

    if (ctx.input.mouse_right_button) {
        yaw_offset_   -= ctx.input.mouse_delta_x * mouse_sensitivity;
        pitch_offset_ -= ctx.input.mouse_delta_y * mouse_sensitivity;
        const float max_pitch = 1.3f; // ~75 degrees
        pitch_offset_ = std::clamp(pitch_offset_, -max_pitch, max_pitch);
    }

    camera_ = metaral::render::make_orbit_camera(coords_, orbit_, yaw_offset_, pitch_offset_);

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
