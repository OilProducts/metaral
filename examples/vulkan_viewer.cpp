#include "metaral/core/coords.hpp"
#include "metaral/platform/platform.hpp"
#include "metaral/render/camera.hpp"
#include "metaral/render/vulkan_renderer.hpp"
#include "metaral/world/world.hpp"

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
};

void VulkanViewer::on_init(const metaral::platform::AppInitContext& ctx) {
    coords_.voxel_size_m = 0.5f;
    coords_.chunk_size = 32;
    coords_.planet_radius_m = 50.0f;
    coords_.planet_center_offset_voxels = {0, 0, 0};

    world_ = std::make_unique<metaral::world::World>(coords_);
    metaral::world::fill_sphere(*world_, 3, /*solid*/ 1, /*empty*/ 0);

    window_width_ = ctx.window_width;
    window_height_ = ctx.window_height;

    renderer_ = std::make_unique<metaral::render::VulkanRenderer>(ctx.vulkan,
                                                                   static_cast<std::uint32_t>(window_width_),
                                                                   static_cast<std::uint32_t>(window_height_));

    metaral::render::OrbitParameters orbit{};
    orbit.altitude_m = 20.0f;
    orbit.latitude_radians = 0.4f;
    orbit.longitude_radians = 0.6f;
    camera_ = metaral::render::make_orbit_camera(coords_, orbit);
}

void VulkanViewer::on_frame(const metaral::platform::FrameContext& ctx) {
    if (ctx.input.key_escape || ctx.input.quit_requested) {
        ctx.request_quit();
        return;
    }

    if (!renderer_) {
        return;
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

