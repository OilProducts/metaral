#include "metaral/core/coords.hpp"
#include "metaral/platform/platform.hpp"
#include "metaral/render/camera.hpp"
#include "metaral/render/raymarch.hpp"
#include "metaral/world/world.hpp"

#include <SDL3/SDL.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>

namespace {

constexpr int kFrameWidth  = 800;
constexpr int kFrameHeight = 600;

struct Pixel {
    std::uint8_t r = 0;
    std::uint8_t g = 0;
    std::uint8_t b = 0;
    std::uint8_t a = 255;
};

class MinimalViewer final : public metaral::platform::IApp {
public:
    void on_init(const metaral::platform::AppInitContext&) override;
    void on_frame(const metaral::platform::FrameContext& ctx) override;
    void on_shutdown() override;

private:
    void render_frame();

    metaral::core::CoordinateConfig coords_{};
    std::unique_ptr<metaral::world::World> world_;

    SDL_Window* window_ = nullptr;
    SDL_Renderer* renderer_ = nullptr;
    SDL_Texture* texture_ = nullptr;

    std::vector<Pixel> framebuffer_{};
    metaral::render::Camera camera_{};
};

void MinimalViewer::on_init(const metaral::platform::AppInitContext& ctx) {
    coords_.voxel_size_m = 0.5f;
    coords_.chunk_size = 32;
    coords_.planet_radius_m = 50.0f;
    coords_.planet_center_offset_voxels = {0, 0, 0};

    world_ = std::make_unique<metaral::world::World>(coords_);
    metaral::world::fill_sphere(*world_, 3, /*solid*/ 1, /*empty*/ 0);

    window_ = static_cast<SDL_Window*>(ctx.native_window);
    renderer_ = SDL_CreateRenderer(window_, nullptr);
    texture_ = SDL_CreateTexture(renderer_, SDL_PIXELFORMAT_RGBA32, SDL_TEXTUREACCESS_STREAMING, kFrameWidth, kFrameHeight);
    framebuffer_.resize(kFrameWidth * kFrameHeight);

    metaral::render::OrbitParameters orbit{};
    orbit.altitude_m = 20.0f;
    orbit.latitude_radians = 0.4f;
    orbit.longitude_radians = 0.6f;
    camera_ = metaral::render::make_orbit_camera(coords_, orbit);
}

void MinimalViewer::render_frame() {
    using namespace metaral;

    const render::RaymarchSettings settings{
        .max_distance_m = 200.0f,
        .step_size_m = 0.5f,
    };

    auto normalize = [](const core::PlanetPosition& v) {
        const float len = metaral::core::length(v);
        if (len < 1e-6f) {
            return core::PlanetPosition{0.0f, 1.0f, 0.0f};
        }
        const float inv = 1.0f / len;
        return core::PlanetPosition{v.x * inv, v.y * inv, v.z * inv};
    };

    auto cross = [](const core::PlanetPosition& a, const core::PlanetPosition& b) {
        return core::PlanetPosition{
            a.y * b.z - a.z * b.y,
            a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x,
        };
    };

    const float aspect = static_cast<float>(kFrameWidth) / static_cast<float>(kFrameHeight);
    const float scale = std::tan(camera_.fov_y_radians * 0.5f);
    const core::PlanetPosition right = normalize(cross(camera_.forward, camera_.up));
    const core::PlanetPosition up = camera_.up;

    for (int y = 0; y < kFrameHeight; ++y) {
        for (int x = 0; x < kFrameWidth; ++x) {
            const float ndc_x = (2.0f * static_cast<float>(x) / kFrameWidth - 1.0f);
            const float ndc_y = (1.0f - 2.0f * static_cast<float>(y) / kFrameHeight);
            const float u = ndc_x * aspect * scale;
            const float v = ndc_y * scale;

            core::PlanetPosition dir{
                camera_.forward.x + u * right.x + v * up.x,
                camera_.forward.y + u * right.y + v * up.y,
                camera_.forward.z + u * right.z + v * up.z,
            };
            dir = normalize(dir);

            const auto result = render::march_ray(*world_, camera_.position, dir, settings);
            Pixel color{};
            if (result.hit) {
                const float t = std::clamp(result.height / 20.0f + 0.5f, 0.0f, 1.0f);
                color.r = static_cast<std::uint8_t>(200.0f * t);
                color.g = static_cast<std::uint8_t>(180.0f * (1.0f - t));
                color.b = 50;
            } else {
                color.r = 4;
                color.g = 6;
                color.b = 18;
            }
            framebuffer_[y * kFrameWidth + x] = color;
        }
    }

    SDL_UpdateTexture(texture_, nullptr, framebuffer_.data(), kFrameWidth * sizeof(Pixel));
    SDL_RenderClear(renderer_);
    SDL_RenderTexture(renderer_, texture_, nullptr, nullptr);
    SDL_RenderPresent(renderer_);
}

void MinimalViewer::on_frame(const metaral::platform::FrameContext& ctx) {
    if (ctx.input.key_escape || ctx.input.quit_requested) {
        ctx.request_quit();
        return;
    }

    render_frame();
}

void MinimalViewer::on_shutdown() {
    framebuffer_.clear();

    if (texture_) {
        SDL_DestroyTexture(texture_);
        texture_ = nullptr;
    }
    if (renderer_) {
        SDL_DestroyRenderer(renderer_);
        renderer_ = nullptr;
    }
    window_ = nullptr;
    world_.reset();
}

} // namespace

int main() {
    MinimalViewer app;
    metaral::platform::WindowConfig config;
    config.width = kFrameWidth;
    config.height = kFrameHeight;
    config.enable_vulkan = false;
    config.title = "Metaral Minimal Viewer";
    return metaral::platform::run_app(app, config);
}
