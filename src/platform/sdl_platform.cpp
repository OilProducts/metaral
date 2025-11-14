#include "metaral/platform/platform.hpp"

#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>

#include <chrono>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace metaral::platform {

namespace {
constexpr Uint32 build_window_flags(const WindowConfig& config) {
    Uint32 flags = SDL_WINDOW_RESIZABLE;
    if (config.enable_vulkan) {
        flags |= SDL_WINDOW_VULKAN;
    }
#ifdef SDL_WINDOW_HIGH_PIXEL_DENSITY
    if (config.high_pixel_density) {
        flags |= SDL_WINDOW_HIGH_PIXEL_DENSITY;
    }
#endif
    return flags;
}

std::runtime_error make_sdl_error(const std::string& context) {
    return std::runtime_error(context + ": " + SDL_GetError());
}
} // namespace

struct SdlPlatform::Impl {
    WindowConfig config{};
    SDL_Window* window = nullptr;
    bool sdl_initialized = false;
};

SdlPlatform::SdlPlatform(const WindowConfig& config)
    : impl_(std::make_unique<Impl>()) {
    impl_->config = config;

    if (!SDL_Init(SDL_INIT_VIDEO | SDL_INIT_EVENTS)) {
        throw make_sdl_error("SDL_Init");
    }

    impl_->sdl_initialized = true;

    impl_->window = SDL_CreateWindow(
        config.title.c_str(),
        config.width,
        config.height,
        build_window_flags(config));

    if (!impl_->window) {
        throw make_sdl_error("SDL_CreateWindow");
    }
}

SdlPlatform::~SdlPlatform() {
    if (impl_->window) {
        SDL_DestroyWindow(impl_->window);
        impl_->window = nullptr;
    }

    if (impl_->sdl_initialized) {
        SDL_Quit();
        impl_->sdl_initialized = false;
    }
}

bool SdlPlatform::poll_event(Event& out_event) {
    SDL_Event event;
    while (SDL_PollEvent(&event)) {
        switch (event.type) {
        case SDL_EVENT_QUIT:
            out_event.type = EventType::Quit;
            return true;
        case SDL_EVENT_WINDOW_RESIZED:
            out_event.type = EventType::WindowResized;
            out_event.window_resized.width = event.window.data1;
            out_event.window_resized.height = event.window.data2;
            impl_->config.width = event.window.data1;
            impl_->config.height = event.window.data2;
            return true;
        case SDL_EVENT_KEY_DOWN:
            out_event.type = EventType::KeyDown;
            out_event.key.keycode = event.key.key;
            out_event.key.scancode = event.key.scancode;
            out_event.key.repeat = event.key.repeat != 0;
            return true;
        case SDL_EVENT_KEY_UP:
            out_event.type = EventType::KeyUp;
            out_event.key.keycode = event.key.key;
            out_event.key.scancode = event.key.scancode;
            out_event.key.repeat = false;
            return true;
        default:
            break;
        }
    }

    out_event.type = EventType::None;
    return false;
}

void SdlPlatform::pump_events() {
    SDL_PumpEvents();
}

void SdlPlatform::request_quit() {
    SDL_Event quit_event{};
    quit_event.type = SDL_EVENT_QUIT;
    SDL_PushEvent(&quit_event);
}

std::vector<std::string> SdlPlatform::required_vulkan_instance_extensions() const {
    if (!impl_->config.enable_vulkan) {
        return {};
    }

    uint32_t count = 0;
    const char* const* names = SDL_Vulkan_GetInstanceExtensions(&count);
    if (!names) {
        throw make_sdl_error("SDL_Vulkan_GetInstanceExtensions");
    }

    std::vector<std::string> extensions;
    extensions.reserve(count);
    for (uint32_t i = 0; i < count; ++i) {
        const char* name = names[i];
        extensions.emplace_back(name);
    }

    return extensions;
}

VkSurfaceKHR SdlPlatform::create_vulkan_surface(VkInstance instance) const {
    if (!impl_->config.enable_vulkan) {
        throw std::runtime_error("Vulkan surface requested but WindowConfig::enable_vulkan is false");
    }

    VkSurfaceKHR surface = VK_NULL_HANDLE;
    if (!SDL_Vulkan_CreateSurface(impl_->window, instance, nullptr, &surface)) {
        throw make_sdl_error("SDL_Vulkan_CreateSurface");
    }

    return surface;
}

int SdlPlatform::width() const noexcept {
    return impl_->config.width;
}

int SdlPlatform::height() const noexcept {
    return impl_->config.height;
}

void* SdlPlatform::native_window_handle() const noexcept {
    return impl_->window;
}

} // namespace metaral::platform

namespace metaral::platform {

namespace {

struct LoopState {
    bool running = true;
    bool escape_down = false;
    bool resized_this_frame = false;
    bool quit_requested = false;
    int window_width = 0;
    int window_height = 0;
};

void handle_event(const Event& event, LoopState& state) {
    switch (event.type) {
    case EventType::Quit:
        state.quit_requested = true;
        break;
    case EventType::WindowResized:
        state.window_width = event.window_resized.width;
        state.window_height = event.window_resized.height;
        state.resized_this_frame = true;
        break;
    case EventType::KeyDown:
        if (event.key.keycode == SDLK_ESCAPE) {
            state.escape_down = true;
        }
        break;
    case EventType::KeyUp:
        if (event.key.keycode == SDLK_ESCAPE) {
            state.escape_down = false;
        }
        break;
    default:
        break;
    }
}

} // namespace

int run_app(IApp& app, const WindowConfig& cfg) {
    SdlPlatform platform(cfg);

    LoopState state{};
    state.window_width = platform.width();
    state.window_height = platform.height();

    VulkanContext vulkan_context;
    vulkan_context.required_instance_extensions = [&platform]() {
        return platform.required_vulkan_instance_extensions();
    };
    vulkan_context.create_surface = [&platform](VkInstance instance) {
        return platform.create_vulkan_surface(instance);
    };

    AppInitContext init_ctx{
        .vulkan = vulkan_context,
        .window_width = state.window_width,
        .window_height = state.window_height,
        .native_window = platform.native_window_handle(),
    };

    app.on_init(init_ctx);

    using clock = std::chrono::steady_clock;
    auto last_tick = clock::now();

    while (state.running) {
        state.resized_this_frame = false;
        state.quit_requested = false;

        Event event;
        while (platform.poll_event(event)) {
            handle_event(event, state);
        }

        auto now = clock::now();
        float dt = std::chrono::duration<float>(now - last_tick).count();
        last_tick = now;

        FrameInput frame_input{};
        frame_input.quit_requested = state.quit_requested;
        frame_input.key_escape = state.escape_down;
        frame_input.window_resized = state.resized_this_frame;

        FrameContext frame_ctx{};
        frame_ctx.dt_seconds = dt;
        frame_ctx.input = frame_input;
        frame_ctx.window_width = state.window_width;
        frame_ctx.window_height = state.window_height;
        frame_ctx.request_quit = [&state]() { state.running = false; };

        app.on_frame(frame_ctx);

        if (state.quit_requested) {
            state.running = false;
        }
    }

    app.on_shutdown();
    return 0;
}

} // namespace metaral::platform
