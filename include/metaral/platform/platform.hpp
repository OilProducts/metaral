#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#if __has_include(<vulkan/vulkan.h>)
#include <vulkan/vulkan.h>
#else
struct VkInstance_T;
using VkInstance = VkInstance_T*;
struct VkSurfaceKHR_T;
using VkSurfaceKHR = VkSurfaceKHR_T*;
#endif

namespace metaral::platform {

enum class EventType {
    None,
    Quit,
    WindowResized,
    KeyDown,
    KeyUp,
    MouseMotion,
    MouseButtonDown,
    MouseButtonUp,
};

struct WindowResizedEvent {
    int width = 0;
    int height = 0;
};

struct KeyEvent {
    int keycode = 0;
    int scancode = 0;
    bool repeat = false;
};

struct MouseMotionEvent {
    int dx = 0;
    int dy = 0;
};

struct MouseButtonEvent {
    int button = 0;
    bool pressed = false;
};

struct Event {
    EventType type = EventType::None;
    WindowResizedEvent window_resized{};
    KeyEvent key{};
    MouseMotionEvent mouse_motion{};
    MouseButtonEvent mouse_button{};
};

struct WindowConfig {
    int width = 1280;
    int height = 720;
    bool resizable = true;
    bool enable_vulkan = true;
    bool high_pixel_density = true;
    std::string title = "Metaral";
};

struct FrameInput {
    bool quit_requested = false;
    bool key_escape = false;
    bool window_resized = false;
    bool key_w = false;
    bool key_a = false;
    bool key_s = false;
    bool key_d = false;
    bool key_space = false;
    bool key_shift = false;
    bool key_tab_pressed = false;
    bool key_f_pressed = false;
    bool key_1_pressed = false;
    bool key_2_pressed = false;
    bool key_3_pressed = false;
    bool key_plus_pressed = false;
    bool key_minus_pressed = false;
    bool key_comma_pressed = false;
    bool key_period_pressed = false;
    bool key_bracket_left_pressed = false;
    bool key_bracket_right_pressed = false;
    bool mouse_left_button = false;
    bool mouse_right_button = false;
    float mouse_delta_x = 0.0f;
    float mouse_delta_y = 0.0f;
};

struct FrameContext {
    float dt_seconds = 0.0f;
    FrameInput input{};
    int window_width = 0;
    int window_height = 0;
    std::function<void()> request_quit;
};

struct VulkanContext {
    std::function<std::vector<std::string>()> required_instance_extensions;
    std::function<VkSurfaceKHR(VkInstance)> create_surface;
};

struct AppInitContext {
    VulkanContext vulkan;
    int window_width = 0;
    int window_height = 0;
    void* native_window = nullptr; // backend-specific handle (SDL_Window*)
};

class IApp {
public:
    virtual ~IApp() = default;
    virtual void on_init(const AppInitContext& ctx) { (void)ctx; }
    virtual void on_frame(const FrameContext& ctx) = 0;
    virtual void on_shutdown() {}
};

int run_app(IApp& app, const WindowConfig& cfg);

class SdlPlatform {
public:
    explicit SdlPlatform(const WindowConfig& config);
    ~SdlPlatform();

    SdlPlatform(SdlPlatform&&) noexcept = default;
    SdlPlatform& operator=(SdlPlatform&&) noexcept = default;

    SdlPlatform(const SdlPlatform&) = delete;
    SdlPlatform& operator=(const SdlPlatform&) = delete;

    [[nodiscard]] bool poll_event(Event& out_event);
    void pump_events();
    void request_quit();

    [[nodiscard]] std::vector<std::string> required_vulkan_instance_extensions() const;
    [[nodiscard]] VkSurfaceKHR create_vulkan_surface(VkInstance instance) const;

    [[nodiscard]] int width() const noexcept;
    [[nodiscard]] int height() const noexcept;
    [[nodiscard]] void* native_window_handle() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace metaral::platform
