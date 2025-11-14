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

struct Event {
    EventType type = EventType::None;
    WindowResizedEvent window_resized{};
    KeyEvent key{};
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

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace metaral::platform
