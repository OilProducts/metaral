#pragma once

#include "metaral/render/camera.hpp"
#include "metaral/world/world.hpp"
#include "metaral/platform/platform.hpp"

#ifdef METARAL_ENABLE_VULKAN
#include <vulkan/vulkan.h>
#endif

#include <memory>

namespace metaral::render {

#ifdef METARAL_ENABLE_VULKAN

// Very small, initial Vulkan renderer stub.
// Goal: own a Vulkan device + swapchain and be able to render
// a fullscreen triangle (raymarching will come later).

class VulkanRenderer {
public:
    VulkanRenderer(const platform::VulkanContext& ctx,
                   VkSurfaceKHR surface,
                   std::uint32_t width,
                   std::uint32_t height);
    ~VulkanRenderer();

    VulkanRenderer(const VulkanRenderer&) = delete;
    VulkanRenderer& operator=(const VulkanRenderer&) = delete;

    void resize(std::uint32_t width, std::uint32_t height);
    void draw_frame(const Camera& camera, const world::World& world);
    void wait_idle();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

#endif // METARAL_ENABLE_VULKAN

} // namespace metaral::render

