#include "metaral/render/vulkan_renderer.hpp"

#ifdef METARAL_ENABLE_VULKAN

#include <array>
#include <stdexcept>
#include <vector>

namespace metaral::render {

struct VulkanRenderer::Impl {
    platform::VulkanContext hooks;
    VkSurfaceKHR surface = VK_NULL_HANDLE;
    std::uint32_t width = 0;
    std::uint32_t height = 0;

    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physical_device = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue graphics_queue = VK_NULL_HANDLE;
    uint32_t graphics_queue_family = 0;

    VkSwapchainKHR swapchain = VK_NULL_HANDLE;
    VkFormat swapchain_format = VK_FORMAT_B8G8R8A8_UNORM;
    VkExtent2D swapchain_extent{};
    std::vector<VkImage> swapchain_images;
    std::vector<VkImageView> swapchain_image_views;

    VkRenderPass render_pass = VK_NULL_HANDLE;
    std::vector<VkFramebuffer> framebuffers;

    VkCommandPool command_pool = VK_NULL_HANDLE;
    std::vector<VkCommandBuffer> command_buffers;

    VkSemaphore image_available = VK_NULL_HANDLE;
    VkSemaphore render_finished = VK_NULL_HANDLE;
    VkFence in_flight = VK_NULL_HANDLE;
};

namespace {

void vk_check(VkResult res, const char* what) {
    if (res != VK_SUCCESS) {
        throw std::runtime_error(std::string("Vulkan error in ") + what);
    }
}

} // namespace

VulkanRenderer::VulkanRenderer(const platform::VulkanContext& ctx,
                               VkSurfaceKHR surface,
                               std::uint32_t width,
                               std::uint32_t height)
    : impl_(std::make_unique<Impl>()) {
    impl_->hooks = ctx;
    impl_->surface = surface;
    impl_->width = width;
    impl_->height = height;

    // NOTE: For now, this is only a stub and does not fully initialize
    // the Vulkan instance/device/swapchain. The intent is to grow this
    // step-by-step into a working fullscreen-triangle renderer.
}

VulkanRenderer::~VulkanRenderer() = default;

void VulkanRenderer::resize(std::uint32_t width, std::uint32_t height) {
    impl_->width = width;
    impl_->height = height;
    // Swapchain recreation will be added later.
}

void VulkanRenderer::draw_frame(const Camera&, const world::World&) {
    // No-op for now; pipeline and command buffer recording will be added
    // when the basic Vulkan setup is in place.
}

void VulkanRenderer::wait_idle() {
    if (impl_->device != VK_NULL_HANDLE) {
        vkDeviceWaitIdle(impl_->device);
    }
}

} // namespace metaral::render

#endif // METARAL_ENABLE_VULKAN

