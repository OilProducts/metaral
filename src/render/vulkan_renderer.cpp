#include "metaral/render/vulkan_renderer.hpp"

#ifdef METARAL_ENABLE_VULKAN

#include <algorithm>
#include <array>
#include <cstdio>
#include <stdexcept>
#include <string>
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

    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
};

namespace {

using Impl = VulkanRenderer::Impl;

[[noreturn]] void vk_throw(const char* what, VkResult res) {
    throw std::runtime_error(std::string("Vulkan error in ") + what + ": " + std::to_string(res));
}

void vk_check(VkResult res, const char* what) {
    if (res != VK_SUCCESS) {
        vk_throw(what, res);
    }
}

bool supports_presentation(VkPhysicalDevice device,
                           uint32_t queue_family,
                           VkSurfaceKHR surface) {
    VkBool32 supported = VK_FALSE;
    vkGetPhysicalDeviceSurfaceSupportKHR(device, queue_family, surface, &supported);
    return supported == VK_TRUE;
}

VkPhysicalDevice pick_physical_device(Impl& impl) {
    uint32_t device_count = 0;
    vkEnumeratePhysicalDevices(impl.instance, &device_count, nullptr);
    if (device_count == 0) {
        throw std::runtime_error("No Vulkan physical devices found");
    }

    std::vector<VkPhysicalDevice> devices(device_count);
    vkEnumeratePhysicalDevices(impl.instance, &device_count, devices.data());

    for (VkPhysicalDevice dev : devices) {
        uint32_t family_count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(dev, &family_count, nullptr);
        std::vector<VkQueueFamilyProperties> families(family_count);
        vkGetPhysicalDeviceQueueFamilyProperties(dev, &family_count, families.data());

        for (uint32_t i = 0; i < family_count; ++i) {
            if ((families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) &&
                supports_presentation(dev, i, impl.surface)) {
                impl.graphics_queue_family = i;
                return dev;
            }
        }
    }

    throw std::runtime_error("Failed to find suitable Vulkan physical device with graphics+present queue");
}

void create_logical_device_and_queue(Impl& impl) {
    const float queue_priority = 1.0f;
    VkDeviceQueueCreateInfo queue_info{};
    queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_info.queueFamilyIndex = impl.graphics_queue_family;
    queue_info.queueCount = 1;
    queue_info.pQueuePriorities = &queue_priority;

    const char* device_extensions[] = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};

    VkDeviceCreateInfo device_info{};
    device_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    device_info.queueCreateInfoCount = 1;
    device_info.pQueueCreateInfos = &queue_info;
    device_info.enabledExtensionCount = 1;
    device_info.ppEnabledExtensionNames = device_extensions;

    vk_check(vkCreateDevice(impl.physical_device, &device_info, nullptr, &impl.device), "vkCreateDevice");
    vkGetDeviceQueue(impl.device, impl.graphics_queue_family, 0, &impl.graphics_queue);
}

struct SwapchainSupportDetails {
    VkSurfaceCapabilitiesKHR capabilities{};
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR> present_modes;
};

SwapchainSupportDetails query_swapchain_support(const Impl& impl) {
    SwapchainSupportDetails details{};
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(impl.physical_device, impl.surface, &details.capabilities);

    uint32_t format_count = 0;
    vkGetPhysicalDeviceSurfaceFormatsKHR(impl.physical_device, impl.surface, &format_count, nullptr);
    details.formats.resize(format_count);
    vkGetPhysicalDeviceSurfaceFormatsKHR(impl.physical_device, impl.surface, &format_count, details.formats.data());

    uint32_t present_count = 0;
    vkGetPhysicalDeviceSurfacePresentModesKHR(impl.physical_device, impl.surface, &present_count, nullptr);
    details.present_modes.resize(present_count);
    vkGetPhysicalDeviceSurfacePresentModesKHR(impl.physical_device, impl.surface, &present_count, details.present_modes.data());

    return details;
}

VkSurfaceFormatKHR choose_surface_format(const std::vector<VkSurfaceFormatKHR>& formats) {
    for (const auto& f : formats) {
        if (f.format == VK_FORMAT_B8G8R8A8_UNORM &&
            f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
            return f;
        }
    }
    return formats.empty() ? VkSurfaceFormatKHR{VK_FORMAT_B8G8R8A8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR}
                           : formats[0];
}

VkPresentModeKHR choose_present_mode(const std::vector<VkPresentModeKHR>& modes) {
    for (const auto m : modes) {
        if (m == VK_PRESENT_MODE_MAILBOX_KHR) {
            return m;
        }
    }
    return VK_PRESENT_MODE_FIFO_KHR;
}

VkExtent2D choose_extent(const Impl& impl, const VkSurfaceCapabilitiesKHR& caps) {
    if (caps.currentExtent.width != UINT32_MAX) {
        return caps.currentExtent;
    }

    VkExtent2D actual{impl.width, impl.height};
    actual.width = std::max(caps.minImageExtent.width,
                            std::min(caps.maxImageExtent.width, actual.width));
    actual.height = std::max(caps.minImageExtent.height,
                             std::min(caps.maxImageExtent.height, actual.height));
    return actual;
}

void create_swapchain(Impl& impl) {
    SwapchainSupportDetails support = query_swapchain_support(impl);
    VkSurfaceFormatKHR surface_format = choose_surface_format(support.formats);
    VkPresentModeKHR present_mode = choose_present_mode(support.present_modes);
    VkExtent2D extent = choose_extent(impl, support.capabilities);

    uint32_t image_count = support.capabilities.minImageCount + 1;
    if (support.capabilities.maxImageCount > 0 && image_count > support.capabilities.maxImageCount) {
        image_count = support.capabilities.maxImageCount;
    }

    VkSwapchainCreateInfoKHR info{};
    info.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    info.surface = impl.surface;
    info.minImageCount = image_count;
    info.imageFormat = surface_format.format;
    info.imageColorSpace = surface_format.colorSpace;
    info.imageExtent = extent;
    info.imageArrayLayers = 1;
    info.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    info.preTransform = support.capabilities.currentTransform;
    info.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    info.presentMode = present_mode;
    info.clipped = VK_TRUE;
    info.oldSwapchain = VK_NULL_HANDLE;

    vk_check(vkCreateSwapchainKHR(impl.device, &info, nullptr, &impl.swapchain), "vkCreateSwapchainKHR");

    impl.swapchain_extent = extent;
    impl.swapchain_format = surface_format.format;

    uint32_t actual_image_count = 0;
    vkGetSwapchainImagesKHR(impl.device, impl.swapchain, &actual_image_count, nullptr);
    impl.swapchain_images.resize(actual_image_count);
    vkGetSwapchainImagesKHR(impl.device, impl.swapchain, &actual_image_count, impl.swapchain_images.data());
}

void create_image_views(Impl& impl) {
    impl.swapchain_image_views.resize(impl.swapchain_images.size());

    for (std::size_t i = 0; i < impl.swapchain_images.size(); ++i) {
        VkImageViewCreateInfo view{};
        view.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        view.image = impl.swapchain_images[i];
        view.viewType = VK_IMAGE_VIEW_TYPE_2D;
        view.format = impl.swapchain_format;
        view.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
        view.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
        view.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
        view.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
        view.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        view.subresourceRange.baseMipLevel = 0;
        view.subresourceRange.levelCount = 1;
        view.subresourceRange.baseArrayLayer = 0;
        view.subresourceRange.layerCount = 1;

        vk_check(vkCreateImageView(impl.device, &view, nullptr, &impl.swapchain_image_views[i]), "vkCreateImageView");
    }
}

void create_render_pass(Impl& impl) {
    VkAttachmentDescription color_attachment{};
    color_attachment.format = impl.swapchain_format;
    color_attachment.samples = VK_SAMPLE_COUNT_1_BIT;
    color_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    color_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    color_attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    color_attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    color_attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    color_attachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    VkAttachmentReference color_ref{};
    color_ref.attachment = 0;
    color_ref.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &color_ref;

    VkRenderPassCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    info.attachmentCount = 1;
    info.pAttachments = &color_attachment;
    info.subpassCount = 1;
    info.pSubpasses = &subpass;

    vk_check(vkCreateRenderPass(impl.device, &info, nullptr, &impl.render_pass), "vkCreateRenderPass");
}

void create_framebuffers(Impl& impl) {
    impl.framebuffers.resize(impl.swapchain_image_views.size());

    for (std::size_t i = 0; i < impl.swapchain_image_views.size(); ++i) {
        VkImageView attachments[] = {impl.swapchain_image_views[i]};

        VkFramebufferCreateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        info.renderPass = impl.render_pass;
        info.attachmentCount = 1;
        info.pAttachments = attachments;
        info.width = impl.swapchain_extent.width;
        info.height = impl.swapchain_extent.height;
        info.layers = 1;

        vk_check(vkCreateFramebuffer(impl.device, &info, nullptr, &impl.framebuffers[i]), "vkCreateFramebuffer");
    }
}

std::vector<char> read_file(const char* path) {
    std::FILE* f = std::fopen(path, "rb");
    if (!f) {
        throw std::runtime_error(std::string("Failed to open shader file: ") + path);
    }
    std::fseek(f, 0, SEEK_END);
    long size = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    std::vector<char> data(static_cast<std::size_t>(size));
    std::fread(data.data(), 1, data.size(), f);
    std::fclose(f);
    return data;
}

VkShaderModule create_shader_module(VkDevice device, const char* path) {
    std::vector<char> bytes = read_file(path);

    VkShaderModuleCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    info.codeSize = bytes.size();
    info.pCode = reinterpret_cast<const uint32_t*>(bytes.data());

    VkShaderModule module = VK_NULL_HANDLE;
    vk_check(vkCreateShaderModule(device, &info, nullptr, &module), "vkCreateShaderModule");
    return module;
}

void create_pipeline(Impl& impl) {
    // Push constant block layout: CameraParams
    VkPushConstantRange range{};
    range.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    range.offset = 0;
    range.size = sizeof(float) * 4 * 4; // 4 vec4s

    VkPipelineLayoutCreateInfo layout_info{};
    layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layout_info.pushConstantRangeCount = 1;
    layout_info.pPushConstantRanges = &range;

    vk_check(vkCreatePipelineLayout(impl.device, &layout_info, nullptr, &impl.pipeline_layout), "vkCreatePipelineLayout");

    VkShaderModule vert = create_shader_module(impl.device, METARAL_SHADER_DIR "/fullscreen_triangle.vert.spv");
    VkShaderModule frag = create_shader_module(impl.device, METARAL_SHADER_DIR "/analytic_sphere.frag.spv");

    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vert;
    stages[0].pName = "main";

    stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = frag;
    stages[1].pName = "main";

    VkPipelineVertexInputStateCreateInfo vi{};
    vi.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

    VkPipelineInputAssemblyStateCreateInfo ia{};
    ia.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkViewport viewport{};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = static_cast<float>(impl.swapchain_extent.width);
    viewport.height = static_cast<float>(impl.swapchain_extent.height);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;

    VkRect2D scissor{};
    scissor.offset = {0, 0};
    scissor.extent = impl.swapchain_extent;

    VkPipelineViewportStateCreateInfo vp{};
    vp.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    vp.viewportCount = 1;
    vp.pViewports = &viewport;
    vp.scissorCount = 1;
    vp.pScissors = &scissor;

    VkPipelineRasterizationStateCreateInfo rs{};
    rs.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rs.polygonMode = VK_POLYGON_MODE_FILL;
    rs.cullMode = VK_CULL_MODE_NONE;
    rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rs.lineWidth = 1.0f;

    VkPipelineMultisampleStateCreateInfo ms{};
    ms.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineColorBlendAttachmentState blend_attachment{};
    blend_attachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                      VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

    VkPipelineColorBlendStateCreateInfo blend{};
    blend.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    blend.attachmentCount = 1;
    blend.pAttachments = &blend_attachment;

    VkGraphicsPipelineCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    info.stageCount = 2;
    info.pStages = stages;
    info.pVertexInputState = &vi;
    info.pInputAssemblyState = &ia;
    info.pViewportState = &vp;
    info.pRasterizationState = &rs;
    info.pMultisampleState = &ms;
    info.pColorBlendState = &blend;
    info.layout = impl.pipeline_layout;
    info.renderPass = impl.render_pass;
    info.subpass = 0;

    vk_check(vkCreateGraphicsPipelines(impl.device, VK_NULL_HANDLE, 1, &info, nullptr, &impl.pipeline), "vkCreateGraphicsPipelines");

    vkDestroyShaderModule(impl.device, vert, nullptr);
    vkDestroyShaderModule(impl.device, frag, nullptr);
}

void create_command_pool_and_buffers(Impl& impl) {
    VkCommandPoolCreateInfo pool_info{};
    pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pool_info.queueFamilyIndex = impl.graphics_queue_family;

    vk_check(vkCreateCommandPool(impl.device, &pool_info, nullptr, &impl.command_pool), "vkCreateCommandPool");

    impl.command_buffers.resize(impl.framebuffers.size());

    VkCommandBufferAllocateInfo alloc_info{};
    alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    alloc_info.commandPool = impl.command_pool;
    alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc_info.commandBufferCount = static_cast<uint32_t>(impl.command_buffers.size());

    vk_check(vkAllocateCommandBuffers(impl.device, &alloc_info, impl.command_buffers.data()), "vkAllocateCommandBuffers");
}

void record_command_buffers(Impl& impl) {
    // No-op: command buffers are recorded per-frame in draw_frame
    (void)impl;
}

void create_sync_objects(Impl& impl) {
    VkSemaphoreCreateInfo sem_info{};
    sem_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

    VkFenceCreateInfo fence_info{};
    fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fence_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    vk_check(vkCreateSemaphore(impl.device, &sem_info, nullptr, &impl.image_available), "vkCreateSemaphore(image_available)");
    vk_check(vkCreateSemaphore(impl.device, &sem_info, nullptr, &impl.render_finished), "vkCreateSemaphore(render_finished)");
    vk_check(vkCreateFence(impl.device, &fence_info, nullptr, &impl.in_flight), "vkCreateFence");
}

} // namespace

VulkanRenderer::VulkanRenderer(const platform::VulkanContext& ctx,
                               std::uint32_t width,
                               std::uint32_t height)
    : impl_(std::make_unique<Impl>()) {
    impl_->hooks = ctx;
    impl_->width = width;
    impl_->height = height;

    // Instance
    std::vector<std::string> ext_names = impl_->hooks.required_instance_extensions();
    std::vector<const char*> ext_cstrs;
    ext_cstrs.reserve(ext_names.size());
    for (const auto& e : ext_names) {
        ext_cstrs.push_back(e.c_str());
    }

    VkApplicationInfo app_info{};
    app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.pApplicationName = "Metaral";
    app_info.applicationVersion = VK_MAKE_VERSION(0, 1, 0);
    app_info.pEngineName = "MetaralEngine";
    app_info.engineVersion = VK_MAKE_VERSION(0, 1, 0);
    app_info.apiVersion = VK_API_VERSION_1_2;

    VkInstanceCreateInfo create_info{};
    create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    create_info.pApplicationInfo = &app_info;
    create_info.enabledExtensionCount = static_cast<uint32_t>(ext_cstrs.size());
    create_info.ppEnabledExtensionNames = ext_cstrs.data();

    vk_check(vkCreateInstance(&create_info, nullptr, &impl_->instance), "vkCreateInstance");

    // Surface from platform hook
    impl_->surface = impl_->hooks.create_surface(impl_->instance);

    // Physical device + logical device
    impl_->physical_device = pick_physical_device(*impl_);
    create_logical_device_and_queue(*impl_);

    // Swapchain and associated resources
    create_swapchain(*impl_);
    create_image_views(*impl_);
    create_render_pass(*impl_);
    create_pipeline(*impl_);
    create_framebuffers(*impl_);
    create_command_pool_and_buffers(*impl_);
    create_sync_objects(*impl_);
}

VulkanRenderer::~VulkanRenderer() {
    if (impl_->device != VK_NULL_HANDLE) {
        vkDeviceWaitIdle(impl_->device);
    }

    if (impl_->swapchain != VK_NULL_HANDLE) {
        vkDestroySwapchainKHR(impl_->device, impl_->swapchain, nullptr);
    }
    if (impl_->pipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(impl_->device, impl_->pipeline, nullptr);
    }
    if (impl_->pipeline_layout != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(impl_->device, impl_->pipeline_layout, nullptr);
    }
    for (auto fb : impl_->framebuffers) {
        vkDestroyFramebuffer(impl_->device, fb, nullptr);
    }
    if (impl_->render_pass != VK_NULL_HANDLE) {
        vkDestroyRenderPass(impl_->device, impl_->render_pass, nullptr);
    }
    for (auto view : impl_->swapchain_image_views) {
        vkDestroyImageView(impl_->device, view, nullptr);
    }
    if (impl_->command_pool != VK_NULL_HANDLE) {
        vkDestroyCommandPool(impl_->device, impl_->command_pool, nullptr);
    }
    if (impl_->image_available != VK_NULL_HANDLE) {
        vkDestroySemaphore(impl_->device, impl_->image_available, nullptr);
    }
    if (impl_->render_finished != VK_NULL_HANDLE) {
        vkDestroySemaphore(impl_->device, impl_->render_finished, nullptr);
    }
    if (impl_->in_flight != VK_NULL_HANDLE) {
        vkDestroyFence(impl_->device, impl_->in_flight, nullptr);
    }
    if (impl_->device != VK_NULL_HANDLE) {
        vkDestroyDevice(impl_->device, nullptr);
    }
    if (impl_->surface != VK_NULL_HANDLE) {
        vkDestroySurfaceKHR(impl_->instance, impl_->surface, nullptr);
    }
    if (impl_->instance != VK_NULL_HANDLE) {
        vkDestroyInstance(impl_->instance, nullptr);
    }
}

void VulkanRenderer::resize(std::uint32_t width, std::uint32_t height) {
    impl_->width = width;
    impl_->height = height;
    if (impl_->device == VK_NULL_HANDLE) {
        return;
    }

    vkDeviceWaitIdle(impl_->device);

    for (auto fb : impl_->framebuffers) {
        vkDestroyFramebuffer(impl_->device, fb, nullptr);
    }
    impl_->framebuffers.clear();

    for (auto view : impl_->swapchain_image_views) {
        vkDestroyImageView(impl_->device, view, nullptr);
    }
    impl_->swapchain_image_views.clear();

    if (impl_->swapchain != VK_NULL_HANDLE) {
        vkDestroySwapchainKHR(impl_->device, impl_->swapchain, nullptr);
        impl_->swapchain = VK_NULL_HANDLE;
    }

    create_swapchain(*impl_);
    create_image_views(*impl_);
    create_framebuffers(*impl_);
}
struct CameraPush {
    float camPos[3];      float planetRadius;
    float forward[3];     float fovY;
    float right[3];       float aspect;
    float up[3];          float pad1;
};

core::PlanetPosition normalize_vec(const core::PlanetPosition& v) {
    const float len = metaral::core::length(v);
    if (len < 1e-6f) {
        return {0.0f, 1.0f, 0.0f};
    }
    const float inv = 1.0f / len;
    return {v.x * inv, v.y * inv, v.z * inv};
}

core::PlanetPosition cross_vec(const core::PlanetPosition& a, const core::PlanetPosition& b) {
    return {
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x,
    };
}

void VulkanRenderer::draw_frame(const Camera& camera, const world::World& world) {
    if (impl_->device == VK_NULL_HANDLE || impl_->swapchain == VK_NULL_HANDLE) {
        return;
    }

    vkWaitForFences(impl_->device, 1, &impl_->in_flight, VK_TRUE, UINT64_MAX);
    vkResetFences(impl_->device, 1, &impl_->in_flight);

    uint32_t image_index = 0;
    VkResult res = vkAcquireNextImageKHR(impl_->device,
                                         impl_->swapchain,
                                         UINT64_MAX,
                                         impl_->image_available,
                                         VK_NULL_HANDLE,
                                         &image_index);
    if (res == VK_ERROR_OUT_OF_DATE_KHR) {
        resize(impl_->width, impl_->height);
        return;
    }
    vk_check(res, "vkAcquireNextImageKHR");

    // Build push constants for this frame
    core::PlanetPosition fwd = normalize_vec(camera.forward);
    core::PlanetPosition up = normalize_vec(camera.up);
    core::PlanetPosition right = normalize_vec(cross_vec(fwd, up));
    up = normalize_vec(cross_vec(right, fwd));

    CameraPush push{};
    push.camPos[0] = camera.position.x;
    push.camPos[1] = camera.position.y;
    push.camPos[2] = camera.position.z;
    push.planetRadius = world.coords().planet_radius_m;

    push.forward[0] = fwd.x;
    push.forward[1] = fwd.y;
    push.forward[2] = fwd.z;
    push.fovY = camera.fov_y_radians;

    push.right[0] = right.x;
    push.right[1] = right.y;
    push.right[2] = right.z;
    push.aspect = static_cast<float>(impl_->swapchain_extent.width) /
                  static_cast<float>(impl_->swapchain_extent.height);

    push.up[0] = up.x;
    push.up[1] = up.y;
    push.up[2] = up.z;

    // Record command buffer for this image
    VkCommandBuffer cmd = impl_->command_buffers[image_index];
    vk_check(vkResetCommandBuffer(cmd, 0), "vkResetCommandBuffer");

    VkCommandBufferBeginInfo begin_info{};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    vk_check(vkBeginCommandBuffer(cmd, &begin_info), "vkBeginCommandBuffer");

    VkClearValue clear_color{};
    clear_color.color = {{0.02f, 0.04f, 0.1f, 1.0f}};

    VkRenderPassBeginInfo rp_info{};
    rp_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    rp_info.renderPass = impl_->render_pass;
    rp_info.framebuffer = impl_->framebuffers[image_index];
    rp_info.renderArea.offset = {0, 0};
    rp_info.renderArea.extent = impl_->swapchain_extent;
    rp_info.clearValueCount = 1;
    rp_info.pClearValues = &clear_color;

    vkCmdBeginRenderPass(cmd, &rp_info, VK_SUBPASS_CONTENTS_INLINE);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, impl_->pipeline);
    vkCmdPushConstants(cmd,
                       impl_->pipeline_layout,
                       VK_SHADER_STAGE_FRAGMENT_BIT,
                       0,
                       sizeof(CameraPush),
                       &push);
    vkCmdDraw(cmd, 3, 1, 0, 0);
    vkCmdEndRenderPass(cmd);

    vk_check(vkEndCommandBuffer(cmd), "vkEndCommandBuffer");

    VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;

    VkSubmitInfo submit{};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.waitSemaphoreCount = 1;
    submit.pWaitSemaphores = &impl_->image_available;
    submit.pWaitDstStageMask = &wait_stage;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &cmd;
    submit.signalSemaphoreCount = 1;
    submit.pSignalSemaphores = &impl_->render_finished;

    vk_check(vkQueueSubmit(impl_->graphics_queue, 1, &submit, impl_->in_flight), "vkQueueSubmit");

    VkPresentInfoKHR present{};
    present.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    present.waitSemaphoreCount = 1;
    present.pWaitSemaphores = &impl_->render_finished;
    present.swapchainCount = 1;
    present.pSwapchains = &impl_->swapchain;
    present.pImageIndices = &image_index;

    res = vkQueuePresentKHR(impl_->graphics_queue, &present);
    if (res == VK_ERROR_OUT_OF_DATE_KHR || res == VK_SUBOPTIMAL_KHR) {
        resize(impl_->width, impl_->height);
    } else {
        vk_check(res, "vkQueuePresentKHR");
    }
}

void VulkanRenderer::wait_idle() {
    if (impl_->device != VK_NULL_HANDLE) {
        vkDeviceWaitIdle(impl_->device);
    }
}

} // namespace metaral::render

#endif // METARAL_ENABLE_VULKAN
