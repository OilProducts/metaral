#include "metaral/render/vulkan_renderer.hpp"

#ifdef METARAL_ENABLE_VULKAN

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "metaral/render/sdf_grid.hpp"
#include "metaral/render/fluid_compute.hpp"

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

    VkDescriptorSetLayout descriptor_set_layout = VK_NULL_HANDLE;
    VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
    VkDescriptorSet descriptor_set = VK_NULL_HANDLE;

    VkBuffer sdf_buffer = VK_NULL_HANDLE;
    VkDeviceMemory sdf_memory = VK_NULL_HANDLE;
    VkBuffer material_buffer = VK_NULL_HANDLE;
    VkDeviceMemory material_memory = VK_NULL_HANDLE;
    VkBuffer sdf_octree_buffer = VK_NULL_HANDLE;
    VkDeviceMemory sdf_octree_memory = VK_NULL_HANDLE;
    uint32_t sdf_octree_node_count = 0;
    uint32_t sdf_octree_root_index = 0;
    uint32_t sdf_octree_depth = 0;
    uint32_t sdf_dim = 0;
    float sdf_voxel_size = 0.0f;
    float sdf_radius = 0.0f;
    float sdf_half_extent = 0.0f;
    SdfGrid sdf_grid;
    bool sdf_dirty = false; // indicates whether a dirty region is pending
    core::PlanetPosition dirty_min{};
    core::PlanetPosition dirty_max{};
    std::function<void(VkCommandBuffer)> overlay_callback;

    // Fluid compute
    FluidComputeContext fluid;
    sim::SphParams fluid_params{};
    uint32_t fluid_particle_count = 0;
    VkImage fluid_volume_image = VK_NULL_HANDLE;
    VkDeviceMemory fluid_volume_memory = VK_NULL_HANDLE;
    VkImageView fluid_volume_view = VK_NULL_HANDLE;
    VkSampler fluid_volume_sampler = VK_NULL_HANDLE;
    VkExtent3D fluid_volume_extent{128, 128, 64};
    float fluid_cell_size = 0.25f;
    core::PlanetPosition fluid_origin{};
    float fluid_iso = 0.5f;
    VkImageLayout fluid_volume_layout = VK_IMAGE_LAYOUT_UNDEFINED;
    VkBuffer fluid_params_buffer = VK_NULL_HANDLE;
    VkDeviceMemory fluid_params_memory = VK_NULL_HANDLE;
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

uint32_t find_memory_type(const Impl& impl,
                          uint32_t type_bits,
                          VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties memory_properties;
    vkGetPhysicalDeviceMemoryProperties(impl.physical_device, &memory_properties);

    for (uint32_t index = 0; index < memory_properties.memoryTypeCount; ++index) {
        const bool supported = (type_bits & (1u << index)) != 0;
        const bool has_flags =
            (memory_properties.memoryTypes[index].propertyFlags & properties) == properties;
        if (supported && has_flags) {
            return index;
        }
    }

    vk_throw("find_memory_type", VK_ERROR_FEATURE_NOT_PRESENT);
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

    // Prefer robust buffer access to guard against any stray GPU-side OOB reads/writes.
    VkPhysicalDeviceFeatures supported{};
    vkGetPhysicalDeviceFeatures(impl.physical_device, &supported);
    VkPhysicalDeviceFeatures requested{};
    if (supported.robustBufferAccess) {
        requested.robustBufferAccess = VK_TRUE;
        device_info.pEnabledFeatures = &requested;
    }

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
    (void)std::fread(data.data(), 1, data.size(), f);
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

struct CameraPush {
    float camPos[3];      float planetRadius;
    float forward[3];     float fovY;
    float right[3];       float aspect;
    float up[3];          float pad1;
    float gridDim;
    float gridVoxelSize;
    float gridHalfExtent;
    float isoFraction;    // fraction of gridVoxelSize used as SDF iso-offset
    float sunDirection[3]; float sunIntensity;
    float sunColor[3];     float pad2;
    float octreeRootIndex;
    float octreeMaxDepth;
    float octreeNodeCount;
    float octreeEnabled;
};

struct alignas(16) GpuSdfOctreeNode {
    float center[3];
    float half_size;
    float min_distance;
    float max_distance;
    std::uint32_t first_child;
    std::uint32_t occupancy_mask;
    std::uint32_t flags;
    std::uint32_t pad0;
    std::uint32_t pad1;
    std::uint32_t pad2;
};

static_assert(sizeof(GpuSdfOctreeNode) == 48, "GpuSdfOctreeNode size must match GLSL layout");

void create_pipeline(Impl& impl) {
    VkDescriptorSetLayoutBinding bindings[5]{};

    // binding 0: SDF buffer
    bindings[0].binding = 0;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    // binding 1: material buffer
    bindings[1].binding = 1;
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    // binding 2: SDF octree nodes
    bindings[2].binding = 2;
    bindings[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[2].descriptorCount = 1;
    bindings[2].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    // binding 3: fluid density volume
    bindings[3].binding = 3;
    bindings[3].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[3].descriptorCount = 1;
    bindings[3].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    // binding 4: fluid params uniform
    bindings[4].binding = 4;
    bindings[4].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    bindings[4].descriptorCount = 1;
    bindings[4].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    VkDescriptorSetLayoutCreateInfo set_layout_info{};
    set_layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    set_layout_info.bindingCount = 5;
    set_layout_info.pBindings = bindings;

    vk_check(vkCreateDescriptorSetLayout(impl.device,
                                         &set_layout_info,
                                         nullptr,
                                         &impl.descriptor_set_layout),
             "vkCreateDescriptorSetLayout");

    // Push constant block layout: CameraParams (CameraPush on the C++ side)
    VkPushConstantRange range{};
    range.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    range.offset = 0;
    range.size = sizeof(CameraPush);

    VkPipelineLayoutCreateInfo layout_info{};
    layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layout_info.setLayoutCount = 1;
    layout_info.pSetLayouts = &impl.descriptor_set_layout;
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

void destroy_fluid_volume(Impl& impl) {
    if (impl.device == VK_NULL_HANDLE) {
        return;
    }
    if (impl.fluid_params_buffer != VK_NULL_HANDLE) {
        vkDestroyBuffer(impl.device, impl.fluid_params_buffer, nullptr);
        impl.fluid_params_buffer = VK_NULL_HANDLE;
    }
    if (impl.fluid_params_memory != VK_NULL_HANDLE) {
        vkFreeMemory(impl.device, impl.fluid_params_memory, nullptr);
        impl.fluid_params_memory = VK_NULL_HANDLE;
    }
    if (impl.fluid_volume_sampler != VK_NULL_HANDLE) {
        vkDestroySampler(impl.device, impl.fluid_volume_sampler, nullptr);
        impl.fluid_volume_sampler = VK_NULL_HANDLE;
    }
    if (impl.fluid_volume_view != VK_NULL_HANDLE) {
        vkDestroyImageView(impl.device, impl.fluid_volume_view, nullptr);
        impl.fluid_volume_view = VK_NULL_HANDLE;
    }
    if (impl.fluid_volume_image != VK_NULL_HANDLE) {
        vkDestroyImage(impl.device, impl.fluid_volume_image, nullptr);
        impl.fluid_volume_image = VK_NULL_HANDLE;
    }
    if (impl.fluid_volume_memory != VK_NULL_HANDLE) {
        vkFreeMemory(impl.device, impl.fluid_volume_memory, nullptr);
        impl.fluid_volume_memory = VK_NULL_HANDLE;
    }
    impl.fluid_volume_layout = VK_IMAGE_LAYOUT_UNDEFINED;
    impl.fluid.set_density_image(VK_NULL_HANDLE, VK_NULL_HANDLE);
}

void ensure_fluid_volume(Impl& impl) {
    if (impl.device == VK_NULL_HANDLE || impl.fluid_volume_image != VK_NULL_HANDLE) {
        return;
    }

    VkExtent3D extent = impl.fluid_volume_extent;
    if (extent.width == 0 || extent.height == 0 || extent.depth == 0) {
        extent = {128, 128, 64};
        impl.fluid_volume_extent = extent;
    }

    VkImageCreateInfo img{};
    img.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    img.imageType = VK_IMAGE_TYPE_3D;
    img.extent = extent;
    img.mipLevels = 1;
    img.arrayLayers = 1;
    img.format = VK_FORMAT_R16_SFLOAT;
    img.tiling = VK_IMAGE_TILING_OPTIMAL;
    img.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    img.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    img.samples = VK_SAMPLE_COUNT_1_BIT;
    img.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    vk_check(vkCreateImage(impl.device, &img, nullptr, &impl.fluid_volume_image),
             "vkCreateImage(fluid_volume_image)");

    VkMemoryRequirements req{};
    vkGetImageMemoryRequirements(impl.device, impl.fluid_volume_image, &req);

    VkMemoryAllocateInfo alloc{};
    alloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc.allocationSize = req.size;
    alloc.memoryTypeIndex = find_memory_type(
        impl, req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    vk_check(vkAllocateMemory(impl.device, &alloc, nullptr, &impl.fluid_volume_memory),
             "vkAllocateMemory(fluid_volume_memory)");
    vk_check(vkBindImageMemory(impl.device, impl.fluid_volume_image, impl.fluid_volume_memory, 0),
             "vkBindImageMemory(fluid_volume_image)");

    VkImageViewCreateInfo view{};
    view.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    view.image = impl.fluid_volume_image;
    view.viewType = VK_IMAGE_VIEW_TYPE_3D;
    view.format = VK_FORMAT_R16_SFLOAT;
    view.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    view.subresourceRange.baseMipLevel = 0;
    view.subresourceRange.levelCount = 1;
    view.subresourceRange.baseArrayLayer = 0;
    view.subresourceRange.layerCount = 1;
    vk_check(vkCreateImageView(impl.device, &view, nullptr, &impl.fluid_volume_view),
             "vkCreateImageView(fluid_volume_view)");

    VkSamplerCreateInfo samp{};
    samp.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samp.magFilter = VK_FILTER_LINEAR;
    samp.minFilter = VK_FILTER_LINEAR;
    samp.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    samp.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samp.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samp.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samp.minLod = 0.0f;
    samp.maxLod = 0.0f;
    vk_check(vkCreateSampler(impl.device, &samp, nullptr, &impl.fluid_volume_sampler),
             "vkCreateSampler(fluid_volume_sampler)");

    impl.fluid_volume_layout = VK_IMAGE_LAYOUT_UNDEFINED;
    impl.fluid.set_density_image(impl.fluid_volume_image, impl.fluid_volume_view);

    // Allocate a small uniform buffer for fluid params (two vec4).
    VkBufferCreateInfo buf{};
    buf.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buf.size = sizeof(float) * 8;
    buf.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
    buf.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    vk_check(vkCreateBuffer(impl.device, &buf, nullptr, &impl.fluid_params_buffer),
             "vkCreateBuffer(fluid_params_buffer)");

    VkMemoryRequirements bufReq{};
    vkGetBufferMemoryRequirements(impl.device, impl.fluid_params_buffer, &bufReq);

    VkMemoryAllocateInfo bufAlloc{};
    bufAlloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    bufAlloc.allocationSize = bufReq.size;
    bufAlloc.memoryTypeIndex = find_memory_type(
        impl, bufReq.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    vk_check(vkAllocateMemory(impl.device, &bufAlloc, nullptr, &impl.fluid_params_memory),
             "vkAllocateMemory(fluid_params_memory)");
    vk_check(vkBindBufferMemory(impl.device, impl.fluid_params_buffer, impl.fluid_params_memory, 0),
             "vkBindBufferMemory(fluid_params_buffer)");
}

void ensure_sdf_grid(Impl& impl,
                     const world::World& world,
                     const core::CoordinateConfig& cfg,
                     bool force_rebuild) {
    if (impl.device == VK_NULL_HANDLE) {
        return;
    }

    const bool needs_full_rebuild =
        force_rebuild ||
        impl.sdf_dim == 0 ||
        impl.sdf_voxel_size <= 0.0f ||
        impl.sdf_radius != cfg.planet_radius_m;
    const bool need_descriptor_update =
        impl.descriptor_set == VK_NULL_HANDLE ||
        (impl.fluid_volume_view != VK_NULL_HANDLE && impl.fluid_volume_sampler != VK_NULL_HANDLE);

    // If no full rebuild is needed, there's no dirty region, and descriptors
    // are already valid, nothing to do.
    if (!needs_full_rebuild && !impl.sdf_dirty && !need_descriptor_update) {
        return;
    }

    // For a full rebuild, recompute the entire grid on the CPU first.
    if (needs_full_rebuild) {
        build_sdf_grid_from_world(world, cfg, impl.sdf_grid);
        impl.sdf_dim = impl.sdf_grid.dim;
        impl.sdf_voxel_size = impl.sdf_grid.voxel_size;
        impl.sdf_radius = cfg.planet_radius_m;
        impl.sdf_half_extent = impl.sdf_grid.half_extent;
        impl.sdf_dirty = false;
    }

    const std::size_t sdf_bytes_raw =
        impl.sdf_grid.values.size() * sizeof(float);
    const std::size_t sdf_bytes = std::max<std::size_t>(sdf_bytes_raw, sizeof(float));
    const std::size_t mat_count_raw = impl.sdf_grid.materials.size();
    const std::size_t mat_count = std::max<std::size_t>(mat_count_raw, 1);

    // Cache octree metadata from the CPU grid. If the octree is unavailable or
    // failed to build, we simply leave the node count at zero and the shader
    // falls back to dense SDF marching.
    std::size_t octree_node_count = 0;
    if (impl.sdf_grid.has_octree) {
        octree_node_count = impl.sdf_grid.octree.nodes.size();
    }
    if (octree_node_count > static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())) {
        // Extremely unlikely; clamp and effectively disable the octree.
        octree_node_count = 0;
    }
    impl.sdf_octree_node_count = static_cast<std::uint32_t>(octree_node_count);
    impl.sdf_octree_depth = impl.sdf_grid.has_octree ? impl.sdf_grid.octree.depth : 0;
    if (impl.sdf_octree_depth > 0 &&
        impl.sdf_grid.octree.level_offsets.size() >= impl.sdf_octree_depth) {
        impl.sdf_octree_root_index =
            impl.sdf_grid.octree.level_offsets[impl.sdf_octree_depth - 1u];
    } else {
        impl.sdf_octree_root_index = 0;
    }

    const std::size_t octree_bytes =
        (impl.sdf_octree_node_count > 0)
            ? static_cast<std::size_t>(impl.sdf_octree_node_count) * sizeof(GpuSdfOctreeNode)
            : sizeof(GpuSdfOctreeNode);

    if (impl.sdf_buffer == VK_NULL_HANDLE) {
        VkBufferCreateInfo sdf_info{};
        sdf_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        sdf_info.size = sdf_bytes;
        sdf_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        sdf_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        vk_check(vkCreateBuffer(impl.device, &sdf_info, nullptr, &impl.sdf_buffer),
                 "vkCreateBuffer(sdf_buffer)");

        VkMemoryRequirements sdf_requirements{};
        vkGetBufferMemoryRequirements(impl.device, impl.sdf_buffer, &sdf_requirements);

        VkMemoryAllocateInfo sdf_alloc{};
        sdf_alloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        sdf_alloc.allocationSize = sdf_requirements.size;
        sdf_alloc.memoryTypeIndex = find_memory_type(
            impl,
            sdf_requirements.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

        vk_check(vkAllocateMemory(impl.device, &sdf_alloc, nullptr, &impl.sdf_memory),
                 "vkAllocateMemory(sdf_memory)");
        vk_check(vkBindBufferMemory(impl.device, impl.sdf_buffer, impl.sdf_memory, 0),
                 "vkBindBufferMemory(sdf_buffer)");
    }

    if (impl.material_buffer == VK_NULL_HANDLE) {
        VkBufferCreateInfo mat_info{};
        mat_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        mat_info.size = mat_count * sizeof(std::uint32_t);
        mat_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        mat_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        vk_check(vkCreateBuffer(impl.device, &mat_info, nullptr, &impl.material_buffer),
                 "vkCreateBuffer(material_buffer)");

        VkMemoryRequirements mat_requirements{};
        vkGetBufferMemoryRequirements(impl.device, impl.material_buffer, &mat_requirements);

        VkMemoryAllocateInfo mat_alloc{};
        mat_alloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        mat_alloc.allocationSize = mat_requirements.size;
        mat_alloc.memoryTypeIndex = find_memory_type(
            impl,
            mat_requirements.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

        vk_check(vkAllocateMemory(impl.device, &mat_alloc, nullptr, &impl.material_memory),
                 "vkAllocateMemory(material_memory)");
        vk_check(vkBindBufferMemory(impl.device, impl.material_buffer, impl.material_memory, 0),
                 "vkBindBufferMemory(material_buffer)");
    }

    if (impl.sdf_octree_buffer == VK_NULL_HANDLE) {
        VkBufferCreateInfo oct_info{};
        oct_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        oct_info.size = octree_bytes;
        oct_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        oct_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        vk_check(vkCreateBuffer(impl.device, &oct_info, nullptr, &impl.sdf_octree_buffer),
                 "vkCreateBuffer(sdf_octree_buffer)");

        VkMemoryRequirements oct_requirements{};
        vkGetBufferMemoryRequirements(impl.device, impl.sdf_octree_buffer, &oct_requirements);

        VkMemoryAllocateInfo oct_alloc{};
        oct_alloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        oct_alloc.allocationSize = oct_requirements.size;
        oct_alloc.memoryTypeIndex = find_memory_type(
            impl,
            oct_requirements.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

        vk_check(vkAllocateMemory(impl.device, &oct_alloc, nullptr, &impl.sdf_octree_memory),
                 "vkAllocateMemory(sdf_octree_memory)");
        vk_check(vkBindBufferMemory(impl.device, impl.sdf_octree_buffer, impl.sdf_octree_memory, 0),
                 "vkBindBufferMemory(sdf_octree_buffer)");
    }

    float* sdf_gpu = nullptr;
    std::uint32_t* mat_gpu = nullptr;
    GpuSdfOctreeNode* octree_gpu = nullptr;
    if (needs_full_rebuild || impl.sdf_dirty) {
        void* mapped_sdf = nullptr;
        vk_check(vkMapMemory(impl.device,
                             impl.sdf_memory,
                             0,
                             sdf_bytes,
                             0,
                             &mapped_sdf),
                 "vkMapMemory(sdf_memory)");
        sdf_gpu = static_cast<float*>(mapped_sdf);

        void* mapped_mat = nullptr;
        vk_check(vkMapMemory(impl.device,
                             impl.material_memory,
                             0,
                             mat_count * sizeof(std::uint32_t),
                             0,
                             &mapped_mat),
                 "vkMapMemory(material_memory)");
        mat_gpu = static_cast<std::uint32_t*>(mapped_mat);

        if (impl.sdf_octree_node_count > 0) {
            void* mapped_oct = nullptr;
            vk_check(vkMapMemory(impl.device,
                                 impl.sdf_octree_memory,
                                 0,
                                 octree_bytes,
                                 0,
                                 &mapped_oct),
                     "vkMapMemory(sdf_octree_memory)");
            octree_gpu = static_cast<GpuSdfOctreeNode*>(mapped_oct);
        }
    }

    if (needs_full_rebuild && sdf_gpu && mat_gpu) {
        // Full rebuild: mirror the entire CPU grid into the GPU buffers.
        const std::size_t val_count = impl.sdf_grid.values.size();
        for (std::size_t idx = 0; idx < val_count; ++idx) {
            sdf_gpu[idx] = impl.sdf_grid.values[idx];
        }
        const std::size_t mat_count_copy = impl.sdf_grid.materials.size();
        for (std::size_t idx = 0; idx < mat_count_copy; ++idx) {
            mat_gpu[idx] = static_cast<std::uint32_t>(impl.sdf_grid.materials[idx]);
        }
    } else if (impl.sdf_dirty && sdf_gpu && mat_gpu) {
        // Incremental update: only touch samples inside the dirty world-space
        // AABB, updating both CPU grid and GPU buffers.
        update_sdf_region_from_world(world,
                                     cfg,
                                     impl.dirty_min,
                                     impl.dirty_max,
                                     impl.sdf_grid,
                                     sdf_gpu,
                                     mat_gpu);
        impl.sdf_dirty = false;
    }

    // Upload octree nodes if present.
    if (octree_gpu &&
        impl.sdf_grid.has_octree &&
        impl.sdf_octree_node_count <= impl.sdf_grid.octree.nodes.size()) {
        // Upload the entire octree whenever the CPU copy changes; incremental
        // updates are already limited to dirty regions.
        for (std::uint32_t i = 0; i < impl.sdf_octree_node_count; ++i) {
            const SdfOctreeNode& src = impl.sdf_grid.octree.nodes[i];
            GpuSdfOctreeNode& dst = octree_gpu[i];
            dst.center[0] = src.center.x;
            dst.center[1] = src.center.y;
            dst.center[2] = src.center.z;
            dst.half_size = src.half_size;
            dst.min_distance = src.min_distance;
            dst.max_distance = src.max_distance;
            dst.first_child = src.first_child;
            dst.occupancy_mask = src.occupancy_mask;
            dst.flags = src.flags;
            dst.pad0 = 0u;
            dst.pad1 = 0u;
            dst.pad2 = 0u;
        }
    }

    if (sdf_gpu) {
        vkUnmapMemory(impl.device, impl.sdf_memory);
    }
    if (mat_gpu) {
        vkUnmapMemory(impl.device, impl.material_memory);
    }
    if (octree_gpu) {
        vkUnmapMemory(impl.device, impl.sdf_octree_memory);
    }

    impl.sdf_dim = impl.sdf_grid.dim;
    impl.sdf_voxel_size = impl.sdf_grid.voxel_size;
    impl.sdf_radius = cfg.planet_radius_m;
    impl.sdf_half_extent = impl.sdf_grid.half_extent;

    if (impl.descriptor_pool == VK_NULL_HANDLE) {
        VkDescriptorPoolSize pool_sizes[3]{};
        pool_sizes[0].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        pool_sizes[0].descriptorCount = 3;
        pool_sizes[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        pool_sizes[1].descriptorCount = 1;
        pool_sizes[2].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        pool_sizes[2].descriptorCount = 1;

        VkDescriptorPoolCreateInfo pool_info{};
        pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        pool_info.maxSets = 1;
        pool_info.poolSizeCount = 3;
        pool_info.pPoolSizes = pool_sizes;

        vk_check(vkCreateDescriptorPool(impl.device, &pool_info, nullptr, &impl.descriptor_pool),
                 "vkCreateDescriptorPool");
    }

    if (impl.descriptor_set == VK_NULL_HANDLE) {
        VkDescriptorSetAllocateInfo alloc_set{};
        alloc_set.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        alloc_set.descriptorPool = impl.descriptor_pool;
        alloc_set.descriptorSetCount = 1;
        alloc_set.pSetLayouts = &impl.descriptor_set_layout;

        vk_check(vkAllocateDescriptorSets(impl.device, &alloc_set, &impl.descriptor_set),
                 "vkAllocateDescriptorSets");
    }
    if (impl.descriptor_set == VK_NULL_HANDLE) {
        return;
    }

    VkDescriptorBufferInfo sdf_desc{};
    sdf_desc.buffer = impl.sdf_buffer;
    sdf_desc.offset = 0;
    sdf_desc.range = sdf_bytes;

    VkDescriptorBufferInfo mat_desc{};
    mat_desc.buffer = impl.material_buffer;
    mat_desc.offset = 0;
    mat_desc.range = mat_count * sizeof(std::uint32_t);

    VkDescriptorBufferInfo octree_desc{};
    octree_desc.buffer = impl.sdf_octree_buffer;
    octree_desc.offset = 0;
    octree_desc.range = octree_bytes > 0 ? octree_bytes : sizeof(GpuSdfOctreeNode);

    std::vector<VkWriteDescriptorSet> writes;
    writes.reserve(5);

    auto push_buffer_write = [&](uint32_t binding, VkDescriptorBufferInfo* info) {
        if (!info || info->buffer == VK_NULL_HANDLE || info->range == 0) {
            return;
        }
        VkWriteDescriptorSet w{};
        w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w.dstSet = impl.descriptor_set;
        w.dstBinding = binding;
        w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        w.descriptorCount = 1;
        w.pBufferInfo = info;
        writes.push_back(w);
    };

    push_buffer_write(0, &sdf_desc);
    push_buffer_write(1, &mat_desc);
    push_buffer_write(2, &octree_desc);

    if (impl.fluid_volume_view != VK_NULL_HANDLE &&
        impl.fluid_volume_sampler != VK_NULL_HANDLE &&
        impl.fluid_params_buffer != VK_NULL_HANDLE) {
        VkDescriptorImageInfo fluid_img{};
        fluid_img.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        fluid_img.imageView = impl.fluid_volume_view;
        fluid_img.sampler = impl.fluid_volume_sampler;

        VkWriteDescriptorSet wf{};
        wf.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        wf.dstSet = impl.descriptor_set;
        wf.dstBinding = 3;
        wf.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        wf.descriptorCount = 1;
        wf.pImageInfo = &fluid_img;
        writes.push_back(wf);

        VkDescriptorBufferInfo fluid_params_desc{};
        fluid_params_desc.buffer = impl.fluid_params_buffer;
        fluid_params_desc.offset = 0;
        fluid_params_desc.range = sizeof(float) * 8; // two vec4

        if (fluid_params_desc.buffer != VK_NULL_HANDLE && fluid_params_desc.range > 0) {
            VkWriteDescriptorSet wfParams{};
            wfParams.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            wfParams.dstSet = impl.descriptor_set;
            wfParams.dstBinding = 4;
            wfParams.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            wfParams.descriptorCount = 1;
            wfParams.pBufferInfo = &fluid_params_desc;
            writes.push_back(wfParams);
        }
    }

    if (!writes.empty()) {
        vkUpdateDescriptorSets(impl.device,
                               static_cast<uint32_t>(writes.size()),
                               writes.data(),
                               0,
                               nullptr);
    }
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

    // Initialize fluid compute (safe even if unused yet).
    FluidGpuParams fluid_gpu_params{};
    impl_->fluid.initialize(impl_->device,
                            impl_->physical_device,
                            impl_->graphics_queue_family,
                            fluid_gpu_params);

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

    if (impl_->sdf_buffer != VK_NULL_HANDLE) {
        vkDestroyBuffer(impl_->device, impl_->sdf_buffer, nullptr);
    }
    if (impl_->sdf_memory != VK_NULL_HANDLE) {
        vkFreeMemory(impl_->device, impl_->sdf_memory, nullptr);
    }
    if (impl_->sdf_octree_buffer != VK_NULL_HANDLE) {
        vkDestroyBuffer(impl_->device, impl_->sdf_octree_buffer, nullptr);
    }
    if (impl_->sdf_octree_memory != VK_NULL_HANDLE) {
        vkFreeMemory(impl_->device, impl_->sdf_octree_memory, nullptr);
    }
    if (impl_->material_buffer != VK_NULL_HANDLE) {
        vkDestroyBuffer(impl_->device, impl_->material_buffer, nullptr);
    }
    if (impl_->material_memory != VK_NULL_HANDLE) {
        vkFreeMemory(impl_->device, impl_->material_memory, nullptr);
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
    if (impl_->descriptor_pool != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(impl_->device, impl_->descriptor_pool, nullptr);
    }

    // Fluid resources
    destroy_fluid_volume(*impl_);
    impl_->fluid = FluidComputeContext{};
    if (impl_->descriptor_set_layout != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(impl_->device, impl_->descriptor_set_layout, nullptr);
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

void VulkanRenderer::draw_frame(const Camera& camera, const world::World& world, float dt_seconds) {
    if (impl_->device == VK_NULL_HANDLE || impl_->swapchain == VK_NULL_HANDLE) {
        return;
    }

    const bool fluid_ready = impl_->fluid.initialized();

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

    const bool force_rebuild =
        impl_->sdf_dim == 0 ||
        impl_->sdf_voxel_size <= 0.0f ||
        impl_->sdf_radius != world.coords().planet_radius_m;
    ensure_fluid_volume(*impl_);
    ensure_sdf_grid(*impl_, world, world.coords(), force_rebuild);

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
    push.gridDim = static_cast<float>(impl_->sdf_dim);
    push.gridVoxelSize = impl_->sdf_voxel_size;
    push.gridHalfExtent = impl_->sdf_half_extent;
    push.isoFraction = kDefaultSdfIsoFraction;

    // Simple directional "sun" light expressed in planetary/world space.
    core::PlanetPosition sun_dir_ws{0.3f, 0.8f, 0.4f};
    sun_dir_ws = normalize_vec(sun_dir_ws);
    push.sunDirection[0] = sun_dir_ws.x;
    push.sunDirection[1] = sun_dir_ws.y;
    push.sunDirection[2] = sun_dir_ws.z;
    push.sunIntensity = 1.0f;

    // Slightly warm sun color.
    push.sunColor[0] = 1.0f;
    push.sunColor[1] = 0.98f;
    push.sunColor[2] = 0.92f;
    push.pad2 = 0.0f;

    // Octree metadata for the fragment shader. When no octree is available,
    // octreeEnabled is 0 and traversal falls back to dense marching.
    float octree_root_index_f = 0.0f;
    if (impl_->sdf_octree_node_count > 0 &&
        impl_->sdf_grid.has_octree) {
        octree_root_index_f = static_cast<float>(impl_->sdf_octree_root_index);
    }
    push.octreeRootIndex = octree_root_index_f;
    push.octreeMaxDepth = static_cast<float>(impl_->sdf_octree_depth);
    push.octreeNodeCount = static_cast<float>(impl_->sdf_octree_node_count);
    push.octreeEnabled =
        (impl_->sdf_octree_node_count > 0 && impl_->sdf_grid.has_octree) ? 1.0f : 0.0f;

    impl_->fluid_cell_size = (impl_->sdf_voxel_size > 0.0f) ? impl_->sdf_voxel_size : 0.25f;
    const float spanX = impl_->fluid_cell_size * static_cast<float>(impl_->fluid_volume_extent.width);
    const float spanY = impl_->fluid_cell_size * static_cast<float>(impl_->fluid_volume_extent.height);
    const float spanZ = impl_->fluid_cell_size * static_cast<float>(impl_->fluid_volume_extent.depth);
    impl_->fluid_origin = {
        camera.position.x - 0.5f * spanX,
        camera.position.y - 0.5f * spanY,
        camera.position.z - 0.5f * spanZ,
    };
    impl_->fluid_iso = 0.5f;

    FluidVolumeParams volume_params{};
    volume_params.origin = impl_->fluid_origin;
    volume_params.cell_size = impl_->fluid_cell_size;
    volume_params.dim_x = impl_->fluid_volume_extent.width;
    volume_params.dim_y = impl_->fluid_volume_extent.height;
    volume_params.dim_z = impl_->fluid_volume_extent.depth;
    volume_params.iso_threshold = impl_->fluid_iso;
    volume_params.planet_radius = world.coords().planet_radius_m;
    if (impl_->fluid_particle_count == 0) {
        volume_params.dim_x = 0;
        volume_params.dim_y = 0;
        volume_params.dim_z = 0;
    }

    if (impl_->fluid_params_buffer != VK_NULL_HANDLE) {
        struct FluidParamsGpu {
            float originCell[4];
            float dimIso[4];
        } params{};
        params.originCell[0] = impl_->fluid_origin.x;
        params.originCell[1] = impl_->fluid_origin.y;
        params.originCell[2] = impl_->fluid_origin.z;
        params.originCell[3] = impl_->fluid_cell_size;
        params.dimIso[0] = static_cast<float>(volume_params.dim_x);
        params.dimIso[1] = static_cast<float>(volume_params.dim_y);
        params.dimIso[2] = static_cast<float>(volume_params.dim_z);
        params.dimIso[3] = impl_->fluid_iso;

        void* mapped = nullptr;
        vk_check(vkMapMemory(impl_->device, impl_->fluid_params_memory, 0, sizeof(FluidParamsGpu), 0, &mapped),
                 "vkMapMemory(fluid_params)");
        std::memcpy(mapped, &params, sizeof(FluidParamsGpu));
        vkUnmapMemory(impl_->device, impl_->fluid_params_memory);
    }

    // Record command buffer for this image
    VkCommandBuffer cmd = impl_->command_buffers[image_index];
    vk_check(vkResetCommandBuffer(cmd, 0), "vkResetCommandBuffer");

    VkCommandBufferBeginInfo begin_info{};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    vk_check(vkBeginCommandBuffer(cmd, &begin_info), "vkBeginCommandBuffer");

    // Compute pass: run fluid step before the render pass.
    if (impl_->fluid_volume_image != VK_NULL_HANDLE &&
        impl_->fluid_volume_layout != VK_IMAGE_LAYOUT_GENERAL) {
        VkImageMemoryBarrier layout_barrier{};
        layout_barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        layout_barrier.oldLayout = impl_->fluid_volume_layout;
        layout_barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        layout_barrier.srcAccessMask = 0;
        layout_barrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        layout_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        layout_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        layout_barrier.image = impl_->fluid_volume_image;
        layout_barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        layout_barrier.subresourceRange.baseMipLevel = 0;
        layout_barrier.subresourceRange.levelCount = 1;
        layout_barrier.subresourceRange.baseArrayLayer = 0;
        layout_barrier.subresourceRange.layerCount = 1;
        vkCmdPipelineBarrier(cmd,
                             VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             0,
                             0, nullptr,
                             0, nullptr,
                             1, &layout_barrier);
        impl_->fluid_volume_layout = VK_IMAGE_LAYOUT_GENERAL;
    }

    if (fluid_ready && impl_->fluid_particle_count > 0) {
        if (impl_->sdf_buffer != VK_NULL_HANDLE && impl_->sdf_dim > 0) {
            impl_->fluid.set_sdf_buffer(impl_->sdf_buffer,
                                        impl_->sdf_dim,
                                        impl_->sdf_voxel_size,
                                        impl_->sdf_half_extent,
                                        impl_->sdf_radius);
        } else {
            impl_->fluid.set_sdf_buffer(VK_NULL_HANDLE, 0, 0.0f, 0.0f, impl_->sdf_radius);
        }
        if (impl_->sdf_octree_buffer != VK_NULL_HANDLE &&
            impl_->sdf_octree_node_count > 0 &&
            impl_->sdf_grid.has_octree) {
            impl_->fluid.set_sdf_octree(impl_->sdf_octree_buffer,
                                        impl_->sdf_octree_node_count,
                                        impl_->sdf_octree_root_index,
                                        impl_->sdf_octree_depth);
        } else {
            impl_->fluid.set_sdf_octree(VK_NULL_HANDLE, 0, 0, 0);
        }
        impl_->fluid.step(cmd,
                          dt_seconds,
                          impl_->fluid_params,
                          impl_->fluid_particle_count,
                          volume_params);
    }
    if (fluid_ready && impl_->fluid_particle_count > 0 && impl_->fluid_volume_image != VK_NULL_HANDLE) {
        VkImageMemoryBarrier to_read{};
        to_read.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        to_read.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        to_read.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        to_read.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        to_read.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        to_read.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        to_read.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        to_read.image = impl_->fluid_volume_image;
        to_read.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        to_read.subresourceRange.baseMipLevel = 0;
        to_read.subresourceRange.levelCount = 1;
        to_read.subresourceRange.baseArrayLayer = 0;
        to_read.subresourceRange.layerCount = 1;
        vkCmdPipelineBarrier(cmd,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                             0,
                             0, nullptr,
                             0, nullptr,
                             1, &to_read);
    }

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
    if (impl_->descriptor_set != VK_NULL_HANDLE) {
        vkCmdBindDescriptorSets(cmd,
                                VK_PIPELINE_BIND_POINT_GRAPHICS,
                                impl_->pipeline_layout,
                                0,
                                1,
                                &impl_->descriptor_set,
                                0,
                                nullptr);
    }
    vkCmdPushConstants(cmd,
                       impl_->pipeline_layout,
                       VK_SHADER_STAGE_FRAGMENT_BIT,
                       0,
                       sizeof(CameraPush),
                       &push);
    vkCmdDraw(cmd, 3, 1, 0, 0);
    if (impl_->overlay_callback) {
        impl_->overlay_callback(cmd);
    }
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

SdfGridInfo VulkanRenderer::sdf_grid_info() const {
    SdfGridInfo info{};
    info.dim = impl_->sdf_dim;
    info.voxel_size = impl_->sdf_voxel_size;
    info.half_extent = impl_->sdf_half_extent;
    return info;
}

const SdfGrid* VulkanRenderer::sdf_grid() const {
    if (impl_->sdf_dim == 0) {
        return nullptr;
    }
    return &impl_->sdf_grid;
}

VulkanBackendHandles VulkanRenderer::backend_handles() const {
    VulkanBackendHandles handles{};
    handles.instance = impl_->instance;
    handles.physical_device = impl_->physical_device;
    handles.device = impl_->device;
    handles.graphics_queue = impl_->graphics_queue;
    handles.graphics_queue_family = impl_->graphics_queue_family;
    handles.render_pass = impl_->render_pass;
    handles.swapchain_image_count = static_cast<uint32_t>(impl_->swapchain_images.size());
    return handles;
}

void VulkanRenderer::set_overlay_callback(OverlayCallback callback) {
    if (!impl_) {
        return;
    }
    impl_->overlay_callback = std::move(callback);
}

sim::SphParams& VulkanRenderer::fluid_params() noexcept {
    return impl_->fluid_params;
}

void VulkanRenderer::update_fluid_particles(std::span<const sim::FluidParticle> particles) noexcept {
    if (!impl_ || !impl_->fluid.initialized()) {
        return;
    }
    impl_->fluid.upload_particles(particles);
    impl_->fluid_particle_count =
        std::min<uint32_t>(static_cast<uint32_t>(particles.size()), 65536u);
}

void VulkanRenderer::wait_idle() {
    if (impl_->device != VK_NULL_HANDLE) {
        vkDeviceWaitIdle(impl_->device);
    }
}

void VulkanRenderer::mark_sdf_dirty(const core::PlanetPosition& min_p,
                                    const core::PlanetPosition& max_p) {
    if (!impl_) {
        return;
    }
    if (!impl_->sdf_dirty) {
        impl_->dirty_min = min_p;
        impl_->dirty_max = max_p;
        impl_->sdf_dirty = true;
    } else {
        impl_->dirty_min.x = std::min(impl_->dirty_min.x, min_p.x);
        impl_->dirty_min.y = std::min(impl_->dirty_min.y, min_p.y);
        impl_->dirty_min.z = std::min(impl_->dirty_min.z, min_p.z);
        impl_->dirty_max.x = std::max(impl_->dirty_max.x, max_p.x);
        impl_->dirty_max.y = std::max(impl_->dirty_max.y, max_p.y);
        impl_->dirty_max.z = std::max(impl_->dirty_max.z, max_p.z);
    }
}

} // namespace metaral::render

#endif // METARAL_ENABLE_VULKAN
