#pragma once

#include "metaral/render/camera.hpp"
#include "metaral/render/sdf_grid.hpp"
#include "metaral/render/fluid_compute.hpp"
#include "metaral/world/world.hpp"
#include "metaral/platform/platform.hpp"

#ifdef METARAL_ENABLE_VULKAN
#include <vulkan/vulkan.h>
#endif

#include <cstdint>
#include <functional>
#include <memory>

namespace metaral::render {

#ifdef METARAL_ENABLE_VULKAN

struct SdfGridInfo {
    std::uint32_t dim = 0;
    float voxel_size = 0.0f;
    float half_extent = 0.0f;
};

struct VulkanBackendHandles {
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physical_device = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue graphics_queue = VK_NULL_HANDLE;
    uint32_t graphics_queue_family = 0;
    VkRenderPass render_pass = VK_NULL_HANDLE;
    uint32_t swapchain_image_count = 0;
};

using OverlayCallback = std::function<void(VkCommandBuffer)>;

// Very small, initial Vulkan renderer stub.
// Goal: own a Vulkan device + swapchain and be able to render
// a fullscreen triangle (raymarching will come later).

class VulkanRenderer {
public:
    VulkanRenderer(const platform::VulkanContext& ctx,
                   std::uint32_t width,
                   std::uint32_t height);
    ~VulkanRenderer();

    VulkanRenderer(const VulkanRenderer&) = delete;
    VulkanRenderer& operator=(const VulkanRenderer&) = delete;

    void resize(std::uint32_t width, std::uint32_t height);
    void draw_frame(const Camera& camera, const world::World& world, float dt_seconds);
    void wait_idle();

    // Mark the SDF grid as dirty in a given world-space region; the grid will
    // be updated on the next frame. Passing min == max lets you force a full
    // rebuild if desired.
    void mark_sdf_dirty(const core::PlanetPosition& min_p,
                        const core::PlanetPosition& max_p);

    SdfGridInfo sdf_grid_info() const;
    const SdfGrid* sdf_grid() const;
    VulkanBackendHandles backend_handles() const;

    void set_overlay_callback(OverlayCallback callback);

    // Expose fluid parameters for UI toggling.
    sim::SphParams& fluid_params() noexcept;

    // Upload particle data for the fluid compute path (clamped to GPU capacity).
    void update_fluid_particles(std::span<const sim::FluidParticle> particles) noexcept;

    struct Impl;

private:
    std::unique_ptr<Impl> impl_;
};

#endif // METARAL_ENABLE_VULKAN

} // namespace metaral::render
