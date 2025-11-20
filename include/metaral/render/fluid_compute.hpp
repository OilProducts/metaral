// Vulkan-side scaffold for fluid compute pipeline. Right now it only owns the
// compiled shader module paths and provides entry points to initialize GPU
// resources. The actual dispatch sequence will be filled in next.

#pragma once

#include "metaral/core/coords.hpp"
#include "metaral/sim/sph_fluid.hpp"

#ifdef METARAL_ENABLE_VULKAN
#include <vulkan/vulkan.h>
#endif

#include <span>
#include <vector>

namespace metaral::render {

struct FluidGpuParams {
    uint32_t max_particles = 65536;
};

class FluidComputeContext {
public:
#ifdef METARAL_ENABLE_VULKAN
    FluidComputeContext() = default;
    ~FluidComputeContext();

    // Initializes descriptor layouts, pipelines, and GPU buffers. Safe to call
    // no-op when Vulkan is not available.
    void initialize(VkDevice device,
                    VkPhysicalDevice physical,
                    uint32_t queue_family_index,
                    const FluidGpuParams& params);

    bool initialized() const noexcept { return initialized_; }

    // Dispatches a minimal step (external forces + integrate) for `count`
    // particles. More stages will be chained once neighbor tables are ready.
    void step(VkCommandBuffer cmd,
              float dt,
              const sim::SphParams& params,
              uint32_t count);

    // Fills identity ranges/indices so density/pressure kernels can run in
    // a trivial mode (each particle only sees itself) until the hash/range
    // builder is wired in.
    void bootstrap_identity_ranges(uint32_t count);

    // CPU upload path for particle data. Count is clamped to max_particles_
    // provided at init.
    void upload_particles(std::span<const sim::FluidParticle> particles);
#else
    void initialize(...) {}
    void step(...) {}
    void bootstrap_identity_ranges(...) {}
    void upload_particles(...) {}
    bool initialized() const noexcept { return false; }
#endif

private:
#ifdef METARAL_ENABLE_VULKAN
    VkDevice device_ = VK_NULL_HANDLE;
    VkPhysicalDevice physical_ = VK_NULL_HANDLE;
    uint32_t queue_family_ = 0;
    bool initialized_ = false;

    VkPipelineLayout pipeline_layout_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout set_layout_ = VK_NULL_HANDLE;
    VkDescriptorPool descriptor_pool_ = VK_NULL_HANDLE;
    VkDescriptorSet descriptor_set_ = VK_NULL_HANDLE;

    VkPipeline external_forces_ = VK_NULL_HANDLE;
    VkPipeline hash_ = VK_NULL_HANDLE;
    VkPipeline reorder_ = VK_NULL_HANDLE;
    VkPipeline density_ = VK_NULL_HANDLE;
    VkPipeline pressure_ = VK_NULL_HANDLE;
    VkPipeline viscosity_ = VK_NULL_HANDLE;
    VkPipeline integrate_ = VK_NULL_HANDLE;
    VkPipeline bitonic_sort_ = VK_NULL_HANDLE;
    VkPipeline range_mark_ = VK_NULL_HANDLE;
    VkPipeline range_scan_fwd_ = VK_NULL_HANDLE;
    VkPipeline range_scan_bwd_ = VK_NULL_HANDLE;

    VkBuffer positions_a_ = VK_NULL_HANDLE;
    VkBuffer positions_b_ = VK_NULL_HANDLE;
    VkDeviceMemory positions_mem_ = VK_NULL_HANDLE; // shared for A/B
    VkDeviceMemory positions_b_mem_ = VK_NULL_HANDLE; // separate to simplify mapping

    VkBuffer keys_ = VK_NULL_HANDLE;
    VkBuffer indices_ = VK_NULL_HANDLE;
    VkBuffer range_starts_ = VK_NULL_HANDLE;
    VkBuffer range_ends_ = VK_NULL_HANDLE;
    VkBuffer densities_ = VK_NULL_HANDLE;
    VkDeviceMemory keys_mem_ = VK_NULL_HANDLE;
    VkDeviceMemory indices_mem_ = VK_NULL_HANDLE;
    VkDeviceMemory range_starts_mem_ = VK_NULL_HANDLE;
    VkDeviceMemory range_ends_mem_ = VK_NULL_HANDLE;
    VkDeviceMemory densities_mem_ = VK_NULL_HANDLE;

    uint32_t max_particles_ = 0;
    float last_smoothing_radius_ = 0.2f;
    uint32_t last_count_ = 0;
    uint32_t last_padded_ = 0;

    void destroy();
    VkPipeline create_compute_pipeline(const char* spv_path);
    void create_storage_buffer(VkDeviceSize size, VkBuffer& buf, VkDeviceMemory& mem);
#endif
};

} // namespace metaral::render
