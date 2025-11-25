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
    uint32_t max_particles = 655360; // default capacity (10x prior)
};

struct FluidVolumeParams {
    core::PlanetPosition origin{};
    float cell_size = 0.25f;
    uint32_t dim_x = 0;
    uint32_t dim_y = 0;
    uint32_t dim_z = 0;
    float iso_threshold = 0.5f;
    float planet_radius = 0.0f;
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

    // Refresh storage-buffer descriptor bindings defensively.
    void refresh_descriptor_set();

    bool initialized() const noexcept { return initialized_; }

    // Dispatches a minimal step (external forces + integrate) for `count`
    // particles. More stages will be chained once neighbor tables are ready.
    void step(VkCommandBuffer cmd,
              float dt,
              const sim::SphParams& params,
              uint32_t count,
              const FluidVolumeParams& volume);

    // Fills identity ranges/indices so density/pressure kernels can run in
    // a trivial mode (each particle only sees itself) until the hash/range
    // builder is wired in.
    void bootstrap_identity_ranges(uint32_t count);

    // CPU upload path for particle data. Count is clamped to max_particles_
    // provided at init.
    void upload_particles(std::span<const sim::FluidParticle> particles);
    void set_density_image(VkImage image, VkImageView view);
    // Provide SDF buffer for collision; params are the dense grid meta.
    void set_sdf_buffer(VkBuffer sdf_values,
                        uint32_t dim,
                        float voxel_size,
                        float half_extent,
                        float planet_radius);
    void set_sdf_octree(VkBuffer octree_nodes,
                        uint32_t node_count,
                        uint32_t root_index,
                        uint32_t depth);
#else
    void initialize(...) {}
    void step(...) {}
    void bootstrap_identity_ranges(...) {}
    void upload_particles(...) {}
    bool initialized() const noexcept { return false; }
    void set_density_image(...) {}
    void set_sdf_buffer(...) {}
    void set_sdf_octree(...) {}
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
    VkPipeline bucket_count_ = VK_NULL_HANDLE;
    VkPipeline bucket_prefix_ = VK_NULL_HANDLE;
    VkPipeline radix_scatter_ = VK_NULL_HANDLE;
    VkPipeline range_mark_ = VK_NULL_HANDLE;
    VkPipeline range_scan_fwd_ = VK_NULL_HANDLE;
    VkPipeline range_scan_bwd_ = VK_NULL_HANDLE;
    VkPipeline volume_clear_ = VK_NULL_HANDLE;
    VkPipeline volume_splat_ = VK_NULL_HANDLE;

    VkBuffer positions_a_ = VK_NULL_HANDLE;
    VkBuffer positions_b_ = VK_NULL_HANDLE;
    VkDeviceMemory positions_mem_ = VK_NULL_HANDLE; // shared for A/B
    VkDeviceMemory positions_b_mem_ = VK_NULL_HANDLE; // separate to simplify mapping

    VkBuffer keys_ = VK_NULL_HANDLE;
    VkBuffer indices_ = VK_NULL_HANDLE;
    VkBuffer range_starts_ = VK_NULL_HANDLE;
    VkBuffer range_ends_ = VK_NULL_HANDLE;
    VkBuffer densities_ = VK_NULL_HANDLE;
    VkBuffer bucket_counts_ = VK_NULL_HANDLE;
    VkBuffer bucket_offsets_ = VK_NULL_HANDLE;
    VkBuffer sorted_indices_ = VK_NULL_HANDLE;
    VkBuffer params_ubo_ = VK_NULL_HANDLE;
    VkDeviceMemory keys_mem_ = VK_NULL_HANDLE;
    VkDeviceMemory indices_mem_ = VK_NULL_HANDLE;
    VkDeviceMemory range_starts_mem_ = VK_NULL_HANDLE;
    VkDeviceMemory range_ends_mem_ = VK_NULL_HANDLE;
    VkDeviceMemory densities_mem_ = VK_NULL_HANDLE;
    VkDeviceMemory bucket_counts_mem_ = VK_NULL_HANDLE;
    VkDeviceMemory bucket_offsets_mem_ = VK_NULL_HANDLE;
    VkDeviceMemory sorted_indices_mem_ = VK_NULL_HANDLE;
    VkDeviceMemory params_ubo_mem_ = VK_NULL_HANDLE;

    // Optional collision SDF supplied by the renderer.
    VkBuffer sdf_buffer_ = VK_NULL_HANDLE;
    VkDescriptorBufferInfo sdf_buffer_info_{};
    uint32_t sdf_dim_ = 0;
    float sdf_voxel_size_ = 0.0f;
    float sdf_half_extent_ = 0.0f;
    float sdf_planet_radius_ = 0.0f;
    VkBuffer sdf_octree_buffer_ = VK_NULL_HANDLE;
    VkDescriptorBufferInfo sdf_octree_info_{};
    uint32_t sdf_octree_node_count_ = 0;
    uint32_t sdf_octree_root_index_ = 0;
    uint32_t sdf_octree_depth_ = 0;

    uint32_t max_particles_ = 0;
    float last_smoothing_radius_ = 0.2f;
    uint32_t last_count_ = 0;
    uint32_t last_padded_ = 0;
    VkImage density_image_ = VK_NULL_HANDLE;
    VkImageView density_image_view_ = VK_NULL_HANDLE;

    void destroy();
    VkPipeline create_compute_pipeline(const char* spv_path);
    void create_storage_buffer(VkDeviceSize size, VkBuffer& buf, VkDeviceMemory& mem);
#endif
};

} // namespace metaral::render
