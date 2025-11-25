#include "metaral/render/fluid_compute.hpp"

#ifdef METARAL_ENABLE_VULKAN

#include <stdexcept>
#include <vector>
#include <fstream>
#include <cstring>

namespace metaral::render {

namespace {

std::vector<char> read_file(const char* path) {
    std::ifstream file(path, std::ios::ate | std::ios::binary);
    if (!file) {
        throw std::runtime_error(std::string("Failed to open SPV file: ") + path);
    }
    const std::streamsize size = file.tellg();
    std::vector<char> buffer(static_cast<std::size_t>(size));
    file.seekg(0);
    file.read(buffer.data(), size);
    return buffer;
}

uint32_t find_memory_type(VkPhysicalDevice physical,
                          uint32_t type_bits,
                          VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties memory_properties;
    vkGetPhysicalDeviceMemoryProperties(physical, &memory_properties);
    for (uint32_t index = 0; index < memory_properties.memoryTypeCount; ++index) {
        const bool supported = (type_bits & (1u << index)) != 0;
        const bool has_flags =
            (memory_properties.memoryTypes[index].propertyFlags & properties) == properties;
        if (supported && has_flags) {
            return index;
        }
    }
    throw std::runtime_error("Failed to find suitable memory type for fluid buffers");
}

inline uint32_t cpu_hash(int32_t x, int32_t y, int32_t z) {
    // Match the shader hash path; simple integer mix then fold to scalar.
    uint32_t vx = static_cast<uint32_t>(x);
    uint32_t vy = static_cast<uint32_t>(y);
    uint32_t vz = static_cast<uint32_t>(z);
    vx = (vx + 0x7ed55d16u) + (vx << 12u);
    vx = (vx ^ 0xc761c23cu) ^ (vx >> 19u);
    vx = (vx + 0x165667b1u) + (vx << 5u);
    vx = (vx + 0xd3a2646cu) ^ (vx << 9u);
    vx = (vx + 0xfd7046c5u) + (vx << 3u);
    vx = (vx ^ 0xb55a4f09u) ^ (vx >> 16u);

    vy = (vy + 0x7ed55d16u) + (vy << 12u);
    vy = (vy ^ 0xc761c23cu) ^ (vy >> 19u);
    vy = (vy + 0x165667b1u) + (vy << 5u);
    vy = (vy + 0xd3a2646cu) ^ (vy << 9u);
    vy = (vy + 0xfd7046c5u) + (vy << 3u);
    vy = (vy ^ 0xb55a4f09u) ^ (vy >> 16u);

    vz = (vz + 0x7ed55d16u) + (vz << 12u);
    vz = (vz ^ 0xc761c23cu) ^ (vz >> 19u);
    vz = (vz + 0x165667b1u) + (vz << 5u);
    vz = (vz + 0xd3a2646cu) ^ (vz << 9u);
    vz = (vz + 0xfd7046c5u) + (vz << 3u);
    vz = (vz ^ 0xb55a4f09u) ^ (vz >> 16u);

    return vx ^ vy ^ vz;
}

// Keep CPU-side push constants in sync with FluidPush in shaders/fluid_common.glsl
struct FluidPushConstants {
    float deltaTime;
    float smoothingRadius;
    float targetDensity;
    float pressureMultiplier;
    float nearPressureMultiplier;
    float viscosityStrength;
    float gravity;
    uint32_t numParticles;
    uint32_t aux0;
    uint32_t aux1;
    uint32_t aux2;
};
static_assert(sizeof(FluidPushConstants) == 44,
              "CPU/GPU Fluid push constants drifted; update both sides together.");

struct FluidParamsUbo {
    float volumeOriginCell[4];
    float volumeDimIso[4];
    float planetParams[4];
    float sdfParams[4]; // x=dim, y=voxel_size, z=half_extent, w=planet_radius
    float octreeParams[4]; // x=node_count, y=root_index, z=depth, w=enabled
};

} // namespace

FluidComputeContext::~FluidComputeContext() {
    destroy();
}

void FluidComputeContext::destroy() {
    if (!device_) return;

    auto destroy_pipeline = [this](VkPipeline p) {
        if (p) vkDestroyPipeline(device_, p, nullptr);
    };
    destroy_pipeline(external_forces_);
    destroy_pipeline(hash_);
    destroy_pipeline(reorder_);
    destroy_pipeline(density_);
    destroy_pipeline(pressure_);
    destroy_pipeline(viscosity_);
    destroy_pipeline(integrate_);
    destroy_pipeline(bucket_count_);
    destroy_pipeline(bucket_prefix_);
    destroy_pipeline(radix_scatter_);
    destroy_pipeline(range_mark_);
    destroy_pipeline(range_scan_fwd_);
    destroy_pipeline(range_scan_bwd_);
    destroy_pipeline(volume_clear_);
    destroy_pipeline(volume_splat_);

    if (pipeline_layout_) vkDestroyPipelineLayout(device_, pipeline_layout_, nullptr);
    if (set_layout_) vkDestroyDescriptorSetLayout(device_, set_layout_, nullptr);
    if (descriptor_pool_) vkDestroyDescriptorPool(device_, descriptor_pool_, nullptr);

    auto destroy_buffer = [this](VkBuffer b, VkDeviceMemory m) {
        if (b) vkDestroyBuffer(device_, b, nullptr);
        if (m) vkFreeMemory(device_, m, nullptr);
    };
    destroy_buffer(positions_a_, positions_mem_);
    destroy_buffer(positions_b_, positions_b_mem_);
    destroy_buffer(keys_, keys_mem_);
    destroy_buffer(indices_, indices_mem_);
    destroy_buffer(range_starts_, range_starts_mem_);
    destroy_buffer(range_ends_, range_ends_mem_);
    destroy_buffer(densities_, densities_mem_);
    destroy_buffer(bucket_counts_, bucket_counts_mem_);
    destroy_buffer(bucket_offsets_, bucket_offsets_mem_);
    destroy_buffer(sorted_indices_, sorted_indices_mem_);
    destroy_buffer(params_ubo_, params_ubo_mem_);

    sdf_buffer_ = VK_NULL_HANDLE;
    sdf_buffer_info_ = {};
    sdf_dim_ = 0;
    sdf_voxel_size_ = 0.0f;
    sdf_half_extent_ = 0.0f;
    sdf_planet_radius_ = 0.0f;
    sdf_octree_buffer_ = VK_NULL_HANDLE;
    sdf_octree_info_ = {};
    sdf_octree_node_count_ = 0;
    sdf_octree_root_index_ = 0;
    sdf_octree_depth_ = 0;

    device_ = VK_NULL_HANDLE;
    initialized_ = false;
}

void FluidComputeContext::refresh_descriptor_set() {
    if (!device_ || descriptor_set_ == VK_NULL_HANDLE) {
        return;
    }
    // All buffers must be valid before updating descriptors.
    if (positions_a_ == VK_NULL_HANDLE || positions_b_ == VK_NULL_HANDLE ||
        keys_ == VK_NULL_HANDLE || indices_ == VK_NULL_HANDLE ||
        range_starts_ == VK_NULL_HANDLE || range_ends_ == VK_NULL_HANDLE ||
        densities_ == VK_NULL_HANDLE) {
        return;
    }

    const VkDeviceSize particle_bytes = sizeof(float) * 8;
    const VkDeviceSize positions_size = particle_bytes * max_particles_;
    const VkDeviceSize scalar_u_size = sizeof(uint32_t) * max_particles_;
    const VkDeviceSize density_size = sizeof(float) * 2 * max_particles_;

    std::vector<VkDescriptorBufferInfo> infos;
    infos.reserve(15);
    auto add_info = [&](VkBuffer buf, VkDeviceSize size) {
        VkDescriptorBufferInfo bi{};
        bi.buffer = buf;
        bi.offset = 0;
        bi.range = size;
        infos.push_back(bi);
        return &infos.back();
    };

    std::vector<VkWriteDescriptorSet> writes;
    writes.reserve(15);
    auto add_write = [&](uint32_t binding, const VkDescriptorBufferInfo* info_ptr) {
        VkWriteDescriptorSet w{};
        w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w.dstSet = descriptor_set_;
        w.dstBinding = binding;
        w.descriptorCount = 1;
        w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        w.pBufferInfo = info_ptr;
        writes.push_back(w);
    };

    add_write(0, add_info(positions_a_, positions_size));
    add_write(1, add_info(positions_b_, positions_size));
    add_write(2, add_info(keys_, scalar_u_size));
    add_write(3, add_info(indices_, scalar_u_size));
    add_write(4, add_info(positions_a_, positions_size)); // sorted output buffer
    add_write(5, add_info(range_starts_, scalar_u_size));
    add_write(6, add_info(range_ends_, scalar_u_size));
    add_write(7, add_info(densities_, density_size));
    // Bucket offsets/counts and sorted indices (radix)
    if (bucket_offsets_ != VK_NULL_HANDLE) {
        add_write(11, add_info(bucket_offsets_, sizeof(uint32_t) * 256));
    }
    if (bucket_counts_ != VK_NULL_HANDLE) {
        add_write(12, add_info(bucket_counts_, sizeof(uint32_t) * 256));
    }
    if (sorted_indices_ != VK_NULL_HANDLE) {
        add_write(13, add_info(sorted_indices_, scalar_u_size));
    }

    VkDescriptorImageInfo image_info{};
    if (density_image_view_ != VK_NULL_HANDLE) {
        image_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        image_info.imageView = density_image_view_;
        image_info.sampler = VK_NULL_HANDLE;

        VkWriteDescriptorSet img_write{};
        img_write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        img_write.dstSet = descriptor_set_;
        img_write.dstBinding = 8;
        img_write.descriptorCount = 1;
        img_write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        img_write.pImageInfo = &image_info;
        writes.push_back(img_write);
    }

    if (params_ubo_ != VK_NULL_HANDLE) {
        VkDescriptorBufferInfo ubo_info{};
        ubo_info.buffer = params_ubo_;
        ubo_info.offset = 0;
        ubo_info.range = sizeof(FluidParamsUbo);

        VkWriteDescriptorSet ubo_write{};
        ubo_write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        ubo_write.dstSet = descriptor_set_;
        ubo_write.dstBinding = 9;
        ubo_write.descriptorCount = 1;
        ubo_write.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        ubo_write.pBufferInfo = &ubo_info;
        writes.push_back(ubo_write);
    }

    VkDescriptorBufferInfo sdf_info = sdf_buffer_info_;
    VkDescriptorBufferInfo sdf_fallback{};
    if (sdf_info.buffer == VK_NULL_HANDLE && positions_a_ != VK_NULL_HANDLE) {
        sdf_fallback.buffer = positions_a_;
        sdf_fallback.offset = 0;
        sdf_fallback.range = sizeof(float) * 8; // one particle's worth
        sdf_info = sdf_fallback;
    }
    if (sdf_info.buffer != VK_NULL_HANDLE) {
        VkWriteDescriptorSet sdf_write{};
        sdf_write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        sdf_write.dstSet = descriptor_set_;
        sdf_write.dstBinding = 10;
        sdf_write.descriptorCount = 1;
        sdf_write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        sdf_write.pBufferInfo = &sdf_info;
        writes.push_back(sdf_write);
    }

    VkDescriptorBufferInfo oct_info = sdf_octree_info_;
    VkDescriptorBufferInfo oct_fallback{};
    if (oct_info.buffer == VK_NULL_HANDLE && positions_a_ != VK_NULL_HANDLE) {
        oct_fallback.buffer = positions_a_;
        oct_fallback.offset = 0;
        oct_fallback.range = sizeof(float) * 8; // one particle worth as dummy
        oct_info = oct_fallback;
    }
    if (oct_info.buffer != VK_NULL_HANDLE) {
        VkWriteDescriptorSet oct_write{};
        oct_write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        oct_write.dstSet = descriptor_set_;
        oct_write.dstBinding = 14;
        oct_write.descriptorCount = 1;
        oct_write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        oct_write.pBufferInfo = &oct_info;
        writes.push_back(oct_write);
    }

    vkUpdateDescriptorSets(device_, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
}

void FluidComputeContext::set_density_image(VkImage image, VkImageView view) {
    density_image_ = image;
    density_image_view_ = view;
    refresh_descriptor_set();
}

void FluidComputeContext::set_sdf_buffer(VkBuffer sdf_values,
                                         uint32_t dim,
                                         float voxel_size,
                                         float half_extent,
                                         float planet_radius) {
    sdf_buffer_ = sdf_values;
    if (sdf_buffer_ != VK_NULL_HANDLE) {
        sdf_buffer_info_.buffer = sdf_buffer_;
        sdf_buffer_info_.offset = 0;
        sdf_buffer_info_.range = VK_WHOLE_SIZE;
        sdf_dim_ = dim;
        sdf_voxel_size_ = voxel_size;
        sdf_half_extent_ = half_extent;
        sdf_planet_radius_ = planet_radius;
    } else {
        sdf_buffer_info_ = {};
        sdf_dim_ = 0;
        sdf_voxel_size_ = 0.0f;
        sdf_half_extent_ = 0.0f;
        sdf_planet_radius_ = 0.0f;
    }
    refresh_descriptor_set();
}

void FluidComputeContext::set_sdf_octree(VkBuffer octree_nodes,
                                         uint32_t node_count,
                                         uint32_t root_index,
                                         uint32_t depth) {
    sdf_octree_buffer_ = octree_nodes;
    if (sdf_octree_buffer_ != VK_NULL_HANDLE) {
        sdf_octree_info_.buffer = sdf_octree_buffer_;
        sdf_octree_info_.offset = 0;
        sdf_octree_info_.range = VK_WHOLE_SIZE;
        sdf_octree_node_count_ = node_count;
        sdf_octree_root_index_ = root_index;
        sdf_octree_depth_ = depth;
    } else {
        sdf_octree_info_ = {};
        sdf_octree_node_count_ = 0;
        sdf_octree_root_index_ = 0;
        sdf_octree_depth_ = 0;
    }
    refresh_descriptor_set();
}

VkPipeline FluidComputeContext::create_compute_pipeline(const char* spv_path) {
    auto code = read_file(spv_path);

    VkShaderModuleCreateInfo module_info{};
    module_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    module_info.codeSize = code.size();
    module_info.pCode = reinterpret_cast<const uint32_t*>(code.data());

    VkShaderModule module;
    if (vkCreateShaderModule(device_, &module_info, nullptr, &module) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create fluid compute shader module");
    }

    VkPipelineShaderStageCreateInfo stage{};
    stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    stage.module = module;
    stage.pName  = "main";

    VkComputePipelineCreateInfo info{};
    info.sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    info.stage  = stage;
    info.layout = pipeline_layout_;

    VkPipeline pipeline;
    if (vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &info, nullptr, &pipeline) != VK_SUCCESS) {
        vkDestroyShaderModule(device_, module, nullptr);
        throw std::runtime_error("Failed to create fluid compute pipeline");
    }

    vkDestroyShaderModule(device_, module, nullptr);
    return pipeline;
}

void FluidComputeContext::create_storage_buffer(VkDeviceSize size,
                                                VkBuffer& buf,
                                                VkDeviceMemory& mem) {
    VkBufferCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    info.size = size;
    info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateBuffer(device_, &info, nullptr, &buf) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create fluid storage buffer");
    }

    VkMemoryRequirements req{};
    vkGetBufferMemoryRequirements(device_, buf, &req);

    VkMemoryAllocateInfo alloc{};
    alloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc.allocationSize = req.size;
    alloc.memoryTypeIndex = find_memory_type(physical_, req.memoryTypeBits,
                                             VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    if (vkAllocateMemory(device_, &alloc, nullptr, &mem) != VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate fluid buffer memory");
    }

    vkBindBufferMemory(device_, buf, mem, 0);
}

void FluidComputeContext::initialize(VkDevice device,
                                     VkPhysicalDevice physical,
                                     uint32_t queue_family_index,
                                     const FluidGpuParams& params) {
    if (initialized_) {
        return;
    }

    device_ = device;
    physical_ = physical;
    queue_family_ = queue_family_index;
    max_particles_ = params.max_particles;

    // Descriptor set layout (bindings match shader bindings 0-13).
    std::vector<VkDescriptorSetLayoutBinding> bindings;
    auto add_binding = [&](uint32_t binding) {
        VkDescriptorSetLayoutBinding b{};
        b.binding = binding;
        b.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        b.descriptorCount = 1;
        b.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        bindings.push_back(b);
    };
    for (uint32_t b = 0; b <= 7; ++b) {
        add_binding(b);
    }
    VkDescriptorSetLayoutBinding img{};
    img.binding = 8;
    img.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    img.descriptorCount = 1;
    img.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    bindings.push_back(img);
    VkDescriptorSetLayoutBinding ubo{};
    ubo.binding = 9;
    ubo.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    ubo.descriptorCount = 1;
    ubo.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    bindings.push_back(ubo);
    VkDescriptorSetLayoutBinding sdf{};
    sdf.binding = 10;
    sdf.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    sdf.descriptorCount = 1;
    sdf.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    bindings.push_back(sdf);
    VkDescriptorSetLayoutBinding bucketOffsets{};
    bucketOffsets.binding = 11;
    bucketOffsets.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bucketOffsets.descriptorCount = 1;
    bucketOffsets.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    bindings.push_back(bucketOffsets);
    VkDescriptorSetLayoutBinding bucketCounts{};
    bucketCounts.binding = 12;
    bucketCounts.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bucketCounts.descriptorCount = 1;
    bucketCounts.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    bindings.push_back(bucketCounts);
    VkDescriptorSetLayoutBinding sortedIdx{};
    sortedIdx.binding = 13;
    sortedIdx.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    sortedIdx.descriptorCount = 1;
    sortedIdx.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    bindings.push_back(sortedIdx);
    VkDescriptorSetLayoutBinding octreeNodes{};
    octreeNodes.binding = 14;
    octreeNodes.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    octreeNodes.descriptorCount = 1;
    octreeNodes.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    bindings.push_back(octreeNodes);

    VkDescriptorSetLayoutCreateInfo set_info{};
    set_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    set_info.bindingCount = static_cast<uint32_t>(bindings.size());
    set_info.pBindings = bindings.data();
    if (vkCreateDescriptorSetLayout(device_, &set_info, nullptr, &set_layout_) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create fluid descriptor set layout");
    }

    VkPipelineLayoutCreateInfo layout_info{};
    layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    VkPushConstantRange push{};
    push.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    push.offset = 0;
    push.size = sizeof(FluidPushConstants); // matches FluidPush in shader

    layout_info.setLayoutCount = 1;
    layout_info.pSetLayouts = &set_layout_;
    layout_info.pushConstantRangeCount = 1;
    layout_info.pPushConstantRanges = &push;
    if (vkCreatePipelineLayout(device_, &layout_info, nullptr, &pipeline_layout_) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create fluid pipeline layout");
    }

    // Descriptor pool & set
    VkDescriptorPoolSize pool_sizes[3]{};
    pool_sizes[0].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    pool_sizes[0].descriptorCount = 29; // extra buffers for radix + octree
    pool_sizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    pool_sizes[1].descriptorCount = 2;
    pool_sizes[2].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    pool_sizes[2].descriptorCount = 2;

    VkDescriptorPoolCreateInfo pool_info{};
    pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_info.poolSizeCount = 3;
    pool_info.pPoolSizes = pool_sizes;
    pool_info.maxSets = 2;
    if (vkCreateDescriptorPool(device_, &pool_info, nullptr, &descriptor_pool_) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create fluid descriptor pool");
    }

    VkDescriptorSetAllocateInfo alloc_info{};
    alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    alloc_info.descriptorPool = descriptor_pool_;
    alloc_info.descriptorSetCount = 1;
    alloc_info.pSetLayouts = &set_layout_;
    if (vkAllocateDescriptorSets(device_, &alloc_info, &descriptor_set_) != VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate fluid descriptor set");
    }

    // Buffers
    const VkDeviceSize particle_bytes = sizeof(float) * 8 /*vec4 pos + vec4 vel*/;
    const VkDeviceSize positions_size = particle_bytes * max_particles_;
    create_storage_buffer(positions_size, positions_a_, positions_mem_);
    create_storage_buffer(positions_size, positions_b_, positions_b_mem_);

    const VkDeviceSize scalar_u_size = sizeof(uint32_t) * max_particles_;
    create_storage_buffer(scalar_u_size, keys_, keys_mem_);
    create_storage_buffer(scalar_u_size, indices_, indices_mem_);
    create_storage_buffer(scalar_u_size, range_starts_, range_starts_mem_);
    create_storage_buffer(scalar_u_size, range_ends_, range_ends_mem_);
    create_storage_buffer(sizeof(uint32_t) * 256, bucket_counts_, bucket_counts_mem_);
    create_storage_buffer(sizeof(uint32_t) * 256, bucket_offsets_, bucket_offsets_mem_);
    create_storage_buffer(scalar_u_size, sorted_indices_, sorted_indices_mem_);
    const VkDeviceSize density_size = sizeof(float) * 2 * max_particles_;
    create_storage_buffer(density_size, densities_, densities_mem_);

    // Uniform buffer for per-dispatch params (volume + planet).
    VkBufferCreateInfo ubo_info{};
    ubo_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    ubo_info.size = sizeof(FluidParamsUbo);
    ubo_info.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
    ubo_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(device_, &ubo_info, nullptr, &params_ubo_) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create fluid params uniform buffer");
    }
    VkMemoryRequirements ubo_req{};
    vkGetBufferMemoryRequirements(device_, params_ubo_, &ubo_req);

    VkMemoryAllocateInfo ubo_alloc{};
    ubo_alloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    ubo_alloc.allocationSize = ubo_req.size;
    ubo_alloc.memoryTypeIndex = find_memory_type(
        physical_, ubo_req.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (vkAllocateMemory(device_, &ubo_alloc, nullptr, &params_ubo_mem_) != VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate fluid params uniform buffer memory");
    }
    vkBindBufferMemory(device_, params_ubo_, params_ubo_mem_, 0);

    // Descriptor writes
    refresh_descriptor_set();

    // Pipelines
    external_forces_ = create_compute_pipeline(METARAL_SHADER_DIR "/fluid_external_forces.spv");
    hash_            = create_compute_pipeline(METARAL_SHADER_DIR "/fluid_spatial_hash.spv");
    reorder_         = create_compute_pipeline(METARAL_SHADER_DIR "/fluid_reorder.spv");
    density_         = create_compute_pipeline(METARAL_SHADER_DIR "/fluid_density.spv");
    pressure_        = create_compute_pipeline(METARAL_SHADER_DIR "/fluid_pressure.spv");
    viscosity_       = create_compute_pipeline(METARAL_SHADER_DIR "/fluid_viscosity.spv");
    integrate_       = create_compute_pipeline(METARAL_SHADER_DIR "/fluid_integrate.spv");
    bucket_count_    = create_compute_pipeline(METARAL_SHADER_DIR "/fluid_bucket_count.spv");
    bucket_prefix_   = create_compute_pipeline(METARAL_SHADER_DIR "/fluid_bucket_prefix.spv");
    radix_scatter_   = create_compute_pipeline(METARAL_SHADER_DIR "/fluid_radix_scatter.spv");
    range_mark_      = create_compute_pipeline(METARAL_SHADER_DIR "/fluid_range_mark.spv");
    range_scan_fwd_  = create_compute_pipeline(METARAL_SHADER_DIR "/fluid_range_scan_forward.spv");
    range_scan_bwd_  = create_compute_pipeline(METARAL_SHADER_DIR "/fluid_range_scan_backward.spv");
    volume_clear_    = create_compute_pipeline(METARAL_SHADER_DIR "/fluid_clear_volume.spv");
    volume_splat_    = create_compute_pipeline(METARAL_SHADER_DIR "/fluid_splat_volume.spv");

    initialized_ = true;
}

void FluidComputeContext::bootstrap_identity_ranges(uint32_t count) {
    const VkDeviceSize scalar_u_size = sizeof(uint32_t) * count;
    // Fill ranges and indices so that start=i, end=i+1 and indices[i]=i.
    auto fill_u32 = [&](VkDeviceMemory mem, VkBuffer buf, auto generator) {
        void* data = nullptr;
        vkMapMemory(device_, mem, 0, scalar_u_size, 0, &data);
        auto* ptr = static_cast<uint32_t*>(data);
        for (uint32_t i = 0; i < count; ++i) {
            ptr[i] = generator(i);
        }
        vkUnmapMemory(device_, mem);
    };
    fill_u32(indices_mem_, indices_, [](uint32_t i) { return i; });
    fill_u32(range_starts_mem_, range_starts_, [](uint32_t i) { return i; });
    fill_u32(range_ends_mem_, range_ends_, [](uint32_t i) { return i + 1; });
}

void FluidComputeContext::upload_particles(std::span<const sim::FluidParticle> particles) {
    if (!initialized_) return;
    const uint32_t count = std::min<uint32_t>(max_particles_,
                                              static_cast<uint32_t>(particles.size()));
    if (count == 0) {
        return;
    }

    last_count_ = count;
    last_padded_ = 1u << static_cast<uint32_t>(std::ceil(std::log2(std::max(1u, count))));
    if (last_padded_ > max_particles_) {
        last_padded_ = max_particles_;
    }

    const VkDeviceSize particle_bytes = sizeof(float) * 8;
    const VkDeviceSize copy_size = particle_bytes * count;

    auto write_positions = [&](VkDeviceMemory mem) {
        void* data = nullptr;
        vkMapMemory(device_, mem, 0, copy_size, 0, &data);
        auto* out = static_cast<float*>(data);
        for (uint32_t i = 0; i < count; ++i) {
            const auto& p = particles[i];
            out[i * 8 + 0] = p.position.x;
            out[i * 8 + 1] = p.position.y;
            out[i * 8 + 2] = p.position.z;
            out[i * 8 + 3] = 0.0f;
            out[i * 8 + 4] = p.velocity.x;
            out[i * 8 + 5] = p.velocity.y;
            out[i * 8 + 6] = p.velocity.z;
            out[i * 8 + 7] = 0.0f;
        }
        vkUnmapMemory(device_, mem);
    };

    write_positions(positions_mem_);
    write_positions(positions_b_mem_);

    auto fill_u32 = [&](VkDeviceMemory mem, uint32_t value) {
        void* data = nullptr;
        vkMapMemory(device_, mem, 0, sizeof(uint32_t) * last_padded_, 0, &data);
        auto* out = static_cast<uint32_t*>(data);
        for (uint32_t i = 0; i < last_padded_; ++i) {
            out[i] = (i < count) ? (value == UINT32_MAX ? 0u : i) : value;
        }
        vkUnmapMemory(device_, mem);
    };

    fill_u32(keys_mem_, 0xFFFFFFFFu); // hash kernel will overwrite first count entries
    fill_u32(indices_mem_, 0u);
    fill_u32(range_starts_mem_, 0u);
    fill_u32(range_ends_mem_, 0u);
}

void FluidComputeContext::step(VkCommandBuffer cmd,
                               float dt,
                               const sim::SphParams& params,
                               uint32_t count,
                               const FluidVolumeParams& volume) {
    if (!initialized_ || count == 0) {
        return;
    }

    // Ensure descriptors still point at valid buffers (defensive against any prior reinit).
    refresh_descriptor_set();

    const uint32_t group_count = (count + 255u) / 256u;
    const uint32_t padded = (last_padded_ > 0) ? last_padded_ : count;

    FluidPushConstants push{};

    push.deltaTime = dt;
    push.smoothingRadius = params.smoothing_radius;
    push.targetDensity = params.target_density;
    push.pressureMultiplier = params.pressure_multiplier;
    push.nearPressureMultiplier = params.near_pressure_multiplier;
    push.viscosityStrength = params.viscosity_strength;
    push.gravity = params.gravity;
    push.numParticles = count;
    push.aux0 = 0;
    push.aux1 = 0;
    push.aux2 = 0;
    last_smoothing_radius_ = params.smoothing_radius;

    // Update uniform buffer for volume/planet params.
    if (params_ubo_mem_ != VK_NULL_HANDLE) {
        FluidParamsUbo ubo{};
        ubo.volumeOriginCell[0] = volume.origin.x;
        ubo.volumeOriginCell[1] = volume.origin.y;
        ubo.volumeOriginCell[2] = volume.origin.z;
        ubo.volumeOriginCell[3] = volume.cell_size;
        ubo.volumeDimIso[0] = static_cast<float>(volume.dim_x);
        ubo.volumeDimIso[1] = static_cast<float>(volume.dim_y);
        ubo.volumeDimIso[2] = static_cast<float>(volume.dim_z);
        ubo.volumeDimIso[3] = volume.iso_threshold;
        ubo.planetParams[0] = volume.planet_radius;
        ubo.planetParams[1] = params.collision_damping;
        ubo.planetParams[2] = 0.0f;
        ubo.planetParams[3] = 0.0f;
        ubo.sdfParams[0] = static_cast<float>(sdf_dim_);
        ubo.sdfParams[1] = sdf_voxel_size_;
        ubo.sdfParams[2] = sdf_half_extent_;
        ubo.sdfParams[3] = (sdf_planet_radius_ > 0.0f) ? sdf_planet_radius_ : volume.planet_radius;
        ubo.octreeParams[0] = static_cast<float>(sdf_octree_node_count_);
        ubo.octreeParams[1] = static_cast<float>(sdf_octree_root_index_);
        ubo.octreeParams[2] = static_cast<float>(sdf_octree_depth_);
        ubo.octreeParams[3] = (sdf_octree_buffer_ != VK_NULL_HANDLE && sdf_octree_node_count_ > 0) ? 1.0f : 0.0f;

        void* mapped = nullptr;
        vkMapMemory(device_, params_ubo_mem_, 0, sizeof(FluidParamsUbo), 0, &mapped);
        std::memcpy(mapped, &ubo, sizeof(FluidParamsUbo));
        vkUnmapMemory(device_, params_ubo_mem_);
    }

    auto bind_and_dispatch = [&](VkPipeline pipeline, uint32_t groups, const FluidPushConstants& p) {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                pipeline_layout_, 0, 1, &descriptor_set_, 0, nullptr);
        vkCmdPushConstants(cmd, pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(FluidPushConstants), &p);
        vkCmdDispatch(cmd, groups, 1, 1);
    };
    auto bind_and_dispatch_3d = [&](VkPipeline pipeline,
                                    uint32_t gx,
                                    uint32_t gy,
                                    uint32_t gz,
                                    const FluidPushConstants& p) {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                pipeline_layout_, 0, 1, &descriptor_set_, 0, nullptr);
        vkCmdPushConstants(cmd, pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(FluidPushConstants), &p);
        vkCmdDispatch(cmd, gx, gy, gz);
    };

    auto barrier_buffers = [&](std::initializer_list<VkBuffer> bufs) {
        std::vector<VkBufferMemoryBarrier> barriers;
        barriers.reserve(bufs.size());
        for (auto b : bufs) {
            VkBufferMemoryBarrier mb{};
            mb.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
            mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            mb.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            mb.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            mb.buffer = b;
            mb.offset = 0;
            mb.size = VK_WHOLE_SIZE;
            barriers.push_back(mb);
        }
        vkCmdPipelineBarrier(cmd,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             0,
                            0, nullptr,
                            static_cast<uint32_t>(barriers.size()), barriers.data(),
                            0, nullptr);
    };

    auto barrier_image = [&](VkAccessFlags src, VkAccessFlags dst) {
        if (density_image_ == VK_NULL_HANDLE) {
            return;
        }
        VkImageMemoryBarrier ib{};
        ib.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        ib.srcAccessMask = src;
        ib.dstAccessMask = dst;
        ib.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        ib.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        ib.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        ib.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        ib.image = density_image_;
        ib.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        ib.subresourceRange.baseMipLevel = 0;
        ib.subresourceRange.levelCount = 1;
        ib.subresourceRange.baseArrayLayer = 0;
        ib.subresourceRange.layerCount = 1;

        vkCmdPipelineBarrier(cmd,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             0,
                             0, nullptr,
                             0, nullptr,
                             1, &ib);
    };

    // External forces (positions_in -> positions_out)
    bind_and_dispatch(external_forces_, group_count, push);
    barrier_buffers({positions_b_});

    // Hash keys (also writes indices = i)
    bind_and_dispatch(hash_, group_count, push);
    barrier_buffers({keys_, indices_});

    // Radix sort keys/indices by bytes (LSB-first).
    vkCmdFillBuffer(cmd, bucket_counts_, 0, sizeof(uint32_t) * 256, 0);
    VkBufferMemoryBarrier counts_zero{};
    counts_zero.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    counts_zero.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    counts_zero.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    counts_zero.buffer = bucket_counts_;
    counts_zero.offset = 0;
    counts_zero.size = VK_WHOLE_SIZE;
    vkCmdPipelineBarrier(cmd,
                         VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0, 0, nullptr, 1, &counts_zero, 0, nullptr);

    uint32_t shifts[4] = {0u, 8u, 16u, 24u};
    for (uint32_t s = 0; s < 4; ++s) {
        push.aux0 = shifts[s];
        // Count buckets for this byte.
        bind_and_dispatch(bucket_count_, group_count, push);
        barrier_buffers({bucket_counts_});

        // Prefix sums to offsets; also zeros counts for scatter.
        bind_and_dispatch(bucket_prefix_, 1, push);
        barrier_buffers({bucket_counts_, bucket_offsets_});

        // Scatter to sorted_indices_ and rewrite keys in sorted order.
        bind_and_dispatch(radix_scatter_, group_count, push);
        barrier_buffers({keys_, sorted_indices_, bucket_counts_, bucket_offsets_});

        // Copy sorted indices back for next pass / downstream.
        VkBufferCopy idxCopy{};
        idxCopy.srcOffset = 0;
        idxCopy.dstOffset = 0;
        idxCopy.size = sizeof(uint32_t) * count;
        vkCmdCopyBuffer(cmd, sorted_indices_, indices_, 1, &idxCopy);
        VkBufferMemoryBarrier idxBarrier{};
        idxBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        idxBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        idxBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        idxBarrier.buffer = indices_;
        idxBarrier.offset = 0;
        idxBarrier.size = idxCopy.size;
        vkCmdPipelineBarrier(cmd,
                             VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             0, 0, nullptr, 1, &idxBarrier, 0, nullptr);

        // Reset counts for next radix byte.
        vkCmdFillBuffer(cmd, bucket_counts_, 0, sizeof(uint32_t) * 256, 0);
        VkBufferMemoryBarrier resetCounts{};
        resetCounts.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        resetCounts.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        resetCounts.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        resetCounts.buffer = bucket_counts_;
        resetCounts.offset = 0;
        resetCounts.size = VK_WHOLE_SIZE;
        vkCmdPipelineBarrier(cmd,
                             VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             0, 0, nullptr, 1, &resetCounts, 0, nullptr);
    }
    barrier_buffers({keys_, indices_});

    // Mark range starts/ends using sorted keys
    bind_and_dispatch(range_mark_, group_count, push);
    barrier_buffers({range_starts_, range_ends_});

    // Forward scan to propagate starts
    FluidPushConstants scan_push = push;
    for (uint32_t offset = 1; offset < count; offset <<= 1) {
        scan_push.aux0 = offset;
        bind_and_dispatch(range_scan_fwd_, group_count, scan_push);
        barrier_buffers({range_starts_});
    }
    // Backward scan to propagate ends
    for (uint32_t offset = 1; offset < count; offset <<= 1) {
        scan_push.aux0 = offset;
        bind_and_dispatch(range_scan_bwd_, group_count, scan_push);
        barrier_buffers({range_ends_});
    }
    barrier_buffers({range_starts_, range_ends_});

    // Reorder positions into binding 4 (positions_a_)
    bind_and_dispatch(reorder_, group_count, push);
    barrier_buffers({positions_a_});

    // Copy sorted positions back to positions_out buffer (binding 1) for downstream kernels
    VkBufferMemoryBarrier copy_src_barrier{};
    copy_src_barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    copy_src_barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    copy_src_barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    copy_src_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    copy_src_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    copy_src_barrier.buffer = positions_a_;
    copy_src_barrier.offset = 0;
    copy_src_barrier.size = VK_WHOLE_SIZE;
    vkCmdPipelineBarrier(cmd,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT,
                         0, 0, nullptr, 1, &copy_src_barrier, 0, nullptr);

    VkBufferCopy copy{};
    copy.srcOffset = 0;
    copy.dstOffset = 0;
    copy.size = sizeof(float) * 8 * count;
    vkCmdCopyBuffer(cmd, positions_a_, positions_b_, 1, &copy);

    VkBufferMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.buffer = positions_b_;
    barrier.offset = 0;
    barrier.size = copy.size;
    vkCmdPipelineBarrier(cmd,
                         VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0, 0, nullptr, 1, &barrier, 0, nullptr);

    // Density / pressure / viscosity / integrate on sorted buffer
    bind_and_dispatch(density_, group_count, push);
    barrier_buffers({densities_});
    bind_and_dispatch(pressure_, group_count, push);
    barrier_buffers({positions_b_});
    bind_and_dispatch(viscosity_, group_count, push);
    barrier_buffers({positions_b_});
    bind_and_dispatch(integrate_, group_count, push);
    barrier_buffers({positions_b_});

    // Keep positions_in (binding 0) in sync with the post-integrate buffer so
    // the next frame starts from the latest positions/velocities instead of
    // the stale pre-simulation state.
    VkBufferMemoryBarrier copy_back_src{};
    copy_back_src.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    copy_back_src.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    copy_back_src.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    copy_back_src.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    copy_back_src.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    copy_back_src.buffer = positions_b_;
    copy_back_src.offset = 0;
    copy_back_src.size = copy.size;
    vkCmdPipelineBarrier(cmd,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT,
                         0, 0, nullptr, 1, &copy_back_src, 0, nullptr);

    VkBufferCopy copy_back{};
    copy_back.srcOffset = 0;
    copy_back.dstOffset = 0;
    copy_back.size = copy.size;
    vkCmdCopyBuffer(cmd, positions_b_, positions_a_, 1, &copy_back);

    VkBufferMemoryBarrier copy_back_dst{};
    copy_back_dst.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    copy_back_dst.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    copy_back_dst.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    copy_back_dst.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    copy_back_dst.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    copy_back_dst.buffer = positions_a_;
    copy_back_dst.offset = 0;
    copy_back_dst.size = copy_back.size;
    vkCmdPipelineBarrier(cmd,
                         VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0, 0, nullptr, 1, &copy_back_dst, 0, nullptr);

    // Build the GPU density volume so the fragment shader can sample fluids.
    if (density_image_view_ != VK_NULL_HANDLE &&
        volume.dim_x > 0 && volume.dim_y > 0 && volume.dim_z > 0) {
        const uint32_t clear_x = (volume.dim_x + 7u) / 8u;
        const uint32_t clear_y = (volume.dim_y + 7u) / 8u;
        const uint32_t clear_z = (volume.dim_z + 3u) / 4u;
        bind_and_dispatch_3d(volume_clear_, clear_x, clear_y, clear_z, push);
        barrier_image(VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_WRITE_BIT);

        bind_and_dispatch(volume_splat_, group_count, push);
        barrier_image(VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
    }
}

} // namespace metaral::render

#endif // METARAL_ENABLE_VULKAN
