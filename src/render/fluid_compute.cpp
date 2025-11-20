#include "metaral/render/fluid_compute.hpp"

#ifdef METARAL_ENABLE_VULKAN

#include <stdexcept>
#include <vector>
#include <fstream>

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

    device_ = VK_NULL_HANDLE;
    initialized_ = false;
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

    // Descriptor set layout (bindings match shader bindings 0-7).
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
    push.size = sizeof(float) * 7 + sizeof(uint32_t); // matches FluidPush in shader

    layout_info.setLayoutCount = 1;
    layout_info.pSetLayouts = &set_layout_;
    layout_info.pushConstantRangeCount = 1;
    layout_info.pPushConstantRanges = &push;
    if (vkCreatePipelineLayout(device_, &layout_info, nullptr, &pipeline_layout_) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create fluid pipeline layout");
    }

    // Descriptor pool & set
    VkDescriptorPoolSize pool_size{};
    pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    pool_size.descriptorCount = 16; // generous

    VkDescriptorPoolCreateInfo pool_info{};
    pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_info.poolSizeCount = 1;
    pool_info.pPoolSizes = &pool_size;
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
    const VkDeviceSize density_size = sizeof(float) * 2 * max_particles_;
    create_storage_buffer(density_size, densities_, densities_mem_);

    // Descriptor writes
    auto write = [&](uint32_t binding, VkBuffer buf, VkDeviceSize size) {
        VkDescriptorBufferInfo buf_info{};
        buf_info.buffer = buf;
        buf_info.offset = 0;
        buf_info.range = size;

        VkWriteDescriptorSet w{};
        w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w.dstSet = descriptor_set_;
        w.dstBinding = binding;
        w.descriptorCount = 1;
        w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        w.pBufferInfo = &buf_info;
        return w;
    };

    std::vector<VkWriteDescriptorSet> writes;
    writes.push_back(write(0, positions_a_, positions_size));
    writes.push_back(write(1, positions_b_, positions_size));
    writes.push_back(write(2, keys_, scalar_u_size));
    writes.push_back(write(3, indices_, scalar_u_size));
    writes.push_back(write(4, positions_b_, positions_size)); // sorted buffer placeholder
    writes.push_back(write(5, range_starts_, scalar_u_size));
    writes.push_back(write(6, range_ends_, scalar_u_size));
    writes.push_back(write(7, densities_, density_size));

    vkUpdateDescriptorSets(device_, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);

    // Pipelines
    external_forces_ = create_compute_pipeline(METARAL_SHADER_DIR "/fluid_external_forces.spv");
    hash_            = create_compute_pipeline(METARAL_SHADER_DIR "/fluid_spatial_hash.spv");
    reorder_         = create_compute_pipeline(METARAL_SHADER_DIR "/fluid_reorder.spv");
    density_         = create_compute_pipeline(METARAL_SHADER_DIR "/fluid_density.spv");
    pressure_        = create_compute_pipeline(METARAL_SHADER_DIR "/fluid_pressure.spv");
    viscosity_       = create_compute_pipeline(METARAL_SHADER_DIR "/fluid_viscosity.spv");
    integrate_       = create_compute_pipeline(METARAL_SHADER_DIR "/fluid_integrate.spv");

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

    const VkDeviceSize particle_bytes = sizeof(float) * 8;
    const VkDeviceSize copy_size = particle_bytes * count;

    void* data = nullptr;
    vkMapMemory(device_, positions_mem_, 0, copy_size, 0, &data);
    auto* out = static_cast<float*>(data);
    for (uint32_t i = 0; i < count; ++i) {
        const auto& p = particles[i];
        // vec4 pos, vec4 vel
        out[i * 8 + 0] = p.position.x;
        out[i * 8 + 1] = p.position.y;
        out[i * 8 + 2] = p.position.z;
        out[i * 8 + 3] = 0.0f;
        out[i * 8 + 4] = p.velocity.x;
        out[i * 8 + 5] = p.velocity.y;
        out[i * 8 + 6] = p.velocity.z;
        out[i * 8 + 7] = 0.0f;
    }
    vkUnmapMemory(device_, positions_mem_);

    // Naive neighbor ranges for now.
    // TODO: replace with GPU radix sort + prefix offsets over spatial hash.
    bootstrap_identity_ranges(count);
}

void FluidComputeContext::step(VkCommandBuffer cmd,
                               float dt,
                               const sim::SphParams& params,
                               uint32_t count) {
    if (!initialized_ || count == 0) {
        return;
    }

    const uint32_t group_count = (count + 255u) / 256u;

    struct Push {
        float deltaTime;
        float smoothingRadius;
        float targetDensity;
        float pressureMultiplier;
        float nearPressureMultiplier;
        float viscosityStrength;
        float gravity;
        uint32_t numParticles;
    } push;

    push.deltaTime = dt;
    push.smoothingRadius = params.smoothing_radius;
    push.targetDensity = params.target_density;
    push.pressureMultiplier = params.pressure_multiplier;
    push.nearPressureMultiplier = params.near_pressure_multiplier;
    push.viscosityStrength = params.viscosity_strength;
    push.gravity = params.gravity;
    push.numParticles = count;

    auto bind_and_dispatch = [&](VkPipeline pipeline) {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                pipeline_layout_, 0, 1, &descriptor_set_, 0, nullptr);
        vkCmdPushConstants(cmd, pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(Push), &push);
        vkCmdDispatch(cmd, group_count, 1, 1);
    };

    // Minimal sequence until hash/ranges are wired: just gravity + integrate.
    bind_and_dispatch(external_forces_);
    bind_and_dispatch(integrate_);
}

} // namespace metaral::render

#endif // METARAL_ENABLE_VULKAN
