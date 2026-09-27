/*
    src/vulkan_tex.cpp -- hardware texture support for the Vulkan backend

    Copyright (c) 2026 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#if defined(DRJIT_ENABLE_VULKAN)

#include "vulkan.h"
#include "vulkan_tex.h"
#include "vulkan_ts.h"
#include "internal.h"
#include "eval.h"
#include "malloc.h"
#include "log.h"
#include "var.h"
#include "op.h"
#include "tex.h"
#include "util.h"

#include <algorithm>
#include <cstring>

/// Macro to check Vulkan return values inside texture routines
#define vk_tex_check(expr)                                                     \
    do {                                                                       \
        VkResult rv__ = (expr);                                                \
        if (unlikely(rv__ != VK_SUCCESS))                                      \
            jitc_raise("jit_vulkan_tex(): %s failed with %s (%i)!", #expr,     \
                       jitc_vulkan_error_name(rv__), (int) rv__);              \
    } while (0)

// ============================================================================
//  Helpers
// ============================================================================

static void vulkan_tex_formats(const VulkanDevice &dev, int format,
                               size_t channels_storage, int srgb,
                               int compression, size_t *out_storage_channels,
                               VkFormat *out_sampled_fmt,
                               VkFormat *out_unorm_fmt) {
    switch (compression) {
        case 4:
            *out_storage_channels = channels_storage;
            *out_sampled_fmt = VK_FORMAT_BC4_UNORM_BLOCK;
            *out_unorm_fmt = VK_FORMAT_BC4_UNORM_BLOCK;
            return;
        case 5:
            *out_storage_channels = channels_storage;
            *out_sampled_fmt = VK_FORMAT_BC5_UNORM_BLOCK;
            *out_unorm_fmt = VK_FORMAT_BC5_UNORM_BLOCK;
            return;
        case 7:
            *out_storage_channels = channels_storage;
            *out_sampled_fmt = srgb ? VK_FORMAT_BC7_SRGB_BLOCK
                                    : VK_FORMAT_BC7_UNORM_BLOCK;
            *out_unorm_fmt = VK_FORMAT_BC7_UNORM_BLOCK;
            return;
        default:
            break;
    }

    if ((VarType) format == VarType::UInt8) {
        if (srgb) {
            if (channels_storage == 1 && !dev.srgb_r8_supported)
                channels_storage = 4;
            else if (channels_storage == 2 && !dev.srgb_rg8_supported)
                channels_storage = 4;

            *out_storage_channels = channels_storage;
            switch (channels_storage) {
                case 1:
                    *out_sampled_fmt = VK_FORMAT_R8_SRGB;
                    *out_unorm_fmt = VK_FORMAT_R8_UNORM;
                    return;
                case 2:
                    *out_sampled_fmt = VK_FORMAT_R8G8_SRGB;
                    *out_unorm_fmt = VK_FORMAT_R8G8_UNORM;
                    return;
                default:
                    *out_sampled_fmt = VK_FORMAT_R8G8B8A8_SRGB;
                    *out_unorm_fmt = VK_FORMAT_R8G8B8A8_UNORM;
                    return;
            }
        }
        *out_storage_channels = channels_storage;
        switch (channels_storage) {
            case 1:  *out_sampled_fmt = VK_FORMAT_R8_UNORM; break;
            case 2:  *out_sampled_fmt = VK_FORMAT_R8G8_UNORM; break;
            default: *out_sampled_fmt = VK_FORMAT_R8G8B8A8_UNORM; break;
        }
        *out_unorm_fmt = *out_sampled_fmt;
    } else if ((VarType) format == VarType::Float32) {
        *out_storage_channels = channels_storage;
        switch (channels_storage) {
            case 1:  *out_sampled_fmt = VK_FORMAT_R32_SFLOAT; break;
            case 2:  *out_sampled_fmt = VK_FORMAT_R32G32_SFLOAT; break;
            default: *out_sampled_fmt = VK_FORMAT_R32G32B32A32_SFLOAT; break;
        }
        *out_unorm_fmt = *out_sampled_fmt;
    } else { // Float16
        *out_storage_channels = channels_storage;
        switch (channels_storage) {
            case 1:  *out_sampled_fmt = VK_FORMAT_R16_SFLOAT; break;
            case 2:  *out_sampled_fmt = VK_FORMAT_R16G16_SFLOAT; break;
            default: *out_sampled_fmt = VK_FORMAT_R16G16B16A16_SFLOAT; break;
        }
        *out_unorm_fmt = *out_sampled_fmt;
    }
}

struct VulkanTexReleasePayload {
    VulkanTexture *texture;
    size_t index;
};

static void vulkan_tex_install_release(VulkanTexture *tex, size_t i) {
    auto *payload = new VulkanTexReleasePayload({ tex, i });
    jitc_var_set_callback(
        tex->indices[i],
        [](uint32_t, int free, void *cb) {
            if (!free)
                return;
            auto *p = (VulkanTexReleasePayload *) cb;
            VulkanTexture *t = p->texture;
            size_t idx = p->index;

            if (!state.vulkan_devices.empty() &&
                (size_t) t->device_id < state.vulkan_devices.size()) {
                VulkanDevice &dev = state.vulkan_devices[t->device_id];
                VulkanTexResource &r = t->resources[idx];
                if (r.image || r.view || r.memory || r.slot != UINT32_MAX) {
                    jitc_vulkan_tex_free_later(dev, r.image, r.view,
                                               r.storage_view, r.memory,
                                               r.slot);
                    r.image = VK_NULL_HANDLE;
                    r.view = VK_NULL_HANDLE;
                    r.storage_view = VK_NULL_HANDLE;
                    r.memory = VK_NULL_HANDLE;
                    r.slot = UINT32_MAX;
                }
            }

            if (--t->n_referenced_textures == 0)
                delete t;
            delete p;
        },
        (void *) payload, false);
}

// ============================================================================
//  Creation / destruction
// ============================================================================

void *jitc_vulkan_tex_create(size_t ndim, const size_t *shape,
                             size_t n_channels, int format,
                             int filter_mode, int wrap_mode,
                             int writable, int srgb, size_t n_levels,
                             int mip_filter, size_t max_aniso,
                             int compression) {
    if (ndim < 1 || ndim > 3)
        jitc_raise("jit_vulkan_tex_create(): invalid texture dimension!");
    else if (n_channels == 0)
        jitc_raise("jit_vulkan_tex_create(): must have at least 1 channel!");
    for (size_t i = 0; i < ndim; ++i)
        if (shape[i] == 0)
            jitc_raise("jit_vulkan_tex_create(): texture dimensions must be "
                       "nonzero!");
    if (n_levels == 0)
        n_levels = 1;
    if (n_levels > 1 && writable)
        jitc_raise("jit_vulkan_tex_create(): MIP-mapped textures cannot be "
                   "writable!");

    size_t type_size;
    switch ((VarType) format) {
        case VarType::Float32: type_size = sizeof(float); break;
        case VarType::Float16: type_size = sizeof(uint16_t); break;
        case VarType::UInt8:   type_size = sizeof(uint8_t); break;
        default: jitc_raise("jit_vulkan_tex_create(): invalid data type!");
    }

    ThreadState *ts = thread_state(JitBackend::Vulkan);
    VulkanDevice &dev = state.vulkan_devices[ts->device];
    VkDevice device = (VkDevice) dev.device;

    // Validate sampler modes and obtain/create the shared bindless sampler slot
    uint32_t sampler_slot = jitc_vulkan_sampler_slot(
        dev, filter_mode, wrap_mode, mip_filter, max_aniso, n_levels > 1);

    VulkanTexture *tex = new VulkanTexture(
        ts->device, type_size, n_channels, writable != 0, format, srgb != 0);
    tex->ndim = ndim;
    tex->n_levels = n_levels;
    tex->compression = compression;
    for (size_t i = 0; i < ndim; ++i)
        tex->shape[i] = shape[i];

    // Promote 1D to 2D textures (with height = 1), matching Metal
    size_t width  = shape[0];
    size_t height = (ndim >= 2) ? shape[1] : 1;
    size_t depth  = (ndim == 3) ? shape[2] : 1;
    size_t total_bytes = 0;

    for (size_t i = 0; i < tex->n_textures; ++i) {
        size_t ci_req = tex->channels_storage(i);
        size_t ci = ci_req;
        VkFormat sampled_fmt = VK_FORMAT_UNDEFINED,
                 unorm_fmt   = VK_FORMAT_UNDEFINED;
        vulkan_tex_formats(dev, format, ci_req, srgb, compression,
                           &ci, &sampled_fmt, &unorm_fmt);

        bool mutable_srgb_write = (tex->writable && srgb != 0);

        VkImageCreateInfo img_info {};
        img_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        img_info.flags = mutable_srgb_write ? VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT
                                            : 0;
        img_info.imageType = (ndim == 3) ? VK_IMAGE_TYPE_3D : VK_IMAGE_TYPE_2D;
        img_info.format = mutable_srgb_write ? unorm_fmt : sampled_fmt;
        img_info.extent = { (uint32_t) width, (uint32_t) height,
                            (uint32_t) depth };
        img_info.mipLevels = (uint32_t) n_levels;
        img_info.arrayLayers = 1;
        img_info.samples = VK_SAMPLE_COUNT_1_BIT;
        img_info.tiling = VK_IMAGE_TILING_OPTIMAL;
        img_info.usage = VK_IMAGE_USAGE_SAMPLED_BIT |
                         VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                         VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                         (tex->writable ? VK_IMAGE_USAGE_STORAGE_BIT : 0);
        img_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        img_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

        VkImage image = VK_NULL_HANDLE;
        vk_tex_check(vkCreateImage(device, &img_info, nullptr, &image));

        VkMemoryRequirements mem_reqs {};
        vkGetImageMemoryRequirements(device, image, &mem_reqs);

        uint32_t mem_type_idx = UINT32_MAX;
        for (uint32_t m = 0; m < dev.device_mem_type_count; ++m) {
            uint32_t candidate = dev.device_mem_types[m];
            if (mem_reqs.memoryTypeBits & (1u << candidate)) {
                mem_type_idx = candidate;
                break;
            }
        }
        if (mem_type_idx == UINT32_MAX) {
            for (uint32_t m = 0; m < dev.shared_mem_type_count; ++m) {
                uint32_t candidate = dev.shared_mem_types[m];
                if (mem_reqs.memoryTypeBits & (1u << candidate)) {
                    mem_type_idx = candidate;
                    break;
                }
            }
        }
        if (mem_type_idx == UINT32_MAX) {
            vkDestroyImage(device, image, nullptr);
            jitc_raise("jit_vulkan_tex_create(): no compatible memory type for "
                       "texture image!");
        }

        VkMemoryAllocateInfo alloc_info {};
        alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        alloc_info.allocationSize = mem_reqs.size;
        alloc_info.memoryTypeIndex = mem_type_idx;

        VkDeviceMemory memory = VK_NULL_HANDLE;
        VkResult rv = vkAllocateMemory(device, &alloc_info, nullptr, &memory);
        if (rv != VK_SUCCESS) {
            vkDestroyImage(device, image, nullptr);
            vk_tex_check(rv);
        }

        rv = vkBindImageMemory(device, image, memory, 0);
        if (rv != VK_SUCCESS) {
            vkFreeMemory(device, memory, nullptr);
            vkDestroyImage(device, image, nullptr);
            vk_tex_check(rv);
        }

        VkImageViewUsageCreateInfo view_usage_info {};
        view_usage_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_CREATE_INFO;
        view_usage_info.usage = VK_IMAGE_USAGE_SAMPLED_BIT;

        VkImageViewCreateInfo view_info {};
        view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        view_info.pNext = mutable_srgb_write ? &view_usage_info : nullptr;
        view_info.image = image;
        view_info.viewType = (ndim == 3) ? VK_IMAGE_VIEW_TYPE_3D
                                         : VK_IMAGE_VIEW_TYPE_2D;
        view_info.format = sampled_fmt;
        view_info.components = {
            VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY,
            VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY
        };
        view_info.subresourceRange = {
            VK_IMAGE_ASPECT_COLOR_BIT, 0, (uint32_t) n_levels, 0, 1
        };

        VkImageView view = VK_NULL_HANDLE;
        rv = vkCreateImageView(device, &view_info, nullptr, &view);
        if (rv != VK_SUCCESS) {
            vkFreeMemory(device, memory, nullptr);
            vkDestroyImage(device, image, nullptr);
            vk_tex_check(rv);
        }

        VkImageView storage_view = VK_NULL_HANDLE;
        if (mutable_srgb_write) {
            view_info.pNext = nullptr;
            view_info.format = unorm_fmt;
            rv = vkCreateImageView(device, &view_info, nullptr, &storage_view);
            if (rv != VK_SUCCESS) {
                vkDestroyImageView(device, view, nullptr);
                vkFreeMemory(device, memory, nullptr);
                vkDestroyImage(device, image, nullptr);
                vk_tex_check(rv);
            }
        } else if (tex->writable) {
            storage_view = view;
        }

        uint32_t slot = jitc_vulkan_tex_slot_alloc(dev);

        VkDescriptorImageInfo sampled_desc {};
        sampled_desc.imageView = view;
        sampled_desc.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkDescriptorImageInfo storage_desc {};
        storage_desc.imageView = storage_view;
        storage_desc.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkWriteDescriptorSet writes[2] {};
        uint32_t n_writes = 0;

        writes[n_writes].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[n_writes].dstSet = (VkDescriptorSet) dev.descriptor_set;
        writes[n_writes].dstBinding = 0;
        writes[n_writes].dstArrayElement = slot;
        writes[n_writes].descriptorCount = 1;
        writes[n_writes].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        writes[n_writes].pImageInfo = &sampled_desc;
        n_writes++;

        if (tex->writable) {
            writes[n_writes].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[n_writes].dstSet = (VkDescriptorSet) dev.descriptor_set;
            writes[n_writes].dstBinding = 1;
            writes[n_writes].dstArrayElement = slot;
            writes[n_writes].descriptorCount = 1;
            writes[n_writes].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            writes[n_writes].pImageInfo = &storage_desc;
            n_writes++;
        }

        vkUpdateDescriptorSets(device, n_writes, writes, 0, nullptr);

        // Transition all MIP levels to VK_IMAGE_LAYOUT_GENERAL once at creation
        VkCommandBuffer cmd = jitc_vulkan_cmd(dev);
        VkImageMemoryBarrier imb {};
        imb.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        imb.srcAccessMask = 0;
        imb.dstAccessMask =
            VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
        imb.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        imb.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        imb.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        imb.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        imb.image = image;
        imb.subresourceRange = {
            VK_IMAGE_ASPECT_COLOR_BIT, 0, (uint32_t) n_levels, 0, 1
        };
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                             VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0,
                             0, nullptr, 0, nullptr, 1, &imb);
        dev.stream->write_pending = true;
        dev.stream->cmd_count++;

        total_bytes += (size_t) mem_reqs.size;

        VulkanTexResource &r = tex->resources[i];
        r.parent = tex;
        r.image = image;
        r.view = view;
        r.storage_view = storage_view;
        r.memory = memory;
        r.slot = slot;
        r.storage_channels = (uint32_t) ci;

        tex->indices[i] = jitc_var_mem_map(JitBackend::Vulkan, VarType::UInt64,
                                           &tex->resources[i], 1, 0);
        vulkan_tex_install_release(tex, i);
    }

    VulkanTexResource &smp_res = tex->resources[tex->n_textures];
    smp_res.parent = tex;
    smp_res.slot = sampler_slot;
    tex->sampler_index = jitc_var_mem_map(JitBackend::Vulkan, VarType::UInt64,
                                          &smp_res, 1, 0);

    tex->device_bytes = total_bytes;
    jitc_tex_track_usage(JitBackend::Vulkan, (int64_t) total_bytes, 1);

    jitc_log(LogLevel::Debug, "jitc_vulkan_tex_create(): " DRJIT_PTR,
             (uintptr_t) tex);
    return (void *) tex;
}

void *jitc_vulkan_tex_wrap(uintptr_t /*handle*/, size_t /*ndim*/,
                           int /*format*/, int /*writable*/,
                           int /*filter_mode*/, int /*wrap_mode*/,
                           int /*srgb*/) {
    jitc_raise("jit_tex_wrap(): wrapping external textures is not supported "
               "on the Vulkan backend.");
}

uintptr_t jitc_vulkan_tex_native_handle(const void *handle, size_t sub_index) {
    const VulkanTexture &tex = *((const VulkanTexture *) handle);
    if (sub_index >= tex.n_textures)
        jitc_raise("jit_tex_native_handle(): sub-texture index out of range.");
    return (uintptr_t) tex.resources[sub_index].image;
}

void jitc_vulkan_tex_get_shape(const void *handle, size_t *shape) {
    const VulkanTexture &tex = *((const VulkanTexture *) handle);
    for (size_t i = 0; i < tex.ndim; ++i)
        shape[i] = tex.shape[i];
    shape[tex.ndim] = tex.n_channels;
}

void jitc_vulkan_tex_get_indices(const void *handle, uint32_t *indices) {
    if (!handle)
        return;
    const VulkanTexture &tex = *((const VulkanTexture *) handle);
    for (size_t i = 0; i < tex.n_textures; ++i)
        indices[i] = tex.indices[i];
    indices[tex.n_textures] = tex.sampler_index;
}

void jitc_vulkan_tex_destroy(void *handle) {
    if (!handle)
        return;
    jitc_log(LogLevel::Debug, "jitc_vulkan_tex_destroy(" DRJIT_PTR ")",
             (uintptr_t) handle);

    VulkanTexture *tex = (VulkanTexture *) handle;
    const size_t n_textures = tex->n_textures;
    jitc_var_dec_ref(tex->sampler_index);
    for (size_t i = 0; i < n_textures; ++i)
        jitc_var_dec_ref(tex->indices[i]);
}

// ============================================================================
//  Device <-> texture memory transfer
// ============================================================================

static size_t vulkan_tex_n_texels(size_t ndim, const size_t *shape) {
    size_t n = shape[0];
    for (size_t i = 1; i < ndim; ++i)
        n *= shape[i];
    return n;
}

/// Host-side mirror of PushConstants for deinterleave/interleave in
/// resources/vulkan_kernels.glsl
struct VulkanChannelPackParams {
    uint64_t src_addr;
    uint64_t dst_addr;
    uint32_t size;
    uint32_t n_channels;
    uint32_t ci;
    uint32_t tex_base;
    uint32_t c_valid;
};

static void vulkan_channel_pack(VulkanDevice &dev, VulkanKernel kern,
                                const void *src, void *dst, uint32_t n_threads,
                                uint32_t n_channels, uint32_t ci,
                                uint32_t tex_base, uint32_t c_valid) {
    if (n_threads == 0)
        return;

    size_t src_off = 0, dst_off = 0;
    VulkanBuffer *src_buf = jitc_vulkan_find_buffer(src, &src_off);
    VulkanBuffer *dst_buf = jitc_vulkan_find_buffer(dst, &dst_off);
    if (!src_buf || !dst_buf)
        jitc_raise("vulkan_channel_pack(): buffer lookup failed.");

    VkPipeline pipe = jitc_vulkan_pipeline(dev, kern);
    if (!pipe)
        jitc_raise("vulkan_channel_pack(): utility pipeline unavailable.");

    VulkanChannelPackParams params;
    params.src_addr = src_buf->address + src_off;
    params.dst_addr = dst_buf->address + dst_off;
    params.size = n_threads;
    params.n_channels = n_channels;
    params.ci = ci;
    params.tex_base = tex_base;
    params.c_valid = c_valid;

    VkCommandBuffer cmd = jitc_vulkan_cmd(dev);
    jitc_vulkan_access(dev, true);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    vkCmdPushConstants(cmd, (VkPipelineLayout) dev.pipeline_layout,
                       VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(params), &params);

    uint32_t total_groups = ceil_div(n_threads, 256u);
    uint32_t max_x = dev.max_compute_work_group_count[0];
    uint32_t gx = total_groups, gy = 1;
    if (max_x > 0 && total_groups > max_x) {
        gx = max_x;
        gy = ceil_div(total_groups, gx);
    }

    vkCmdDispatch(cmd, gx, gy, 1);

    if (unlikely(jit_flag(JitFlag::LaunchBlocking)))
        jitc_vulkan_sync(false);
    else
        jitc_vulkan_maybe_submit(dev);
}

void jitc_vulkan_tex_memcpy_d2t(const void *src_ptr, void *dst_handle,
                                size_t level) {
    VulkanTexture &tex = *((VulkanTexture *) dst_handle);
    VulkanDevice &dev = state.vulkan_devices[tex.device_id];

    if (level >= tex.n_levels)
        jitc_raise("jit_tex_memcpy_d2t(): MIP level %zu out of range!", level);

    size_t ndim = tex.ndim;
    size_t shape[3];
    tex.level_shape(level, shape);
    size_t n_texels = vulkan_tex_n_texels(ndim, shape);
    size_t type_size = tex.type_size;
    size_t width  = shape[0];
    size_t height = (ndim >= 2) ? shape[1] : 1;
    size_t depth  = (ndim == 3) ? shape[2] : 1;

    if (tex.compression) {
        size_t src_off = 0;
        VulkanBuffer *src_buf = jitc_vulkan_find_buffer(src_ptr, &src_off);
        if (!src_buf)
            jitc_raise("jit_tex_memcpy_d2t(): source pointer is not a Vulkan "
                       "allocation!");

        VkCommandBuffer cmd = jitc_vulkan_cmd(dev);
        jitc_vulkan_access(dev, true);

        VkBufferImageCopy region {};
        region.bufferOffset = src_off;
        region.bufferRowLength = 0;
        region.bufferImageHeight = 0;
        region.imageSubresource = {
            VK_IMAGE_ASPECT_COLOR_BIT, (uint32_t) level, 0, 1
        };
        region.imageOffset = { 0, 0, 0 };
        region.imageExtent = { (uint32_t) width, (uint32_t) height, 1 };

        vkCmdCopyBufferToImage(cmd, src_buf->buffer, tex.resources[0].image,
                               VK_IMAGE_LAYOUT_GENERAL, 1, &region);
        jitc_vulkan_maybe_submit(dev);
        return;
    }

    bool needs_staging =
        (tex.n_textures > 1) ||
        (tex.resources[0].storage_channels != tex.n_channels);
    VulkanKernel deint = (type_size == 1)   ? VulkanKernel::DeinterleaveU8
                         : (type_size == 2) ? VulkanKernel::DeinterleaveU16
                                            : VulkanKernel::DeinterleaveU32;

    std::vector<void *> packed(tex.n_textures, nullptr);
    if (needs_staging) {
        for (size_t i = 0; i < tex.n_textures; ++i) {
            size_t ci = tex.resources[i].storage_channels;
            packed[i] = jitc_malloc(JitBackend::Vulkan,
                                    n_texels * ci * type_size);
            vulkan_channel_pack(dev, deint, src_ptr, packed[i],
                                (uint32_t) (n_texels * ci),
                                (uint32_t) tex.n_channels, (uint32_t) ci,
                                (uint32_t) (i * 4), (uint32_t) tex.channels(i));
        }
    }

    VkCommandBuffer cmd = jitc_vulkan_cmd(dev);
    jitc_vulkan_access(dev, true);

    for (size_t i = 0; i < tex.n_textures; ++i) {
        size_t src_off = 0;
        VulkanBuffer *src_buf = jitc_vulkan_find_buffer(
            needs_staging ? packed[i] : src_ptr, &src_off);
        if (!src_buf)
            jitc_raise("jit_tex_memcpy_d2t(): source pointer is not a Vulkan "
                       "allocation!");

        VkBufferImageCopy region {};
        region.bufferOffset = src_off;
        region.bufferRowLength = 0;
        region.bufferImageHeight = 0;
        region.imageSubresource = {
            VK_IMAGE_ASPECT_COLOR_BIT, (uint32_t) level, 0, 1
        };
        region.imageOffset = { 0, 0, 0 };
        region.imageExtent = {
            (uint32_t) width, (uint32_t) height, (uint32_t) depth
        };

        vkCmdCopyBufferToImage(cmd, src_buf->buffer, tex.resources[i].image,
                               VK_IMAGE_LAYOUT_GENERAL, 1, &region);
    }

    jitc_vulkan_maybe_submit(dev);

    for (void *p : packed)
        jitc_free(p);
}

void jitc_vulkan_tex_memcpy_t2d(const void *src_handle, void *dst_ptr) {
    VulkanTexture &tex = *((VulkanTexture *) src_handle);
    if (tex.compression)
        jitc_raise("jit_tex_memcpy_t2d(): block-compressed textures cannot be "
                   "read back into linear memory.");

    VulkanDevice &dev = state.vulkan_devices[tex.device_id];
    size_t ndim = tex.ndim;
    const size_t *shape = tex.shape;
    size_t n_texels = vulkan_tex_n_texels(ndim, shape);
    size_t type_size = tex.type_size;
    size_t width  = shape[0];
    size_t height = (ndim >= 2) ? shape[1] : 1;
    size_t depth  = (ndim == 3) ? shape[2] : 1;

    bool needs_staging =
        (tex.n_textures > 1) ||
        (tex.resources[0].storage_channels != tex.n_channels);

    if (!needs_staging) {
        size_t dst_off = 0;
        VulkanBuffer *dst_buf = jitc_vulkan_find_buffer(dst_ptr, &dst_off);
        if (!dst_buf)
            jitc_raise("jit_tex_memcpy_t2d(): destination pointer is not a "
                       "Vulkan allocation!");

        VkCommandBuffer cmd = jitc_vulkan_cmd(dev);
        jitc_vulkan_access(dev, true);

        VkBufferImageCopy region {};
        region.bufferOffset = dst_off;
        region.bufferRowLength = 0;
        region.bufferImageHeight = 0;
        region.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
        region.imageOffset = { 0, 0, 0 };
        region.imageExtent = {
            (uint32_t) width, (uint32_t) height, (uint32_t) depth
        };

        vkCmdCopyImageToBuffer(cmd, tex.resources[0].image,
                               VK_IMAGE_LAYOUT_GENERAL, dst_buf->buffer, 1,
                               &region);
        jitc_vulkan_maybe_submit(dev);
        return;
    }

    VulkanKernel inter = (type_size == 1)   ? VulkanKernel::InterleaveU8
                         : (type_size == 2) ? VulkanKernel::InterleaveU16
                                            : VulkanKernel::InterleaveU32;
    std::vector<void *> packed(tex.n_textures, nullptr);
    for (size_t i = 0; i < tex.n_textures; ++i) {
        size_t ci = tex.resources[i].storage_channels;
        packed[i] = jitc_malloc(JitBackend::Vulkan, n_texels * ci * type_size);
    }

    VkCommandBuffer cmd = jitc_vulkan_cmd(dev);
    jitc_vulkan_access(dev, true);

    for (size_t i = 0; i < tex.n_textures; ++i) {
        size_t off = 0;
        VulkanBuffer *packed_buf = jitc_vulkan_find_buffer(packed[i], &off);
        if (!packed_buf)
            jitc_raise("jit_tex_memcpy_t2d(): staging buffer lookup failed!");

        VkBufferImageCopy region {};
        region.bufferOffset = off;
        region.bufferRowLength = 0;
        region.bufferImageHeight = 0;
        region.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
        region.imageOffset = { 0, 0, 0 };
        region.imageExtent = {
            (uint32_t) width, (uint32_t) height, (uint32_t) depth
        };

        vkCmdCopyImageToBuffer(cmd, tex.resources[i].image,
                               VK_IMAGE_LAYOUT_GENERAL, packed_buf->buffer, 1,
                               &region);
    }

    for (size_t i = 0; i < tex.n_textures; ++i) {
        size_t ci = tex.resources[i].storage_channels, c = tex.channels(i);
        vulkan_channel_pack(dev, inter, packed[i], dst_ptr,
                            (uint32_t) (n_texels * c),
                            (uint32_t) tex.n_channels, (uint32_t) ci,
                            (uint32_t) (i * 4), (uint32_t) c);
    }

    for (void *p : packed)
        jitc_free(p);
}

// ============================================================================
//  Lookup / bilerp / write graph builders
// ============================================================================

static Variable jitc_vulkan_tex_check(VarType out_type, size_t ndim,
                                      const uint32_t *pos, size_t n_extra = 0,
                                      const uint32_t *extra = nullptr) {
    uint32_t size = 0;
    bool dirty = false, symbolic = false;

    if (ndim < 1 || ndim > 3)
        jitc_raise("jit_vulkan_tex_check(): invalid texture dimension!");

    uint32_t idx[9];
    size_t n = 0;
    for (size_t i = 0; i < ndim; ++i)
        idx[n++] = pos[i];
    for (size_t i = 0; i < n_extra; ++i)
        idx[n++] = extra[i];

    for (size_t i = 0; i < n; ++i) {
        const Variable *v = jitc_var(idx[i]);
        if ((VarType) v->type != VarType::Float32)
            jitc_raise("jit_vulkan_tex_check(): type mismatch for arg. %zu "
                       "(got %s, expected %s)", i, type_name[v->type],
                       type_name[(int) VarType::Float32]);
        size = std::max(size, v->size);
        dirty |= v->is_dirty();
        symbolic |= (bool) v->symbolic;
    }

    for (size_t i = 0; i < n; ++i) {
        const Variable *v = jitc_var(idx[i]);
        if (v->size != 1 && v->size != size)
            jitc_raise("jit_vulkan_tex_check(): arithmetic involving arrays of "
                       "incompatible size!");
    }

    if (dirty) {
        jitc_eval(thread_state(JitBackend::Vulkan));
        for (size_t i = 0; i < n; ++i) {
            if (jitc_var(idx[i])->is_dirty())
                jitc_raise_dirty_error(idx[i]);
        }
    }

    Variable v;
    v.size = size;
    v.backend = (uint32_t) JitBackend::Vulkan;
    v.symbolic = symbolic;
    v.type = (uint32_t) out_type;
    return v;
}

static TexData *vulkan_tex_coord_data(size_t ndim, const uint32_t *pos) {
    TexData *td = new TexData();
    td->ndim = (uint32_t) ndim;
    for (size_t i = 0; i < ndim; ++i) {
        td->indices[i] = pos[i];
        jitc_var_inc_ref(pos[i]);
    }
    return td;
}

static void vulkan_tex_set_free_callback(uint32_t index, TexData *td) {
    jitc_var_set_callback(
        index,
        [](uint32_t, int free, void *ptr) {
            if (free)
                delete (TexData *) ptr;
        },
        td, true);
}

static uint32_t vulkan_tex_node(VarKind kind, const Variable &tmpl,
                                uint32_t tex_h, uint32_t smp_h,
                                uint32_t active, bool masked, TexData *td) {
    uint32_t node = masked
        ? jitc_var_new_node_3(JitBackend::Vulkan, kind, VarType::Float32,
              tmpl.size, tmpl.symbolic, tex_h, jitc_var(tex_h), smp_h,
              jitc_var(smp_h), active, jitc_var(active), (uintptr_t) td)
        : jitc_var_new_node_2(JitBackend::Vulkan, kind, VarType::Float32,
              tmpl.size, tmpl.symbolic, tex_h, jitc_var(tex_h), smp_h,
              jitc_var(smp_h), (uintptr_t) td);
    vulkan_tex_set_free_callback(node, td);
    return node;
}

static void vulkan_tex_sample_op(VarKind kind, const void *handle,
                                 const uint32_t *pos, size_t n_extra,
                                 const uint32_t *extra, uint32_t active,
                                 uint32_t *out) {
    VulkanTexture &tex = *((VulkanTexture *) handle);
    size_t ndim = tex.ndim;
    if (kind != VarKind::TexLookup && tex.n_levels <= 1)
        jitc_raise("jit_tex_lookup_lod/grad(): texture was created without "
                   "MIP levels!");
    Variable tmpl =
        jitc_vulkan_tex_check(VarType::Float32, ndim, pos, n_extra, extra);

    const Variable *active_v = jitc_var(active);
    bool masked = !(active_v->is_literal() && active_v->literal == 1);

    for (size_t ti = 0; ti < tex.n_textures; ++ti) {
        Ref tex_h = steal(jitc_var_resource_pointer(tex.indices[ti],
                                                    ResourceKind::Texture));
        Ref smp_h = steal(jitc_var_resource_pointer(tex.sampler_index,
                                                    ResourceKind::Sampler));
        TexData *td = vulkan_tex_coord_data(ndim, pos);
        td->n_values = (uint32_t) n_extra;
        for (size_t i = 0; i < n_extra; ++i) {
            td->values[i] = extra[i];
            jitc_var_inc_ref(extra[i]);
        }

        Ref tex_load = steal(vulkan_tex_node(kind, tmpl, tex_h, smp_h, active,
                                             masked, td));

        for (size_t ch = 0; ch < tex.channels(ti); ++ch)
            *out++ = jitc_var_new_node_1(
                JitBackend::Vulkan, VarKind::Extract, VarType::Float32,
                tmpl.size, tmpl.symbolic, tex_load, jitc_var(tex_load),
                (uint64_t) ch);
    }
}

void jitc_vulkan_tex_lookup(const void *handle, const uint32_t *pos,
                            uint32_t active, uint32_t *out) {
    vulkan_tex_sample_op(VarKind::TexLookup, handle, pos, 0, nullptr, active,
                         out);
}

void jitc_vulkan_tex_lookup_lod(const void *handle, const uint32_t *pos,
                                uint32_t lod, uint32_t active, uint32_t *out) {
    vulkan_tex_sample_op(VarKind::TexLookupLod, handle, pos, 1, &lod, active,
                         out);
}

void jitc_vulkan_tex_lookup_grad(const void *handle, const uint32_t *pos,
                                 const uint32_t *ddx, const uint32_t *ddy,
                                 uint32_t active, uint32_t *out) {
    size_t ndim = ((VulkanTexture *) handle)->ndim;
    uint32_t extra[6];
    for (size_t i = 0; i < ndim; ++i) {
        extra[i] = ddx[i];
        extra[ndim + i] = ddy[i];
    }
    vulkan_tex_sample_op(VarKind::TexLookupGrad, handle, pos, 2 * ndim, extra,
                         active, out);
}

void jitc_vulkan_tex_write(void *handle, const uint32_t *pos,
                           const uint32_t *value, uint32_t active) {
    VulkanTexture &tex = *((VulkanTexture *) handle);
    if (!tex.writable)
        jitc_raise("jit_tex_write(): texture was not created with the "
                   "'writable' flag.");

    size_t ndim = tex.ndim, n_channels = tex.n_channels;

    uint32_t size = 1;
    bool dirty = false, symbolic = false;
    auto check = [&](uint32_t idx, VarType expect, const char *what,
                     size_t which) {
        const Variable *v = jitc_var(idx);
        if (expect != VarType::Void && (VarType) v->type != expect)
            jitc_raise("jit_tex_write(): type mismatch for %s %zu (got %s, "
                       "expected %s).", what, which, type_name[v->type],
                       type_name[(int) expect]);
        if (v->size != 1) {
            if (size != 1 && v->size != size)
                jitc_raise("jit_tex_write(): incompatible argument sizes "
                           "(%u vs %u).", size, v->size);
            size = v->size;
        }
        dirty |= v->is_dirty();
        symbolic |= (bool) v->symbolic;
    };
    for (size_t i = 0; i < ndim; ++i)
        check(pos[i], VarType::UInt32, "coordinate", i);
    for (size_t c = 0; c < n_channels; ++c)
        check(value[c], VarType::Float32, "value", c);
    check(active, VarType::Void, "mask", 0);

    if (dirty)
        jitc_eval(thread_state(JitBackend::Vulkan));

    const Variable *active_v = jitc_var(active);
    bool masked = !(active_v->is_literal() && active_v->literal == 1);

    for (size_t ti = 0; ti < tex.n_textures; ++ti) {
        size_t base = ti * 4, nc = tex.channels(ti);
        Ref tex_h = steal(jitc_var_resource_pointer(tex.indices[ti],
                                                    ResourceKind::Texture,
                                                    /*write=*/1));

        TexData *td = vulkan_tex_coord_data(ndim, pos);
        td->n_values = (uint32_t) nc;
        td->comp_bytes = (uint32_t) tex.type_size |
                         ((uint32_t) tex.resources[ti].storage_channels << 8) |
                         (tex.srgb ? 0x10000u : 0u);
        for (size_t c = 0; c < nc; ++c) {
            td->values[c] = value[base + c];
            jitc_var_inc_ref(value[base + c]);
        }

        uint32_t node =
            masked ? jitc_var_new_node_2(
                         JitBackend::Vulkan, VarKind::TexWrite, VarType::Void,
                         size, symbolic, tex_h, jitc_var(tex_h), active,
                         jitc_var(active), (uintptr_t) td)
                   : jitc_var_new_node_1(
                         JitBackend::Vulkan, VarKind::TexWrite, VarType::Void,
                         size, symbolic, tex_h, jitc_var(tex_h),
                         (uintptr_t) td);

        vulkan_tex_set_free_callback(node, td);
        jitc_var_mark_side_effect(node);
    }
}

void jitc_vulkan_tex_bilerp_fetch(const void *handle, const uint32_t *pos,
                                  uint32_t active, uint32_t *out) {
    VulkanTexture &tex = *((VulkanTexture *) handle);
    if (tex.ndim != 2)
        jitc_raise("jitc_vulkan_tex_bilerp_fetch(): only 2D textures are "
                   "supported!");
    size_t ndim = tex.ndim;
    Variable tmpl = jitc_vulkan_tex_check(VarType::Float32, ndim, pos);

    const Variable *active_v = jitc_var(active);
    bool masked = !(active_v->is_literal() && active_v->literal == 1);

    for (size_t ti = 0; ti < tex.n_textures; ++ti) {
        Ref tex_h = steal(jitc_var_resource_pointer(tex.indices[ti],
                                                    ResourceKind::Texture));
        Ref smp_h = steal(jitc_var_resource_pointer(tex.sampler_index,
                                                    ResourceKind::Sampler));

        for (size_t ch = 0; ch < tex.channels(ti); ++ch) {
            TexData *td = vulkan_tex_coord_data(ndim, pos);
            td->component = (uint32_t) ch;

            Ref tex_load = steal(vulkan_tex_node(VarKind::TexFetchBilerp, tmpl,
                                                 tex_h, smp_h, active, masked,
                                                 td));

            for (uint32_t j = 0; j < 4; ++j)
                *out++ = jitc_var_new_node_1(
                    JitBackend::Vulkan, VarKind::Extract, VarType::Float32,
                    tmpl.size, tmpl.symbolic, tex_load, jitc_var(tex_load),
                    (uint64_t) j);
        }
    }
}

#endif // defined(DRJIT_ENABLE_VULKAN)
