/*
    src/vulkan_tex.h -- hardware texture support for the Vulkan backend

    Copyright (c) 2026 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#include "common.h"
#include "tex.h"

#if defined(DRJIT_ENABLE_VULKAN)

#include "vulkan_api.h"
#include <cstdint>
#include <cstddef>
#include <atomic>
#include <memory>
#include <vector>

struct VulkanTexture;

/// Opaque sub-resource of a Vulkan texture: one of the 1/2/4-channel
/// sub-textures or the shared sampler.
struct VulkanTexResource {
    VulkanTexture *parent = nullptr;
    VkImage image = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    VkImageView storage_view = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    uint32_t slot = UINT32_MAX;
    uint32_t storage_channels = 0;
};

/// Host-side state for one Vulkan hardware texture
struct VulkanTexture : TextureBase {
    int device_id = 0;
    int format = 0;
    bool srgb = false;

    /// Pointer backing variable per sub-texture
    std::vector<uint32_t> indices;
    uint32_t sampler_index = 0;

    /// One resource record per sub-texture, plus a trailing sampler record
    std::unique_ptr<VulkanTexResource[]> resources;

    VulkanTexture(int device_id, size_t type_size, size_t n_channels,
                  bool writable, int format, bool srgb)
        : TextureBase(JitBackend::Vulkan, type_size, n_channels, writable),
          device_id(device_id), format(format), srgb(srgb),
          indices(n_textures, 0),
          resources(std::make_unique<VulkanTexResource[]>(n_textures + 1)) { }
};

// ---------------------------------------------------------------------
//  Public texture API
// ---------------------------------------------------------------------

extern void *jitc_vulkan_tex_create(size_t ndim, const size_t *shape,
                                    size_t n_channels, int format,
                                    int filter_mode, int wrap_mode,
                                    int writable, int srgb, size_t n_levels,
                                    int mip_filter, size_t max_aniso,
                                    int compression);
extern void *jitc_vulkan_tex_wrap(uintptr_t handle, size_t ndim, int format,
                                  int writable, int filter_mode, int wrap_mode,
                                  int srgb);
extern uintptr_t jitc_vulkan_tex_native_handle(const void *handle,
                                               size_t sub_index);
extern void jitc_vulkan_tex_get_shape(const void *handle, size_t *shape);
extern void jitc_vulkan_tex_get_indices(const void *handle,
                                        uint32_t *indices);
extern void jitc_vulkan_tex_memcpy_d2t(const void *src_ptr,
                                       void *dst_handle, size_t level);
extern void jitc_vulkan_tex_memcpy_t2d(const void *src_handle,
                                       void *dst_ptr);
extern void jitc_vulkan_tex_lookup(const void *handle,
                                   const uint32_t *pos, uint32_t active,
                                   uint32_t *out);
extern void jitc_vulkan_tex_lookup_lod(const void *handle,
                                       const uint32_t *pos, uint32_t lod,
                                       uint32_t active, uint32_t *out);
extern void jitc_vulkan_tex_lookup_grad(const void *handle,
                                        const uint32_t *pos,
                                        const uint32_t *ddx,
                                        const uint32_t *ddy,
                                        uint32_t active, uint32_t *out);
extern void jitc_vulkan_tex_write(void *handle, const uint32_t *pos,
                                  const uint32_t *value, uint32_t active);
extern void jitc_vulkan_tex_bilerp_fetch(const void *handle,
                                         const uint32_t *pos, uint32_t active,
                                         uint32_t *out);
extern void jitc_vulkan_tex_destroy(void *handle);

#endif // defined(DRJIT_ENABLE_VULKAN)
