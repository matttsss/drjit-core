/*
    src/vulkan_ts.cpp -- ThreadState specialization for the Vulkan backend

    Copyright (c) 2026 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#include "vulkan_ts.h"
#include "vulkan.h"
#include "log.h"
#include <algorithm>
#include <cstring>
#include <memory>

/// Maximum size of a vkCmdUpdateBuffer() command
static constexpr size_t VulkanMaxUpdateSize = 65536;

/// Return the device associated with a thread state
static VulkanDevice &vulkan_device(const ThreadState *ts) {
    return state.vulkan_devices[ts->device];
}

[[noreturn]] static void vulkan_unsupported(const char *func) {
    jitc_raise("%s(): this operation is not supported by the Vulkan backend "
               "yet.", func);
}

/**
 * \brief Look up the Vulkan allocation containing the memory range
 * <tt>[ptr, ptr + size)</tt>
 *
 * Returns \c nullptr if \c ptr is not part of a Vulkan allocation, and raises
 * an exception if the range exceeds the allocation.
 */
static VulkanBuffer *vulkan_lookup(const void *ptr, size_t size,
                                   size_t *offset, const char *func) {
    VulkanBuffer *buf = jitc_vulkan_find_buffer(ptr, offset);
    if (buf && size > buf->size - *offset)
        jitc_raise("%s(): the memory range [" DRJIT_PTR ", " DRJIT_PTR
                   ") exceeds the bounds of the Vulkan allocation at " DRJIT_PTR
                   " (%zu bytes)!",
                   func, (uintptr_t) ptr, (uintptr_t) ptr + size, buf->base,
                   buf->size);
    return buf;
}

/**
 * \brief Fill the memory range <tt>[ptr, ptr + size)</tt> with a pattern of
 * \c isize bytes (1, 2, 4, or 8)
 *
 * This uses transfer commands only: \c vkCmdFillBuffer() writes the 4-byte
 * aligned part of the range, and unaligned bytes at its boundaries are copied
 * from staging memory. Patterns with a period of 8 bytes are written via \c
 * vkCmdUpdateBuffer() or by replicating a staged copy of the pattern.
 */
static void vulkan_fill(VulkanDevice &dev, void *ptr, size_t size,
                        uint32_t isize, const void *src, const char *func) {
    uint8_t pattern[8];
    memcpy(pattern, src, isize);

    uint32_t period = isize;
    if (period == 8 && memcmp(pattern, pattern + 4, 4) == 0)
        period = 4;

    size_t offset = 0;
    if (!vulkan_lookup(ptr, size, &offset, func))
        jitc_raise("%s(): " DRJIT_PTR " is not a Vulkan allocation!", func,
                   (uintptr_t) ptr);

    uint8_t *staging = nullptr;
    void *staging_alloc = nullptr;
    size_t staging_size = 0;

    if (period <= 4) {
        // 4-byte aligned part ("body") of the range
        size_t body_start = (offset + 3) / 4 * 4,
               body_end = (offset + size) / 4 * 4,
               head, tail;

        if (body_start < body_end) {
            head = body_start - offset;
            tail = offset + size - body_end;
        } else {
            head = size;
            tail = 0;
            body_start = body_end = offset + size;
        }

        // Stage the unaligned bytes
        if (head + tail > 0) {
            staging = jitc_vulkan_stage(dev, head + tail);
            for (size_t i = 0; i < head; ++i)
                staging[i] = pattern[i % period];
            for (size_t i = 0; i < tail; ++i)
                staging[head + i] = pattern[(body_end - offset + i) % period];
        }

        // Look up the buffers after jitc_vulkan_stage(), which may release
        // the lock
        VulkanBuffer *buf = jitc_vulkan_find_buffer(ptr),
                     *staging_buf = nullptr;
        size_t staging_offset = 0;
        if (staging)
            staging_buf = jitc_vulkan_find_buffer(staging, &staging_offset);

        VkCommandBuffer cmd = jitc_vulkan_cmd(dev);
        jitc_vulkan_access(dev, true);

        if (body_start < body_end) {
            // The 4-byte word covering an aligned position of the body
            uint8_t word_bytes[4];
            for (size_t i = 0; i < 4; ++i)
                word_bytes[i] = pattern[(body_start - offset + i) % period];

            // (vkCmdFillBuffer() writes 'word' using the host endianness)
            uint32_t word;
            memcpy(&word, word_bytes, 4);
            vkCmdFillBuffer(cmd, buf->buffer, body_start,
                            body_end - body_start, word);
        }

        if (staging) {
            VkBufferCopy regions[2];
            uint32_t n_regions = 0;
            if (head)
                regions[n_regions++] = { staging_offset, offset, head };
            if (tail)
                regions[n_regions++] = { staging_offset + head, body_end,
                                         tail };
            vkCmdCopyBuffer(cmd, staging_buf->buffer, buf->buffer, n_regions,
                            regions);
        }
    } else if (offset % 4 == 0 && size % 4 == 0 &&
               size <= VulkanMaxUpdateSize) {
        // Non-repeating 8-byte pattern, small aligned range: embed the data
        // into the command buffer
        std::unique_ptr<uint8_t[]> data(new uint8_t[size]);
        for (size_t i = 0; i < size; ++i)
            data[i] = pattern[i % 8];

        VulkanBuffer *buf = jitc_vulkan_find_buffer(ptr);
        VkCommandBuffer cmd = jitc_vulkan_cmd(dev);
        jitc_vulkan_access(dev, true);
        vkCmdUpdateBuffer(cmd, buf->buffer, offset, size, data.get());
    } else {
        /* Non-repeating 8-byte pattern: stage a chunk containing multiple
           periods of the pattern and replicate it using a copy command with
           up to 4096 regions */
        size_t chunk = (size + 7) / 8 * 8,
               chunk_min = std::max((size_t) 65536,
                                    ((size + 4095) / 4096 + 7) / 8 * 8);
        chunk = std::min(chunk, chunk_min);

        if (chunk <= VulkanMaxStageSize) {
            staging = jitc_vulkan_stage(dev, chunk);
        } else {
            staging_alloc = jitc_malloc(JitBackend::Vulkan, chunk, true);
            staging = (uint8_t *) staging_alloc;
            staging_size = chunk;
        }

        for (size_t i = 0; i < chunk; ++i)
            staging[i] = pattern[i % 8];

        VulkanBuffer *buf = jitc_vulkan_find_buffer(ptr);
        size_t staging_offset = 0;
        VulkanBuffer *staging_buf =
            jitc_vulkan_find_buffer(staging, &staging_offset);

        size_t n_regions = (size + chunk - 1) / chunk;
        std::unique_ptr<VkBufferCopy[]> regions(new VkBufferCopy[n_regions]);
        for (size_t i = 0; i < n_regions; ++i) {
            size_t start = i * chunk;
            regions[i] = { staging_offset, offset + start,
                           std::min(chunk, size - start) };
        }

        VkCommandBuffer cmd = jitc_vulkan_cmd(dev);
        jitc_vulkan_access(dev, true);
        vkCmdCopyBuffer(cmd, staging_buf->buffer, buf->buffer,
                        (uint32_t) n_regions, regions.get());
    }

    if (staging_alloc) {
        // Released once the command buffer has finished
        jitc_free(staging_alloc);
        dev.stream->cmd_staging += staging_size;
    }

    jitc_vulkan_maybe_submit(dev);
}

/// Record a copy of \c size bytes from host memory to a Vulkan allocation.
/// The source data is captured before the function returns.
static void vulkan_upload(VulkanDevice &dev, void *dst, const void *src,
                          size_t size) {
    size_t dst_offset = 0;
    jitc_vulkan_find_buffer(dst, &dst_offset);

    if (dst_offset % 4 == 0 && size % 4 == 0 && size <= VulkanMaxUpdateSize) {
        // Embed the data into the command buffer
        VulkanBuffer *dst_buf = jitc_vulkan_find_buffer(dst);
        VkCommandBuffer cmd = jitc_vulkan_cmd(dev);
        jitc_vulkan_access(dev, true);
        vkCmdUpdateBuffer(cmd, dst_buf->buffer, dst_offset, size, src);
    } else {
        void *staging_alloc = nullptr;
        uint8_t *staging;

        if (size <= VulkanMaxStageSize) {
            staging = jitc_vulkan_stage(dev, size);
            memcpy(staging, src, size);
        } else {
            staging_alloc = jitc_malloc(JitBackend::Vulkan, size, true);
            staging = (uint8_t *) staging_alloc;

            // Nobody else references the staging buffer yet
            unlock_guard guard(state.lock);
            memcpy(staging, src, size);
        }

        // Look up the buffers after allocating staging memory, which may
        // release the lock
        size_t staging_offset = 0;
        VulkanBuffer *dst_buf = jitc_vulkan_find_buffer(dst),
                     *staging_buf =
                         jitc_vulkan_find_buffer(staging, &staging_offset);

        VkCommandBuffer cmd = jitc_vulkan_cmd(dev);
        jitc_vulkan_access(dev, true);
        VkBufferCopy region { staging_offset, dst_offset, size };
        vkCmdCopyBuffer(cmd, staging_buf->buffer, dst_buf->buffer, 1, &region);

        if (staging_alloc) {
            // Released once the command buffer has finished
            jitc_free(staging_alloc);
            dev.stream->cmd_staging += size;
        }
    }

    jitc_vulkan_maybe_submit(dev);
}

/// Record a copy of \c size bytes between two Vulkan allocations
static void vulkan_copy_buffer(VulkanDevice &dev, void *dst, const void *src,
                               size_t size, const char *func) {
    size_t dst_offset = 0, src_offset = 0;
    VulkanBuffer *dst_buf = jitc_vulkan_find_buffer(dst, &dst_offset),
                 *src_buf = jitc_vulkan_find_buffer(src, &src_offset);

    uintptr_t src_addr = (uintptr_t) src,
              dst_addr = (uintptr_t) dst;

    if (dst_buf == src_buf && src_addr < dst_addr + size && dst_addr < src_addr + size)
        jitc_raise("%s(): the source and destination ranges overlap, which is "
                   "not supported by the Vulkan backend!", func);

    VkCommandBuffer cmd = jitc_vulkan_cmd(dev);
    jitc_vulkan_access(dev, true);
    VkBufferCopy region { src_offset, dst_offset, size };
    vkCmdCopyBuffer(cmd, src_buf->buffer, dst_buf->buffer, 1, &region);

    jitc_vulkan_maybe_submit(dev);
}

/**
 * \brief Copy \c size bytes between arbitrary combinations of host memory and
 * Vulkan allocations
 *
 * Copies to Vulkan allocations are recorded into the command stream, and
 * copies to host memory that is not part of a Vulkan allocation block until
 * all previously issued work has finished. When \c async is \c false, the
 * function waits until the copy has finished.
 */
static void vulkan_copy(VulkanDevice &dev, void *dst, const void *src,
                        size_t size, bool async) {
    const char *func = async ? "jit_memcpy_async" : "jit_memcpy";
    if (size == 0)
        return;

    size_t dst_offset = 0, src_offset = 0;
    VulkanBuffer *dst_buf = vulkan_lookup(dst, size, &dst_offset, func),
                 *src_buf = vulkan_lookup(src, size, &src_offset, func);

    // Host memory only
    if (!dst_buf && !src_buf) {
        memcpy(dst, src, size);
        return;
    }

    bool dst_device = dst_buf && !dst_buf->shared,
         src_device = src_buf && !src_buf->shared;

    // Synchronous copy between host memory and/or shared allocations
    if (!async && !dst_device && !src_device) {
        jitc_vulkan_sync(false);
        memcpy(dst, src, size);
        return;
    }

    if (!dst_buf) {
        // Copy to host memory: must wait for the GPU
        if (src_device) {
            // Stage device memory through a shared allocation
            void *staging = jitc_malloc(JitBackend::Vulkan, size, true);
            vulkan_copy_buffer(dev, staging, src, size, func);
            jitc_vulkan_sync(false);
            memcpy(dst, staging, size);
            jitc_free(staging);
        } else {
            jitc_vulkan_sync(false);
            memcpy(dst, src, size);
        }
        return;
    }

    if (src_buf)
        vulkan_copy_buffer(dev, dst, src, size, func);
    else
        vulkan_upload(dev, dst, src, size);

    if (!async)
        jitc_vulkan_sync(false);
}

void VulkanThreadState::barrier() {
    // Submit the work to overlap its execution with subsequent tracing
    jitc_vulkan_flush();
}

Task *VulkanThreadState::launch(Kernel, KernelKey &, XXH128_hash_t, uint32_t,
                                std::vector<void *> &,
                                const std::vector<uint32_t> &,
                                KernelHistoryEntry *) {
    vulkan_unsupported("jit_launch");
}

void VulkanThreadState::memset_async(void *ptr, uint32_t size, uint32_t isize,
                                     const void *src) {
    if (isize != 1 && isize != 2 && isize != 4 && isize != 8)
        jitc_raise("jit_memset_async(): invalid element size (must be 1, 2, "
                   "4, or 8)!");

    jitc_trace("jit_memset_async(" DRJIT_PTR ", isize=%u, size=%u)",
               (uintptr_t) ptr, isize, size);

    if (size == 0)
        return;

    vulkan_fill(vulkan_device(this), ptr, (size_t) size * isize, isize, src,
                "jit_memset_async");
}

void VulkanThreadState::block_reduce(VarType, ReduceOp, uint32_t, uint32_t,
                                     const void *, void *) {
    vulkan_unsupported("jit_block_reduce");
}

void VulkanThreadState::block_prefix_reduce(VarType, ReduceOp, uint32_t,
                                            uint32_t, bool, bool, const void *,
                                            void *) {
    vulkan_unsupported("jit_block_prefix_reduce");
}

void VulkanThreadState::reduce_dot(VarType, const void *, const void *,
                                   uint32_t, void *) {
    vulkan_unsupported("jit_reduce_dot");
}

void VulkanThreadState::batched_gemm(VarType, bool, bool, uint32_t, uint32_t,
                                     uint32_t, const GemmBatch *, const void *,
                                     const void *, void *) {
    vulkan_unsupported("jit_batched_gemm");
}

uint32_t VulkanThreadState::compress(const uint8_t *, uint32_t, uint32_t *) {
    vulkan_unsupported("jit_compress");
}

uint32_t VulkanThreadState::block_mkperm(const uint32_t *, uint32_t, uint32_t,
                                         uint32_t, uint32_t *, uint32_t *) {
    vulkan_unsupported("jit_mkperm");
}

void VulkanThreadState::memcpy(void *dst, const void *src, size_t size) {
    vulkan_copy(vulkan_device(this), dst, src, size, false);
}

void VulkanThreadState::memcpy_async(void *dst, const void *src, size_t size) {
    vulkan_copy(vulkan_device(this), dst, src, size, true);
}

void VulkanThreadState::poke(void *dst, const void *src, uint32_t size) {
    jitc_log(Debug, "jit_poke(" DRJIT_PTR ", size=%u)", (uintptr_t) dst, size);

    if (size != 1 && size != 2 && size != 4 && size != 8)
        jitc_raise("jit_poke(): only size=1, 2, 4 or 8 are supported!");

    vulkan_fill(vulkan_device(this), dst, size, size, src, "jit_poke");
}

void VulkanThreadState::aggregate(void *, AggregationEntry *, uint32_t) {
    vulkan_unsupported("jit_aggregate");
}

void VulkanThreadState::enqueue_host_func(void (*callback)(void *),
                                          void *payload) {
    jitc_vulkan_enqueue_host_func(vulkan_device(this), callback, payload);
}

void VulkanThreadState::coop_vec_pack(uint32_t, const void *,
                                      const MatrixDescr *, void *,
                                      const MatrixDescr *) {
    vulkan_unsupported("jit_coop_vec_pack");
}
