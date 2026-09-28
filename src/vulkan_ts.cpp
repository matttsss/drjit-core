/*
    src/vulkan_ts.cpp -- ThreadState specialization for the Vulkan backend

    Copyright (c) 2026 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#include "vulkan_ts.h"
#include "vulkan.h"
#include "vulkan_tex.h"
#include "history.h"
#include "var.h"
#include "util.h"
#include "log.h"
#include <algorithm>
#include <cstring>
#include <memory>
#include <tuple>

/// Maximum size of a vkCmdUpdateBuffer() command
static constexpr size_t VulkanMaxUpdateSize = 65536;

/// Workgroup size of precompiled utility kernels in vulkan_kernels.glsl
static constexpr uint32_t VulkanWorkgroupSize = 256;

/// Contiguous elements each thread of a chunked reduction/scan processes
static constexpr uint32_t VulkanGrain = 4;

/// Elements covered by one forward scan workgroup (4 tiles * 256 threads * 4 elements)
static constexpr uint32_t VulkanScanChunkSize = 4096;

/// Elements covered by one reduction/dot workgroup (4 iterations * 1024)
static constexpr uint32_t VulkanChunkSize = 4096;

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
                        uint32_t isize, const void *src, const char *func,
                        VulkanThreadState *ts = nullptr,
                        KernelType hist_type = KernelType::Memset,
                        uint32_t hist_width = 0) {
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
    bool record_hist = ts && (jit_flags() & (uint32_t) JitFlag::KernelHistory);
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    void *qp = nullptr;
    uint32_t qi = 0;

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

        cmd = jitc_vulkan_cmd(dev);
        jitc_vulkan_access(dev, true);
        if (unlikely(record_hist))
            std::tie(qp, qi) = jitc_vulkan_history_begin(dev, cmd);

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
        cmd = jitc_vulkan_cmd(dev);
        jitc_vulkan_access(dev, true);
        if (unlikely(record_hist))
            std::tie(qp, qi) = jitc_vulkan_history_begin(dev, cmd);
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

        cmd = jitc_vulkan_cmd(dev);
        jitc_vulkan_access(dev, true);
        if (unlikely(record_hist))
            std::tie(qp, qi) = jitc_vulkan_history_begin(dev, cmd);
        vkCmdCopyBuffer(cmd, staging_buf->buffer, buf->buffer,
                        (uint32_t) n_regions, regions.get());
    }

    if (unlikely(record_hist)) {
        KernelHistoryEntry meta = {};
        meta.backend = JitBackend::Vulkan;
        meta.type = hist_type;
        meta.recording_mode = ts->recording_mode;
        meta.size = hist_width;
        meta.input_count = 1;
        meta.output_count = 1;
        KernelHistoryEntryImpl *e = jitc_kernel_history_append(meta);
        jitc_vulkan_history_end(dev, cmd, qp, qi, e);
    }

    if (staging_alloc) {
        // Released once the command buffer has finished
        jitc_free(staging_alloc);
        dev.stream->cmd_staging += staging_size;
    }

    if (unlikely(jit_flag(JitFlag::LaunchBlocking)))
        jitc_vulkan_sync(false);
    else
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

Task *VulkanThreadState::launch(Kernel kernel, KernelKey & /*key*/,
                                XXH128_hash_t /*hash*/, uint32_t size,
                                std::vector<void *> &kernel_params,
                                const std::vector<uint32_t> & /*kernel_param_ids*/,
                                KernelHistoryEntry *kernel_history_entry) {
    if (size == 0)
        return nullptr;

    VulkanDevice &dev = vulkan_device(this);
    size_t n_params = kernel_params.size();
    const KernelParamInfo *info = kernel.param_info;
    if (info) {
        for (size_t i = 1; i < n_params; ++i) {
            void *owner = kernel_params[i];
            if (!owner)
                continue;
            switch ((ResourceKind) info[i].kind) {
                case ResourceKind::Accel:
                    kernel_params[i] =
                        (void *) (uintptr_t) ((VulkanScene *) owner)->tlas_address;
                    break;
                case ResourceKind::IFT:
                    kernel_params[i] =
                        (void *) (uintptr_t) ((VulkanScene *) owner)->isect_table_address;
                    break;
                case ResourceKind::Texture:
                case ResourceKind::Sampler:
                    kernel_params[i] =
                        (void *) (uintptr_t) ((VulkanTexResource *) owner)->slot;
                    break;
                default:
                    break;
            }
        }
    }

    size_t params_bytes = n_params * sizeof(void *);
    bool params_in_pc = params_bytes <= dev.max_push_constants_size;

    // In jitc_vulkan_assemble(), %Params always declares at least 1 element in
    // args[] (16 bytes total) even when n_params == 1.
    void *pc_fallback[2] = { n_params > 0 ? kernel_params[0] : nullptr,
                             nullptr };
    const void *pc_data = n_params == 1 ? (const void *) pc_fallback
                                        : (const void *) kernel_params.data();
    size_t pc_size = n_params == 1 ? 2 * sizeof(void *) : params_bytes;

    uint64_t params_device_addr = 0;
    void *large_staging = nullptr;

    if (!params_in_pc) {
        // Reserve staging memory *before* jitc_vulkan_cmd(), since
        // jitc_vulkan_stage() or jitc_malloc() may allocate a new buffer.
        void *staging = nullptr;
        if (pc_size <= VulkanMaxStageSize) {
            staging = jitc_vulkan_stage(dev, pc_size);
        } else {
            large_staging = jitc_malloc(JitBackend::Vulkan, pc_size, true);
            staging = large_staging;
        }
        std::memcpy(staging, pc_data, pc_size);

        size_t staging_offset = 0;
        VulkanBuffer *staging_buf =
            jitc_vulkan_find_buffer(staging, &staging_offset);
        if (unlikely(!staging_buf))
            jitc_fail("VulkanThreadState::launch(): staging buffer not found!");
        params_device_addr = staging_buf->address + staging_offset;
    }

    VkCommandBuffer cmd = jitc_vulkan_cmd(dev);
    jitc_vulkan_access(dev, true);

    void *qp = nullptr;
    uint32_t qi = 0;
    if (unlikely(kernel_history_entry))
        std::tie(qp, qi) = jitc_vulkan_history_begin(dev, cmd);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                      (VkPipeline) kernel.vulkan.pipeline);

    if (params_in_pc) {
        vkCmdPushConstants(cmd, (VkPipelineLayout) kernel.vulkan.layout,
                           VK_SHADER_STAGE_COMPUTE_BIT, 0,
                           (uint32_t) pc_size, pc_data);
    } else {
        vkCmdPushConstants(cmd, (VkPipelineLayout) kernel.vulkan.layout,
                           VK_SHADER_STAGE_COMPUTE_BIT, 0,
                           sizeof(uint64_t), &params_device_addr);
    }

    uint32_t wg_size = kernel.vulkan.workgroup_size;
    uint32_t group_count = (size + wg_size - 1) / wg_size;
    if (dev.max_compute_work_group_count[0] > 0 &&
        group_count > dev.max_compute_work_group_count[0])
        group_count = dev.max_compute_work_group_count[0];

    vkCmdDispatch(cmd, group_count, 1, 1);

    if (unlikely(kernel_history_entry)) {
        kernel_history_entry->size = size;
        KernelHistoryEntryImpl *e =
            jitc_kernel_history_append(*kernel_history_entry);
        jitc_vulkan_history_end(dev, cmd, qp, qi, e);
    }

    if (unlikely(large_staging))
        jitc_free(large_staging);

    if (unlikely(jit_flag(JitFlag::LaunchBlocking)))
        jitc_vulkan_sync(false);
    else
        jitc_vulkan_maybe_submit(dev);

    return nullptr;
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

    uint64_t zero = 0;
    bool is_64bit_pattern = (isize == 8 && memcmp(src, &zero, 8) != 0);
    vulkan_fill(vulkan_device(this), ptr, (size_t) size * isize, isize, src,
                "jit_memset_async", is_64bit_pattern ? this : nullptr,
                KernelType::Memset, size);
}

// ====================================================================
//          Reductions, prefix scans, compaction & permutations
// ====================================================================

static VarType vulkan_make_int_unsigned(VarType type) {
    switch (type) {
        case VarType::Int8:  return VarType::UInt8;
        case VarType::Int16: return VarType::UInt16;
        case VarType::Int32: return VarType::UInt32;
        case VarType::Int64: return VarType::UInt64;
        default: return type;
    }
}

static uint64_t vulkan_resolve(const void *ptr, size_t size, const char *func) {
    if (!ptr && size == 0)
        return 0;
    size_t offset = 0;
    VulkanBuffer *buf = vulkan_lookup(ptr, size, &offset, func);
    if (!buf)
        jitc_raise("%s(): " DRJIT_PTR " is not a Vulkan allocation!", func,
                   (uintptr_t) ptr);
    return buf->address + offset;
}

template <typename Params>
static void vulkan_dispatch_groups(VulkanDevice &dev, KernelType type,
                                   KernelRecordingMode recording_mode,
                                   VkPipeline pipeline, const Params &params,
                                   uint32_t total_groups, uint32_t width) {
    if (total_groups == 0)
        return;

    uint32_t flags = jit_flags();
    VkCommandBuffer cmd = jitc_vulkan_cmd(dev);
    jitc_vulkan_access(dev, true);

    void *qp = nullptr;
    uint32_t qi = 0;
    if (unlikely(flags & (uint32_t) JitFlag::KernelHistory))
        std::tie(qp, qi) = jitc_vulkan_history_begin(dev, cmd);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdPushConstants(cmd, (VkPipelineLayout) dev.pipeline_layout,
                       VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(Params), &params);

    uint32_t max_x = dev.max_compute_work_group_count[0];
    uint32_t gx = total_groups, gy = 1;
    if (max_x > 0 && total_groups > max_x) {
        gx = max_x;
        gy = ceil_div(total_groups, gx);
    }

    vkCmdDispatch(cmd, gx, gy, 1);

    if (unlikely(flags & (uint32_t) JitFlag::KernelHistory)) {
        KernelHistoryEntry meta = {};
        meta.backend = JitBackend::Vulkan;
        meta.type = type;
        meta.recording_mode = recording_mode;
        meta.size = width;
        meta.input_count = 1;
        meta.output_count = 1;

        KernelHistoryEntryImpl *e = jitc_kernel_history_append(meta);
        jitc_vulkan_history_end(dev, cmd, qp, qi, e);
    }

    if (unlikely(flags & (uint32_t) JitFlag::LaunchBlocking))
        jitc_vulkan_sync(false);
    else
        jitc_vulkan_maybe_submit(dev);
}

template <typename Params>
static void vulkan_dispatch_threads(VulkanDevice &dev, KernelType type,
                                    KernelRecordingMode recording_mode,
                                    VkPipeline pipeline, const Params &params,
                                    uint32_t thread_count, uint32_t width) {
    vulkan_dispatch_groups(dev, type, recording_mode, pipeline, params,
                           ceil_div(thread_count, VulkanWorkgroupSize), width);
}

// Host-side mirrors of the push constant structs in resources/vulkan_kernels.glsl

struct VulkanCompressScatterParams {
    uint64_t in_addr, prefix_addr, out_addr, count_addr;
    uint32_t size, chunk_size, chunk_count, chunks_scanned;
};

struct VulkanMkpermParams {
    uint64_t values_addr, buckets_addr, perm_addr;
    uint32_t size;
};

struct VulkanMkpermOffsetsParams {
    uint64_t buckets_addr, offsets_addr, counter_addr;
    uint32_t bucket_count, perm_size, row_stride;
};

struct VulkanMkpermTinyParams {
    uint64_t values_addr, buckets_addr, perm_addr;
    uint32_t size, size_per_block, bucket_count, block_size;
    uint32_t n_blocks, gpu_blocks_per_group, warp_count, rows_per_group;
};

struct VulkanBlockReduceParams {
    uint64_t in_addr, out_addr;
    uint32_t size, block_size, block_count, chunk_size, chunks_per_block;
};

struct VulkanBlockPrefixReduceParams {
    uint64_t in_addr, out_addr, prefixes_addr;
    uint32_t size, block_size, block_count, chunk_size, chunks_per_block;
    uint32_t exclusive, reverse;
};

struct VulkanReduceDotParams {
    uint64_t in1_addr, in2_addr, out_addr;
    uint32_t size, chunk_size, chunk_count;
};

void VulkanThreadState::block_reduce(VarType vt, ReduceOp op, uint32_t size,
                                     uint32_t block_size, const void *in,
                                     void *out) {
    if (size == 0)
        return;
    if (block_size == 0 || block_size > size)
        jitc_raise("jit_block_reduce(): invalid block size "
                   "(size=%u, block_size=%u)!", size, block_size);

    uint32_t tsize = type_size[(int) vt];
    if (block_size == 1) {
        if (in != out)
            memcpy_async(out, in, (size_t) size * tsize);
        return;
    }

    if (op == ReduceOp::Add || op == ReduceOp::Mul ||
        op == ReduceOp::Or || op == ReduceOp::And)
        vt = vulkan_make_int_unsigned(vt);

    VulkanDevice &dev = vulkan_device(this);

    uint32_t block_count = ceil_div(size, block_size);

    bool sg_coop = block_size >= 16 &&
                   (block_size < 256 || (block_size <= 1024 && block_count >= 64)) &&
                   dev.subgroup_size >= 16 &&
                   VulkanWorkgroupSize % dev.subgroup_size == 0;
    bool small = block_size < 16 || sg_coop;

    VkPipeline pipe = jitc_vulkan_reduce_pipeline(
        dev, small ? VulkanReduceKind::Small : VulkanReduceKind::Chunk, op, vt);
    if (!pipe)
        jitc_raise("jit_block_reduce(): unsupported type '%s' for op '%s' "
                   "on the Vulkan backend.",
                   type_name[(int) vt], red_name[(int) op]);

    uint32_t chunk_size = sg_coop ? 1u : 0u,
             chunks_per_block = 1,
             chunk_count = block_count;
    if (!small) {
        chunk_size = std::min(block_size, VulkanChunkSize);
        chunks_per_block = ceil_div(block_size, chunk_size);
        chunk_count = block_count * chunks_per_block;
    }

    void *dst = out;
    if (chunks_per_block > 1)
        dst = jitc_malloc(JitBackend::Vulkan, (size_t) chunk_count * tsize);

    VulkanBlockReduceParams params;
    params.in_addr = vulkan_resolve(in, (size_t) size * tsize, "jit_block_reduce");
    params.out_addr =
        vulkan_resolve(dst, (size_t) chunk_count * tsize, "jit_block_reduce");
    params.size = size;
    params.block_size = block_size;
    params.block_count = block_count;
    params.chunk_size = chunk_size;
    params.chunks_per_block = chunks_per_block;

    jitc_log(Debug,
             "jit_block_reduce(" DRJIT_PTR " -> " DRJIT_PTR
             ", type=%s, op=%s, size=%u, block_size=%u, block_count=%u, "
             "chunk_size=%u, chunks_per_block=%u)",
             (uintptr_t) in, (uintptr_t) out, type_name[(int) vt],
             red_name[(int) op], size, block_size, block_count, chunk_size,
             chunks_per_block);

    if (sg_coop) {
        uint32_t sgs_per_wg = VulkanWorkgroupSize / dev.subgroup_size;
        vulkan_dispatch_groups(dev, KernelType::BlockReduce,
                               this->recording_mode, pipe, params,
                               ceil_div(block_count, sgs_per_wg), size);
    } else if (small) {
        vulkan_dispatch_threads(dev, KernelType::BlockReduce,
                                this->recording_mode, pipe, params,
                                block_count, size);
    } else {
        vulkan_dispatch_groups(dev, KernelType::BlockReduce,
                               this->recording_mode, pipe, params,
                               chunk_count, size);
    }

    if (chunks_per_block > 1) {
        block_reduce(vt, op, chunk_count, chunks_per_block, dst, out);
        jitc_free(dst);
    }
}

static void vulkan_block_prefix_reduce(VulkanThreadState *ts, VarType vt,
                                       VarType vta, ReduceOp op, uint32_t size,
                                       uint32_t block_size, bool exclusive,
                                       bool reverse, const void *in,
                                       void *out,
                                       KernelType ktype = KernelType::BlockPrefixReduce) {
    VulkanDevice &dev = vulkan_device(ts);
    uint32_t tsize_in = type_size[(int) vt];
    uint32_t tsize_out = (vt == VarType::UInt8 && vta == VarType::UInt32)
                             ? sizeof(uint32_t)
                             : tsize_in;

    uint32_t block_count = ceil_div(size, block_size);
    bool small = block_size < 64 && block_count >= 64;

    VkPipeline scan_pipe = jitc_vulkan_reduce_pipeline(
        dev, small ? VulkanReduceKind::SmallScan : VulkanReduceKind::Scan, op,
        vt);
    if (!scan_pipe)
        jitc_raise("jit_block_prefix_reduce(): unsupported type '%s' for "
                   "op '%s' on the Vulkan backend.",
                   type_name[(int) vt], red_name[(int) op]);

    uint32_t max_chunk = reverse ? (VulkanWorkgroupSize * VulkanGrain)
                                 : VulkanScanChunkSize;
    uint32_t chunk_size = std::min(block_size, max_chunk),
             chunks_per_block = small ? 1u : ceil_div(block_size, chunk_size),
             chunk_count = block_count * chunks_per_block;

    jitc_log(Debug,
             "jit_block_prefix_reduce(" DRJIT_PTR " -> " DRJIT_PTR
             ", type=%s, op=%s, size=%u, block_size=%u, exclusive=%i, "
             "reverse=%i, block_count=%u, chunk_size=%u, "
             "chunks_per_block=%u)",
             (uintptr_t) in, (uintptr_t) out, type_name[(int) vt],
             red_name[(int) op], size, block_size, (int) exclusive,
             (int) reverse, block_count, chunk_size, chunks_per_block);

    void *temp = nullptr;
    if (chunks_per_block > 1) {
        VkPipeline red_pipe = jitc_vulkan_reduce_pipeline(
            dev,
            vta != vt ? VulkanReduceKind::WideChunk : VulkanReduceKind::Chunk,
            op, vt);
        if (!red_pipe)
            jitc_raise("jit_block_prefix_reduce(): unsupported type '%s' "
                       "for op '%s' on the Vulkan backend.",
                       type_name[(int) vt], red_name[(int) op]);

        size_t temp_bytes = (size_t) chunk_count * type_size[(int) vta];
        temp = jitc_malloc(JitBackend::Vulkan, temp_bytes);

        VulkanBlockReduceParams rparams;
        rparams.in_addr =
            vulkan_resolve(in, (size_t) size * tsize_in, "jit_block_prefix_reduce");
        rparams.out_addr =
            vulkan_resolve(temp, temp_bytes, "jit_block_prefix_reduce");
        rparams.size = size;
        rparams.block_size = block_size;
        rparams.block_count = block_count;
        rparams.chunk_size = chunk_size;
        rparams.chunks_per_block = chunks_per_block;

        vulkan_dispatch_groups(dev, ktype, ts->recording_mode, red_pipe,
                               rparams, chunk_count, size);

        vulkan_block_prefix_reduce(ts, vta, vta, op, chunk_count,
                                   chunks_per_block, /* exclusive = */ true,
                                   reverse, temp, temp, ktype);
    }

    // Handle in-place reverse scan over multiple elements in a single workgroup
    // safely even if input and output alias. (Each workgroup reads all 4 per-thread
    // elements into registers before writing any outputs.)
    VulkanBlockPrefixReduceParams params;
    params.in_addr =
        vulkan_resolve(in, (size_t) size * tsize_in, "jit_block_prefix_reduce");
    params.out_addr =
        vulkan_resolve(out, (size_t) size * tsize_out, "jit_block_prefix_reduce");
    params.prefixes_addr =
        temp ? vulkan_resolve(temp, (size_t) chunk_count * type_size[(int) vta],
                              "jit_block_prefix_reduce")
             : params.out_addr;
    params.size = size;
    params.block_size = block_size;
    params.block_count = block_count;
    params.chunk_size = chunk_size;
    params.chunks_per_block = chunks_per_block;
    params.exclusive = exclusive ? 1u : 0u;
    params.reverse = reverse ? 1u : 0u;

    if (small)
        vulkan_dispatch_threads(dev, ktype, ts->recording_mode, scan_pipe,
                                params, block_count, size);
    else
        vulkan_dispatch_groups(dev, ktype, ts->recording_mode, scan_pipe,
                               params, chunk_count, size);

    if (temp)
        jitc_free(temp);
}

void VulkanThreadState::block_prefix_reduce(VarType vt, ReduceOp op,
                                            uint32_t size, uint32_t block_size,
                                            bool exclusive, bool reverse,
                                            const void *in, void *out) {
    if (size == 0)
        return;
    if (block_size == 0 || block_size > size)
        jitc_raise("jit_block_prefix_reduce(): invalid "
                   "block_size=%u for size=%u.", block_size, size);

    uint32_t tsize = type_size[(int) vt];
    if (block_size == 1) {
        if (exclusive) {
            uint64_t ident = jitc_reduce_identity(vt, op);
            memset_async(out, size, tsize, &ident);
        } else if (in != out) {
            memcpy_async(out, in, (size_t) size * tsize);
        }
        return;
    }

    if (op == ReduceOp::Add || op == ReduceOp::Mul ||
        op == ReduceOp::Or || op == ReduceOp::And)
        vt = vulkan_make_int_unsigned(vt);

    VarType vta = vt;
    if (vt == VarType::Float16 &&
        (op == ReduceOp::Add || op == ReduceOp::Mul))
        vta = VarType::Float32;

    vulkan_block_prefix_reduce(this, vt, vta, op, size, block_size,
                               exclusive, reverse, in, out,
                               KernelType::BlockPrefixReduce);
}

void VulkanThreadState::reduce_dot(VarType vt, const void *ptr_1,
                                   const void *ptr_2, uint32_t size,
                                   void *out) {
    if (size == 0)
        return;

    VulkanDevice &dev = vulkan_device(this);
    VkPipeline pipe =
        jitc_vulkan_reduce_pipeline(dev, VulkanReduceKind::Dot, ReduceOp::Add, vt);
    if (!pipe)
        jitc_raise("jit_reduce_dot(): unsupported type %s on the Vulkan backend.",
                   type_name[(int) vt]);

    uint32_t tsize = type_size[(int) vt];
    uint32_t chunk_size = std::min(size, VulkanChunkSize),
             chunk_count = ceil_div(size, chunk_size);

    jitc_log(Debug,
             "jit_reduce_dot(" DRJIT_PTR ", " DRJIT_PTR
             ", type=%s, size=%u, chunk_count=%u)",
             (uintptr_t) ptr_1, (uintptr_t) ptr_2, type_name[(int) vt], size,
             chunk_count);

    void *dst = out;
    if (chunk_count > 1)
        dst = jitc_malloc(JitBackend::Vulkan, (size_t) chunk_count * tsize);

    VulkanReduceDotParams params;
    params.in1_addr =
        vulkan_resolve(ptr_1, (size_t) size * tsize, "jit_reduce_dot");
    params.in2_addr =
        vulkan_resolve(ptr_2, (size_t) size * tsize, "jit_reduce_dot");
    params.out_addr =
        vulkan_resolve(dst, (size_t) chunk_count * tsize, "jit_reduce_dot");
    params.size = size;
    params.chunk_size = chunk_size;
    params.chunk_count = chunk_count;

    vulkan_dispatch_groups(dev, KernelType::Dot, this->recording_mode, pipe,
                           params, chunk_count, size);

    if (chunk_count > 1) {
        block_reduce(vt, ReduceOp::Add, chunk_count, chunk_count, dst, out);
        jitc_free(dst);
    }
}

void VulkanThreadState::batched_gemm(VarType, bool, bool, uint32_t, uint32_t,
                                     uint32_t, const GemmBatch *, const void *,
                                     const void *, void *) {
    vulkan_unsupported("jit_batched_gemm");
}

uint32_t VulkanThreadState::compress(const uint8_t *in, uint32_t size,
                                     uint32_t *out) {
    if (size == 0)
        return 0;

    VulkanDevice &dev = vulkan_device(this);

    jitc_log(Debug, "jit_compress(" DRJIT_PTR " -> " DRJIT_PTR ", size=%u)",
             (uintptr_t) in, (uintptr_t) out, size);

    VkPipeline scatter_pipe =
        jitc_vulkan_pipeline(dev, VulkanKernel::CompressScatter);
    if (!scatter_pipe)
        jitc_raise("jit_compress(): failed to create compress_scatter pipeline!");

    uint32_t chunk_size = VulkanChunkSize;
    uint32_t chunk_count = ceil_div(size, chunk_size);
    uint32_t chunks_scanned = 0;

    void *count =
        jitc_malloc(JitBackend::Vulkan, sizeof(uint32_t), /*shared=*/true);
    void *chunk_buf = nullptr;

    uint64_t in_addr = vulkan_resolve(in, size, "jit_compress");
    uint64_t out_addr =
        vulkan_resolve(out, (size_t) size * sizeof(uint32_t), "jit_compress");
    uint64_t count_addr =
        vulkan_resolve(count, sizeof(uint32_t), "jit_compress");
    uint64_t prefix_addr = out_addr;

    if (chunk_count > 1) {
        VkPipeline red_pipe = jitc_vulkan_reduce_pipeline(
            dev, VulkanReduceKind::WideChunk, ReduceOp::Add, VarType::UInt8);
        if (!red_pipe)
            jitc_raise("jit_compress(): failed to create chunk reduction pipeline!");

        size_t chunk_bytes = (size_t) chunk_count * sizeof(uint32_t);
        chunk_buf = jitc_malloc(JitBackend::Vulkan, chunk_bytes);
        prefix_addr = vulkan_resolve(chunk_buf, chunk_bytes, "jit_compress");

        VulkanBlockReduceParams rparams;
        rparams.in_addr = in_addr;
        rparams.out_addr = prefix_addr;
        rparams.size = size;
        rparams.block_size = size;
        rparams.block_count = 1;
        rparams.chunk_size = chunk_size;
        rparams.chunks_per_block = chunk_count;

        vulkan_dispatch_groups(dev, KernelType::Compress, this->recording_mode,
                               red_pipe, rparams, chunk_count, size);

        if (chunk_count > VulkanScanChunkSize) {
            vulkan_block_prefix_reduce(this, VarType::UInt32, VarType::UInt32,
                                       ReduceOp::Add, chunk_count, chunk_count,
                                       /* exclusive = */ true,
                                       /* reverse = */ false, chunk_buf,
                                       chunk_buf, KernelType::Compress);
            chunks_scanned = 1;
        }
    }

    VulkanCompressScatterParams params;
    params.in_addr = in_addr;
    params.prefix_addr = prefix_addr;
    params.out_addr = out_addr;
    params.count_addr = count_addr;
    params.size = size;
    params.chunk_size = chunk_size;
    params.chunk_count = chunk_count;
    params.chunks_scanned = chunks_scanned;

    vulkan_dispatch_groups(dev, KernelType::Compress, this->recording_mode,
                           scatter_pipe, params, chunk_count, size);

    jitc_vulkan_sync(false);
    uint32_t result = *(const uint32_t *) count;
    if (chunk_buf)
        jitc_free(chunk_buf);
    jitc_free(count);
    return result;
}

static void vulkan_mkperm_detect_offsets(VulkanThreadState *ts,
                                         VulkanDevice &dev, void *buckets,
                                         size_t buckets_bytes,
                                         uint32_t bucket_count, uint32_t size,
                                         uint32_t row_stride,
                                         uint32_t *offsets_dev,
                                         void **counter_out) {
    void *counter =
        jitc_malloc(JitBackend::Vulkan, sizeof(uint32_t), /*shared=*/true);
    *(uint32_t *) counter = 0;
    *counter_out = counter;

    VkPipeline pipe =
        jitc_vulkan_pipeline(dev, VulkanKernel::MkpermDetectOffsets);
    if (!pipe)
        jitc_raise("jit_block_mkperm(): failed to create mkperm_detect_offsets "
                   "pipeline!");

    size_t offsets_bytes = (size_t) bucket_count * 4 * sizeof(uint32_t);
    VulkanMkpermOffsetsParams params;
    params.buckets_addr =
        vulkan_resolve(buckets, buckets_bytes, "jit_block_mkperm");
    params.offsets_addr =
        vulkan_resolve(offsets_dev, offsets_bytes, "jit_block_mkperm");
    params.counter_addr =
        vulkan_resolve(counter, sizeof(uint32_t), "jit_block_mkperm");
    params.bucket_count = bucket_count;
    params.perm_size = size;
    params.row_stride = row_stride;

    vulkan_dispatch_threads(dev, KernelType::MkPerm, ts->recording_mode, pipe,
                            params, bucket_count, size);
}

static uint32_t vulkan_mkperm_finish(VulkanThreadState *ts, void *buckets,
                                     uint32_t bucket_count, uint32_t *offsets,
                                     uint32_t *offsets_staging,
                                     VulkanBuffer *offsets_buf, void *counter) {
    jitc_free(buckets);

    uint32_t unique_count = 0;
    if (offsets) {
        jitc_vulkan_sync(false);
        unique_count = *(const uint32_t *) counter;
        if (offsets_staging) {
            offsets_staging[4 * bucket_count] = unique_count;
            std::memcpy(offsets, offsets_staging,
                        ((size_t) bucket_count * 4 + 1) * sizeof(uint32_t));
            jitc_free(offsets_staging);
        } else if (offsets_buf->shared) {
            offsets[4 * bucket_count] = unique_count;
        } else {
            ts->poke(offsets + 4 * bucket_count, &unique_count,
                     sizeof(uint32_t));
            jitc_vulkan_sync(false);
        }
        jitc_free(counter);
    }

    return unique_count;
}

uint32_t VulkanThreadState::block_mkperm(const uint32_t *values, uint32_t size,
                                         uint32_t block_size,
                                         uint32_t bucket_count, uint32_t *perm,
                                         uint32_t *offsets) {
    if (size == 0)
        return 0;
    if (bucket_count == 0)
        jitc_fail("jit_block_mkperm(): bucket_count cannot be zero!");

    VulkanDevice &dev = vulkan_device(this);
    const uint32_t warp = dev.subgroup_size ? dev.subgroup_size : 32u;

    size_t offsets_bytes = ((size_t) bucket_count * 4 + 1) * sizeof(uint32_t);
    size_t offsets_off = 0;
    VulkanBuffer *offsets_buf =
        offsets ? vulkan_lookup(offsets, offsets_bytes, &offsets_off,
                                "jit_block_mkperm")
                : nullptr;
    uint32_t *offsets_staging = nullptr;
    uint32_t *offsets_dev = offsets;
    if (offsets && !offsets_buf) {
        offsets_staging = (uint32_t *) jitc_malloc(
            JitBackend::Vulkan, offsets_bytes, /*shared=*/true);
        offsets_dev = offsets_staging;
    }

    // ---- Stable "tiny" path: per-subgroup shared memory histograms ----
    // vulkan_kernels.glsl declares 'shared uint sh[2048]' (8 KiB).
    uint32_t max_subgroups = std::max(1u, VulkanWorkgroupSize / warp);
    uint32_t max_warps_mem = 2048u / bucket_count;
    uint32_t warp_count = std::min({ 8u, max_subgroups, max_warps_mem });

    if (warp_count >= 1) {
        uint32_t thread_count = warp_count * warp;
        uint32_t n_blocks = ceil_div(size, block_size);

        const uint32_t ELEMS_PER_THREAD = 16;
        uint32_t elems_per_blk = thread_count * ELEMS_PER_THREAD;
        uint32_t gpu_blocks_per_group =
            std::max(1u, ceil_div(block_size, elems_per_blk));

        uint32_t max_cells = std::max(1u, size / 4u);
        uint64_t denom = (uint64_t) n_blocks * warp_count * bucket_count;
        uint32_t max_sub = (uint32_t) std::max<uint64_t>(1, max_cells / denom);
        gpu_blocks_per_group = std::min(gpu_blocks_per_group, max_sub);

        uint32_t size_per_block =
            align_up(ceil_div(block_size, gpu_blocks_per_group), warp);
        uint32_t rows_per_group = gpu_blocks_per_group * warp_count;
        uint32_t seg = rows_per_group * bucket_count;
        uint32_t total_cells = n_blocks * seg;
        uint32_t total_groups = n_blocks * gpu_blocks_per_group;

        if (offsets && n_blocks != 1)
            jitc_raise("jit_block_mkperm(): offset extraction requires "
                       "block_size == size.");

        jitc_log(Debug,
                 "jit_block_mkperm(" DRJIT_PTR ", size=%u, block_size=%u, "
                 "bucket_count=%u, variant=tiny, blocks=%u, sub/block=%u, "
                 "warps=%u, size_per_block=%u)",
                 (uintptr_t) values, size, block_size, bucket_count, n_blocks,
                 gpu_blocks_per_group, warp_count, size_per_block);

        size_t buckets_bytes = (size_t) total_cells * sizeof(uint32_t);
        void *buckets = jitc_malloc(JitBackend::Vulkan, buckets_bytes);

        VkPipeline p1_tiny =
            jitc_vulkan_pipeline(dev, VulkanKernel::MkpermPhase1Tiny);
        VkPipeline p4_tiny =
            jitc_vulkan_pipeline(dev, VulkanKernel::MkpermPhase4Tiny);
        if (!p1_tiny || !p4_tiny)
            jitc_raise("jit_block_mkperm(): failed to create tiny mkperm "
                       "pipelines!");

        VulkanMkpermTinyParams tp;
        tp.values_addr = vulkan_resolve(
            values, (size_t) size * sizeof(uint32_t), "jit_block_mkperm");
        tp.buckets_addr =
            vulkan_resolve(buckets, buckets_bytes, "jit_block_mkperm");
        tp.perm_addr = 0;
        tp.size = size;
        tp.size_per_block = size_per_block;
        tp.bucket_count = bucket_count;
        tp.block_size = block_size;
        tp.n_blocks = n_blocks;
        tp.gpu_blocks_per_group = gpu_blocks_per_group;
        tp.warp_count = warp_count;
        tp.rows_per_group = rows_per_group;

        // Phase 1: per-subgroup histograms -> bucket-major global layout
        vulkan_dispatch_groups(dev, KernelType::MkPerm, this->recording_mode,
                               p1_tiny, tp, total_groups, size);

        // Phase 2: per-group exclusive prefix sum (segment length = seg)
        block_prefix_reduce(VarType::UInt32, ReduceOp::Add, total_cells, seg,
                            /*exclusive=*/true, /*reverse=*/false, buckets,
                            buckets);

        // Phase 2.5 (optional, single block): detect non-empty buckets
        void *counter = nullptr;
        if (offsets)
            vulkan_mkperm_detect_offsets(this, dev, buckets, buckets_bytes,
                                         bucket_count, size, rows_per_group,
                                         offsets_dev, &counter);

        // Phase 4: scatter the permutation
        tp.perm_addr = vulkan_resolve(perm, (size_t) size * sizeof(uint32_t),
                                      "jit_block_mkperm");
        vulkan_dispatch_groups(dev, KernelType::MkPerm, this->recording_mode,
                               p4_tiny, tp, total_groups, size);

        return vulkan_mkperm_finish(this, buckets, bucket_count, offsets,
                                    offsets_staging, offsets_buf, counter);
    }

    // ---- Fallback: global-atomic path (single block only) ----
    if (block_size != size)
        jitc_raise("jit_block_mkperm(): bucket_count=%u exceeds the stable "
                   "shared-memory path, and the global-atomic fallback does "
                   "not support per-block permutations (block_size != size) "
                   "on Vulkan.",
                   bucket_count);

    jitc_log(Debug,
             "jit_block_mkperm(" DRJIT_PTR ", size=%u, block_size=%u, "
             "bucket_count=%u, variant=global)",
             (uintptr_t) values, size, block_size, bucket_count);

    size_t bucket_bytes = (size_t) bucket_count * sizeof(uint32_t);
    void *buckets = jitc_malloc(JitBackend::Vulkan, bucket_bytes);
    uint32_t zero = 0;
    vulkan_fill(dev, buckets, bucket_bytes, sizeof(uint32_t), &zero,
                "jit_block_mkperm");

    VkPipeline p1 = jitc_vulkan_pipeline(dev, VulkanKernel::MkpermPhase1);
    VkPipeline p3 = jitc_vulkan_pipeline(dev, VulkanKernel::MkpermPhase3);
    if (!p1 || !p3)
        jitc_raise("jit_block_mkperm(): failed to create global mkperm "
                   "pipelines!");

    VulkanMkpermParams params;
    params.values_addr = vulkan_resolve(
        values, (size_t) size * sizeof(uint32_t), "jit_block_mkperm");
    params.buckets_addr =
        vulkan_resolve(buckets, bucket_bytes, "jit_block_mkperm");
    params.perm_addr = 0;
    params.size = size;

    // Phase 1: Build histogram via global atomics
    vulkan_dispatch_threads(dev, KernelType::MkPerm, this->recording_mode, p1,
                            params, size, size);

    // Phase 2: Exclusive prefix sum over histogram
    block_prefix_reduce(VarType::UInt32, ReduceOp::Add, bucket_count,
                        bucket_count, true, false, buckets, buckets);

    // Phase 2.5 (optional): Detect non-empty buckets
    void *counter = nullptr;
    if (offsets)
        vulkan_mkperm_detect_offsets(this, dev, buckets, bucket_bytes,
                                     bucket_count, size, /* row_stride = */ 1,
                                     offsets_dev, &counter);

    // Phase 3: Scatter indices using atomics on the prefix-summed histogram
    params.perm_addr = vulkan_resolve(perm, (size_t) size * sizeof(uint32_t),
                                      "jit_block_mkperm");
    vulkan_dispatch_threads(dev, KernelType::MkPerm, this->recording_mode, p3,
                            params, size, size);

    return vulkan_mkperm_finish(this, buckets, bucket_count, offsets,
                                offsets_staging, offsets_buf, counter);
}

void VulkanThreadState::memcpy(void *dst, const void *src, size_t size) {
    vulkan_copy(vulkan_device(this), dst, src, size, false);
}

void VulkanThreadState::memcpy_async(void *dst, const void *src, size_t size) {
    if (size == 0)
        return;
    vulkan_copy(vulkan_device(this), dst, src, size, true);
}

void VulkanThreadState::poke(void *dst, const void *src, uint32_t size) {
    jitc_log(Debug, "jit_poke(" DRJIT_PTR ", size=%u)", (uintptr_t) dst, size);

    if (size != 1 && size != 2 && size != 4 && size != 8)
        jitc_raise("jit_poke(): only size=1, 2, 4 or 8 are supported!");

    vulkan_fill(vulkan_device(this), dst, size, size, src, "jit_poke",
                this, KernelType::Poke, 1);
}

void VulkanThreadState::aggregate(void *dst, AggregationEntry *agg,
                                  uint32_t size) {
    if (size == 0)
        return;

    VulkanDevice &dev = vulkan_device(this);

    jitc_log(InfoSym,
             "jit_aggregate(" DRJIT_PTR " -> " DRJIT_PTR ", size=%u)",
             (uintptr_t) agg, (uintptr_t) dst, size);

    size_t max_end = 0;
    size_t lit_bytes = 0;
    for (uint32_t i = 0; i < size; ++i) {
        const AggregationEntry &e = agg[i];
        uint32_t esize = (uint32_t) std::abs((int) e.size);
        size_t end = (size_t) e.offset + esize;
        if (end > max_end)
            max_end = end;
        if (e.size > 0)
            lit_bytes += esize;
    }

    size_t dst_offset = 0;
    if (!vulkan_lookup(dst, max_end, &dst_offset, "jit_aggregate"))
        jitc_raise("jit_aggregate(): " DRJIT_PTR " is not a Vulkan allocation!",
                   (uintptr_t) dst);

    uint8_t *staging = nullptr;
    void *staging_alloc = nullptr;
    if (lit_bytes > 0) {
        if (lit_bytes <= VulkanMaxStageSize) {
            staging = jitc_vulkan_stage(dev, lit_bytes);
        } else {
            staging_alloc = jitc_malloc(JitBackend::Vulkan, lit_bytes, true);
            staging = (uint8_t *) staging_alloc;
        }

        size_t pos = 0;
        for (uint32_t i = 0; i < size; ++i) {
            const AggregationEntry &e = agg[i];
            if (e.size <= 0)
                continue;
            uintptr_t val = (uintptr_t) e.src;
            if (e.size == 8 && val) {
                switch ((ResourceKind) e.resource_kind) {
                    case ResourceKind::Buffer: {
                        size_t off = 0;
                        if (VulkanBuffer *b = jitc_vulkan_find_buffer(e.src, &off))
                            val = (uintptr_t) (b->address + off);
                        break;
                    }
                    case ResourceKind::Accel:
                        val = (uintptr_t) ((VulkanScene *) e.src)->tlas_address;
                        break;
                    case ResourceKind::IFT:
                        val = (uintptr_t) ((VulkanScene *) e.src)->isect_table_address;
                        break;
                    case ResourceKind::Texture:
                    case ResourceKind::Sampler:
                        val = (uintptr_t) ((VulkanTexResource *) e.src)->slot;
                        break;
                    default:
                        break;
                }
            }
            switch (e.size) {
                case 1: {
                    uint8_t v = (uint8_t) val;
                    std::memcpy(staging + pos, &v, 1);
                    pos += 1;
                    break;
                }
                case 2: {
                    uint16_t v = (uint16_t) val;
                    std::memcpy(staging + pos, &v, 2);
                    pos += 2;
                    break;
                }
                case 4: {
                    uint32_t v = (uint32_t) val;
                    std::memcpy(staging + pos, &v, 4);
                    pos += 4;
                    break;
                }
                case 8: {
                    uint64_t v = (uint64_t) val;
                    std::memcpy(staging + pos, &v, 8);
                    pos += 8;
                    break;
                }
                default:
                    jitc_fail("jit_aggregate(): invalid literal entry size %d!",
                              (int) e.size);
            }
        }
    }

    // Look up buffers after allocating staging memory, which may release the lock
    VulkanBuffer *dst_buf = jitc_vulkan_find_buffer(dst, &dst_offset),
                 *staging_buf = nullptr;
    size_t staging_offset = 0;
    if (staging)
        staging_buf = jitc_vulkan_find_buffer(staging, &staging_offset);

    uint32_t flags = jit_flags();
    VkCommandBuffer cmd = jitc_vulkan_cmd(dev);
    jitc_vulkan_access(dev, true);

    void *qp = nullptr;
    uint32_t qi = 0;
    if (unlikely(flags & (uint32_t) JitFlag::KernelHistory))
        std::tie(qp, qi) = jitc_vulkan_history_begin(dev, cmd);

    if (staging) {
        std::vector<VkBufferCopy> regions;
        size_t pos = 0;
        for (uint32_t i = 0; i < size; ++i) {
            const AggregationEntry &e = agg[i];
            if (e.size <= 0)
                continue;
            uint32_t esize = (uint32_t) e.size;
            VkDeviceSize src_off = staging_offset + pos;
            VkDeviceSize dst_off = dst_offset + e.offset;
            if (!regions.empty() &&
                regions.back().srcOffset + regions.back().size == src_off &&
                regions.back().dstOffset + regions.back().size == dst_off) {
                regions.back().size += esize;
            } else {
                regions.push_back({ src_off, dst_off, esize });
            }
            pos += esize;
        }
        if (!regions.empty())
            vkCmdCopyBuffer(cmd, staging_buf->buffer, dst_buf->buffer,
                            (uint32_t) regions.size(), regions.data());
    }

    for (uint32_t i = 0; i < size; ++i) {
        const AggregationEntry &e = agg[i];
        if (e.size >= 0)
            continue;
        uint32_t esize = (uint32_t) (-e.size);
        size_t src_offset = 0;
        VulkanBuffer *src_buf =
            vulkan_lookup(e.src, esize, &src_offset, "jit_aggregate");
        if (!src_buf)
            jitc_raise("jit_aggregate(): source pointer " DRJIT_PTR
                       " is not a Vulkan allocation!",
                       (uintptr_t) e.src);
        VkBufferCopy region { src_offset, dst_offset + e.offset, esize };
        vkCmdCopyBuffer(cmd, src_buf->buffer, dst_buf->buffer, 1, &region);
    }

    if (unlikely(flags & (uint32_t) JitFlag::KernelHistory)) {
        KernelHistoryEntry meta = {};
        meta.backend = JitBackend::Vulkan;
        meta.type = KernelType::Aggregate;
        meta.recording_mode = this->recording_mode;
        meta.size = 1;
        meta.input_count = 1;
        meta.output_count = 1;

        KernelHistoryEntryImpl *e = jitc_kernel_history_append(meta);
        jitc_vulkan_history_end(dev, cmd, qp, qi, e);
    }

    if (staging_alloc) {
        jitc_free(staging_alloc);
        dev.stream->cmd_staging += lit_bytes;
    }

    if (unlikely(flags & (uint32_t) JitFlag::LaunchBlocking))
        jitc_vulkan_sync(false);
    else
        jitc_vulkan_maybe_submit(dev);
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
