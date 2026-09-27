/*
    src/vulkan_ts.h -- ThreadState specialization for the Vulkan backend

    Copyright (c) 2026 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#include "internal.h"

#if defined(DRJIT_ENABLE_VULKAN)

/**
 * \brief Thread state of the Vulkan backend
 *
 * All thread states record their work into the command stream of the
 * (single) Vulkan device while holding the Dr.Jit lock (see \ref
 * VulkanStream in vulkan.h). The thread state therefore holds no
 * synchronization state of its own.
 */
struct VulkanThreadState final : ThreadState {
    /// The Vulkan backend uses barrier() as a hint to submit its work
    void barrier() override;

    Task *launch(Kernel kernel, KernelKey &key, XXH128_hash_t hash,
                 uint32_t size, std::vector<void *> &kernel_params,
                 const std::vector<uint32_t> &kernel_param_ids,
                 KernelHistoryEntry *kernel_history_entry) override;

    /// Fill a memory region with constants of a given type
    void memset_async(void *ptr, uint32_t size, uint32_t isize,
                      const void *src) override;

    void block_reduce(VarType vt, ReduceOp op, uint32_t size,
                      uint32_t block_size, const void *in, void *out) override;

    void block_prefix_reduce(VarType vt, ReduceOp op, uint32_t size,
                             uint32_t block_size, bool exclusive, bool reverse,
                             const void *in, void *out) override;

    void reduce_dot(VarType type, const void *ptr_1, const void *ptr_2,
                    uint32_t size, void *out) override;

    void batched_gemm(VarType vt, bool At, bool Bt, uint32_t M, uint32_t N,
                      uint32_t K, const GemmBatch *batch, const void *A,
                      const void *B, void *C) override;

    uint32_t compress(const uint8_t *in, uint32_t size, uint32_t *out) override;

    uint32_t block_mkperm(const uint32_t *values, uint32_t size,
                          uint32_t block_size, uint32_t bucket_count,
                          uint32_t *perm, uint32_t *offsets) override;

    /// Synchronous copy (ordered after all previously issued work)
    void memcpy(void *dst, const void *src, size_t size) override;

    /// Asynchronous copy. Copies from device memory to host memory that is
    /// not a shared allocation are synchronous.
    void memcpy_async(void *dst, const void *src, size_t size) override;

    /// Asynchronously update a single element in memory
    void poke(void *dst, const void *src, uint32_t size) override;

    void aggregate(void *dst, AggregationEntry *agg, uint32_t size) override;

    /// Enqueue a host callback that runs once all previously issued work has
    /// finished. It runs during a later synchronization (e.g. \ref
    /// jit_sync_thread()), or right away if no work is pending.
    void enqueue_host_func(void (*callback)(void *), void *payload) override;

    void coop_vec_pack(uint32_t count, const void *in, const MatrixDescr *in_d,
                       void *out, const MatrixDescr *out_d) override;

    /// Nesting depth of active VulkanHistoryScope guards
    uint32_t vulkan_history_depth = 0;
};

#endif // defined(DRJIT_ENABLE_VULKAN)
