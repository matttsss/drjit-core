/*
    src/vulkan.h -- Vulkan backend: device context, command submission,
    synchronization, and memory allocation

    Copyright (c) 2026 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#include "internal.h"

#if defined(DRJIT_ENABLE_VULKAN)

#include "vulkan_api.h"
#include "malloc.h"
#include <deque>
#include <vector>

/**
 * \brief Record of a memory allocation made by the Vulkan backend
 *
 * Each allocation consists of a dedicated \c VkDeviceMemory block that is
 * bound to a \c VkBuffer covering all of it. Dr.Jit identifies allocations via
 * the pointer \c base:
 *
 * - Shared (host-visible) allocations are persistently mapped, and \c base is
 *   their mapped host address.
 *
 * - Device-local allocations are not accessible from the host, and \c base
 *   equals their device address (\c VkDeviceAddress).
 *
 * The device address of a pointer \c p within the allocation is
 * <tt>address + (p - base)</tt>.
 */
struct VulkanBuffer {
    uintptr_t base;
    size_t size;
    VkDevice device;
    VkBuffer buffer;
    VkDeviceMemory memory;
    VkDeviceAddress address;
    bool shared;
    std::vector<VkDeviceMemory> sparse_memories;
};


/// Host callback registered via \ref jit_enqueue_host_func()
struct VulkanCallback {
    void (*func)(void *);
    void *payload;
};

/// List of allocations to be released via \ref jitc_malloc_release()
using VulkanFreeList = std::vector<std::pair<AllocInfo, void *>>;

/// Texture resources to be destroyed and slot to be recycled once GPU work finishes
struct VulkanTexDestroy {
    VkImage image = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    VkImageView storage_view = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    uint32_t slot = UINT32_MAX;
};

/// A command buffer that was submitted to the queue and may still execute
struct VulkanSubmission {
    VkCommandBuffer cmd;

    /// Value of the device's timeline semaphore signaled upon completion
    uint64_t value;

    /// Allocations to release when the work has finished
    VulkanFreeList frees;

    /// Texture images/views/memory/slots to release when the work has finished
    std::vector<VulkanTexDestroy> tex_frees;

    /// Timestamp query ranges to recycle when the work has finished
    std::vector<std::pair<void *, uint32_t>> queries;

    /// Host callbacks to run when the work has finished
    std::vector<VulkanCallback> callbacks;
};

/**
 * \brief Command stream of a Vulkan device
 *
 * All thread states using a device record their work into a single command
 * buffer (\c cmd) while holding the Dr.Jit lock, which resembles the use of
 * a single CUDA stream per device. The command buffer is submitted to the
 * queue when the work is flushed (e.g., at the end of \ref jit_eval()) or
 * when the host must wait for it. Each submission signals the next value of
 * the device's timeline semaphore. Signal operations of a queue complete in
 * submission order, hence the counter value of the semaphore reveals which
 * submissions have finished.
 *
 * The stream tracks the memory ranges read and written by the commands
 * recorded since the last pipeline barrier, and it inserts a new (global)
 * barrier whenever a command would otherwise race with an earlier one.
 */
struct VulkanStream {
    /// Command pool of the device
    VkCommandPool pool = nullptr;

    /// Command buffer being recorded (or \c nullptr)
    VkCommandBuffer cmd = nullptr;

    /// Number of commands and bytes of staging memory used by 'cmd'
    uint32_t cmd_count = 0;
    size_t cmd_staging = 0;

    /// Command buffers that can be reused
    std::vector<VkCommandBuffer> cmd_free;

    /// Submitted work in submission order
    std::deque<VulkanSubmission> in_flight;

    /// Allocations, textures, queries, and callbacks waiting for the completion of 'cmd'
    VulkanFreeList pending_frees;
    std::vector<VulkanTexDestroy> pending_tex_frees;
    std::vector<std::pair<void *, uint32_t>> pending_queries;
    std::vector<VulkanCallback> pending_callbacks;

    /// Host callbacks of finished work that have not run yet
    std::vector<VulkanCallback> ready_callbacks;

    /// Has any command written to memory since the last pipeline barrier?
    bool write_pending = false;

    /// Must the next command be preceded by a barrier? (e.g. at the
    /// beginning of a command buffer, to order it after earlier submissions)
    bool barrier_pending = true;

    /// Timeline semaphore value of the most recent finished submission
    uint64_t completed_value = 0;

    /// Shared memory chunk for staging small uploads (see jitc_vulkan_stage())
    uint8_t *arena = nullptr;
    size_t arena_offset = 0;
};

// ---------------------------------------------------------------------
//  Initialization & synchronization. The functions below must be called
//  while holding the Dr.Jit lock.
// ---------------------------------------------------------------------

/// Select a compatible device and initialize the Vulkan backend
extern bool jitc_vulkan_init();

/// Wait for all work and release the Vulkan device (full shutdown)
extern void jitc_vulkan_shutdown();

/// Create a thread state that issues work to the Vulkan device
extern ThreadState *jitc_vulkan_thread_state_new();

/// Submit all recorded work without waiting for it
extern void jitc_vulkan_flush();

/**
 * \brief Submit all recorded work and wait for it to finish
 *
 * When \c release_lock is set, the function temporarily releases the Dr.Jit
 * lock while waiting and subsequently runs the host callbacks of finished
 * work (also without holding the lock). Otherwise, it waits while holding
 * the lock and leaves host callbacks for a later synchronization.
 */
extern void jitc_vulkan_sync(bool release_lock = true);

/**
 * \brief Wait until the Vulkan backend is idle
 *
 * In contrast to \ref jitc_vulkan_sync(), this function also waits for work
 * that host callbacks or staging buffer releases enqueue while it runs.
 */
extern void jitc_vulkan_sync_all();

// ---------------------------------------------------------------------
//  Command stream (see VulkanStream)
// ---------------------------------------------------------------------

/**
 * \brief Return the command buffer being recorded, beginning a new one if
 * needed
 *
 * This function never submits work and never releases the Dr.Jit lock.
 */
extern VkCommandBuffer jitc_vulkan_cmd(VulkanDevice &dev);

/**
 * \brief Declare a memory access by the next command
 *
 * Inserts a pipeline barrier if the command conflicts with earlier commands
 * (e.g. read-after-write or write-after-write hazards, or at the start of a
 * command buffer). Must be called after \ref jitc_vulkan_cmd().
 */
extern void jitc_vulkan_access(VulkanDevice &dev, bool is_write = true);

/// Submit the command buffer being recorded (if any)
extern void jitc_vulkan_submit(VulkanDevice &dev);

/// Submit the command buffer if it has accumulated a lot of work
extern void jitc_vulkan_maybe_submit(VulkanDevice &dev);

/// Recycle the resources of finished submissions (does not block)
extern void jitc_vulkan_retire(VulkanDevice &dev);

/**
 * \brief Wait until the timeline semaphore of \c dev reaches \c value
 *
 * The Dr.Jit lock is temporarily released while waiting if \c release_lock
 * is set.
 */
extern void jitc_vulkan_wait(VulkanDevice &dev, uint64_t value,
                             bool release_lock);

/// Run the host callbacks of finished work (temporarily releases the lock)
extern void jitc_vulkan_run_callbacks(VulkanDevice &dev);

/// Enqueue a host callback that runs once all previously issued work is done
extern void jitc_vulkan_enqueue_host_func(VulkanDevice &dev,
                                          void (*func)(void *),
                                          void *payload);

/**
 * \brief Release a shared allocation once the GPU no longer accesses it
 *
 * The allocation is returned to the allocation cache when all previously
 * issued work (including work that has not been submitted yet) has finished.
 */
extern void jitc_vulkan_free_later(AllocInfo info, void *ptr);

/// Allocate a bindless descriptor slot for a sub-texture on \c dev
extern uint32_t jitc_vulkan_tex_slot_alloc(VulkanDevice &dev);

/// Destroy a texture image/view/memory and recycle its descriptor slot once
/// all previously issued GPU work has finished
extern void jitc_vulkan_tex_free_later(VulkanDevice &dev, VkImage image,
                                       VkImageView view,
                                       VkImageView storage_view,
                                       VkDeviceMemory memory, uint32_t slot);

/// Return the bindless sampler descriptor slot for the requested sampler
/// parameters, creating the VkSampler on first use
extern uint32_t jitc_vulkan_sampler_slot(VulkanDevice &dev, int filter_mode,
                                         int wrap_mode, int mip_filter = 1,
                                         size_t max_aniso = 1,
                                         bool mipmapped = false);

struct KernelHistoryEntryImpl;

/// Allocate a 2-query timestamp range on \c dev and record the start timestamp on \c cmd
extern std::pair<void *, uint32_t>
jitc_vulkan_history_begin(VulkanDevice &dev, VkCommandBuffer cmd);

/// Record the end timestamp on \c cmd and attach the query range to \c entry
extern void jitc_vulkan_history_end(VulkanDevice &dev, VkCommandBuffer cmd,
                                    void *query_pool, uint32_t query_index,
                                    KernelHistoryEntryImpl *entry);

/// Wait for the command buffer containing the query range to complete and return elapsed time (ms)
extern float jitc_vulkan_history_wait(int device_id, void *query_pool,
                                      uint32_t query_index);

/// Recycle a timestamp query range back to \c dev (immediately if \c timed, or once in-flight work finishes)
extern void jitc_vulkan_history_free(int device_id, void *query_pool,
                                     uint32_t query_index, bool timed);

/**
 * \brief Reserve \c size bytes (<= 16 KiB) of shared staging memory
 *
 * The memory stays valid until the command buffer being recorded has
 * finished. This function may allocate memory and must therefore be called
 * *before* \ref jitc_vulkan_cmd(), and at most once per operation (an
 * operation that needs staging memory for multiple commands should reserve
 * it with a single call).
 */
extern uint8_t *jitc_vulkan_stage(VulkanDevice &dev, size_t size);

/// Maximum size of a reservation made via \ref jitc_vulkan_stage()
constexpr size_t VulkanMaxStageSize = 16384;

// ---------------------------------------------------------------------
//  Memory allocation & buffer registry
// ---------------------------------------------------------------------

/**
 * \brief Allocate a buffer on the given device (see \ref VulkanBuffer)
 *
 * Returns \c nullptr on failure. The ``vulkan_*`` functions lack the
 * ``jitc_`` prefix to indicate that they are called *without* holding the
 * Dr.Jit lock.
 */
extern VulkanBuffer *vulkan_buffer_new(const VulkanDevice &dev, size_t size,
                                       bool shared);

/// Release a buffer created by \ref vulkan_buffer_new() (no lock needed)
extern void vulkan_buffer_free(VulkanBuffer *buffer);

/**
 * \brief Register a buffer for use with \ref jitc_vulkan_find_buffer()
 *
 * Returns \c false if the address range of the buffer overlaps with that of
 * a registered buffer, in which case the buffer is not registered.
 */
extern bool jitc_vulkan_register_buffer(VulkanBuffer *buffer);

/// Unregister the buffer whose base address is \c ptr (or return \c nullptr)
extern VulkanBuffer *jitc_vulkan_unregister_buffer(const void *ptr);

/**
 * \brief Find the registered buffer containing the address \c ptr
 *
 * Returns \c nullptr if \c ptr does not lie within a buffer. Otherwise, the
 * byte offset of \c ptr from the start of the buffer is written to \c offset
 * (if specified).
 */
extern VulkanBuffer *jitc_vulkan_find_buffer(const void *ptr,
                                             size_t *offset = nullptr);

/// Is \c ptr the base address of a registered buffer?
extern bool jitc_vulkan_owns(const void *ptr);

// ---------------------------------------------------------------------
//  Utility compute pipelines
// ---------------------------------------------------------------------

/// Return the utility compute pipeline for \c kernel, creating it on first use
extern VkPipeline jitc_vulkan_pipeline(VulkanDevice &dev, VulkanKernel kernel);

/// Return the block (prefix) reduction compute pipeline for the requested
/// kernel family, operation, and type, creating it on first use (or \c nullptr
/// if unsupported)
extern VkPipeline jitc_vulkan_reduce_pipeline(VulkanDevice &dev,
                                              VulkanReduceKind kind,
                                              ReduceOp op, VarType vt);

// ---------------------------------------------------------------------
//  Ray tracing scene & native handle accessors
// ---------------------------------------------------------------------

struct VulkanScene {
    int device_id = 0;
    VkDevice device = nullptr;
    VkAccelerationStructureKHR tlas = nullptr;
    uint64_t tlas_address = 0;

    /// Host-visible shared buffer of uint64_t[isect_count], populated by
    /// jit_isect_bind() with each binding's b->record.data device address
    void *isect_table = nullptr;
    uint64_t isect_table_address = 0;
    uint32_t isect_count = 0;
    uint32_t geometry_types_mask = 0x1;

    /// Cached VarKind::Evaluated backing variables for ResourceKind::Accel and
    /// ResourceKind::IFT handles (used by RecordThreadState / dr.freeze)
    uint32_t owner_handle = 0;
    uint32_t ift_handle = 0;

    /// Optional user-supplied cleanup callback invoked when the scene expires
    void (*cleanup_callback)(void *) = nullptr;
    void *cleanup_payload = nullptr;
};

extern uint32_t jitc_vulkan_configure_scene(void *tlas,
                                            uint32_t n_isect_entries,
                                            uint32_t geometry_types_mask);
extern void jitc_vulkan_scene_set_cleanup(uint32_t scene_index,
                                          void (*callback)(void *),
                                          void *payload);
extern uint32_t jitc_vulkan_scene_owner_handle(uint32_t scene_index);
extern VulkanScene *jitc_vulkan_get_scene(uint32_t scene_index);
extern void jitc_vulkan_ray_trace(uint32_t n_args, uint32_t *args,
                                  uint32_t mask, uint32_t *out,
                                  uint32_t n_out, uint32_t scene, int shadow);

#endif // defined(DRJIT_ENABLE_VULKAN)

