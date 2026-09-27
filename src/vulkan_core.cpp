/*
    src/vulkan_core.cpp -- Vulkan backend: device initialization, command
    submission, synchronization, and memory allocation

    Copyright (c) 2026 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#include "vulkan.h"
#include "vulkan_ts.h"
#include "var.h"
#include "trace.h"
#include "io.h"
#include "unit.h"
#include "util.h"
#include "log.h"
#include "resources/vulkan_kernels.h"
#include <algorithm>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <memory>
#if !defined(_WIN32)
#  include <unistd.h>
#endif

/// Size of the shared memory chunks used by jitc_vulkan_stage()
static constexpr size_t VulkanArenaSize = 65536;


/// Submit the command buffer being recorded once it contains this many
/// commands or references this much staging memory (see
/// jitc_vulkan_maybe_submit())
static constexpr uint32_t VulkanMaxCmdCount = 1024;
static constexpr size_t VulkanMaxCmdStaging = 16 * 1024 * 1024;

/// Set when jitc_vulkan_init() failed (avoids repeated warnings). Reset by
/// jitc_vulkan_shutdown().
static bool vulkan_init_failed = false;

// ====================================================================
//                          Buffer registry
// ====================================================================

/// Registered buffers, indexed by their base address
static std::map<uintptr_t, VulkanBuffer *> vulkan_buffers;

/// Lookup table for base addresses (fast path of jitc_vulkan_find_buffer())
static tsl::robin_map<uintptr_t, VulkanBuffer *, UInt64Hasher>
    vulkan_buffers_lut;

bool jitc_vulkan_register_buffer(VulkanBuffer *buffer) {
    uintptr_t start = buffer->base, end = start + buffer->size;

    // Reject buffers whose address range overlaps with a registered buffer
    auto it = vulkan_buffers.lower_bound(start);
    if (it != vulkan_buffers.end() && it->first < end)
        return false;
    if (it != vulkan_buffers.begin()) {
        auto prev = std::prev(it);
        if (prev->first + prev->second->size > start)
            return false;
    }

    vulkan_buffers.emplace_hint(it, start, buffer);
    vulkan_buffers_lut.emplace(start, buffer);
    return true;
}

VulkanBuffer *jitc_vulkan_unregister_buffer(const void *ptr) {
    auto it = vulkan_buffers_lut.find((uintptr_t) ptr);
    if (it == vulkan_buffers_lut.end())
        return nullptr;

    VulkanBuffer *buffer = it->second;
    vulkan_buffers_lut.erase(it);
    vulkan_buffers.erase((uintptr_t) ptr);
    return buffer;
}

VulkanBuffer *jitc_vulkan_find_buffer(const void *ptr, size_t *offset) {
    uintptr_t addr = (uintptr_t) ptr;
    VulkanBuffer *buffer;

    auto it = vulkan_buffers_lut.find(addr);
    if (it != vulkan_buffers_lut.end()) {
        buffer = it->second;
    } else {
        // Find the last buffer starting at or before 'addr'
        auto it2 = vulkan_buffers.upper_bound(addr);
        if (it2 == vulkan_buffers.begin())
            return nullptr;
        --it2;
        if (addr - it2->first >= it2->second->size)
            return nullptr;
        buffer = it2->second;
    }

    if (offset)
        *offset = (size_t) (addr - buffer->base);

    return buffer;
}

bool jitc_vulkan_owns(const void *ptr) {
    return vulkan_buffers_lut.find((uintptr_t) ptr) != vulkan_buffers_lut.end();
}

// ====================================================================
//                         Memory allocation
// ====================================================================

VulkanBuffer *vulkan_buffer_new(const VulkanDevice &dev, size_t size,
                                bool shared) {
    jitc_assert(size > 0 && (size & (size - 1)) == 0,
                "vulkan_buffer_new(): buffer size (%zu) must be a power of two!", size);

    VkDevice device = (VkDevice) dev.device;

    VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                               VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                               VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                               VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    if (dev.supports_accel) {
        usage |= VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
                 VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR;
    }

    VkBufferCreateInfo buffer_info {};
    buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_info.size = size;
    buffer_info.usage = usage;
    buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkBuffer buffer = VK_NULL_HANDLE;
    if (vkCreateBuffer(device, &buffer_info, nullptr, &buffer) != VK_SUCCESS)
        return nullptr;

    VkMemoryRequirements reqs {};
    vkGetBufferMemoryRequirements(device, buffer, &reqs);

    // Try the compatible memory types in order of decreasing preference
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkMemoryAllocateFlagsInfo flags_info {};
    flags_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
    flags_info.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;

    VkMemoryAllocateInfo alloc_info {};
    alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc_info.pNext = &flags_info;
    alloc_info.allocationSize = reqs.size;

    const uint8_t *types =
        shared ? dev.shared_mem_types : dev.device_mem_types;
    uint32_t type_count =
        shared ? dev.shared_mem_type_count : dev.device_mem_type_count;

    for (uint32_t i = 0; i < type_count; ++i) {
        if (!(reqs.memoryTypeBits & (1u << types[i])))
            continue;
        alloc_info.memoryTypeIndex = types[i];
        if (vkAllocateMemory(device, &alloc_info, nullptr, &memory) ==
            VK_SUCCESS)
            break;
        memory = VK_NULL_HANDLE;
    }

    void *mapped = nullptr;
    VkDeviceAddress address = 0;
    bool success =
        memory &&
        vkBindBufferMemory(device, buffer, memory, 0) == VK_SUCCESS &&
        (!shared || vkMapMemory(device, memory, 0, VK_WHOLE_SIZE, 0,
                                &mapped) == VK_SUCCESS);

    if (success) {
        VkBufferDeviceAddressInfo address_info {};
        address_info.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
        address_info.buffer = buffer;
        address = vkGetBufferDeviceAddress(device, &address_info);
        success = address != 0;
    }

    if (success) {
        VulkanBuffer *result = new VulkanBuffer();
        result->base = shared ? (uintptr_t) mapped : (uintptr_t) address;
        result->size = size;
        result->device = device;
        result->buffer = buffer;
        result->memory = memory;
        result->address = address;
        result->shared = shared;
        return result;
    }

    // Direct allocation failed. Clean up temporary buffer & memory if any.
    if (buffer) {
        vkDestroyBuffer(device, buffer, nullptr);
        buffer = VK_NULL_HANDLE;
    }
    if (memory) {
        vkFreeMemory(device, memory, nullptr);
        memory = VK_NULL_HANDLE;
    }

    // Fall back to sparse buffer binding if supported and not shared
    if (dev.sparse_binding && !shared) {
        VkBufferCreateInfo sparse_info {};
        sparse_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        sparse_info.flags = VK_BUFFER_CREATE_SPARSE_BINDING_BIT |
                            VK_BUFFER_CREATE_SPARSE_RESIDENCY_BIT;
        sparse_info.size = size;
        sparse_info.usage = usage;
        sparse_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        if (vkCreateBuffer(device, &sparse_info, nullptr, &buffer) !=
            VK_SUCCESS)
            return nullptr;

        vkGetBufferMemoryRequirements(device, buffer, &reqs);

        uint32_t mem_type_index = UINT32_MAX;
        for (uint32_t i = 0; i < dev.device_mem_type_count; ++i) {
            if (reqs.memoryTypeBits & (1u << dev.device_mem_types[i])) {
                mem_type_index = dev.device_mem_types[i];
                break;
            }
        }
        if (mem_type_index == UINT32_MAX) {
            vkDestroyBuffer(device, buffer, nullptr);
            return nullptr;
        }

        VkDeviceSize max_chunk = dev.max_memory_allocation_size;
        if (reqs.alignment > 0)
            max_chunk = (max_chunk / reqs.alignment) * reqs.alignment;
        if (max_chunk == 0) {
            vkDestroyBuffer(device, buffer, nullptr);
            return nullptr;
        }

        std::vector<VkDeviceMemory> chunks;
        std::vector<VkSparseMemoryBind> binds;
        VkDeviceSize offset = 0;
        bool chunk_ok = true;

        while (offset < reqs.size) {
            VkDeviceSize cur_chunk = std::min(max_chunk, reqs.size - offset);

            VkMemoryAllocateFlagsInfo chunk_flags {};
            chunk_flags.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
            chunk_flags.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;

            VkMemoryAllocateInfo chunk_alloc {};
            chunk_alloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
            chunk_alloc.pNext = &chunk_flags;
            chunk_alloc.allocationSize = cur_chunk;
            chunk_alloc.memoryTypeIndex = mem_type_index;

            VkDeviceMemory chunk_mem = VK_NULL_HANDLE;
            if (vkAllocateMemory(device, &chunk_alloc, nullptr, &chunk_mem) !=
                VK_SUCCESS) {
                chunk_ok = false;
                break;
            }
            chunks.push_back(chunk_mem);

            VkSparseMemoryBind bind {};
            bind.resourceOffset = offset;
            bind.size = cur_chunk;
            bind.memory = chunk_mem;
            bind.memoryOffset = 0;
            bind.flags = 0;
            binds.push_back(bind);

            offset += cur_chunk;
        }

        if (chunk_ok) {
            VkSparseBufferMemoryBindInfo buf_bind {};
            buf_bind.buffer = buffer;
            buf_bind.bindCount = (uint32_t) binds.size();
            buf_bind.pBinds = binds.data();

            VkBindSparseInfo bind_info {};
            bind_info.sType = VK_STRUCTURE_TYPE_BIND_SPARSE_INFO;
            bind_info.bufferBindCount = 1;
            bind_info.pBufferBinds = &buf_bind;

            VkResult rv;
            {
                lock_guard guard(state.lock);
                rv = vkQueueBindSparse((VkQueue) dev.queue, 1, &bind_info,
                                       VK_NULL_HANDLE);
                if (rv == VK_SUCCESS)
                    rv = vkQueueWaitIdle((VkQueue) dev.queue);
            }

            if (rv == VK_SUCCESS) {
                VkBufferDeviceAddressInfo address_info {};
                address_info.sType =
                    VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
                address_info.buffer = buffer;
                address = vkGetBufferDeviceAddress(device, &address_info);
            }
        }

        if (address != 0) {
            VulkanBuffer *result = new VulkanBuffer();
            result->base = (uintptr_t) address;
            result->size = size;
            result->device = device;
            result->buffer = buffer;
            result->memory = VK_NULL_HANDLE;
            result->address = address;
            result->shared = false;
            result->sparse_memories = std::move(chunks);
            return result;
        }

        vkDestroyBuffer(device, buffer, nullptr);
        for (VkDeviceMemory mem : chunks)
            vkFreeMemory(device, mem, nullptr);
    }

    return nullptr;
}

void vulkan_buffer_free(VulkanBuffer *buffer) {
    vkDestroyBuffer(buffer->device, buffer->buffer, nullptr);
    if (buffer->memory)
        vkFreeMemory(buffer->device, buffer->memory, nullptr); // (also unmaps)
    for (VkDeviceMemory mem : buffer->sparse_memories)
        vkFreeMemory(buffer->device, mem, nullptr);
    delete buffer;
}

// ====================================================================
//                    Command submission & synchronization
// ====================================================================

/// Retire the submissions whose timeline value is <= 'value'
static void vulkan_retire_until(VulkanDevice &dev, uint64_t value) {
    VulkanStream &s = *dev.stream;
    if (value > s.completed_value)
        s.completed_value = value;

    while (!s.in_flight.empty() &&
           s.in_flight.front().value <= s.completed_value) {
        VulkanSubmission &sub = s.in_flight.front();
        s.cmd_free.push_back(sub.cmd);
        for (auto [info, ptr] : sub.frees)
            jitc_malloc_release(info, ptr);
        s.ready_callbacks.insert(s.ready_callbacks.end(),
                                 sub.callbacks.begin(), sub.callbacks.end());
        s.in_flight.pop_front();
    }
}

void jitc_vulkan_retire(VulkanDevice &dev) {
    if (dev.stream->in_flight.empty())
        return;

    uint64_t value = 0;
    vk_check(vkGetSemaphoreCounterValue((VkDevice) dev.device,
                                        (VkSemaphore) dev.timeline, &value));
    vulkan_retire_until(dev, value);
}

VkCommandBuffer jitc_vulkan_cmd(VulkanDevice &dev) {
    VulkanStream &s = *dev.stream;
    if (s.cmd)
        return s.cmd;

    // Recycle the command buffers of finished submissions
    jitc_vulkan_retire(dev);

    VkCommandBuffer cmd;
    if (!s.cmd_free.empty()) {
        cmd = s.cmd_free.back();
        s.cmd_free.pop_back();
    } else {
        VkCommandBufferAllocateInfo alloc_info {};
        alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        alloc_info.commandPool = s.pool;
        alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        alloc_info.commandBufferCount = 1;
        vk_check(vkAllocateCommandBuffers((VkDevice) dev.device, &alloc_info,
                                          &cmd));
    }

    // The command pool permits resetting individual command buffers, hence
    // vkBeginCommandBuffer() implicitly resets recycled ones
    VkCommandBufferBeginInfo begin_info {};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VkResult rv = vkBeginCommandBuffer(cmd, &begin_info);
    if (rv != VK_SUCCESS) {
        s.cmd_free.push_back(cmd);
        jitc_vulkan_raise(rv, __FILE__, __LINE__);
    }

    s.cmd = cmd;
    s.cmd_count = 0;
    s.barrier_pending = true;
    s.write_pending = false;
    return cmd;
}

void jitc_vulkan_access(VulkanDevice &dev, bool is_write) {
    VulkanStream &s = *dev.stream;

    if (s.barrier_pending || s.write_pending) {
        /* A global memory barrier that orders all preceding commands before
           all subsequent ones. At the beginning of a command buffer, this
           also orders the commands after those of earlier submissions. */
        VkMemoryBarrier mb {};
        mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        mb.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
        vkCmdPipelineBarrier(s.cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                             VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &mb, 0,
                             nullptr, 0, nullptr);
        s.barrier_pending = false;
        s.write_pending = false;
    }

    if (is_write)
        s.write_pending = true;

    s.cmd_count++;
}

void jitc_vulkan_submit(VulkanDevice &dev) {
    VulkanStream &s = *dev.stream;
    if (!s.cmd)
        return;

    // The staging memory of the arena is only referenced by the commands
    // recorded so far. Release it once they have finished.
    if (s.arena) {
        uint8_t *arena = s.arena;
        s.arena = nullptr;
        s.arena_offset = 0;
        jitc_free(arena);
    }

    VkCommandBuffer cmd = s.cmd;

    // Make the results visible to the host
    VkMemoryBarrier mb {};
    mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    mb.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_HOST_READ_BIT | VK_ACCESS_HOST_WRITE_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                         VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &mb, 0, nullptr, 0,
                         nullptr);

    s.cmd = nullptr;
    s.cmd_count = 0;
    s.cmd_staging = 0;
    s.write_pending = false;
    s.barrier_pending = true;

    uint64_t value = dev.timeline_value + 1;
    VkResult rv = vkEndCommandBuffer(cmd);

    if (rv == VK_SUCCESS) {
        VkSemaphore timeline = (VkSemaphore) dev.timeline;

        VkTimelineSemaphoreSubmitInfo timeline_info {};
        timeline_info.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
        timeline_info.signalSemaphoreValueCount = 1;
        timeline_info.pSignalSemaphoreValues = &value;

        VkSubmitInfo submit_info {};
        submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit_info.pNext = &timeline_info;
        submit_info.commandBufferCount = 1;
        submit_info.pCommandBuffers = &cmd;
        submit_info.signalSemaphoreCount = 1;
        submit_info.pSignalSemaphores = &timeline;

        rv = vkQueueSubmit((VkQueue) dev.queue, 1, &submit_info,
                           VK_NULL_HANDLE);
    }

    if (rv != VK_SUCCESS) {
        /* The recorded work is lost. Recycle the command buffer and release
           the associated resources once the previous work has finished. */
        s.cmd_free.push_back(cmd);
        if (!s.in_flight.empty()) {
            VulkanSubmission &last = s.in_flight.back();
            last.frees.insert(last.frees.end(), s.pending_frees.begin(),
                              s.pending_frees.end());
            last.callbacks.insert(last.callbacks.end(),
                                  s.pending_callbacks.begin(),
                                  s.pending_callbacks.end());
        } else {
            for (auto [info, ptr] : s.pending_frees)
                jitc_malloc_release(info, ptr);
            s.ready_callbacks.insert(s.ready_callbacks.end(),
                                     s.pending_callbacks.begin(),
                                     s.pending_callbacks.end());
        }
        s.pending_frees.clear();
        s.pending_callbacks.clear();
        jitc_vulkan_raise(rv, __FILE__, __LINE__);
    }

    VulkanSubmission sub;
    sub.cmd = cmd;
    sub.value = value;
    sub.frees.swap(s.pending_frees);
    sub.callbacks.swap(s.pending_callbacks);
    s.in_flight.push_back(std::move(sub));
    dev.timeline_value = value;
}

void jitc_vulkan_maybe_submit(VulkanDevice &dev) {
    const VulkanStream &s = *dev.stream;
    if (s.cmd && (s.cmd_count >= VulkanMaxCmdCount ||
                  s.cmd_staging >= VulkanMaxCmdStaging))
        jitc_vulkan_submit(dev);
}

void jitc_vulkan_wait(VulkanDevice &dev, uint64_t value, bool release_lock) {
    if (value <= dev.stream->completed_value)
        return;

    VkSemaphore timeline = (VkSemaphore) dev.timeline;
    VkSemaphoreWaitInfo wait_info {};
    wait_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
    wait_info.semaphoreCount = 1;
    wait_info.pSemaphores = &timeline;
    wait_info.pValues = &value;

    VkResult rv;
    if (release_lock) {
        unlock_guard guard(state.lock);
        rv = vkWaitSemaphores((VkDevice) dev.device, &wait_info, UINT64_MAX);
    } else {
        rv = vkWaitSemaphores((VkDevice) dev.device, &wait_info, UINT64_MAX);
    }

    if (rv != VK_SUCCESS)
        jitc_vulkan_raise(rv, __FILE__, __LINE__);

    vulkan_retire_until(dev, value);
}

void jitc_vulkan_run_callbacks(VulkanDevice &dev) {
    VulkanStream &s = *dev.stream;

    /* Callbacks run without holding the Dr.Jit lock, since they may call
       into Dr.Jit (possibly from other threads). Consequently, the callbacks
       of concurrent synchronizations of different threads may run
       concurrently as well. */
    while (!s.ready_callbacks.empty()) {
        std::vector<VulkanCallback> callbacks;
        callbacks.swap(s.ready_callbacks);

        size_t i = 0;
        try {
            unlock_guard guard(state.lock);
            for (; i < callbacks.size(); ++i)
                callbacks[i].func(callbacks[i].payload);
        } catch (...) {
            // Keep the remaining callbacks for a later synchronization
            s.ready_callbacks.insert(s.ready_callbacks.begin(),
                                     callbacks.begin() + i + 1,
                                     callbacks.end());
            throw;
        }
    }
}

void jitc_vulkan_enqueue_host_func(VulkanDevice &dev, void (*func)(void *),
                                   void *payload) {
    VulkanStream &s = *dev.stream;
    jitc_vulkan_retire(dev);

    VulkanCallback cb { func, payload };
    if (s.cmd) {
        s.pending_callbacks.push_back(cb);
    } else if (!s.in_flight.empty()) {
        s.in_flight.back().callbacks.push_back(cb);
    } else {
        // No unfinished work. Run the callback right away (after any other
        // callbacks that are ready to run)
        s.ready_callbacks.push_back(cb);
        jitc_vulkan_run_callbacks(dev);
    }
}

void jitc_vulkan_free_later(AllocInfo info, void *ptr) {
    auto [size, backend, shared, device] = alloc_info_decode(info);
    (void) size; (void) backend; (void) shared;

    VulkanStream &s = *state.vulkan_devices[device].stream;
    if (s.cmd)
        s.pending_frees.emplace_back(info, ptr);
    else if (!s.in_flight.empty())
        s.in_flight.back().frees.emplace_back(info, ptr);
    else
        jitc_malloc_release(info, ptr);
}

uint8_t *jitc_vulkan_stage(VulkanDevice &dev, size_t size) {
    VulkanStream &s = *dev.stream;
    size = (size + 15) / 16 * 16;

    if (size > VulkanMaxStageSize)
        jitc_fail("jitc_vulkan_stage(): requested too much memory!");

    while (!s.arena || s.arena_offset + size > VulkanArenaSize) {
        if (s.arena) {
            // Release the arena once the work referencing it has finished
            uint8_t *arena = s.arena;
            s.arena = nullptr;
            jitc_free(arena);
        }

        // This may temporarily release the lock
        uint8_t *arena =
            (uint8_t *) jitc_malloc(JitBackend::Vulkan, VulkanArenaSize, true);

        if (s.arena) {
            // Another thread installed an arena in the meantime
            jitc_free(arena);
            continue;
        }

        s.arena = arena;
        s.arena_offset = 0;
    }

    uint8_t *result = s.arena + s.arena_offset;
    s.arena_offset += size;
    s.cmd_staging += size;
    return result;
}

void jitc_vulkan_flush() {
    for (VulkanDevice &dev : state.vulkan_devices) {
        jitc_vulkan_submit(dev);
        jitc_vulkan_retire(dev);
    }
}

void jitc_vulkan_sync(bool release_lock) {
    for (VulkanDevice &dev : state.vulkan_devices) {
        jitc_vulkan_submit(dev);
        jitc_vulkan_wait(dev, dev.timeline_value, release_lock);
        if (release_lock)
            jitc_vulkan_run_callbacks(dev);
    }
}

void jitc_vulkan_sync_all() {
    for (VulkanDevice &dev : state.vulkan_devices) {
        VulkanStream &s = *dev.stream;

        /* Host callbacks may enqueue further work, hence repeat until the
           stream is idle. Afterwards, all allocations released via
           jitc_vulkan_free_later() have been returned to the cache. */
        while (true) {
            if (s.arena && !s.cmd) {
                uint8_t *arena = s.arena;
                s.arena = nullptr;
                s.arena_offset = 0;
                jitc_free(arena);
            }

            jitc_vulkan_submit(dev);

            if (s.in_flight.empty() && s.ready_callbacks.empty() && !s.cmd &&
                !s.arena)
                break;

            jitc_vulkan_wait(dev, dev.timeline_value, true);
            jitc_vulkan_run_callbacks(dev);
        }
    }
}

// ====================================================================
//                    Device initialization & shutdown
// ====================================================================

/// Information about a physical device gathered by vulkan_query_device()
struct VulkanDeviceInfo {
    VkPhysicalDevice handle = nullptr;
    VkPhysicalDeviceProperties props {};
    VkPhysicalDeviceVulkan11Properties props11 {};
    VkPhysicalDeviceVulkan12Properties props12 {};
    VkPhysicalDeviceVulkan13Properties props13 {};
    VkPhysicalDeviceFeatures2 features {};
    VkPhysicalDeviceVulkan11Features features11 {};
    VkPhysicalDeviceVulkan12Features features12 {};
    VkPhysicalDeviceVulkan13Features features13 {};
    VkPhysicalDeviceShaderAtomicFloatFeaturesEXT atomic_float {};
    VkPhysicalDeviceAccelerationStructureFeaturesKHR accel {};
    VkPhysicalDeviceRayQueryFeaturesKHR ray_query {};
    VkPhysicalDeviceMemoryProperties mem_props {};

    /// Index of the first queue family supporting compute (or UINT32_MAX)
    uint32_t queue_family = UINT32_MAX;
    uint32_t queue_flags = 0;

    /// Supported device extensions
    bool has_portability_subset = false, has_atomic_float = false,
         has_accel = false, has_ray_query = false,
         has_deferred_host_operations = false;

    /// Memory types for device-local and shared allocations (see
    /// vulkan_rank_memory_types())
    uint8_t device_mem_types[32] { }, shared_mem_types[32] { };
    uint32_t device_mem_type_count = 0, shared_mem_type_count = 0;

    /// Size of the largest device-local memory heap
    uint64_t device_memory = 0;

    /// Reason why the device is incompatible with Dr.Jit (or nullptr)
    const char *reason = nullptr;
};

/// Rank the memory types of a device for device-local and shared allocations
static void vulkan_rank_memory_types(VulkanDeviceInfo &info) {
    const VkPhysicalDeviceMemoryProperties &mp = info.mem_props;
    const VkMemoryPropertyFlags excluded =
        VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT |
        VK_MEMORY_PROPERTY_PROTECTED_BIT |
        VK_MEMORY_PROPERTY_DEVICE_COHERENT_BIT_AMD |
        VK_MEMORY_PROPERTY_DEVICE_UNCACHED_BIT_AMD;

    struct Candidate {
        uint32_t index, score;
        VkDeviceSize heap_size;
    };

    std::vector<Candidate> device_types, shared_types;
    for (uint32_t i = 0; i < mp.memoryTypeCount && i < 32; ++i) {
        VkMemoryPropertyFlags flags = mp.memoryTypes[i].propertyFlags;
        VkDeviceSize heap_size =
            mp.memoryHeaps[mp.memoryTypes[i].heapIndex].size;
        if (flags & excluded)
            continue;

        bool device_local = flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
             host_visible = flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
             host_coherent = flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
             host_cached = flags & VK_MEMORY_PROPERTY_HOST_CACHED_BIT;

        // Device-local memory: prefer memory that is not host-visible (e.g.,
        // avoid the small PCIe BAR window of discrete GPUs)
        if (device_local)
            device_types.push_back({ i, host_visible ? 1u : 0u, heap_size });

        // Shared memory: prefer cached system memory, which is fast to read
        // from the host
        if (host_visible && host_coherent)
            shared_types.push_back(
                { i, (device_local ? 2u : 0u) + (host_cached ? 0u : 1u),
                  heap_size });

        if (device_local && heap_size > info.device_memory)
            info.device_memory = heap_size;
    }

    auto compare = [](const Candidate &a, const Candidate &b) {
        if (a.score != b.score)
            return a.score < b.score;
        if (a.heap_size != b.heap_size)
            return a.heap_size > b.heap_size;
        return a.index < b.index;
    };

    std::sort(device_types.begin(), device_types.end(), compare);
    std::sort(shared_types.begin(), shared_types.end(), compare);

    for (const Candidate &c : device_types)
        info.device_mem_types[info.device_mem_type_count++] = (uint8_t) c.index;
    for (const Candidate &c : shared_types)
        info.shared_mem_types[info.shared_mem_type_count++] = (uint8_t) c.index;
}

/// Query the properties and features of a physical device and check whether
/// it is compatible with Dr.Jit
static void vulkan_query_device(VkPhysicalDevice handle,
                                VulkanDeviceInfo &info) {
    info.handle = handle;
    vkGetPhysicalDeviceProperties(handle, &info.props);
    vkGetPhysicalDeviceMemoryProperties(handle, &info.mem_props);
    vulkan_rank_memory_types(info);

    if (info.props.apiVersion < VK_API_VERSION_1_3) {
        info.reason = "Vulkan 1.3 is not supported";
        return;
    }

    // Device extensions
    uint32_t n_extensions = 0;
    std::vector<VkExtensionProperties> extensions;
    VkResult rv = vkEnumerateDeviceExtensionProperties(handle, nullptr,
                                                       &n_extensions, nullptr);
    if (rv == VK_SUCCESS && n_extensions > 0) {
        extensions.resize(n_extensions);
        rv = vkEnumerateDeviceExtensionProperties(handle, nullptr, &n_extensions,
                                                  extensions.data());
        if (rv != VK_SUCCESS && rv != VK_INCOMPLETE)
            n_extensions = 0;
        extensions.resize(n_extensions);
    }

    for (const VkExtensionProperties &ext : extensions) {
        const char *name = ext.extensionName;
        if (strcmp(name, VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME) == 0)
            info.has_portability_subset = true;
        else if (strcmp(name, VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME) == 0)
            info.has_atomic_float = true;
        else if (strcmp(name, VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME) == 0)
            info.has_accel = true;
        else if (strcmp(name, VK_KHR_RAY_QUERY_EXTENSION_NAME) == 0)
            info.has_ray_query = true;
        else if (strcmp(name, VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME) == 0)
            info.has_deferred_host_operations = true;
    }

    // Properties
    VkPhysicalDeviceProperties2 props2 {};
    props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    props2.pNext = &info.props11;
    info.props11.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_PROPERTIES;
    info.props11.pNext = &info.props12;
    info.props12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_PROPERTIES;
    info.props12.pNext = &info.props13;
    info.props13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_PROPERTIES;
    info.props13.pNext = nullptr;
    vkGetPhysicalDeviceProperties2(handle, &props2);

    // Features
    void **next = &info.features.pNext;
    info.features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    info.features11.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
    info.features12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    info.features13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
    *next = &info.features11; next = &info.features11.pNext;
    *next = &info.features12; next = &info.features12.pNext;
    *next = &info.features13; next = &info.features13.pNext;

    if (info.has_atomic_float) {
        info.atomic_float.sType =
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT;
        *next = &info.atomic_float; next = &info.atomic_float.pNext;
    }

    if (info.has_accel) {
        info.accel.sType =
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR;
        *next = &info.accel; next = &info.accel.pNext;
    }

    if (info.has_ray_query) {
        info.ray_query.sType =
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR;
        *next = &info.ray_query; next = &info.ray_query.pNext;
    }

    *next = nullptr;
    vkGetPhysicalDeviceFeatures2(handle, &info.features);

    // The feature/property structures remain valid, but their chains don't
    info.features.pNext = info.features11.pNext = info.features12.pNext =
        info.features13.pNext = info.atomic_float.pNext = info.accel.pNext =
        info.ray_query.pNext = info.props11.pNext = info.props12.pNext =
        info.props13.pNext = nullptr;

    // Find a queue family supporting compute
    uint32_t n_families = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(handle, &n_families, nullptr);
    std::vector<VkQueueFamilyProperties> families(n_families);
    vkGetPhysicalDeviceQueueFamilyProperties(handle, &n_families,
                                             families.data());
    for (uint32_t i = 0; i < n_families; ++i) {
        if ((families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) &&
            families[i].queueCount > 0) {
            info.queue_family = i;
            info.queue_flags = families[i].queueFlags;
            break;
        }
    }

    if (info.queue_family == UINT32_MAX)
        info.reason = "no queue family supports compute operations";
    else if (!info.features12.bufferDeviceAddress)
        info.reason = "the 'bufferDeviceAddress' feature is not supported";
    else if (!info.features12.timelineSemaphore)
        info.reason = "the 'timelineSemaphore' feature is not supported";
    else if (!info.features12.scalarBlockLayout)
        info.reason = "the 'scalarBlockLayout' feature is not supported";
    else if (!info.features.features.shaderInt64)
        info.reason = "the 'shaderInt64' feature is not supported";
    else if (!info.features13.maintenance4)
        info.reason = "the 'maintenance4' feature is not supported";
    else if (info.device_mem_type_count == 0)
        info.reason = "no device-local memory type";
    else if (info.shared_mem_type_count == 0)
        info.reason = "no host-visible and host-coherent memory type";
}

static const char *vulkan_device_type_name(uint32_t type) {
    switch (type) {
        case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: return "integrated GPU";
        case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU: return "discrete GPU";
        case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU: return "virtual GPU";
        case VK_PHYSICAL_DEVICE_TYPE_CPU: return "CPU";
        default: return "other";
    }
}

/// Preference of device types when selecting a device (lower is better)
static uint32_t vulkan_device_type_rank(uint32_t type) {
    switch (type) {
        case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU: return 0;
        case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: return 1;
        case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU: return 2;
        case VK_PHYSICAL_DEVICE_TYPE_CPU: return 4;
        default: return 3;
    }
}

/// Log a warning and disable the Vulkan backend
#if defined(__GNUC__)
    __attribute__((__format__ (__printf__, 1, 2)))
#endif
static bool vulkan_init_fail(const char *fmt, ...) {
    char buf[512];
    va_list args;
    va_start(args, fmt);
    vsnprintf(buf, sizeof(buf), fmt, args);
    va_end(args);
    jitc_log(Warn, "jit_vulkan_init(): %s -- disabling the Vulkan backend!",
             buf);
    vulkan_init_failed = true;
    return false;
}

/// Create a logical device, queue, timeline semaphore, and command pool
static bool vulkan_create_device(const VulkanDeviceInfo &info, int index) {
    float priority = 1.f;
    VkDeviceQueueCreateInfo queue_info {};
    queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_info.queueFamilyIndex = info.queue_family;
    queue_info.queueCount = 1;
    queue_info.pQueuePriorities = &priority;

    // Enable the required features and the supported optional ones
    VkPhysicalDeviceFeatures2 features {};
    VkPhysicalDeviceVulkan11Features features11 {};
    VkPhysicalDeviceVulkan12Features features12 {};
    VkPhysicalDeviceVulkan13Features features13 {};
    VkPhysicalDeviceShaderAtomicFloatFeaturesEXT atomic_float {};
    VkPhysicalDeviceAccelerationStructureFeaturesKHR accel {};
    VkPhysicalDeviceRayQueryFeaturesKHR ray_query {};
    features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    features11.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
    features12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    features13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
    atomic_float.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT;
    accel.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR;
    ray_query.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR;
    features.pNext = &features11;
    features11.pNext = &features12;
    features12.pNext = &features13;
    void **next = &features13.pNext;

    const VkPhysicalDeviceFeatures &f = info.features.features;
    const VkPhysicalDeviceVulkan11Features &f11 = info.features11;
    const VkPhysicalDeviceVulkan12Features &f12 = info.features12;
    const VkPhysicalDeviceShaderAtomicFloatFeaturesEXT &faf = info.atomic_float;

    features.features.shaderInt64 = VK_TRUE;
    features.features.shaderInt16 = f.shaderInt16;
    features.features.shaderFloat64 = f.shaderFloat64;
    bool sparse_binding = f.sparseBinding && f.sparseResidencyBuffer &&
                          (info.queue_flags & VK_QUEUE_SPARSE_BINDING_BIT);
    if (sparse_binding) {
        features.features.sparseBinding = VK_TRUE;
        features.features.sparseResidencyBuffer = VK_TRUE;
    }
    features11.storageBuffer16BitAccess = f11.storageBuffer16BitAccess;
    features12.bufferDeviceAddress = VK_TRUE;
    features12.timelineSemaphore = VK_TRUE;
    features12.scalarBlockLayout = VK_TRUE;
    features12.shaderInt8 = f12.shaderInt8;
    features12.shaderFloat16 = f12.shaderFloat16;
    features12.storageBuffer8BitAccess = f12.storageBuffer8BitAccess;
    features12.shaderBufferInt64Atomics = f12.shaderBufferInt64Atomics;
    features12.shaderSubgroupExtendedTypes = f12.shaderSubgroupExtendedTypes;
    features12.vulkanMemoryModel = f12.vulkanMemoryModel;
    features12.vulkanMemoryModelDeviceScope = f12.vulkanMemoryModelDeviceScope;
    features13.maintenance4 = VK_TRUE;
    features13.synchronization2 = info.features13.synchronization2;

    const char *extensions[8] = { };
    uint32_t n_extensions = 0;

    // Non-conformant implementations (e.g. MoltenVK) require this extension
    if (info.has_portability_subset)
        extensions[n_extensions++] = VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME;

    bool atomic_float_add =
        info.has_atomic_float && (faf.shaderBufferFloat32AtomicAdd ||
                                  faf.shaderBufferFloat64AtomicAdd);
    if (atomic_float_add) {
        extensions[n_extensions++] = VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME;
        atomic_float.shaderBufferFloat32Atomics = faf.shaderBufferFloat32Atomics;
        atomic_float.shaderBufferFloat32AtomicAdd =
            faf.shaderBufferFloat32AtomicAdd;
        atomic_float.shaderBufferFloat64Atomics = faf.shaderBufferFloat64Atomics;
        atomic_float.shaderBufferFloat64AtomicAdd =
            faf.shaderBufferFloat64AtomicAdd;
        *next = &atomic_float;
        next = &atomic_float.pNext;
    }

    bool supports_accel = jitc_vulkan_has_accel_api && info.has_accel &&
                          info.has_deferred_host_operations &&
                          info.accel.accelerationStructure;
    bool supports_ray_query =
        supports_accel && info.has_ray_query && info.ray_query.rayQuery;

    if (supports_accel) {
        extensions[n_extensions++] =
            VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME;
        extensions[n_extensions++] =
            VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME;
        accel.accelerationStructure = VK_TRUE;
        *next = &accel;
        next = &accel.pNext;
    }

    if (supports_ray_query) {
        extensions[n_extensions++] = VK_KHR_RAY_QUERY_EXTENSION_NAME;
        ray_query.rayQuery = VK_TRUE;
        *next = &ray_query;
        next = &ray_query.pNext;
    }

    VkDeviceCreateInfo device_info {};
    device_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    device_info.pNext = &features;
    device_info.queueCreateInfoCount = 1;
    device_info.pQueueCreateInfos = &queue_info;
    device_info.enabledExtensionCount = n_extensions;
    device_info.ppEnabledExtensionNames = extensions;

    VkDevice device = VK_NULL_HANDLE;
    VkResult rv = vkCreateDevice(info.handle, &device_info, nullptr, &device);
    if (rv != VK_SUCCESS)
        return vulkan_init_fail("vkCreateDevice() failed: %s",
                                jitc_vulkan_error_name(rv));

    VkQueue queue = VK_NULL_HANDLE;
    vkGetDeviceQueue(device, info.queue_family, 0, &queue);

    // Timeline semaphore signaled by every submission
    VkSemaphoreTypeCreateInfo semaphore_type_info {};
    semaphore_type_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
    semaphore_type_info.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    semaphore_type_info.initialValue = 0;

    VkSemaphoreCreateInfo semaphore_info {};
    semaphore_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    semaphore_info.pNext = &semaphore_type_info;

    VkSemaphore timeline = VK_NULL_HANDLE;
    rv = vkCreateSemaphore(device, &semaphore_info, nullptr, &timeline);
    if (rv != VK_SUCCESS) {
        vkDestroyDevice(device, nullptr);
        return vulkan_init_fail("vkCreateSemaphore() failed: %s",
                                jitc_vulkan_error_name(rv));
    }

    VkCommandPoolCreateInfo pool_info {};
    pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pool_info.queueFamilyIndex = info.queue_family;

    VkCommandPool pool = VK_NULL_HANDLE;
    rv = vkCreateCommandPool(device, &pool_info, nullptr, &pool);
    if (rv != VK_SUCCESS) {
        vkDestroySemaphore(device, timeline, nullptr);
        vkDestroyDevice(device, nullptr);
        return vulkan_init_fail("vkCreateCommandPool() failed: %s",
                                jitc_vulkan_error_name(rv));
    }

    const VkPhysicalDeviceLimits &limits = info.props.limits;

    // Shared compute pipeline layout with push constant range covering the
    // device's maximum push constant size
    VkPushConstantRange pc_range {};
    pc_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pc_range.offset = 0;
    pc_range.size = limits.maxPushConstantsSize;

    VkPipelineLayoutCreateInfo layout_info {};
    layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layout_info.pushConstantRangeCount = 1;
    layout_info.pPushConstantRanges = &pc_range;

    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
    rv = vkCreatePipelineLayout(device, &layout_info, nullptr,
                                &pipeline_layout);
    if (rv != VK_SUCCESS) {
        vkDestroyCommandPool(device, pool, nullptr);
        vkDestroySemaphore(device, timeline, nullptr);
        vkDestroyDevice(device, nullptr);
        return vulkan_init_fail("vkCreatePipelineLayout() failed: %s",
                                jitc_vulkan_error_name(rv));
    }

    // Load persistent driver pipeline cache from ~/.drjit/ if available
    std::vector<uint8_t> cache_data;
    if (const char *dir = jitc_cache_dir()) {
        const uint8_t *u = info.props.pipelineCacheUUID;
        char name[64];
        snprintf(name, sizeof(name),
                 "vulkan_pipeline_cache_%02x%02x%02x%02x%02x%02x%02x%02x"
                 "%02x%02x%02x%02x%02x%02x%02x%02x.bin",
                 u[0], u[1], u[2], u[3], u[4], u[5], u[6], u[7],
                 u[8], u[9], u[10], u[11], u[12], u[13], u[14], u[15]);
        std::filesystem::path cache_path = std::filesystem::path(dir) / name;
        std::ifstream file(cache_path, std::ios::binary | std::ios::ate);
        if (file) {
            std::streamsize fsize = file.tellg();
            if (fsize > 0 && fsize <= (std::streamsize) (64 * 1024 * 1024)) {
                cache_data.resize((size_t) fsize);
                file.seekg(0);
                if (!file.read((char *) cache_data.data(), fsize))
                    cache_data.clear();
            }
        }
    }

    VkPipelineCacheCreateInfo cache_info {};
    cache_info.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
    cache_info.initialDataSize = cache_data.size();
    cache_info.pInitialData = cache_data.empty() ? nullptr : cache_data.data();

    VkPipelineCache pipeline_cache = VK_NULL_HANDLE;
    rv = vkCreatePipelineCache(device, &cache_info, nullptr, &pipeline_cache);
    if (rv != VK_SUCCESS && !cache_data.empty()) {
        cache_info.initialDataSize = 0;
        cache_info.pInitialData = nullptr;
        rv = vkCreatePipelineCache(device, &cache_info, nullptr,
                                   &pipeline_cache);
    }
    if (rv != VK_SUCCESS)
        pipeline_cache = VK_NULL_HANDLE;

    VulkanDevice dev;
    dev.physical_device = info.handle;
    dev.device = device;
    dev.queue = queue;
    dev.timeline = timeline;
    dev.pipeline_layout = pipeline_layout;
    dev.pipeline_cache = pipeline_cache;
    dev.timeline_value = 0;
    dev.stream = new VulkanStream();
    dev.stream->pool = pool;
    dev.index = index;
    dev.queue_family = info.queue_family;
    dev.api_version = std::min(info.props.apiVersion, jitc_vulkan_api_version);
    dev.vendor_id = info.props.vendorID;
    dev.device_id = info.props.deviceID;
    dev.driver_version = info.props.driverVersion;
    dev.device_type = (uint32_t) info.props.deviceType;
    memcpy(dev.pipeline_cache_uuid, info.props.pipelineCacheUUID,
           sizeof(dev.pipeline_cache_uuid));
    dev.subgroup_size = info.props11.subgroupSize;
    dev.max_push_constants_size = limits.maxPushConstantsSize;
    dev.max_compute_shared_memory_size = limits.maxComputeSharedMemorySize;
    dev.max_compute_work_group_invocations =
        limits.maxComputeWorkGroupInvocations;
    for (int i = 0; i < 3; ++i) {
        dev.max_compute_work_group_count[i] = limits.maxComputeWorkGroupCount[i];
        dev.max_compute_work_group_size[i] = limits.maxComputeWorkGroupSize[i];
    }
    dev.max_memory_allocation_count = limits.maxMemoryAllocationCount;
    dev.max_storage_buffer_range = limits.maxStorageBufferRange;
    dev.max_memory_allocation_size = info.props11.maxMemoryAllocationSize;
    dev.max_buffer_size = info.props13.maxBufferSize;
    dev.sparse_binding = sparse_binding;
    memcpy(dev.device_mem_types, info.device_mem_types,
           sizeof(dev.device_mem_types));
    memcpy(dev.shared_mem_types, info.shared_mem_types,
           sizeof(dev.shared_mem_types));
    dev.device_mem_type_count = info.device_mem_type_count;
    dev.shared_mem_type_count = info.shared_mem_type_count;

    dev.shader_int8 = f12.shaderInt8;
    dev.shader_int16 = f.shaderInt16;
    dev.shader_float16 = f12.shaderFloat16;
    dev.shader_float64 = f.shaderFloat64;
    dev.storage_buffer_8bit = f12.storageBuffer8BitAccess;
    dev.storage_buffer_16bit = f11.storageBuffer16BitAccess;
    dev.shader_int64_atomics = f12.shaderBufferInt64Atomics;
    dev.shader_float32_atomic_add =
        atomic_float_add && faf.shaderBufferFloat32AtomicAdd;
    dev.shader_float64_atomic_add =
        atomic_float_add && faf.shaderBufferFloat64AtomicAdd;
    dev.vulkan_memory_model = f12.vulkanMemoryModel;
    dev.supports_accel = supports_accel;
    dev.supports_ray_query = supports_ray_query;
    dev.name = strdup(info.props.deviceName);

    state.vulkan_devices.push_back(dev);

    jitc_log(Info,
             "jit_vulkan_init(): using device %i (\"%s\", Vulkan %u.%u, "
             "subgroup size %u, %s max. allocation)%s%s%s%s%s%s%s%s%s%s",
             index, dev.name, VK_API_VERSION_MAJOR(dev.api_version),
             VK_API_VERSION_MINOR(dev.api_version), dev.subgroup_size,
             jitc_mem_string(dev.max_memory_allocation_size).c_str(),
             dev.shader_int8 ? ", int8" : "",
             dev.shader_int16 ? ", int16" : "",
             dev.shader_float16 ? ", fp16" : "",
             dev.shader_float64 ? ", fp64" : "",
             dev.shader_int64_atomics ? ", int64 atomics" : "",
             dev.shader_float32_atomic_add ? ", fp32 atomic add" : "",
             dev.shader_float64_atomic_add ? ", fp64 atomic add" : "",
             dev.supports_accel ? ", accel. structures" : "",
             dev.supports_ray_query ? ", ray queries" : "",
             dev.sparse_binding ? ", sparse binding" : "");

    return true;
}

// ====================================================================
//                   Precompiled utility compute pipelines
// ====================================================================

/// Decompressed builtin SPIR-V utility kernel blob
static char *jitc_vulkan_kernels = nullptr;

static const char *vulkan_kernel_names[(uint32_t) VulkanKernel::Count] = {
    "compress_scatter",
    "mkperm_phase_1",
    "mkperm_phase_3",
    "mkperm_detect_offsets",
    "mkperm_phase_1_tiny",
    "mkperm_phase_4_tiny"
};

static VkPipeline vulkan_create_utility_pipeline(VulkanDevice &dev,
                                                 const char *name) {
    if (!jitc_vulkan_kernels)
        return VK_NULL_HANDLE;

    size_t name_len = strlen(name);
    const char *p = vulkan_kernels_list;
    uint32_t idx = 0;
    const uint32_t *code = nullptr;
    size_t code_size = 0;

    while (const char *comma = strchr(p, ',')) {
        if ((size_t) (comma - p) == name_len && memcmp(p, name, name_len) == 0) {
            code = (const uint32_t *) (jitc_vulkan_kernels +
                                       vulkan_kernels_offsets[idx]);
            code_size = vulkan_kernels_sizes[idx];
            break;
        }
        p = comma + 1;
        ++idx;
    }

    if (!code || code_size == 0)
        return VK_NULL_HANDLE;

    VkDevice device = (VkDevice) dev.device;

    VkShaderModuleCreateInfo mod_info {};
    mod_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    mod_info.codeSize = code_size;
    mod_info.pCode = code;

    VkShaderModule mod = VK_NULL_HANDLE;
    VkResult rv = vkCreateShaderModule(device, &mod_info, nullptr, &mod);
    if (rv != VK_SUCCESS) {
        jitc_log(Warn, "vulkan_create_utility_pipeline(%s): "
                       "vkCreateShaderModule() failed: %s",
                 name, jitc_vulkan_error_name(rv));
        return VK_NULL_HANDLE;
    }

    VkComputePipelineCreateInfo pipe_info {};
    pipe_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipe_info.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipe_info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipe_info.stage.module = mod;
    pipe_info.stage.pName = "main";
    pipe_info.layout = (VkPipelineLayout) dev.pipeline_layout;

    VkPipeline pipeline = VK_NULL_HANDLE;
    rv = vkCreateComputePipelines(device, (VkPipelineCache) dev.pipeline_cache,
                                  1, &pipe_info, nullptr, &pipeline);
    vkDestroyShaderModule(device, mod, nullptr);

    if (rv != VK_SUCCESS) {
        jitc_log(Warn, "vulkan_create_utility_pipeline(%s): "
                       "vkCreateComputePipelines() failed: %s",
                 name, jitc_vulkan_error_name(rv));
        return VK_NULL_HANDLE;
    }

    return pipeline;
}

VkPipeline jitc_vulkan_pipeline(VulkanDevice &dev, VulkanKernel kernel) {
    void *&slot = dev.pipelines[(uint32_t) kernel];
    if (!slot)
        slot = vulkan_create_utility_pipeline(
            dev, vulkan_kernel_names[(uint32_t) kernel]);
    return (VkPipeline) slot;
}

static const char *vulkan_reduce_type_name(VarType vt) {
    switch (vt) {
        case VarType::Bool:
        case VarType::UInt8:   return "u8";
        case VarType::Float16: return "f16";
        case VarType::Float32: return "f32";
        case VarType::Float64: return "f64";
        case VarType::UInt32:  return "u32";
        case VarType::Int32:   return "i32";
        case VarType::UInt64:  return "u64";
        case VarType::Int64:   return "i64";
        default:               return nullptr;
    }
}

VkPipeline jitc_vulkan_reduce_pipeline(VulkanDevice &dev,
                                       VulkanReduceKind kind,
                                       ReduceOp op, VarType vt) {
    void *&slot = dev.reduce_pipelines[(uint32_t) kind]
                                      [(uint32_t) op]
                                      [(uint32_t) vt];
    if (slot)
        return (VkPipeline) slot;

    if (vt == VarType::Float64 && !dev.shader_float64)
        return VK_NULL_HANDLE;
    if (vt == VarType::Float16 &&
        (!dev.shader_float16 || !dev.storage_buffer_16bit))
        return VK_NULL_HANDLE;
    if ((vt == VarType::UInt8 || vt == VarType::Bool) &&
        (!dev.shader_int8 || !dev.storage_buffer_8bit))
        return VK_NULL_HANDLE;

    const char *tname = vulkan_reduce_type_name(vt);
    if (!tname)
        return VK_NULL_HANDLE;

    const char *prefix = nullptr;
    switch (kind) {
        case VulkanReduceKind::Small:     prefix = "block_reduce_small"; break;
        case VulkanReduceKind::Chunk:     prefix = "block_reduce"; break;
        case VulkanReduceKind::WideChunk: prefix = "block_reduce_wide"; break;
        case VulkanReduceKind::SmallScan: prefix = "block_prefix_reduce_small"; break;
        case VulkanReduceKind::Scan:      prefix = "block_prefix_reduce"; break;
        case VulkanReduceKind::Dot:       prefix = "reduce_dot"; break;
        default: return VK_NULL_HANDLE;
    }

    char name[64];
    if (kind == VulkanReduceKind::Dot)
        snprintf(name, sizeof(name), "%s_%s", prefix, tname);
    else
        snprintf(name, sizeof(name), "%s_%s_%s",
                 prefix, red_name[(int) op], tname);

    slot = vulkan_create_utility_pipeline(dev, name);
    return (VkPipeline) slot;
}

bool jitc_vulkan_init() {
    if (!state.vulkan_devices.empty())
        return true;

    if (vulkan_init_failed || !jitc_vulkan_instance)
        return false;

    uint32_t n_devices = 0;
    std::vector<VkPhysicalDevice> handles;
    VkResult rv =
        vkEnumeratePhysicalDevices(jitc_vulkan_instance, &n_devices, nullptr);
    if (rv == VK_SUCCESS && n_devices > 0) {
        handles.resize(n_devices);
        rv = vkEnumeratePhysicalDevices(jitc_vulkan_instance, &n_devices,
                                        handles.data());
        if (rv != VK_SUCCESS && rv != VK_INCOMPLETE)
            n_devices = 0;
        handles.resize(n_devices);
    }

    if (n_devices == 0)
        return vulkan_init_fail("no Vulkan devices found");

    // The structures contain internal pointers, hence allocate them in place
    std::unique_ptr<VulkanDeviceInfo[]> infos(new VulkanDeviceInfo[n_devices]);
    int selected = -1;

    for (uint32_t i = 0; i < n_devices; ++i) {
        VulkanDeviceInfo &info = infos[i];
        vulkan_query_device(handles[i], info);

        uint32_t version = info.props.apiVersion;
        jitc_log(Info,
                 " - Found Vulkan device %u: \"%s\" (%s, Vulkan %u.%u.%u, "
                 "%s device mem.)",
                 i, info.props.deviceName,
                 vulkan_device_type_name((uint32_t) info.props.deviceType),
                 VK_API_VERSION_MAJOR(version), VK_API_VERSION_MINOR(version),
                 VK_API_VERSION_PATCH(version),
                 jitc_mem_string(info.device_memory).c_str());

        if (info.reason) {
            jitc_log(Warn, " - Warning: %s, skipping ..", info.reason);
            continue;
        }

        if (selected < 0 ||
            vulkan_device_type_rank((uint32_t) info.props.deviceType) <
                vulkan_device_type_rank(
                    (uint32_t) infos[selected].props.deviceType))
            selected = (int) i;
    }

    // Optional override of the device selection
    const char *env = getenv("DRJIT_VULKAN_DEVICE");
    if (env && *env) {
        char *end = nullptr;
        long value = strtol(env, &end, 10);
        if (*end != '\0' || value < 0 || value >= (long) n_devices)
            return vulkan_init_fail(
                "invalid value DRJIT_VULKAN_DEVICE=\"%s\" (must be a device "
                "index between 0 and %u)", env, n_devices - 1);

        if (infos[value].reason)
            return vulkan_init_fail(
                "the device selected via DRJIT_VULKAN_DEVICE=%ld is "
                "incompatible with Dr.Jit (%s)", value, infos[value].reason);

        selected = (int) value;
    }

    if (selected < 0)
        return vulkan_init_fail("no compatible Vulkan devices found");

    if (!jitc_vulkan_kernels) {
        jitc_vulkan_kernels = jitc_lz4_inflate(
            vulkan_kernels, vulkan_kernels_size_compressed,
            vulkan_kernels_size_uncompressed, "Vulkan utility kernels");
    }

    return vulkan_create_device(infos[selected], selected);
}

void jitc_vulkan_shutdown() {
    vulkan_init_failed = false;

    if (jitc_vulkan_kernels) {
        free(jitc_vulkan_kernels);
        jitc_vulkan_kernels = nullptr;
    }

    if (state.vulkan_devices.empty())
        return;

    jitc_vulkan_sync_all();

    for (VulkanDevice &dev : state.vulkan_devices)
        vkDeviceWaitIdle((VkDevice) dev.device);

    // Evict cached VkShaderModules before destroying the Vulkan device
    jitc_unit_cache_flush((int) JitBackend::Vulkan);

    // Release cached allocations (normally, jitc_malloc_shutdown() already
    // did so)
    std::vector<void *> cached;
    /* Critical section */ {
        lock_guard guard(state.alloc_free_lock);
        for (auto it = state.alloc_free.begin(); it != state.alloc_free.end();
             ++it) {
            auto [size, backend, shared, device] = alloc_info_decode(it->first);
            (void) size; (void) shared; (void) device;
            if (backend != JitBackend::Vulkan)
                continue;
            cached.insert(cached.end(), it->second.begin(), it->second.end());
            it.value().clear();
        }
    }

    for (void *ptr : cached) {
        if (VulkanBuffer *buffer = jitc_vulkan_unregister_buffer(ptr))
            vulkan_buffer_free(buffer);
    }

    /* Release buffers that were leaked by the application. Their records in
       'state.alloc_used' become stale, which jitc_free() and jitc_malloc()
       take into account. */
    if (!vulkan_buffers.empty()) {
        jitc_log(Info, "jit_vulkan_shutdown(): releasing %zu leaked buffer%s ..",
                 vulkan_buffers.size(), vulkan_buffers.size() > 1 ? "s" : "");
        for (auto &kv : vulkan_buffers)
            vulkan_buffer_free(kv.second);
        vulkan_buffers.clear();
        vulkan_buffers_lut.clear();
    }

    // No Vulkan memory remains allocated
    state.alloc_allocated[(int) JitBackend::Vulkan] = 0;

    for (VulkanDevice &dev : state.vulkan_devices) {
        VkDevice device = (VkDevice) dev.device;

        for (uint32_t i = 0; i < (uint32_t) VulkanKernel::Count; ++i) {
            if (dev.pipelines[i])
                vkDestroyPipeline(device, (VkPipeline) dev.pipelines[i],
                                  nullptr);
        }

        for (uint32_t i = 0; i < (uint32_t) VulkanReduceKind::Count; ++i)
            for (uint32_t j = 0; j < (uint32_t) ReduceOp::Count; ++j)
                for (uint32_t k = 0; k < (uint32_t) VarType::Count; ++k)
                    if (dev.reduce_pipelines[i][j][k])
                        vkDestroyPipeline(
                            device, (VkPipeline) dev.reduce_pipelines[i][j][k],
                            nullptr);

        if (dev.pipeline_cache) {
            VkPipelineCache cache = (VkPipelineCache) dev.pipeline_cache;
            const char *dir = jitc_cache_dir();
            if (dir && jitc_cache_writable()) {
                size_t cache_size = 0;
                if (vkGetPipelineCacheData(device, cache, &cache_size,
                                           nullptr) == VK_SUCCESS &&
                    cache_size > 0) {
                    std::vector<uint8_t> data(cache_size);
                    if (vkGetPipelineCacheData(device, cache, &cache_size,
                                               data.data()) == VK_SUCCESS) {
                        const uint8_t *u = dev.pipeline_cache_uuid;
                        char name[64], tmp_suffix[24];
                        snprintf(name, sizeof(name),
                                 "vulkan_pipeline_cache_"
                                 "%02x%02x%02x%02x%02x%02x%02x%02x"
                                 "%02x%02x%02x%02x%02x%02x%02x%02x.bin",
                                 u[0], u[1], u[2], u[3], u[4], u[5], u[6], u[7],
                                 u[8], u[9], u[10], u[11], u[12], u[13], u[14],
                                 u[15]);
#if defined(_WIN32)
                        snprintf(tmp_suffix, sizeof(tmp_suffix), ".%u.tmp",
                                 (unsigned) GetCurrentProcessId());
#else
                        snprintf(tmp_suffix, sizeof(tmp_suffix), ".%u.tmp",
                                 (unsigned) getpid());
#endif
                        std::filesystem::path cache_path =
                                                  std::filesystem::path(dir) /
                                                  name,
                                              tmp_path = cache_path;
                        tmp_path += tmp_suffix;
                        std::ofstream file(tmp_path,
                                           std::ios::binary | std::ios::trunc);
                        if (file.write((const char *) data.data(), cache_size)) {
                            file.close();
                            if (file.good()) {
                                std::error_code ec;
                                std::filesystem::rename(tmp_path, cache_path,
                                                        ec);
                                if (ec)
                                    std::filesystem::remove(tmp_path, ec);
                            }
                        } else {
                            file.close();
                            std::error_code ec;
                            std::filesystem::remove(tmp_path, ec);
                        }
                    }
                }
            }
            vkDestroyPipelineCache(device, cache, nullptr);
        }

        if (dev.pipeline_layout)
            vkDestroyPipelineLayout(device,
                                    (VkPipelineLayout) dev.pipeline_layout,
                                    nullptr);

        vkDestroyCommandPool(device, dev.stream->pool, nullptr);
        delete dev.stream;
        vkDestroySemaphore(device, (VkSemaphore) dev.timeline, nullptr);
        vkDestroyDevice(device, nullptr);
        free(dev.name);
    }

    state.vulkan_devices.clear();
    jitc_log(Info, "jit_vulkan_shutdown(): done");
}

ThreadState *jitc_vulkan_thread_state_new() {
    return new VulkanThreadState();
}

// ====================================================================
//                        Ray tracing scene & queries
// ====================================================================

static uint32_t jitc_vulkan_make_resource_handle(void *ptr, ResourceKind kind) {
    if (!ptr)
        return 0;
    uint32_t backing = jitc_var_mem_map(JitBackend::Vulkan, VarType::UInt64,
                                        ptr, 1, /*free=*/0);
    uint32_t handle = jitc_var_resource_pointer(backing, kind);
    jitc_var_dec_ref(backing);
    return handle;
}

static uint32_t jitc_vulkan_scene_resource_handle(VulkanScene *scene,
                                                  ResourceKind kind) {
    if (!scene)
        return 0;
    uint32_t &slot = (kind == ResourceKind::IFT) ? scene->ift_handle
                                                 : scene->owner_handle;
    if (!slot)
        slot = jitc_vulkan_make_resource_handle(scene, kind);
    jitc_var_inc_ref(slot);
    return slot;
}

uint32_t jitc_vulkan_configure_scene(void *tlas_ptr,
                                     uint32_t n_isect_entries,
                                     uint32_t geometry_types_mask) {
    if (!tlas_ptr)
        jitc_raise("jit_vulkan_configure_scene(): 'tlas' must not be NULL.");

    ThreadState *ts = thread_state(JitBackend::Vulkan);
    VulkanDevice &dev = state.vulkan_devices[ts->device];
    if (!dev.supports_ray_query)
        jitc_raise("jit_vulkan_configure_scene(): the active Vulkan device "
                   "does not support ray queries.");

    VkAccelerationStructureDeviceAddressInfoKHR addr_info {};
    addr_info.sType =
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
    addr_info.accelerationStructure = (VkAccelerationStructureKHR) tlas_ptr;
    VkDeviceAddress tlas_addr = vkGetAccelerationStructureDeviceAddressKHR(
        (VkDevice) dev.device, &addr_info);
    if (!tlas_addr)
        jitc_raise("jit_vulkan_configure_scene(): failed to query the "
                   "device address of the TLAS.");

    VulkanScene *scene = new VulkanScene();
    scene->device_id = ts->device;
    scene->device = (VkDevice) dev.device;
    scene->tlas = (VkAccelerationStructureKHR) tlas_ptr;
    scene->tlas_address = (uint64_t) tlas_addr;
    scene->isect_count = n_isect_entries;
    scene->geometry_types_mask = geometry_types_mask;

    if (n_isect_entries > 0) {
        size_t table_bytes = (size_t) n_isect_entries * sizeof(uint64_t);
        scene->isect_table =
            jitc_malloc(JitBackend::Vulkan, table_bytes, /*shared=*/true);
        memset(scene->isect_table, 0, table_bytes);
        size_t offset = 0;
        VulkanBuffer *buf = jitc_vulkan_find_buffer(scene->isect_table, &offset);
        if (!buf)
            jitc_fail("jit_vulkan_configure_scene(): could not resolve buffer "
                      "for isect_table.");
        scene->isect_table_address = (uint64_t) buf->address + offset;
    }

    uint32_t index =
        jitc_var_new_node_0(JitBackend::Vulkan, VarKind::Nop,
                            VarType::Void, 1, 0, (uintptr_t) scene);

    auto callback = [](uint32_t /*index*/, int free, void *ptr) {
        if (!free)
            return;
        auto *s = (VulkanScene *) ptr;
        jitc_log(InfoSym, "jit_vulkan_configure_scene(): freeing VulkanScene "
                          "(n_isect=%u)", s->isect_count);

        if (s->owner_handle)
            jitc_var_dec_ref(s->owner_handle);
        if (s->ift_handle)
            jitc_var_dec_ref(s->ift_handle);
        if (s->isect_table)
            jitc_free(s->isect_table);

        if (s->cleanup_callback) {
            unlock_guard guard(state.lock);
            s->cleanup_callback(s->cleanup_payload);
        }

        delete s;
    };

    jitc_var_set_callback(index, callback, scene, true);
    return index;
}

VulkanScene *jitc_vulkan_get_scene(uint32_t scene_index) {
    Variable *v = scene_index ? jitc_var(scene_index) : nullptr;
    if (!v || (VarKind) v->kind != VarKind::Nop ||
        (VarType) v->type != VarType::Void)
        jitc_fail("jitc_vulkan_get_scene(): r%u does not wrap a Vulkan scene.",
                  scene_index);
    return (VulkanScene *) v->literal;
}

uint32_t jitc_vulkan_scene_owner_handle(uint32_t scene_index) {
    VulkanScene *scene = jitc_vulkan_get_scene(scene_index);
    return jitc_var_mem_map(JitBackend::Vulkan, VarType::UInt64,
                            (void *) scene, 1, /*free=*/0);
}

void jitc_vulkan_ray_trace(uint32_t n_args, uint32_t *args,
                           uint32_t mask, uint32_t *out,
                           uint32_t n_out, uint32_t scene, int shadow) {
    if (n_args != 9 && n_args != 10)
        jitc_raise("jit_vulkan_ray_trace(): expected 9 or 10 ray arguments, "
                   "got %u.", n_args);
    if (n_args == 10 && jitc_var_type(args[9]) != VarType::UInt32)
        jitc_raise("jit_vulkan_ray_trace(): the ray visibility mask "
                   "(argument 10) must be of type UInt32.");
    if (n_out != 8)
        jitc_raise("jit_vulkan_ray_trace(): expected 8 outputs, got %u.",
                   n_out);
    if (!scene)
        jitc_raise("jit_vulkan_ray_trace(): a valid scene_index "
                   "(returned by jit_vulkan_configure_scene) is required.");

    if (jitc_var_type(scene) != VarType::Void)
        jitc_raise("jit_vulkan_ray_trace(): type mismatch for scene argument!");

    VarType vt = jitc_var_type(args[0]);
    if (vt != VarType::Float32 && vt != VarType::Float64)
        jitc_raise("jit_vulkan_ray_trace(): ray origin/direction/distance "
                   "arguments must be of type Float32 or Float64.");
    for (uint32_t i = 1; i < 9; ++i) {
        if (jitc_var_type(args[i]) != vt)
            jitc_raise("jit_vulkan_ray_trace(): ray arguments 0..8 must all "
                       "have the same floating-point type.");
    }

    uint32_t size = 0;
    bool symbolic = false;
    for (uint32_t i = 0; i < n_args; ++i) {
        const Variable *vi = jitc_var(args[i]);
        size = std::max(size, vi->size);
        symbolic |= (bool) vi->symbolic;
    }
    {
        const Variable *vm = jitc_var(mask);
        size = std::max(size, vm->size);
        symbolic |= (bool) vm->symbolic;
    }
    if (size == 0)
        size = 1;

    // Apply mask stack
    Ref valid = steal(jitc_var_mask_apply(mask, size));

    // Build TraceData with ray parameter indices
    TraceData *td = new TraceData();
    td->shadow = shadow != 0;
    td->indices.reserve(n_args);
    for (uint32_t i = 0; i < n_args; ++i) {
        td->indices.push_back(args[i]);
        jitc_var_inc_ref(args[i]);
    }

    // Build acceleration-structure and intersection-function-table handles.
    // The scene reference in dep[1] keeps VulkanScene alive.
    VulkanScene *scene_obj = jitc_vulkan_get_scene(scene);
    Ref accel_h = steal(jitc_vulkan_scene_resource_handle(scene_obj,
                                                          ResourceKind::Accel));
    Ref ift_h = steal(jitc_vulkan_scene_resource_handle(
        scene_obj->isect_count ? scene_obj : nullptr, ResourceKind::IFT));

    // dep[0]=valid, dep[1]=scene, dep[2]=accel handle, dep[3]=IFT.
    Ref trace;
    if (ift_h)
        trace = steal(jitc_var_new_node_4(
            JitBackend::Vulkan, VarKind::TraceRay, VarType::Void, size, symbolic,
            valid, jitc_var(valid), scene, jitc_var(scene),
            accel_h, jitc_var(accel_h), ift_h, jitc_var(ift_h),
            (uintptr_t) td));
    else
        trace = steal(jitc_var_new_node_3(
            JitBackend::Vulkan, VarKind::TraceRay, VarType::Void, size, symbolic,
            valid, jitc_var(valid), scene, jitc_var(scene),
            accel_h, jitc_var(accel_h), (uintptr_t) td));

    // Register cleanup callback
    jitc_var_set_callback(
        trace,
        [](uint32_t, int free, void *ptr) {
            if (free)
                delete (TraceData *) ptr;
        },
        td, true);

    // Create Extract children for each output
    VarType out_types[8] = {
        VarType::Bool,   // valid
        vt,              // distance
        vt,              // bary_u
        vt,              // bary_v
        VarType::UInt32, // instance_id (raw TLAS instance index)
        VarType::UInt32, // primitive_id
        VarType::UInt32, // geometry_id
        VarType::UInt32  // user-provided instance ID
    };

    for (uint32_t i = 0; i < 8; ++i)
        out[i] = jitc_var_new_node_1(
            JitBackend::Vulkan, VarKind::Extract, out_types[i],
            size, symbolic, trace, jitc_var(trace), (uint64_t) i);
}

