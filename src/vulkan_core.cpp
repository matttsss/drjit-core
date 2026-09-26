/*
    src/vulkan_core.cpp -- Vulkan backend: device initialization, command
    submission, synchronization, and memory allocation

    Copyright (c) 2026 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#include "vulkan.h"
#include "vulkan_ts.h"
#include "log.h"
#include <algorithm>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <map>
#include <memory>

/// Size of the shared memory chunks used by jitc_vulkan_stage()
static constexpr size_t VulkanArenaSize = 65536;

/// Maximum number of memory ranges tracked per hazard list. Commands with
/// more complex access patterns are separated by a barrier.
static constexpr size_t VulkanMaxHazardRanges = 16;

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
    VkDevice device = (VkDevice) dev.device;

    VkBufferCreateInfo buffer_info {};
    buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_info.size = size;
    buffer_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                        VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                        VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkBuffer buffer = VK_NULL_HANDLE;
    if (vkCreateBuffer(device, &buffer_info, nullptr, &buffer) != VK_SUCCESS)
        return nullptr;

    VkMemoryRequirements reqs {};
    vkGetBufferMemoryRequirements(device, buffer, &reqs);

    // Try the compatible memory types in order of decreasing preference
    VkDeviceMemory memory = VK_NULL_HANDLE;
    if (reqs.size <= dev.max_memory_allocation_size) {
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

    if (!success) {
        vkDestroyBuffer(device, buffer, nullptr);
        if (memory)
            vkFreeMemory(device, memory, nullptr); // (also unmaps)
        return nullptr;
    }

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

void vulkan_buffer_free(VulkanBuffer *buffer) {
    vkDestroyBuffer(buffer->device, buffer->buffer, nullptr);
    vkFreeMemory(buffer->device, buffer->memory, nullptr); // (also unmaps)
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
    s.reads.clear();
    s.writes.clear();
    return cmd;
}

/// Does the range 'r' overlap with any of the ranges in 'list'?
static bool vulkan_overlaps(const std::vector<VulkanRange> &list,
                            const VulkanRange &r) {
    for (const VulkanRange &r2 : list) {
        if (r.start < r2.end && r2.start < r.end)
            return true;
    }
    return false;
}

void jitc_vulkan_access(VulkanDevice &dev, const VulkanRange *reads,
                        uint32_t n_reads, const VulkanRange *writes,
                        uint32_t n_writes) {
    VulkanStream &s = *dev.stream;
    bool barrier = s.barrier_pending;

    // Read-after-write hazards
    for (uint32_t i = 0; i < n_reads && !barrier; ++i)
        barrier = vulkan_overlaps(s.writes, reads[i]);

    // Write-after-write and write-after-read hazards
    for (uint32_t i = 0; i < n_writes && !barrier; ++i)
        barrier = vulkan_overlaps(s.writes, writes[i]) ||
                  vulkan_overlaps(s.reads, writes[i]);

    // Limit the cost of hazard tracking
    if (s.reads.size() + n_reads > VulkanMaxHazardRanges ||
        s.writes.size() + n_writes > VulkanMaxHazardRanges)
        barrier = true;

    if (barrier) {
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
        s.reads.clear();
        s.writes.clear();
        s.barrier_pending = false;
    }

    s.reads.insert(s.reads.end(), reads, reads + n_reads);
    s.writes.insert(s.writes.end(), writes, writes + n_writes);
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
    s.reads.clear();
    s.writes.clear();

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
    features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    features11.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
    features12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    features13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
    atomic_float.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT;
    features.pNext = &features11;
    features11.pNext = &features12;
    features12.pNext = &features13;

    const VkPhysicalDeviceFeatures &f = info.features.features;
    const VkPhysicalDeviceVulkan11Features &f11 = info.features11;
    const VkPhysicalDeviceVulkan12Features &f12 = info.features12;
    const VkPhysicalDeviceShaderAtomicFloatFeaturesEXT &faf = info.atomic_float;

    features.features.shaderInt64 = VK_TRUE;
    features.features.shaderInt16 = f.shaderInt16;
    features.features.shaderFloat64 = f.shaderFloat64;
    features11.storageBuffer16BitAccess = f11.storageBuffer16BitAccess;
    features12.bufferDeviceAddress = VK_TRUE;
    features12.timelineSemaphore = VK_TRUE;
    features12.scalarBlockLayout = VK_TRUE;
    features12.shaderInt8 = f12.shaderInt8;
    features12.shaderFloat16 = f12.shaderFloat16;
    features12.storageBuffer8BitAccess = f12.storageBuffer8BitAccess;
    features12.shaderBufferInt64Atomics = f12.shaderBufferInt64Atomics;
    features12.vulkanMemoryModel = f12.vulkanMemoryModel;
    features12.vulkanMemoryModelDeviceScope = f12.vulkanMemoryModelDeviceScope;
    features13.maintenance4 = VK_TRUE;
    features13.synchronization2 = info.features13.synchronization2;

    const char *extensions[2] = { };
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
        features13.pNext = &atomic_float;
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

    VulkanDevice dev;
    dev.physical_device = info.handle;
    dev.device = device;
    dev.queue = queue;
    dev.timeline = timeline;
    dev.timeline_value = 0;
    dev.stream = new VulkanStream();
    dev.stream->pool = pool;
    dev.index = index;
    dev.queue_family = info.queue_family;
    dev.api_version = std::min(info.props.apiVersion, jitc_vulkan_api_version);
    dev.vendor_id = info.props.vendorID;
    dev.device_id = info.props.deviceID;
    dev.device_type = (uint32_t) info.props.deviceType;
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
    dev.supports_accel = jitc_vulkan_has_accel_api && info.has_accel &&
                         info.has_deferred_host_operations &&
                         info.accel.accelerationStructure;
    dev.supports_ray_query =
        dev.supports_accel && info.has_ray_query && info.ray_query.rayQuery;
    dev.name = strdup(info.props.deviceName);

    state.vulkan_devices.push_back(dev);

    jitc_log(Info,
             "jit_vulkan_init(): using device %i (\"%s\", Vulkan %u.%u, "
             "subgroup size %u, %s max. allocation)%s%s%s%s%s%s%s%s%s",
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
             dev.supports_ray_query ? ", ray queries" : "");

    return true;
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

    return vulkan_create_device(infos[selected], selected);
}

void jitc_vulkan_shutdown() {
    vulkan_init_failed = false;

    if (state.vulkan_devices.empty())
        return;

    jitc_vulkan_sync_all();

    for (VulkanDevice &dev : state.vulkan_devices)
        vkDeviceWaitIdle((VkDevice) dev.device);

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
