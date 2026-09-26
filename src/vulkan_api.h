/*
    src/vulkan_api.h -- Low-level interface to the Vulkan loader & SPIRV-Tools

    Dr.Jit always loads the Vulkan loader (``libvulkan.so.1``,
    ``vulkan-1.dll``, ``libvulkan.1.dylib``) and the SPIRV-Tools shared
    library (``libSPIRV-Tools-shared.so`` etc.) dynamically at run time. With
    ``DRJIT_DYNAMIC_VULKAN`` (the default), this header provides self-contained
    declarations of the subset of the Vulkan API used by Dr.Jit so that no
    Vulkan SDK is needed at build time. Otherwise, it includes the official
    Vulkan headers (``VK_NO_PROTOTYPES`` mode). The declarations below follow
    the Vulkan specification; ``tests/vulkan_layout.cpp`` cross-checks their
    layout against the official headers whenever they are available.

    All Vulkan entry points are called through function pointers that are
    resolved via ``vkGetInstanceProcAddr()`` on the Dr.Jit ``VkInstance``.
    Device-level functions obtained in this way dispatch to the right driver
    for any ``VkDevice`` of that instance.

    Copyright (c) 2026 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#include <stdint.h>
#include <stddef.h>

#if !defined(DRJIT_DYNAMIC_VULKAN)
#  if !defined(VK_NO_PROTOTYPES)
#    define VK_NO_PROTOTYPES
#  endif
#  include <vulkan/vulkan.h>
#else

static_assert(sizeof(void *) == 8,
              "The Dr.Jit Vulkan backend requires a 64-bit platform.");

// ====================================================================
//            Self-contained subset of the Vulkan 1.3 API
// ====================================================================

#if !defined(VKAPI_CALL)
#  if defined(_WIN32)
#    define VKAPI_ATTR
#    define VKAPI_CALL __stdcall
#    define VKAPI_PTR  VKAPI_CALL
#  else
#    define VKAPI_ATTR
#    define VKAPI_CALL
#    define VKAPI_PTR
#  endif
#endif

#if !defined(VK_MAKE_API_VERSION)
#  define VK_MAKE_API_VERSION(variant, major, minor, patch)                    \
      ((((uint32_t) (variant)) << 29U) | (((uint32_t) (major)) << 22U) |       \
       (((uint32_t) (minor)) << 12U) | ((uint32_t) (patch)))
#  define VK_API_VERSION_VARIANT(version) ((uint32_t) (version) >> 29U)
#  define VK_API_VERSION_MAJOR(version) (((uint32_t) (version) >> 22U) & 0x7FU)
#  define VK_API_VERSION_MINOR(version) (((uint32_t) (version) >> 12U) & 0x3FFU)
#  define VK_API_VERSION_PATCH(version) ((uint32_t) (version) & 0xFFFU)
#  define VK_API_VERSION_1_0 VK_MAKE_API_VERSION(0, 1, 0, 0)
#  define VK_API_VERSION_1_1 VK_MAKE_API_VERSION(0, 1, 1, 0)
#  define VK_API_VERSION_1_2 VK_MAKE_API_VERSION(0, 1, 2, 0)
#  define VK_API_VERSION_1_3 VK_MAKE_API_VERSION(0, 1, 3, 0)
#endif

#if !defined(VK_NULL_HANDLE)
#  define VK_NULL_HANDLE nullptr
#endif

#if !defined(VK_TRUE)
#  define VK_TRUE                          1U
#  define VK_FALSE                         0U
#  define VK_WHOLE_SIZE                    (~0ULL)
#  define VK_QUEUE_FAMILY_IGNORED          (~0U)
#  define VK_MAX_PHYSICAL_DEVICE_NAME_SIZE 256U
#  define VK_UUID_SIZE                     16U
#  define VK_LUID_SIZE                     8U
#  define VK_MAX_EXTENSION_NAME_SIZE       256U
#  define VK_MAX_DESCRIPTION_SIZE          256U
#  define VK_MAX_MEMORY_TYPES              32U
#  define VK_MAX_MEMORY_HEAPS              16U
#  define VK_MAX_DRIVER_NAME_SIZE          256U
#  define VK_MAX_DRIVER_INFO_SIZE          256U
#endif

#if !defined(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME)
#  define VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME "VK_KHR_acceleration_structure"
#  define VK_KHR_RAY_QUERY_EXTENSION_NAME "VK_KHR_ray_query"
#  define VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME "VK_KHR_deferred_host_operations"
#  define VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME "VK_EXT_shader_atomic_float"
#  define VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME "VK_KHR_portability_enumeration"
#endif

#define DRJIT_VK_HANDLE(name) typedef struct name##_T *name;

// Dispatchable handles
DRJIT_VK_HANDLE(VkInstance)
DRJIT_VK_HANDLE(VkPhysicalDevice)
DRJIT_VK_HANDLE(VkDevice)
DRJIT_VK_HANDLE(VkQueue)
DRJIT_VK_HANDLE(VkCommandBuffer)

// Non-dispatchable handles (pointer-sized on 64-bit platforms)
DRJIT_VK_HANDLE(VkSemaphore)
DRJIT_VK_HANDLE(VkFence)
DRJIT_VK_HANDLE(VkDeviceMemory)
DRJIT_VK_HANDLE(VkBuffer)
DRJIT_VK_HANDLE(VkImage)
DRJIT_VK_HANDLE(VkShaderModule)
DRJIT_VK_HANDLE(VkPipelineCache)
DRJIT_VK_HANDLE(VkPipelineLayout)
DRJIT_VK_HANDLE(VkPipeline)
DRJIT_VK_HANDLE(VkDescriptorSetLayout)
DRJIT_VK_HANDLE(VkCommandPool)
DRJIT_VK_HANDLE(VkAccelerationStructureKHR)

#undef DRJIT_VK_HANDLE

typedef uint32_t VkFlags;
typedef uint64_t VkFlags64;
typedef uint32_t VkBool32;
typedef uint64_t VkDeviceSize;
typedef uint64_t VkDeviceAddress;

// Structures that Dr.Jit only ever references via (null) pointers
struct VkAllocationCallbacks;
struct VkCommandBufferInheritanceInfo;
struct VkImageMemoryBarrier;

// -------------------------------- Enumerations --------------------------------

typedef enum VkResult {
    VK_SUCCESS = 0,
    VK_NOT_READY = 1,
    VK_TIMEOUT = 2,
    VK_EVENT_SET = 3,
    VK_EVENT_RESET = 4,
    VK_INCOMPLETE = 5,
    VK_ERROR_OUT_OF_HOST_MEMORY = -1,
    VK_ERROR_OUT_OF_DEVICE_MEMORY = -2,
    VK_ERROR_INITIALIZATION_FAILED = -3,
    VK_ERROR_DEVICE_LOST = -4,
    VK_ERROR_MEMORY_MAP_FAILED = -5,
    VK_ERROR_LAYER_NOT_PRESENT = -6,
    VK_ERROR_EXTENSION_NOT_PRESENT = -7,
    VK_ERROR_FEATURE_NOT_PRESENT = -8,
    VK_ERROR_INCOMPATIBLE_DRIVER = -9,
    VK_ERROR_TOO_MANY_OBJECTS = -10,
    VK_ERROR_FORMAT_NOT_SUPPORTED = -11,
    VK_ERROR_FRAGMENTED_POOL = -12,
    VK_ERROR_UNKNOWN = -13,
    VK_ERROR_OUT_OF_POOL_MEMORY = -1000069000,
    VK_ERROR_INVALID_EXTERNAL_HANDLE = -1000072003,
    VK_ERROR_FRAGMENTATION = -1000161000,
    VK_ERROR_INVALID_OPAQUE_CAPTURE_ADDRESS = -1000257000,
    VK_PIPELINE_COMPILE_REQUIRED = 1000297000,
    VK_ERROR_VALIDATION_FAILED_EXT = -1000011001,
    VK_RESULT_MAX_ENUM = 0x7FFFFFFF
} VkResult;

typedef enum VkStructureType {
    VK_STRUCTURE_TYPE_APPLICATION_INFO = 0,
    VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO = 1,
    VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO = 2,
    VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO = 3,
    VK_STRUCTURE_TYPE_SUBMIT_INFO = 4,
    VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO = 5,
    VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE = 6,
    VK_STRUCTURE_TYPE_BIND_SPARSE_INFO = 7,
    VK_STRUCTURE_TYPE_FENCE_CREATE_INFO = 8,
    VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO = 9,
    VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO = 12,
    VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO = 16,
    VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO = 17,
    VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO = 18,
    VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO = 29,
    VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO = 30,
    VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO = 39,
    VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO = 40,
    VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO = 42,
    VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER = 44,
    VK_STRUCTURE_TYPE_MEMORY_BARRIER = 46,
    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES = 49,
    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_PROPERTIES = 50,
    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES = 51,
    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_PROPERTIES = 52,
    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES = 53,
    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_PROPERTIES = 54,
    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2 = 1000059000,
    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2 = 1000059001,
    VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO = 1000060000,
    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES = 1000094000,
    VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR = 1000150000,
    VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR = 1000150002,
    VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_AABBS_DATA_KHR = 1000150003,
    VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR = 1000150004,
    VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR = 1000150005,
    VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR = 1000150006,
    VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR = 1000150007,
    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR = 1000150013,
    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR = 1000150014,
    VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR = 1000150017,
    VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR = 1000150020,
    VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO = 1000207002,
    VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO = 1000207003,
    VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO = 1000207004,
    VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO = 1000207005,
    VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO = 1000244001,
    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT = 1000260000,
    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR = 1000348013,
    VK_STRUCTURE_TYPE_MAX_ENUM = 0x7FFFFFFF
} VkStructureType;

typedef enum VkPhysicalDeviceType {
    VK_PHYSICAL_DEVICE_TYPE_OTHER = 0,
    VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU = 1,
    VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU = 2,
    VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU = 3,
    VK_PHYSICAL_DEVICE_TYPE_CPU = 4,
    VK_PHYSICAL_DEVICE_TYPE_MAX_ENUM = 0x7FFFFFFF
} VkPhysicalDeviceType;

typedef enum VkInstanceCreateFlagBits {
    VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR = 0x00000001,
    VK_INSTANCE_CREATE_FLAG_BITS_MAX_ENUM = 0x7FFFFFFF
} VkInstanceCreateFlagBits;
typedef VkFlags VkInstanceCreateFlags;

typedef enum VkQueueFlagBits {
    VK_QUEUE_GRAPHICS_BIT = 0x00000001,
    VK_QUEUE_COMPUTE_BIT = 0x00000002,
    VK_QUEUE_TRANSFER_BIT = 0x00000004,
    VK_QUEUE_SPARSE_BINDING_BIT = 0x00000008,
    VK_QUEUE_PROTECTED_BIT = 0x00000010,
    VK_QUEUE_FLAG_BITS_MAX_ENUM = 0x7FFFFFFF
} VkQueueFlagBits;
typedef VkFlags VkQueueFlags;

typedef enum VkMemoryPropertyFlagBits {
    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT = 0x00000001,
    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT = 0x00000002,
    VK_MEMORY_PROPERTY_HOST_COHERENT_BIT = 0x00000004,
    VK_MEMORY_PROPERTY_HOST_CACHED_BIT = 0x00000008,
    VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT = 0x00000010,
    VK_MEMORY_PROPERTY_PROTECTED_BIT = 0x00000020,
    VK_MEMORY_PROPERTY_DEVICE_COHERENT_BIT_AMD = 0x00000040,
    VK_MEMORY_PROPERTY_DEVICE_UNCACHED_BIT_AMD = 0x00000080,
    VK_MEMORY_PROPERTY_FLAG_BITS_MAX_ENUM = 0x7FFFFFFF
} VkMemoryPropertyFlagBits;
typedef VkFlags VkMemoryPropertyFlags;

typedef enum VkMemoryHeapFlagBits {
    VK_MEMORY_HEAP_DEVICE_LOCAL_BIT = 0x00000001,
    VK_MEMORY_HEAP_MULTI_INSTANCE_BIT = 0x00000002,
    VK_MEMORY_HEAP_FLAG_BITS_MAX_ENUM = 0x7FFFFFFF
} VkMemoryHeapFlagBits;
typedef VkFlags VkMemoryHeapFlags;

typedef enum VkMemoryAllocateFlagBits {
    VK_MEMORY_ALLOCATE_DEVICE_MASK_BIT = 0x00000001,
    VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT = 0x00000002,
    VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_CAPTURE_REPLAY_BIT = 0x00000004,
    VK_MEMORY_ALLOCATE_FLAG_BITS_MAX_ENUM = 0x7FFFFFFF
} VkMemoryAllocateFlagBits;
typedef VkFlags VkMemoryAllocateFlags;

typedef enum VkSparseMemoryBindFlagBits {
    VK_SPARSE_MEMORY_BIND_METADATA_BIT = 0x00000001,
    VK_SPARSE_MEMORY_BIND_FLAG_BITS_MAX_ENUM = 0x7FFFFFFF
} VkSparseMemoryBindFlagBits;
typedef VkFlags VkSparseMemoryBindFlags;

typedef enum VkImageAspectFlagBits {
    VK_IMAGE_ASPECT_COLOR_BIT = 0x00000001,
    VK_IMAGE_ASPECT_DEPTH_BIT = 0x00000002,
    VK_IMAGE_ASPECT_STENCIL_BIT = 0x00000004,
    VK_IMAGE_ASPECT_METADATA_BIT = 0x00000008,
    VK_IMAGE_ASPECT_PLANE_0_BIT = 0x00000010,
    VK_IMAGE_ASPECT_PLANE_1_BIT = 0x00000020,
    VK_IMAGE_ASPECT_PLANE_2_BIT = 0x00000040,
    VK_IMAGE_ASPECT_FLAG_BITS_MAX_ENUM = 0x7FFFFFFF
} VkImageAspectFlagBits;
typedef VkFlags VkImageAspectFlags;

typedef enum VkBufferUsageFlagBits {
    VK_BUFFER_USAGE_TRANSFER_SRC_BIT = 0x00000001,
    VK_BUFFER_USAGE_TRANSFER_DST_BIT = 0x00000002,
    VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT = 0x00000004,
    VK_BUFFER_USAGE_STORAGE_TEXEL_BUFFER_BIT = 0x00000008,
    VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT = 0x00000010,
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT = 0x00000020,
    VK_BUFFER_USAGE_INDEX_BUFFER_BIT = 0x00000040,
    VK_BUFFER_USAGE_VERTEX_BUFFER_BIT = 0x00000080,
    VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT = 0x00000100,
    VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR = 0x00000400,
    VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT = 0x00020000,
    VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR = 0x00080000,
    VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR = 0x00100000,
    VK_BUFFER_USAGE_FLAG_BITS_MAX_ENUM = 0x7FFFFFFF
} VkBufferUsageFlagBits;
typedef VkFlags VkBufferUsageFlags;
typedef enum VkBufferCreateFlagBits {
    VK_BUFFER_CREATE_SPARSE_BINDING_BIT = 0x00000001,
    VK_BUFFER_CREATE_SPARSE_RESIDENCY_BIT = 0x00000002,
    VK_BUFFER_CREATE_SPARSE_ALIASED_BIT = 0x00000004,
    VK_BUFFER_CREATE_PROTECTED_BIT = 0x00000008,
    VK_BUFFER_CREATE_DEVICE_ADDRESS_CAPTURE_REPLAY_BIT = 0x00000010,
    VK_BUFFER_CREATE_FLAG_BITS_MAX_ENUM = 0x7FFFFFFF
} VkBufferCreateFlagBits;
typedef VkFlags VkBufferCreateFlags;

typedef enum VkSharingMode {
    VK_SHARING_MODE_EXCLUSIVE = 0,
    VK_SHARING_MODE_CONCURRENT = 1,
    VK_SHARING_MODE_MAX_ENUM = 0x7FFFFFFF
} VkSharingMode;

typedef enum VkCommandPoolCreateFlagBits {
    VK_COMMAND_POOL_CREATE_TRANSIENT_BIT = 0x00000001,
    VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT = 0x00000002,
    VK_COMMAND_POOL_CREATE_PROTECTED_BIT = 0x00000004,
    VK_COMMAND_POOL_CREATE_FLAG_BITS_MAX_ENUM = 0x7FFFFFFF
} VkCommandPoolCreateFlagBits;
typedef VkFlags VkCommandPoolCreateFlags;
typedef VkFlags VkCommandPoolResetFlags;
typedef VkFlags VkCommandBufferResetFlags;

typedef enum VkCommandBufferLevel {
    VK_COMMAND_BUFFER_LEVEL_PRIMARY = 0,
    VK_COMMAND_BUFFER_LEVEL_SECONDARY = 1,
    VK_COMMAND_BUFFER_LEVEL_MAX_ENUM = 0x7FFFFFFF
} VkCommandBufferLevel;

typedef enum VkCommandBufferUsageFlagBits {
    VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT = 0x00000001,
    VK_COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT = 0x00000002,
    VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT = 0x00000004,
    VK_COMMAND_BUFFER_USAGE_FLAG_BITS_MAX_ENUM = 0x7FFFFFFF
} VkCommandBufferUsageFlagBits;
typedef VkFlags VkCommandBufferUsageFlags;

typedef enum VkPipelineStageFlagBits {
    VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT = 0x00000001,
    VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT = 0x00000002,
    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT = 0x00000800,
    VK_PIPELINE_STAGE_TRANSFER_BIT = 0x00001000,
    VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT = 0x00002000,
    VK_PIPELINE_STAGE_HOST_BIT = 0x00004000,
    VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT = 0x00008000,
    VK_PIPELINE_STAGE_ALL_COMMANDS_BIT = 0x00010000,
    VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR = 0x02000000,
    VK_PIPELINE_STAGE_FLAG_BITS_MAX_ENUM = 0x7FFFFFFF
} VkPipelineStageFlagBits;
typedef VkFlags VkPipelineStageFlags;

typedef enum VkAccessFlagBits {
    VK_ACCESS_INDIRECT_COMMAND_READ_BIT = 0x00000001,
    VK_ACCESS_UNIFORM_READ_BIT = 0x00000008,
    VK_ACCESS_SHADER_READ_BIT = 0x00000020,
    VK_ACCESS_SHADER_WRITE_BIT = 0x00000040,
    VK_ACCESS_TRANSFER_READ_BIT = 0x00000800,
    VK_ACCESS_TRANSFER_WRITE_BIT = 0x00001000,
    VK_ACCESS_HOST_READ_BIT = 0x00002000,
    VK_ACCESS_HOST_WRITE_BIT = 0x00004000,
    VK_ACCESS_MEMORY_READ_BIT = 0x00008000,
    VK_ACCESS_MEMORY_WRITE_BIT = 0x00010000,
    VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR = 0x00200000,
    VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR = 0x00400000,
    VK_ACCESS_FLAG_BITS_MAX_ENUM = 0x7FFFFFFF
} VkAccessFlagBits;
typedef VkFlags VkAccessFlags;

typedef enum VkDependencyFlagBits {
    VK_DEPENDENCY_BY_REGION_BIT = 0x00000001,
    VK_DEPENDENCY_FLAG_BITS_MAX_ENUM = 0x7FFFFFFF
} VkDependencyFlagBits;
typedef VkFlags VkDependencyFlags;

typedef enum VkSemaphoreType {
    VK_SEMAPHORE_TYPE_BINARY = 0,
    VK_SEMAPHORE_TYPE_TIMELINE = 1,
    VK_SEMAPHORE_TYPE_MAX_ENUM = 0x7FFFFFFF
} VkSemaphoreType;

typedef enum VkSemaphoreWaitFlagBits {
    VK_SEMAPHORE_WAIT_ANY_BIT = 0x00000001,
    VK_SEMAPHORE_WAIT_FLAG_BITS_MAX_ENUM = 0x7FFFFFFF
} VkSemaphoreWaitFlagBits;
typedef VkFlags VkSemaphoreWaitFlags;
typedef VkFlags VkSemaphoreCreateFlags;
typedef VkFlags VkFenceCreateFlags;

typedef enum VkShaderStageFlagBits {
    VK_SHADER_STAGE_VERTEX_BIT = 0x00000001,
    VK_SHADER_STAGE_FRAGMENT_BIT = 0x00000010,
    VK_SHADER_STAGE_COMPUTE_BIT = 0x00000020,
    VK_SHADER_STAGE_ALL_GRAPHICS = 0x0000001F,
    VK_SHADER_STAGE_ALL = 0x7FFFFFFF,
    VK_SHADER_STAGE_FLAG_BITS_MAX_ENUM = 0x7FFFFFFF
} VkShaderStageFlagBits;
typedef VkFlags VkShaderStageFlags;

typedef enum VkSubgroupFeatureFlagBits {
    VK_SUBGROUP_FEATURE_BASIC_BIT = 0x00000001,
    VK_SUBGROUP_FEATURE_VOTE_BIT = 0x00000002,
    VK_SUBGROUP_FEATURE_ARITHMETIC_BIT = 0x00000004,
    VK_SUBGROUP_FEATURE_BALLOT_BIT = 0x00000008,
    VK_SUBGROUP_FEATURE_SHUFFLE_BIT = 0x00000010,
    VK_SUBGROUP_FEATURE_SHUFFLE_RELATIVE_BIT = 0x00000020,
    VK_SUBGROUP_FEATURE_CLUSTERED_BIT = 0x00000040,
    VK_SUBGROUP_FEATURE_QUAD_BIT = 0x00000080,
    VK_SUBGROUP_FEATURE_FLAG_BITS_MAX_ENUM = 0x7FFFFFFF
} VkSubgroupFeatureFlagBits;
typedef VkFlags VkSubgroupFeatureFlags;

typedef enum VkPipelineBindPoint {
    VK_PIPELINE_BIND_POINT_GRAPHICS = 0,
    VK_PIPELINE_BIND_POINT_COMPUTE = 1,
    VK_PIPELINE_BIND_POINT_MAX_ENUM = 0x7FFFFFFF
} VkPipelineBindPoint;

typedef enum VkPipelineCreateFlagBits {
    VK_PIPELINE_CREATE_DISABLE_OPTIMIZATION_BIT = 0x00000001,
    VK_PIPELINE_CREATE_FLAG_BITS_MAX_ENUM = 0x7FFFFFFF
} VkPipelineCreateFlagBits;
typedef VkFlags VkPipelineCreateFlags;
typedef VkFlags VkPipelineShaderStageCreateFlags;
typedef VkFlags VkPipelineLayoutCreateFlags;
typedef VkFlags VkPipelineCacheCreateFlags;
typedef VkFlags VkShaderModuleCreateFlags;
typedef VkFlags VkDeviceCreateFlags;
typedef VkFlags VkDeviceQueueCreateFlags;
typedef VkFlags VkMemoryMapFlags;
typedef VkFlags VkSampleCountFlags;
typedef VkFlags VkResolveModeFlags;

typedef enum VkPointClippingBehavior {
    VK_POINT_CLIPPING_BEHAVIOR_ALL_CLIP_PLANES = 0,
    VK_POINT_CLIPPING_BEHAVIOR_USER_CLIP_PLANES_ONLY = 1,
    VK_POINT_CLIPPING_BEHAVIOR_MAX_ENUM = 0x7FFFFFFF
} VkPointClippingBehavior;

typedef enum VkDriverId {
    VK_DRIVER_ID_NVIDIA_PROPRIETARY = 4,
    VK_DRIVER_ID_MESA_LLVMPIPE = 13,
    VK_DRIVER_ID_MAX_ENUM = 0x7FFFFFFF
} VkDriverId;

typedef enum VkShaderFloatControlsIndependence {
    VK_SHADER_FLOAT_CONTROLS_INDEPENDENCE_32_BIT_ONLY = 0,
    VK_SHADER_FLOAT_CONTROLS_INDEPENDENCE_ALL = 1,
    VK_SHADER_FLOAT_CONTROLS_INDEPENDENCE_NONE = 2,
    VK_SHADER_FLOAT_CONTROLS_INDEPENDENCE_MAX_ENUM = 0x7FFFFFFF
} VkShaderFloatControlsIndependence;

typedef enum VkFormat {
    VK_FORMAT_UNDEFINED = 0,
    VK_FORMAT_R32G32_SFLOAT = 103,
    VK_FORMAT_R32G32B32_SFLOAT = 106,
    VK_FORMAT_R32G32B32A32_SFLOAT = 109,
    VK_FORMAT_MAX_ENUM = 0x7FFFFFFF
} VkFormat;

typedef enum VkIndexType {
    VK_INDEX_TYPE_UINT16 = 0,
    VK_INDEX_TYPE_UINT32 = 1,
    VK_INDEX_TYPE_NONE_KHR = 1000165000,
    VK_INDEX_TYPE_MAX_ENUM = 0x7FFFFFFF
} VkIndexType;

// VK_KHR_acceleration_structure
typedef enum VkAccelerationStructureTypeKHR {
    VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR = 0,
    VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR = 1,
    VK_ACCELERATION_STRUCTURE_TYPE_GENERIC_KHR = 2,
    VK_ACCELERATION_STRUCTURE_TYPE_MAX_ENUM_KHR = 0x7FFFFFFF
} VkAccelerationStructureTypeKHR;

typedef enum VkGeometryTypeKHR {
    VK_GEOMETRY_TYPE_TRIANGLES_KHR = 0,
    VK_GEOMETRY_TYPE_AABBS_KHR = 1,
    VK_GEOMETRY_TYPE_INSTANCES_KHR = 2,
    VK_GEOMETRY_TYPE_MAX_ENUM_KHR = 0x7FFFFFFF
} VkGeometryTypeKHR;

typedef enum VkBuildAccelerationStructureModeKHR {
    VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR = 0,
    VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR = 1,
    VK_BUILD_ACCELERATION_STRUCTURE_MODE_MAX_ENUM_KHR = 0x7FFFFFFF
} VkBuildAccelerationStructureModeKHR;

typedef enum VkAccelerationStructureBuildTypeKHR {
    VK_ACCELERATION_STRUCTURE_BUILD_TYPE_HOST_KHR = 0,
    VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR = 1,
    VK_ACCELERATION_STRUCTURE_BUILD_TYPE_HOST_OR_DEVICE_KHR = 2,
    VK_ACCELERATION_STRUCTURE_BUILD_TYPE_MAX_ENUM_KHR = 0x7FFFFFFF
} VkAccelerationStructureBuildTypeKHR;

typedef enum VkGeometryFlagBitsKHR {
    VK_GEOMETRY_OPAQUE_BIT_KHR = 0x00000001,
    VK_GEOMETRY_NO_DUPLICATE_ANY_HIT_INVOCATION_BIT_KHR = 0x00000002,
    VK_GEOMETRY_FLAG_BITS_MAX_ENUM_KHR = 0x7FFFFFFF
} VkGeometryFlagBitsKHR;
typedef VkFlags VkGeometryFlagsKHR;

typedef enum VkGeometryInstanceFlagBitsKHR {
    VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR = 0x00000001,
    VK_GEOMETRY_INSTANCE_TRIANGLE_FLIP_FACING_BIT_KHR = 0x00000002,
    VK_GEOMETRY_INSTANCE_FORCE_OPAQUE_BIT_KHR = 0x00000004,
    VK_GEOMETRY_INSTANCE_FORCE_NO_OPAQUE_BIT_KHR = 0x00000008,
    VK_GEOMETRY_INSTANCE_FLAG_BITS_MAX_ENUM_KHR = 0x7FFFFFFF
} VkGeometryInstanceFlagBitsKHR;
typedef VkFlags VkGeometryInstanceFlagsKHR;

typedef enum VkBuildAccelerationStructureFlagBitsKHR {
    VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR = 0x00000001,
    VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_COMPACTION_BIT_KHR = 0x00000002,
    VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR = 0x00000004,
    VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR = 0x00000008,
    VK_BUILD_ACCELERATION_STRUCTURE_LOW_MEMORY_BIT_KHR = 0x00000010,
    VK_BUILD_ACCELERATION_STRUCTURE_FLAG_BITS_MAX_ENUM_KHR = 0x7FFFFFFF
} VkBuildAccelerationStructureFlagBitsKHR;
typedef VkFlags VkBuildAccelerationStructureFlagsKHR;
typedef VkFlags VkAccelerationStructureCreateFlagsKHR;

// --------------------------------- Structures ---------------------------------

typedef struct VkExtent3D {
    uint32_t width;
    uint32_t height;
    uint32_t depth;
} VkExtent3D;

typedef struct VkOffset3D {
    int32_t x;
    int32_t y;
    int32_t z;
} VkOffset3D;

typedef struct VkApplicationInfo {
    VkStructureType sType;
    const void *pNext;
    const char *pApplicationName;
    uint32_t applicationVersion;
    const char *pEngineName;
    uint32_t engineVersion;
    uint32_t apiVersion;
} VkApplicationInfo;

typedef struct VkInstanceCreateInfo {
    VkStructureType sType;
    const void *pNext;
    VkInstanceCreateFlags flags;
    const VkApplicationInfo *pApplicationInfo;
    uint32_t enabledLayerCount;
    const char *const *ppEnabledLayerNames;
    uint32_t enabledExtensionCount;
    const char *const *ppEnabledExtensionNames;
} VkInstanceCreateInfo;

typedef struct VkExtensionProperties {
    char extensionName[VK_MAX_EXTENSION_NAME_SIZE];
    uint32_t specVersion;
} VkExtensionProperties;

typedef struct VkLayerProperties {
    char layerName[VK_MAX_EXTENSION_NAME_SIZE];
    uint32_t specVersion;
    uint32_t implementationVersion;
    char description[VK_MAX_DESCRIPTION_SIZE];
} VkLayerProperties;

typedef struct VkPhysicalDeviceLimits {
    uint32_t maxImageDimension1D;
    uint32_t maxImageDimension2D;
    uint32_t maxImageDimension3D;
    uint32_t maxImageDimensionCube;
    uint32_t maxImageArrayLayers;
    uint32_t maxTexelBufferElements;
    uint32_t maxUniformBufferRange;
    uint32_t maxStorageBufferRange;
    uint32_t maxPushConstantsSize;
    uint32_t maxMemoryAllocationCount;
    uint32_t maxSamplerAllocationCount;
    VkDeviceSize bufferImageGranularity;
    VkDeviceSize sparseAddressSpaceSize;
    uint32_t maxBoundDescriptorSets;
    uint32_t maxPerStageDescriptorSamplers;
    uint32_t maxPerStageDescriptorUniformBuffers;
    uint32_t maxPerStageDescriptorStorageBuffers;
    uint32_t maxPerStageDescriptorSampledImages;
    uint32_t maxPerStageDescriptorStorageImages;
    uint32_t maxPerStageDescriptorInputAttachments;
    uint32_t maxPerStageResources;
    uint32_t maxDescriptorSetSamplers;
    uint32_t maxDescriptorSetUniformBuffers;
    uint32_t maxDescriptorSetUniformBuffersDynamic;
    uint32_t maxDescriptorSetStorageBuffers;
    uint32_t maxDescriptorSetStorageBuffersDynamic;
    uint32_t maxDescriptorSetSampledImages;
    uint32_t maxDescriptorSetStorageImages;
    uint32_t maxDescriptorSetInputAttachments;
    uint32_t maxVertexInputAttributes;
    uint32_t maxVertexInputBindings;
    uint32_t maxVertexInputAttributeOffset;
    uint32_t maxVertexInputBindingStride;
    uint32_t maxVertexOutputComponents;
    uint32_t maxTessellationGenerationLevel;
    uint32_t maxTessellationPatchSize;
    uint32_t maxTessellationControlPerVertexInputComponents;
    uint32_t maxTessellationControlPerVertexOutputComponents;
    uint32_t maxTessellationControlPerPatchOutputComponents;
    uint32_t maxTessellationControlTotalOutputComponents;
    uint32_t maxTessellationEvaluationInputComponents;
    uint32_t maxTessellationEvaluationOutputComponents;
    uint32_t maxGeometryShaderInvocations;
    uint32_t maxGeometryInputComponents;
    uint32_t maxGeometryOutputComponents;
    uint32_t maxGeometryOutputVertices;
    uint32_t maxGeometryTotalOutputComponents;
    uint32_t maxFragmentInputComponents;
    uint32_t maxFragmentOutputAttachments;
    uint32_t maxFragmentDualSrcAttachments;
    uint32_t maxFragmentCombinedOutputResources;
    uint32_t maxComputeSharedMemorySize;
    uint32_t maxComputeWorkGroupCount[3];
    uint32_t maxComputeWorkGroupInvocations;
    uint32_t maxComputeWorkGroupSize[3];
    uint32_t subPixelPrecisionBits;
    uint32_t subTexelPrecisionBits;
    uint32_t mipmapPrecisionBits;
    uint32_t maxDrawIndexedIndexValue;
    uint32_t maxDrawIndirectCount;
    float maxSamplerLodBias;
    float maxSamplerAnisotropy;
    uint32_t maxViewports;
    uint32_t maxViewportDimensions[2];
    float viewportBoundsRange[2];
    uint32_t viewportSubPixelBits;
    size_t minMemoryMapAlignment;
    VkDeviceSize minTexelBufferOffsetAlignment;
    VkDeviceSize minUniformBufferOffsetAlignment;
    VkDeviceSize minStorageBufferOffsetAlignment;
    int32_t minTexelOffset;
    uint32_t maxTexelOffset;
    int32_t minTexelGatherOffset;
    uint32_t maxTexelGatherOffset;
    float minInterpolationOffset;
    float maxInterpolationOffset;
    uint32_t subPixelInterpolationOffsetBits;
    uint32_t maxFramebufferWidth;
    uint32_t maxFramebufferHeight;
    uint32_t maxFramebufferLayers;
    VkSampleCountFlags framebufferColorSampleCounts;
    VkSampleCountFlags framebufferDepthSampleCounts;
    VkSampleCountFlags framebufferStencilSampleCounts;
    VkSampleCountFlags framebufferNoAttachmentsSampleCounts;
    uint32_t maxColorAttachments;
    VkSampleCountFlags sampledImageColorSampleCounts;
    VkSampleCountFlags sampledImageIntegerSampleCounts;
    VkSampleCountFlags sampledImageDepthSampleCounts;
    VkSampleCountFlags sampledImageStencilSampleCounts;
    VkSampleCountFlags storageImageSampleCounts;
    uint32_t maxSampleMaskWords;
    VkBool32 timestampComputeAndGraphics;
    float timestampPeriod;
    uint32_t maxClipDistances;
    uint32_t maxCullDistances;
    uint32_t maxCombinedClipAndCullDistances;
    uint32_t discreteQueuePriorities;
    float pointSizeRange[2];
    float lineWidthRange[2];
    float pointSizeGranularity;
    float lineWidthGranularity;
    VkBool32 strictLines;
    VkBool32 standardSampleLocations;
    VkDeviceSize optimalBufferCopyOffsetAlignment;
    VkDeviceSize optimalBufferCopyRowPitchAlignment;
    VkDeviceSize nonCoherentAtomSize;
} VkPhysicalDeviceLimits;

typedef struct VkPhysicalDeviceSparseProperties {
    VkBool32 residencyStandard2DBlockShape;
    VkBool32 residencyStandard2DMultisampleBlockShape;
    VkBool32 residencyStandard3DBlockShape;
    VkBool32 residencyAlignedMipSize;
    VkBool32 residencyNonResidentStrict;
} VkPhysicalDeviceSparseProperties;

typedef struct VkPhysicalDeviceProperties {
    uint32_t apiVersion;
    uint32_t driverVersion;
    uint32_t vendorID;
    uint32_t deviceID;
    VkPhysicalDeviceType deviceType;
    char deviceName[VK_MAX_PHYSICAL_DEVICE_NAME_SIZE];
    uint8_t pipelineCacheUUID[VK_UUID_SIZE];
    VkPhysicalDeviceLimits limits;
    VkPhysicalDeviceSparseProperties sparseProperties;
} VkPhysicalDeviceProperties;

typedef struct VkPhysicalDeviceProperties2 {
    VkStructureType sType;
    void *pNext;
    VkPhysicalDeviceProperties properties;
} VkPhysicalDeviceProperties2;

typedef struct VkPhysicalDeviceFeatures {
    VkBool32 robustBufferAccess;
    VkBool32 fullDrawIndexUint32;
    VkBool32 imageCubeArray;
    VkBool32 independentBlend;
    VkBool32 geometryShader;
    VkBool32 tessellationShader;
    VkBool32 sampleRateShading;
    VkBool32 dualSrcBlend;
    VkBool32 logicOp;
    VkBool32 multiDrawIndirect;
    VkBool32 drawIndirectFirstInstance;
    VkBool32 depthClamp;
    VkBool32 depthBiasClamp;
    VkBool32 fillModeNonSolid;
    VkBool32 depthBounds;
    VkBool32 wideLines;
    VkBool32 largePoints;
    VkBool32 alphaToOne;
    VkBool32 multiViewport;
    VkBool32 samplerAnisotropy;
    VkBool32 textureCompressionETC2;
    VkBool32 textureCompressionASTC_LDR;
    VkBool32 textureCompressionBC;
    VkBool32 occlusionQueryPrecise;
    VkBool32 pipelineStatisticsQuery;
    VkBool32 vertexPipelineStoresAndAtomics;
    VkBool32 fragmentStoresAndAtomics;
    VkBool32 shaderTessellationAndGeometryPointSize;
    VkBool32 shaderImageGatherExtended;
    VkBool32 shaderStorageImageExtendedFormats;
    VkBool32 shaderStorageImageMultisample;
    VkBool32 shaderStorageImageReadWithoutFormat;
    VkBool32 shaderStorageImageWriteWithoutFormat;
    VkBool32 shaderUniformBufferArrayDynamicIndexing;
    VkBool32 shaderSampledImageArrayDynamicIndexing;
    VkBool32 shaderStorageBufferArrayDynamicIndexing;
    VkBool32 shaderStorageImageArrayDynamicIndexing;
    VkBool32 shaderClipDistance;
    VkBool32 shaderCullDistance;
    VkBool32 shaderFloat64;
    VkBool32 shaderInt64;
    VkBool32 shaderInt16;
    VkBool32 shaderResourceResidency;
    VkBool32 shaderResourceMinLod;
    VkBool32 sparseBinding;
    VkBool32 sparseResidencyBuffer;
    VkBool32 sparseResidencyImage2D;
    VkBool32 sparseResidencyImage3D;
    VkBool32 sparseResidency2Samples;
    VkBool32 sparseResidency4Samples;
    VkBool32 sparseResidency8Samples;
    VkBool32 sparseResidency16Samples;
    VkBool32 sparseResidencyAliased;
    VkBool32 variableMultisampleRate;
    VkBool32 inheritedQueries;
} VkPhysicalDeviceFeatures;

typedef struct VkPhysicalDeviceFeatures2 {
    VkStructureType sType;
    void *pNext;
    VkPhysicalDeviceFeatures features;
} VkPhysicalDeviceFeatures2;

typedef struct VkPhysicalDeviceVulkan11Features {
    VkStructureType sType;
    void *pNext;
    VkBool32 storageBuffer16BitAccess;
    VkBool32 uniformAndStorageBuffer16BitAccess;
    VkBool32 storagePushConstant16;
    VkBool32 storageInputOutput16;
    VkBool32 multiview;
    VkBool32 multiviewGeometryShader;
    VkBool32 multiviewTessellationShader;
    VkBool32 variablePointersStorageBuffer;
    VkBool32 variablePointers;
    VkBool32 protectedMemory;
    VkBool32 samplerYcbcrConversion;
    VkBool32 shaderDrawParameters;
} VkPhysicalDeviceVulkan11Features;

typedef struct VkPhysicalDeviceVulkan11Properties {
    VkStructureType sType;
    void *pNext;
    uint8_t deviceUUID[VK_UUID_SIZE];
    uint8_t driverUUID[VK_UUID_SIZE];
    uint8_t deviceLUID[VK_LUID_SIZE];
    uint32_t deviceNodeMask;
    VkBool32 deviceLUIDValid;
    uint32_t subgroupSize;
    VkShaderStageFlags subgroupSupportedStages;
    VkSubgroupFeatureFlags subgroupSupportedOperations;
    VkBool32 subgroupQuadOperationsInAllStages;
    VkPointClippingBehavior pointClippingBehavior;
    uint32_t maxMultiviewViewCount;
    uint32_t maxMultiviewInstanceIndex;
    VkBool32 protectedNoFault;
    uint32_t maxPerSetDescriptors;
    VkDeviceSize maxMemoryAllocationSize;
} VkPhysicalDeviceVulkan11Properties;

typedef struct VkPhysicalDeviceVulkan12Features {
    VkStructureType sType;
    void *pNext;
    VkBool32 samplerMirrorClampToEdge;
    VkBool32 drawIndirectCount;
    VkBool32 storageBuffer8BitAccess;
    VkBool32 uniformAndStorageBuffer8BitAccess;
    VkBool32 storagePushConstant8;
    VkBool32 shaderBufferInt64Atomics;
    VkBool32 shaderSharedInt64Atomics;
    VkBool32 shaderFloat16;
    VkBool32 shaderInt8;
    VkBool32 descriptorIndexing;
    VkBool32 shaderInputAttachmentArrayDynamicIndexing;
    VkBool32 shaderUniformTexelBufferArrayDynamicIndexing;
    VkBool32 shaderStorageTexelBufferArrayDynamicIndexing;
    VkBool32 shaderUniformBufferArrayNonUniformIndexing;
    VkBool32 shaderSampledImageArrayNonUniformIndexing;
    VkBool32 shaderStorageBufferArrayNonUniformIndexing;
    VkBool32 shaderStorageImageArrayNonUniformIndexing;
    VkBool32 shaderInputAttachmentArrayNonUniformIndexing;
    VkBool32 shaderUniformTexelBufferArrayNonUniformIndexing;
    VkBool32 shaderStorageTexelBufferArrayNonUniformIndexing;
    VkBool32 descriptorBindingUniformBufferUpdateAfterBind;
    VkBool32 descriptorBindingSampledImageUpdateAfterBind;
    VkBool32 descriptorBindingStorageImageUpdateAfterBind;
    VkBool32 descriptorBindingStorageBufferUpdateAfterBind;
    VkBool32 descriptorBindingUniformTexelBufferUpdateAfterBind;
    VkBool32 descriptorBindingStorageTexelBufferUpdateAfterBind;
    VkBool32 descriptorBindingUpdateUnusedWhilePending;
    VkBool32 descriptorBindingPartiallyBound;
    VkBool32 descriptorBindingVariableDescriptorCount;
    VkBool32 runtimeDescriptorArray;
    VkBool32 samplerFilterMinmax;
    VkBool32 scalarBlockLayout;
    VkBool32 imagelessFramebuffer;
    VkBool32 uniformBufferStandardLayout;
    VkBool32 shaderSubgroupExtendedTypes;
    VkBool32 separateDepthStencilLayouts;
    VkBool32 hostQueryReset;
    VkBool32 timelineSemaphore;
    VkBool32 bufferDeviceAddress;
    VkBool32 bufferDeviceAddressCaptureReplay;
    VkBool32 bufferDeviceAddressMultiDevice;
    VkBool32 vulkanMemoryModel;
    VkBool32 vulkanMemoryModelDeviceScope;
    VkBool32 vulkanMemoryModelAvailabilityVisibilityChains;
    VkBool32 shaderOutputViewportIndex;
    VkBool32 shaderOutputLayer;
    VkBool32 subgroupBroadcastDynamicId;
} VkPhysicalDeviceVulkan12Features;

typedef struct VkConformanceVersion {
    uint8_t major;
    uint8_t minor;
    uint8_t subminor;
    uint8_t patch;
} VkConformanceVersion;

typedef struct VkPhysicalDeviceVulkan12Properties {
    VkStructureType sType;
    void *pNext;
    VkDriverId driverID;
    char driverName[VK_MAX_DRIVER_NAME_SIZE];
    char driverInfo[VK_MAX_DRIVER_INFO_SIZE];
    VkConformanceVersion conformanceVersion;
    VkShaderFloatControlsIndependence denormBehaviorIndependence;
    VkShaderFloatControlsIndependence roundingModeIndependence;
    VkBool32 shaderSignedZeroInfNanPreserveFloat16;
    VkBool32 shaderSignedZeroInfNanPreserveFloat32;
    VkBool32 shaderSignedZeroInfNanPreserveFloat64;
    VkBool32 shaderDenormPreserveFloat16;
    VkBool32 shaderDenormPreserveFloat32;
    VkBool32 shaderDenormPreserveFloat64;
    VkBool32 shaderDenormFlushToZeroFloat16;
    VkBool32 shaderDenormFlushToZeroFloat32;
    VkBool32 shaderDenormFlushToZeroFloat64;
    VkBool32 shaderRoundingModeRTEFloat16;
    VkBool32 shaderRoundingModeRTEFloat32;
    VkBool32 shaderRoundingModeRTEFloat64;
    VkBool32 shaderRoundingModeRTZFloat16;
    VkBool32 shaderRoundingModeRTZFloat32;
    VkBool32 shaderRoundingModeRTZFloat64;
    uint32_t maxUpdateAfterBindDescriptorsInAllPools;
    VkBool32 shaderUniformBufferArrayNonUniformIndexingNative;
    VkBool32 shaderSampledImageArrayNonUniformIndexingNative;
    VkBool32 shaderStorageBufferArrayNonUniformIndexingNative;
    VkBool32 shaderStorageImageArrayNonUniformIndexingNative;
    VkBool32 shaderInputAttachmentArrayNonUniformIndexingNative;
    VkBool32 robustBufferAccessUpdateAfterBind;
    VkBool32 quadDivergentImplicitLod;
    uint32_t maxPerStageDescriptorUpdateAfterBindSamplers;
    uint32_t maxPerStageDescriptorUpdateAfterBindUniformBuffers;
    uint32_t maxPerStageDescriptorUpdateAfterBindStorageBuffers;
    uint32_t maxPerStageDescriptorUpdateAfterBindSampledImages;
    uint32_t maxPerStageDescriptorUpdateAfterBindStorageImages;
    uint32_t maxPerStageDescriptorUpdateAfterBindInputAttachments;
    uint32_t maxPerStageUpdateAfterBindResources;
    uint32_t maxDescriptorSetUpdateAfterBindSamplers;
    uint32_t maxDescriptorSetUpdateAfterBindUniformBuffers;
    uint32_t maxDescriptorSetUpdateAfterBindUniformBuffersDynamic;
    uint32_t maxDescriptorSetUpdateAfterBindStorageBuffers;
    uint32_t maxDescriptorSetUpdateAfterBindStorageBuffersDynamic;
    uint32_t maxDescriptorSetUpdateAfterBindSampledImages;
    uint32_t maxDescriptorSetUpdateAfterBindStorageImages;
    uint32_t maxDescriptorSetUpdateAfterBindInputAttachments;
    VkResolveModeFlags supportedDepthResolveModes;
    VkResolveModeFlags supportedStencilResolveModes;
    VkBool32 independentResolveNone;
    VkBool32 independentResolve;
    VkBool32 filterMinmaxSingleComponentFormats;
    VkBool32 filterMinmaxImageComponentMapping;
    uint64_t maxTimelineSemaphoreValueDifference;
    VkSampleCountFlags framebufferIntegerColorSampleCounts;
} VkPhysicalDeviceVulkan12Properties;

typedef struct VkPhysicalDeviceVulkan13Features {
    VkStructureType sType;
    void *pNext;
    VkBool32 robustImageAccess;
    VkBool32 inlineUniformBlock;
    VkBool32 descriptorBindingInlineUniformBlockUpdateAfterBind;
    VkBool32 pipelineCreationCacheControl;
    VkBool32 privateData;
    VkBool32 shaderDemoteToHelperInvocation;
    VkBool32 shaderTerminateInvocation;
    VkBool32 subgroupSizeControl;
    VkBool32 computeFullSubgroups;
    VkBool32 synchronization2;
    VkBool32 textureCompressionASTC_HDR;
    VkBool32 shaderZeroInitializeWorkgroupMemory;
    VkBool32 dynamicRendering;
    VkBool32 shaderIntegerDotProduct;
    VkBool32 maintenance4;
} VkPhysicalDeviceVulkan13Features;

typedef struct VkPhysicalDeviceVulkan13Properties {
    VkStructureType sType;
    void *pNext;
    uint32_t minSubgroupSize;
    uint32_t maxSubgroupSize;
    uint32_t maxComputeWorkgroupSubgroups;
    VkShaderStageFlags requiredSubgroupSizeStages;
    uint32_t maxInlineUniformBlockSize;
    uint32_t maxPerStageDescriptorInlineUniformBlocks;
    uint32_t maxPerStageDescriptorUpdateAfterBindInlineUniformBlocks;
    uint32_t maxDescriptorSetInlineUniformBlocks;
    uint32_t maxDescriptorSetUpdateAfterBindInlineUniformBlocks;
    uint32_t maxInlineUniformTotalSize;
    VkBool32 integerDotProduct8BitUnsignedAccelerated;
    VkBool32 integerDotProduct8BitSignedAccelerated;
    VkBool32 integerDotProduct8BitMixedSignednessAccelerated;
    VkBool32 integerDotProduct4x8BitPackedUnsignedAccelerated;
    VkBool32 integerDotProduct4x8BitPackedSignedAccelerated;
    VkBool32 integerDotProduct4x8BitPackedMixedSignednessAccelerated;
    VkBool32 integerDotProduct16BitUnsignedAccelerated;
    VkBool32 integerDotProduct16BitSignedAccelerated;
    VkBool32 integerDotProduct16BitMixedSignednessAccelerated;
    VkBool32 integerDotProduct32BitUnsignedAccelerated;
    VkBool32 integerDotProduct32BitSignedAccelerated;
    VkBool32 integerDotProduct32BitMixedSignednessAccelerated;
    VkBool32 integerDotProduct64BitUnsignedAccelerated;
    VkBool32 integerDotProduct64BitSignedAccelerated;
    VkBool32 integerDotProduct64BitMixedSignednessAccelerated;
    VkBool32 integerDotProductAccumulatingSaturating8BitUnsignedAccelerated;
    VkBool32 integerDotProductAccumulatingSaturating8BitSignedAccelerated;
    VkBool32 integerDotProductAccumulatingSaturating8BitMixedSignednessAccelerated;
    VkBool32 integerDotProductAccumulatingSaturating4x8BitPackedUnsignedAccelerated;
    VkBool32 integerDotProductAccumulatingSaturating4x8BitPackedSignedAccelerated;
    VkBool32 integerDotProductAccumulatingSaturating4x8BitPackedMixedSignednessAccelerated;
    VkBool32 integerDotProductAccumulatingSaturating16BitUnsignedAccelerated;
    VkBool32 integerDotProductAccumulatingSaturating16BitSignedAccelerated;
    VkBool32 integerDotProductAccumulatingSaturating16BitMixedSignednessAccelerated;
    VkBool32 integerDotProductAccumulatingSaturating32BitUnsignedAccelerated;
    VkBool32 integerDotProductAccumulatingSaturating32BitSignedAccelerated;
    VkBool32 integerDotProductAccumulatingSaturating32BitMixedSignednessAccelerated;
    VkBool32 integerDotProductAccumulatingSaturating64BitUnsignedAccelerated;
    VkBool32 integerDotProductAccumulatingSaturating64BitSignedAccelerated;
    VkBool32 integerDotProductAccumulatingSaturating64BitMixedSignednessAccelerated;
    VkDeviceSize storageTexelBufferOffsetAlignmentBytes;
    VkBool32 storageTexelBufferOffsetSingleTexelAlignment;
    VkDeviceSize uniformTexelBufferOffsetAlignmentBytes;
    VkBool32 uniformTexelBufferOffsetSingleTexelAlignment;
    VkDeviceSize maxBufferSize;
} VkPhysicalDeviceVulkan13Properties;

typedef struct VkPhysicalDeviceSubgroupProperties {
    VkStructureType sType;
    void *pNext;
    uint32_t subgroupSize;
    VkShaderStageFlags supportedStages;
    VkSubgroupFeatureFlags supportedOperations;
    VkBool32 quadOperationsInAllStages;
} VkPhysicalDeviceSubgroupProperties;

typedef struct VkPhysicalDeviceShaderAtomicFloatFeaturesEXT {
    VkStructureType sType;
    void *pNext;
    VkBool32 shaderBufferFloat32Atomics;
    VkBool32 shaderBufferFloat32AtomicAdd;
    VkBool32 shaderBufferFloat64Atomics;
    VkBool32 shaderBufferFloat64AtomicAdd;
    VkBool32 shaderSharedFloat32Atomics;
    VkBool32 shaderSharedFloat32AtomicAdd;
    VkBool32 shaderSharedFloat64Atomics;
    VkBool32 shaderSharedFloat64AtomicAdd;
    VkBool32 shaderImageFloat32Atomics;
    VkBool32 shaderImageFloat32AtomicAdd;
    VkBool32 sparseImageFloat32Atomics;
    VkBool32 sparseImageFloat32AtomicAdd;
} VkPhysicalDeviceShaderAtomicFloatFeaturesEXT;

typedef struct VkQueueFamilyProperties {
    VkQueueFlags queueFlags;
    uint32_t queueCount;
    uint32_t timestampValidBits;
    VkExtent3D minImageTransferGranularity;
} VkQueueFamilyProperties;

typedef struct VkMemoryType {
    VkMemoryPropertyFlags propertyFlags;
    uint32_t heapIndex;
} VkMemoryType;

typedef struct VkMemoryHeap {
    VkDeviceSize size;
    VkMemoryHeapFlags flags;
} VkMemoryHeap;

typedef struct VkPhysicalDeviceMemoryProperties {
    uint32_t memoryTypeCount;
    VkMemoryType memoryTypes[VK_MAX_MEMORY_TYPES];
    uint32_t memoryHeapCount;
    VkMemoryHeap memoryHeaps[VK_MAX_MEMORY_HEAPS];
} VkPhysicalDeviceMemoryProperties;

typedef struct VkDeviceQueueCreateInfo {
    VkStructureType sType;
    const void *pNext;
    VkDeviceQueueCreateFlags flags;
    uint32_t queueFamilyIndex;
    uint32_t queueCount;
    const float *pQueuePriorities;
} VkDeviceQueueCreateInfo;

typedef struct VkDeviceCreateInfo {
    VkStructureType sType;
    const void *pNext;
    VkDeviceCreateFlags flags;
    uint32_t queueCreateInfoCount;
    const VkDeviceQueueCreateInfo *pQueueCreateInfos;
    uint32_t enabledLayerCount;
    const char *const *ppEnabledLayerNames;
    uint32_t enabledExtensionCount;
    const char *const *ppEnabledExtensionNames;
    const VkPhysicalDeviceFeatures *pEnabledFeatures;
} VkDeviceCreateInfo;

typedef struct VkCommandPoolCreateInfo {
    VkStructureType sType;
    const void *pNext;
    VkCommandPoolCreateFlags flags;
    uint32_t queueFamilyIndex;
} VkCommandPoolCreateInfo;

typedef struct VkCommandBufferAllocateInfo {
    VkStructureType sType;
    const void *pNext;
    VkCommandPool commandPool;
    VkCommandBufferLevel level;
    uint32_t commandBufferCount;
} VkCommandBufferAllocateInfo;

typedef struct VkCommandBufferBeginInfo {
    VkStructureType sType;
    const void *pNext;
    VkCommandBufferUsageFlags flags;
    const VkCommandBufferInheritanceInfo *pInheritanceInfo;
} VkCommandBufferBeginInfo;

typedef struct VkSubmitInfo {
    VkStructureType sType;
    const void *pNext;
    uint32_t waitSemaphoreCount;
    const VkSemaphore *pWaitSemaphores;
    const VkPipelineStageFlags *pWaitDstStageMask;
    uint32_t commandBufferCount;
    const VkCommandBuffer *pCommandBuffers;
    uint32_t signalSemaphoreCount;
    const VkSemaphore *pSignalSemaphores;
} VkSubmitInfo;

typedef struct VkTimelineSemaphoreSubmitInfo {
    VkStructureType sType;
    const void *pNext;
    uint32_t waitSemaphoreValueCount;
    const uint64_t *pWaitSemaphoreValues;
    uint32_t signalSemaphoreValueCount;
    const uint64_t *pSignalSemaphoreValues;
} VkTimelineSemaphoreSubmitInfo;

typedef struct VkSemaphoreCreateInfo {
    VkStructureType sType;
    const void *pNext;
    VkSemaphoreCreateFlags flags;
} VkSemaphoreCreateInfo;

typedef struct VkSemaphoreTypeCreateInfo {
    VkStructureType sType;
    const void *pNext;
    VkSemaphoreType semaphoreType;
    uint64_t initialValue;
} VkSemaphoreTypeCreateInfo;

typedef struct VkSemaphoreWaitInfo {
    VkStructureType sType;
    const void *pNext;
    VkSemaphoreWaitFlags flags;
    uint32_t semaphoreCount;
    const VkSemaphore *pSemaphores;
    const uint64_t *pValues;
} VkSemaphoreWaitInfo;

typedef struct VkSemaphoreSignalInfo {
    VkStructureType sType;
    const void *pNext;
    VkSemaphore semaphore;
    uint64_t value;
} VkSemaphoreSignalInfo;

typedef struct VkFenceCreateInfo {
    VkStructureType sType;
    const void *pNext;
    VkFenceCreateFlags flags;
} VkFenceCreateInfo;

typedef struct VkBufferCreateInfo {
    VkStructureType sType;
    const void *pNext;
    VkBufferCreateFlags flags;
    VkDeviceSize size;
    VkBufferUsageFlags usage;
    VkSharingMode sharingMode;
    uint32_t queueFamilyIndexCount;
    const uint32_t *pQueueFamilyIndices;
} VkBufferCreateInfo;

typedef struct VkMemoryRequirements {
    VkDeviceSize size;
    VkDeviceSize alignment;
    uint32_t memoryTypeBits;
} VkMemoryRequirements;

typedef struct VkImageSubresource {
    VkImageAspectFlags aspectMask;
    uint32_t mipLevel;
    uint32_t arrayLayer;
} VkImageSubresource;

typedef struct VkSparseMemoryBind {
    VkDeviceSize resourceOffset;
    VkDeviceSize size;
    VkDeviceMemory memory;
    VkDeviceSize memoryOffset;
    VkSparseMemoryBindFlags flags;
} VkSparseMemoryBind;

typedef struct VkSparseBufferMemoryBindInfo {
    VkBuffer buffer;
    uint32_t bindCount;
    const VkSparseMemoryBind *pBinds;
} VkSparseBufferMemoryBindInfo;

typedef struct VkSparseImageOpaqueMemoryBindInfo {
    VkImage image;
    uint32_t bindCount;
    const VkSparseMemoryBind *pBinds;
} VkSparseImageOpaqueMemoryBindInfo;

typedef struct VkSparseImageMemoryBind {
    VkImageSubresource subresource;
    VkOffset3D offset;
    VkExtent3D extent;
    VkDeviceMemory memory;
    VkDeviceSize memoryOffset;
    VkSparseMemoryBindFlags flags;
} VkSparseImageMemoryBind;

typedef struct VkSparseImageMemoryBindInfo {
    VkImage image;
    uint32_t bindCount;
    const VkSparseImageMemoryBind *pBinds;
} VkSparseImageMemoryBindInfo;

typedef struct VkBindSparseInfo {
    VkStructureType sType;
    const void *pNext;
    uint32_t waitSemaphoreCount;
    const VkSemaphore *pWaitSemaphores;
    uint32_t bufferBindCount;
    const VkSparseBufferMemoryBindInfo *pBufferBinds;
    uint32_t imageOpaqueBindCount;
    const VkSparseImageOpaqueMemoryBindInfo *pImageOpaqueBinds;
    uint32_t imageBindCount;
    const VkSparseImageMemoryBindInfo *pImageBinds;
    uint32_t signalSemaphoreCount;
    const VkSemaphore *pSignalSemaphores;
} VkBindSparseInfo;

typedef struct VkMemoryAllocateInfo {
    VkStructureType sType;
    const void *pNext;
    VkDeviceSize allocationSize;
    uint32_t memoryTypeIndex;
} VkMemoryAllocateInfo;

typedef struct VkMemoryAllocateFlagsInfo {
    VkStructureType sType;
    const void *pNext;
    VkMemoryAllocateFlags flags;
    uint32_t deviceMask;
} VkMemoryAllocateFlagsInfo;

typedef struct VkBufferDeviceAddressInfo {
    VkStructureType sType;
    const void *pNext;
    VkBuffer buffer;
} VkBufferDeviceAddressInfo;

typedef struct VkMappedMemoryRange {
    VkStructureType sType;
    const void *pNext;
    VkDeviceMemory memory;
    VkDeviceSize offset;
    VkDeviceSize size;
} VkMappedMemoryRange;

typedef struct VkBufferCopy {
    VkDeviceSize srcOffset;
    VkDeviceSize dstOffset;
    VkDeviceSize size;
} VkBufferCopy;

typedef struct VkMemoryBarrier {
    VkStructureType sType;
    const void *pNext;
    VkAccessFlags srcAccessMask;
    VkAccessFlags dstAccessMask;
} VkMemoryBarrier;

typedef struct VkBufferMemoryBarrier {
    VkStructureType sType;
    const void *pNext;
    VkAccessFlags srcAccessMask;
    VkAccessFlags dstAccessMask;
    uint32_t srcQueueFamilyIndex;
    uint32_t dstQueueFamilyIndex;
    VkBuffer buffer;
    VkDeviceSize offset;
    VkDeviceSize size;
} VkBufferMemoryBarrier;

typedef struct VkShaderModuleCreateInfo {
    VkStructureType sType;
    const void *pNext;
    VkShaderModuleCreateFlags flags;
    size_t codeSize;
    const uint32_t *pCode;
} VkShaderModuleCreateInfo;

typedef struct VkSpecializationMapEntry {
    uint32_t constantID;
    uint32_t offset;
    size_t size;
} VkSpecializationMapEntry;

typedef struct VkSpecializationInfo {
    uint32_t mapEntryCount;
    const VkSpecializationMapEntry *pMapEntries;
    size_t dataSize;
    const void *pData;
} VkSpecializationInfo;

typedef struct VkPipelineShaderStageCreateInfo {
    VkStructureType sType;
    const void *pNext;
    VkPipelineShaderStageCreateFlags flags;
    VkShaderStageFlagBits stage;
    VkShaderModule module;
    const char *pName;
    const VkSpecializationInfo *pSpecializationInfo;
} VkPipelineShaderStageCreateInfo;

typedef struct VkComputePipelineCreateInfo {
    VkStructureType sType;
    const void *pNext;
    VkPipelineCreateFlags flags;
    VkPipelineShaderStageCreateInfo stage;
    VkPipelineLayout layout;
    VkPipeline basePipelineHandle;
    int32_t basePipelineIndex;
} VkComputePipelineCreateInfo;

typedef struct VkPushConstantRange {
    VkShaderStageFlags stageFlags;
    uint32_t offset;
    uint32_t size;
} VkPushConstantRange;

typedef struct VkPipelineLayoutCreateInfo {
    VkStructureType sType;
    const void *pNext;
    VkPipelineLayoutCreateFlags flags;
    uint32_t setLayoutCount;
    const VkDescriptorSetLayout *pSetLayouts;
    uint32_t pushConstantRangeCount;
    const VkPushConstantRange *pPushConstantRanges;
} VkPipelineLayoutCreateInfo;

typedef struct VkPipelineCacheCreateInfo {
    VkStructureType sType;
    const void *pNext;
    VkPipelineCacheCreateFlags flags;
    size_t initialDataSize;
    const void *pInitialData;
} VkPipelineCacheCreateInfo;

// VK_KHR_acceleration_structure & VK_KHR_ray_query
typedef union VkDeviceOrHostAddressConstKHR {
    VkDeviceAddress deviceAddress;
    const void *hostAddress;
} VkDeviceOrHostAddressConstKHR;

typedef union VkDeviceOrHostAddressKHR {
    VkDeviceAddress deviceAddress;
    void *hostAddress;
} VkDeviceOrHostAddressKHR;

typedef struct VkAccelerationStructureBuildRangeInfoKHR {
    uint32_t primitiveCount;
    uint32_t primitiveOffset;
    uint32_t firstVertex;
    uint32_t transformOffset;
} VkAccelerationStructureBuildRangeInfoKHR;

typedef struct VkAccelerationStructureGeometryTrianglesDataKHR {
    VkStructureType sType;
    const void *pNext;
    VkFormat vertexFormat;
    VkDeviceOrHostAddressConstKHR vertexData;
    VkDeviceSize vertexStride;
    uint32_t maxVertex;
    VkIndexType indexType;
    VkDeviceOrHostAddressConstKHR indexData;
    VkDeviceOrHostAddressConstKHR transformData;
} VkAccelerationStructureGeometryTrianglesDataKHR;

typedef struct VkAccelerationStructureGeometryAabbsDataKHR {
    VkStructureType sType;
    const void *pNext;
    VkDeviceOrHostAddressConstKHR data;
    VkDeviceSize stride;
} VkAccelerationStructureGeometryAabbsDataKHR;

typedef struct VkAccelerationStructureGeometryInstancesDataKHR {
    VkStructureType sType;
    const void *pNext;
    VkBool32 arrayOfPointers;
    VkDeviceOrHostAddressConstKHR data;
} VkAccelerationStructureGeometryInstancesDataKHR;

typedef union VkAccelerationStructureGeometryDataKHR {
    VkAccelerationStructureGeometryTrianglesDataKHR triangles;
    VkAccelerationStructureGeometryAabbsDataKHR aabbs;
    VkAccelerationStructureGeometryInstancesDataKHR instances;
} VkAccelerationStructureGeometryDataKHR;

typedef struct VkAccelerationStructureGeometryKHR {
    VkStructureType sType;
    const void *pNext;
    VkGeometryTypeKHR geometryType;
    VkAccelerationStructureGeometryDataKHR geometry;
    VkGeometryFlagsKHR flags;
} VkAccelerationStructureGeometryKHR;

typedef struct VkAccelerationStructureBuildGeometryInfoKHR {
    VkStructureType sType;
    const void *pNext;
    VkAccelerationStructureTypeKHR type;
    VkBuildAccelerationStructureFlagsKHR flags;
    VkBuildAccelerationStructureModeKHR mode;
    VkAccelerationStructureKHR srcAccelerationStructure;
    VkAccelerationStructureKHR dstAccelerationStructure;
    uint32_t geometryCount;
    const VkAccelerationStructureGeometryKHR *pGeometries;
    const VkAccelerationStructureGeometryKHR *const *ppGeometries;
    VkDeviceOrHostAddressKHR scratchData;
} VkAccelerationStructureBuildGeometryInfoKHR;

typedef struct VkAccelerationStructureCreateInfoKHR {
    VkStructureType sType;
    const void *pNext;
    VkAccelerationStructureCreateFlagsKHR createFlags;
    VkBuffer buffer;
    VkDeviceSize offset;
    VkDeviceSize size;
    VkAccelerationStructureTypeKHR type;
    VkDeviceAddress deviceAddress;
} VkAccelerationStructureCreateInfoKHR;

typedef struct VkAccelerationStructureDeviceAddressInfoKHR {
    VkStructureType sType;
    const void *pNext;
    VkAccelerationStructureKHR accelerationStructure;
} VkAccelerationStructureDeviceAddressInfoKHR;

typedef struct VkAccelerationStructureBuildSizesInfoKHR {
    VkStructureType sType;
    void *pNext;
    VkDeviceSize accelerationStructureSize;
    VkDeviceSize updateScratchSize;
    VkDeviceSize buildScratchSize;
} VkAccelerationStructureBuildSizesInfoKHR;

typedef struct VkTransformMatrixKHR {
    float matrix[3][4];
} VkTransformMatrixKHR;

typedef struct VkAabbPositionsKHR {
    float minX;
    float minY;
    float minZ;
    float maxX;
    float maxY;
    float maxZ;
} VkAabbPositionsKHR;

typedef struct VkAccelerationStructureInstanceKHR {
    VkTransformMatrixKHR transform;
    uint32_t instanceCustomIndex : 24;
    uint32_t mask : 8;
    uint32_t instanceShaderBindingTableRecordOffset : 24;
    VkGeometryInstanceFlagsKHR flags : 8;
    uint64_t accelerationStructureReference;
} VkAccelerationStructureInstanceKHR;

typedef struct VkPhysicalDeviceAccelerationStructureFeaturesKHR {
    VkStructureType sType;
    void *pNext;
    VkBool32 accelerationStructure;
    VkBool32 accelerationStructureCaptureReplay;
    VkBool32 accelerationStructureIndirectBuild;
    VkBool32 accelerationStructureHostCommands;
    VkBool32 descriptorBindingAccelerationStructureUpdateAfterBind;
} VkPhysicalDeviceAccelerationStructureFeaturesKHR;

typedef struct VkPhysicalDeviceAccelerationStructurePropertiesKHR {
    VkStructureType sType;
    void *pNext;
    uint64_t maxGeometryCount;
    uint64_t maxInstanceCount;
    uint64_t maxPrimitiveCount;
    uint32_t maxPerStageDescriptorAccelerationStructures;
    uint32_t maxPerStageDescriptorUpdateAfterBindAccelerationStructures;
    uint32_t maxDescriptorSetAccelerationStructures;
    uint32_t maxDescriptorSetUpdateAfterBindAccelerationStructures;
    uint32_t minAccelerationStructureScratchOffsetAlignment;
} VkPhysicalDeviceAccelerationStructurePropertiesKHR;

typedef struct VkPhysicalDeviceRayQueryFeaturesKHR {
    VkStructureType sType;
    void *pNext;
    VkBool32 rayQuery;
} VkPhysicalDeviceRayQueryFeaturesKHR;

// ------------------------- Function pointer types -------------------------

typedef void (VKAPI_PTR *PFN_vkVoidFunction)(void);
typedef PFN_vkVoidFunction (VKAPI_PTR *PFN_vkGetInstanceProcAddr)(VkInstance instance, const char *pName);
typedef PFN_vkVoidFunction (VKAPI_PTR *PFN_vkGetDeviceProcAddr)(VkDevice device, const char *pName);
typedef VkResult (VKAPI_PTR *PFN_vkCreateInstance)(const VkInstanceCreateInfo *pCreateInfo, const VkAllocationCallbacks *pAllocator, VkInstance *pInstance);
typedef VkResult (VKAPI_PTR *PFN_vkEnumerateInstanceVersion)(uint32_t *pApiVersion);
typedef VkResult (VKAPI_PTR *PFN_vkEnumerateInstanceExtensionProperties)(const char *pLayerName, uint32_t *pPropertyCount, VkExtensionProperties *pProperties);
typedef VkResult (VKAPI_PTR *PFN_vkEnumerateInstanceLayerProperties)(uint32_t *pPropertyCount, VkLayerProperties *pProperties);
typedef void (VKAPI_PTR *PFN_vkDestroyInstance)(VkInstance instance, const VkAllocationCallbacks *pAllocator);
typedef VkResult (VKAPI_PTR *PFN_vkEnumeratePhysicalDevices)(VkInstance instance, uint32_t *pPhysicalDeviceCount, VkPhysicalDevice *pPhysicalDevices);
typedef void (VKAPI_PTR *PFN_vkGetPhysicalDeviceProperties)(VkPhysicalDevice physicalDevice, VkPhysicalDeviceProperties *pProperties);
typedef void (VKAPI_PTR *PFN_vkGetPhysicalDeviceProperties2)(VkPhysicalDevice physicalDevice, VkPhysicalDeviceProperties2 *pProperties);
typedef void (VKAPI_PTR *PFN_vkGetPhysicalDeviceFeatures2)(VkPhysicalDevice physicalDevice, VkPhysicalDeviceFeatures2 *pFeatures);
typedef void (VKAPI_PTR *PFN_vkGetPhysicalDeviceQueueFamilyProperties)(VkPhysicalDevice physicalDevice, uint32_t *pQueueFamilyPropertyCount, VkQueueFamilyProperties *pQueueFamilyProperties);
typedef void (VKAPI_PTR *PFN_vkGetPhysicalDeviceMemoryProperties)(VkPhysicalDevice physicalDevice, VkPhysicalDeviceMemoryProperties *pMemoryProperties);
typedef VkResult (VKAPI_PTR *PFN_vkEnumerateDeviceExtensionProperties)(VkPhysicalDevice physicalDevice, const char *pLayerName, uint32_t *pPropertyCount, VkExtensionProperties *pProperties);
typedef VkResult (VKAPI_PTR *PFN_vkCreateDevice)(VkPhysicalDevice physicalDevice, const VkDeviceCreateInfo *pCreateInfo, const VkAllocationCallbacks *pAllocator, VkDevice *pDevice);
typedef void (VKAPI_PTR *PFN_vkDestroyDevice)(VkDevice device, const VkAllocationCallbacks *pAllocator);
typedef VkResult (VKAPI_PTR *PFN_vkDeviceWaitIdle)(VkDevice device);
typedef void (VKAPI_PTR *PFN_vkGetDeviceQueue)(VkDevice device, uint32_t queueFamilyIndex, uint32_t queueIndex, VkQueue *pQueue);
typedef VkResult (VKAPI_PTR *PFN_vkCreateCommandPool)(VkDevice device, const VkCommandPoolCreateInfo *pCreateInfo, const VkAllocationCallbacks *pAllocator, VkCommandPool *pCommandPool);
typedef void (VKAPI_PTR *PFN_vkDestroyCommandPool)(VkDevice device, VkCommandPool commandPool, const VkAllocationCallbacks *pAllocator);
typedef VkResult (VKAPI_PTR *PFN_vkResetCommandPool)(VkDevice device, VkCommandPool commandPool, VkCommandPoolResetFlags flags);
typedef VkResult (VKAPI_PTR *PFN_vkAllocateCommandBuffers)(VkDevice device, const VkCommandBufferAllocateInfo *pAllocateInfo, VkCommandBuffer *pCommandBuffers);
typedef void (VKAPI_PTR *PFN_vkFreeCommandBuffers)(VkDevice device, VkCommandPool commandPool, uint32_t commandBufferCount, const VkCommandBuffer *pCommandBuffers);
typedef VkResult (VKAPI_PTR *PFN_vkBeginCommandBuffer)(VkCommandBuffer commandBuffer, const VkCommandBufferBeginInfo *pBeginInfo);
typedef VkResult (VKAPI_PTR *PFN_vkEndCommandBuffer)(VkCommandBuffer commandBuffer);
typedef VkResult (VKAPI_PTR *PFN_vkResetCommandBuffer)(VkCommandBuffer commandBuffer, VkCommandBufferResetFlags flags);
typedef VkResult (VKAPI_PTR *PFN_vkQueueSubmit)(VkQueue queue, uint32_t submitCount, const VkSubmitInfo *pSubmits, VkFence fence);
typedef VkResult (VKAPI_PTR *PFN_vkQueueWaitIdle)(VkQueue queue);
typedef VkResult (VKAPI_PTR *PFN_vkQueueBindSparse)(VkQueue queue, uint32_t bindInfoCount, const VkBindSparseInfo *pBindInfo, VkFence fence);
typedef VkResult (VKAPI_PTR *PFN_vkCreateSemaphore)(VkDevice device, const VkSemaphoreCreateInfo *pCreateInfo, const VkAllocationCallbacks *pAllocator, VkSemaphore *pSemaphore);
typedef void (VKAPI_PTR *PFN_vkDestroySemaphore)(VkDevice device, VkSemaphore semaphore, const VkAllocationCallbacks *pAllocator);
typedef VkResult (VKAPI_PTR *PFN_vkWaitSemaphores)(VkDevice device, const VkSemaphoreWaitInfo *pWaitInfo, uint64_t timeout);
typedef VkResult (VKAPI_PTR *PFN_vkSignalSemaphore)(VkDevice device, const VkSemaphoreSignalInfo *pSignalInfo);
typedef VkResult (VKAPI_PTR *PFN_vkGetSemaphoreCounterValue)(VkDevice device, VkSemaphore semaphore, uint64_t *pValue);
typedef VkResult (VKAPI_PTR *PFN_vkCreateBuffer)(VkDevice device, const VkBufferCreateInfo *pCreateInfo, const VkAllocationCallbacks *pAllocator, VkBuffer *pBuffer);
typedef void (VKAPI_PTR *PFN_vkDestroyBuffer)(VkDevice device, VkBuffer buffer, const VkAllocationCallbacks *pAllocator);
typedef void (VKAPI_PTR *PFN_vkGetBufferMemoryRequirements)(VkDevice device, VkBuffer buffer, VkMemoryRequirements *pMemoryRequirements);
typedef VkDeviceAddress (VKAPI_PTR *PFN_vkGetBufferDeviceAddress)(VkDevice device, const VkBufferDeviceAddressInfo *pInfo);
typedef VkResult (VKAPI_PTR *PFN_vkAllocateMemory)(VkDevice device, const VkMemoryAllocateInfo *pAllocateInfo, const VkAllocationCallbacks *pAllocator, VkDeviceMemory *pMemory);
typedef void (VKAPI_PTR *PFN_vkFreeMemory)(VkDevice device, VkDeviceMemory memory, const VkAllocationCallbacks *pAllocator);
typedef VkResult (VKAPI_PTR *PFN_vkBindBufferMemory)(VkDevice device, VkBuffer buffer, VkDeviceMemory memory, VkDeviceSize memoryOffset);
typedef VkResult (VKAPI_PTR *PFN_vkMapMemory)(VkDevice device, VkDeviceMemory memory, VkDeviceSize offset, VkDeviceSize size, VkMemoryMapFlags flags, void **ppData);
typedef void (VKAPI_PTR *PFN_vkUnmapMemory)(VkDevice device, VkDeviceMemory memory);
typedef VkResult (VKAPI_PTR *PFN_vkFlushMappedMemoryRanges)(VkDevice device, uint32_t memoryRangeCount, const VkMappedMemoryRange *pMemoryRanges);
typedef VkResult (VKAPI_PTR *PFN_vkInvalidateMappedMemoryRanges)(VkDevice device, uint32_t memoryRangeCount, const VkMappedMemoryRange *pMemoryRanges);
typedef VkResult (VKAPI_PTR *PFN_vkCreateShaderModule)(VkDevice device, const VkShaderModuleCreateInfo *pCreateInfo, const VkAllocationCallbacks *pAllocator, VkShaderModule *pShaderModule);
typedef void (VKAPI_PTR *PFN_vkDestroyShaderModule)(VkDevice device, VkShaderModule shaderModule, const VkAllocationCallbacks *pAllocator);
typedef VkResult (VKAPI_PTR *PFN_vkCreatePipelineLayout)(VkDevice device, const VkPipelineLayoutCreateInfo *pCreateInfo, const VkAllocationCallbacks *pAllocator, VkPipelineLayout *pPipelineLayout);
typedef void (VKAPI_PTR *PFN_vkDestroyPipelineLayout)(VkDevice device, VkPipelineLayout pipelineLayout, const VkAllocationCallbacks *pAllocator);
typedef VkResult (VKAPI_PTR *PFN_vkCreateComputePipelines)(VkDevice device, VkPipelineCache pipelineCache, uint32_t createInfoCount, const VkComputePipelineCreateInfo *pCreateInfos, const VkAllocationCallbacks *pAllocator, VkPipeline *pPipelines);
typedef void (VKAPI_PTR *PFN_vkDestroyPipeline)(VkDevice device, VkPipeline pipeline, const VkAllocationCallbacks *pAllocator);
typedef VkResult (VKAPI_PTR *PFN_vkCreatePipelineCache)(VkDevice device, const VkPipelineCacheCreateInfo *pCreateInfo, const VkAllocationCallbacks *pAllocator, VkPipelineCache *pPipelineCache);
typedef void (VKAPI_PTR *PFN_vkDestroyPipelineCache)(VkDevice device, VkPipelineCache pipelineCache, const VkAllocationCallbacks *pAllocator);
typedef VkResult (VKAPI_PTR *PFN_vkGetPipelineCacheData)(VkDevice device, VkPipelineCache pipelineCache, size_t *pDataSize, void *pData);
typedef void (VKAPI_PTR *PFN_vkCmdBindPipeline)(VkCommandBuffer commandBuffer, VkPipelineBindPoint pipelineBindPoint, VkPipeline pipeline);
typedef void (VKAPI_PTR *PFN_vkCmdPushConstants)(VkCommandBuffer commandBuffer, VkPipelineLayout layout, VkShaderStageFlags stageFlags, uint32_t offset, uint32_t size, const void *pValues);
typedef void (VKAPI_PTR *PFN_vkCmdDispatch)(VkCommandBuffer commandBuffer, uint32_t groupCountX, uint32_t groupCountY, uint32_t groupCountZ);
typedef void (VKAPI_PTR *PFN_vkCmdPipelineBarrier)(VkCommandBuffer commandBuffer, VkPipelineStageFlags srcStageMask, VkPipelineStageFlags dstStageMask, VkDependencyFlags dependencyFlags, uint32_t memoryBarrierCount, const VkMemoryBarrier *pMemoryBarriers, uint32_t bufferMemoryBarrierCount, const VkBufferMemoryBarrier *pBufferMemoryBarriers, uint32_t imageMemoryBarrierCount, const VkImageMemoryBarrier *pImageMemoryBarriers);
typedef void (VKAPI_PTR *PFN_vkCmdCopyBuffer)(VkCommandBuffer commandBuffer, VkBuffer srcBuffer, VkBuffer dstBuffer, uint32_t regionCount, const VkBufferCopy *pRegions);
typedef void (VKAPI_PTR *PFN_vkCmdFillBuffer)(VkCommandBuffer commandBuffer, VkBuffer dstBuffer, VkDeviceSize dstOffset, VkDeviceSize size, uint32_t data);
typedef void (VKAPI_PTR *PFN_vkCmdUpdateBuffer)(VkCommandBuffer commandBuffer, VkBuffer dstBuffer, VkDeviceSize dstOffset, VkDeviceSize dataSize, const void *pData);
typedef VkResult (VKAPI_PTR *PFN_vkCreateAccelerationStructureKHR)(VkDevice device, const VkAccelerationStructureCreateInfoKHR *pCreateInfo, const VkAllocationCallbacks *pAllocator, VkAccelerationStructureKHR *pAccelerationStructure);
typedef void (VKAPI_PTR *PFN_vkDestroyAccelerationStructureKHR)(VkDevice device, VkAccelerationStructureKHR accelerationStructure, const VkAllocationCallbacks *pAllocator);
typedef void (VKAPI_PTR *PFN_vkGetAccelerationStructureBuildSizesKHR)(VkDevice device, VkAccelerationStructureBuildTypeKHR buildType, const VkAccelerationStructureBuildGeometryInfoKHR *pBuildInfo, const uint32_t *pMaxPrimitiveCounts, VkAccelerationStructureBuildSizesInfoKHR *pSizeInfo);
typedef VkDeviceAddress (VKAPI_PTR *PFN_vkGetAccelerationStructureDeviceAddressKHR)(VkDevice device, const VkAccelerationStructureDeviceAddressInfoKHR *pInfo);
typedef void (VKAPI_PTR *PFN_vkCmdBuildAccelerationStructuresKHR)(VkCommandBuffer commandBuffer, uint32_t infoCount, const VkAccelerationStructureBuildGeometryInfoKHR *pInfos, const VkAccelerationStructureBuildRangeInfoKHR *const *ppBuildRangeInfos);

#endif // DRJIT_DYNAMIC_VULKAN

// Beta extension: <vulkan/vulkan.h> only declares it given VK_ENABLE_BETA_EXTENSIONS
#if !defined(VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME)
#  define VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME "VK_KHR_portability_subset"
#endif

// ====================================================================
//     SPIRV-Tools C API (always resolved dynamically at run time)
// ====================================================================

typedef struct spv_context_t spv_context_t;
typedef spv_context_t *spv_context;
typedef const spv_context_t *spv_const_context;

typedef struct spv_binary_t {
    uint32_t *code;
    size_t wordCount;
} spv_binary_t;
typedef spv_binary_t *spv_binary;

typedef struct spv_text_t {
    const char *str;
    size_t length;
} spv_text_t;
typedef spv_text_t *spv_text;

typedef struct spv_position_t {
    size_t line;
    size_t column;
    size_t index;
} spv_position_t;

typedef struct spv_diagnostic_t {
    spv_position_t position;
    char *error;
    bool isTextSource;
} spv_diagnostic_t;
typedef spv_diagnostic_t *spv_diagnostic;

typedef enum spv_result_t {
    SPV_SUCCESS = 0,
    SPV_UNSUPPORTED = 1,
    SPV_END_OF_STREAM = 2,
    SPV_WARNING = 3,
    SPV_FAILED_MATCH = 4,
    SPV_REQUESTED_TERMINATION = 5,
    SPV_ERROR_INTERNAL = -1,
    SPV_ERROR_OUT_OF_MEMORY = -2,
    SPV_ERROR_INVALID_POINTER = -3,
    SPV_ERROR_INVALID_BINARY = -4,
    SPV_ERROR_INVALID_TEXT = -5,
    SPV_ERROR_INVALID_TABLE = -6,
    SPV_ERROR_INVALID_VALUE = -7,
    SPV_ERROR_INVALID_DIAGNOSTIC = -8,
    SPV_ERROR_INVALID_LOOKUP = -9,
    SPV_ERROR_INVALID_ID = -10,
    SPV_ERROR_INVALID_CFG = -11,
    SPV_ERROR_INVALID_LAYOUT = -12,
    SPV_ERROR_INVALID_CAPABILITY = -13,
    SPV_ERROR_INVALID_DATA = -14,
    SPV_ERROR_MISSING_EXTENSION = -15,
    SPV_ERROR_WRONG_VERSION = -16,
    SPV_FORCE_32BIT_spv_result_t = 0x7FFFFFFF
} spv_result_t;

/// Subset of SPIRV-Tools' 'spv_target_env' enumeration (explicit values)
typedef enum spv_target_env {
    SPV_ENV_UNIVERSAL_1_5 = 22,
    SPV_ENV_VULKAN_1_2 = 23, // SPIR-V 1.5
    SPV_ENV_UNIVERSAL_1_6 = 24,
    SPV_ENV_VULKAN_1_3 = 25  // SPIR-V 1.6
} spv_target_env;

/// Subset of SPIRV-Tools' 'spv_binary_to_text_options_t' enumeration
typedef enum spv_binary_to_text_options_t {
    SPV_BINARY_TO_TEXT_OPTION_NONE = 1 << 0,
    SPV_BINARY_TO_TEXT_OPTION_INDENT = 1 << 3,
    SPV_BINARY_TO_TEXT_OPTION_FRIENDLY_NAMES = 1 << 6,
    SPV_FORCE_32BIT_spv_binary_to_text_options_t = 0x7FFFFFFF
} spv_binary_to_text_options_t;

typedef spv_context (*PFN_spvContextCreate)(spv_target_env env);
typedef void (*PFN_spvContextDestroy)(spv_context context);
typedef spv_result_t (*PFN_spvTextToBinary)(const spv_const_context context,
                                            const char *text,
                                            const size_t length,
                                            spv_binary *binary,
                                            spv_diagnostic *diagnostic);
typedef void (*PFN_spvBinaryDestroy)(spv_binary binary);
typedef void (*PFN_spvDiagnosticDestroy)(spv_diagnostic diagnostic);
typedef spv_result_t (*PFN_spvBinaryToText)(const spv_const_context context,
                                            const uint32_t *binary,
                                            const size_t word_count,
                                            const uint32_t options,
                                            spv_text *text,
                                            spv_diagnostic *diagnostic);
typedef void (*PFN_spvTextDestroy)(spv_text text);
typedef spv_result_t (*PFN_spvValidateBinary)(const spv_const_context context,
                                              const uint32_t *words,
                                              const size_t num_words,
                                              spv_diagnostic *diagnostic);
typedef const char *(*PFN_spvSoftwareVersionString)(void);

// ====================================================================
//              Function pointers resolved at run time
// ====================================================================

/// Global functions (resolved via vkGetInstanceProcAddr(nullptr, ..))
#define DRJIT_VK_GLOBAL_FUNCS(X)                                               \
    X(vkCreateInstance)                                                        \
    X(vkEnumerateInstanceVersion)                                              \
    X(vkEnumerateInstanceExtensionProperties)                                  \
    X(vkEnumerateInstanceLayerProperties)

/// Instance-level functions (required)
#define DRJIT_VK_INSTANCE_FUNCS(X)                                             \
    X(vkDestroyInstance)                                                       \
    X(vkEnumeratePhysicalDevices)                                              \
    X(vkGetPhysicalDeviceProperties)                                           \
    X(vkGetPhysicalDeviceProperties2)                                          \
    X(vkGetPhysicalDeviceFeatures2)                                            \
    X(vkGetPhysicalDeviceQueueFamilyProperties)                                \
    X(vkGetPhysicalDeviceMemoryProperties)                                     \
    X(vkEnumerateDeviceExtensionProperties)                                    \
    X(vkCreateDevice)                                                          \
    X(vkGetDeviceProcAddr)

/// Device-level functions of Vulkan 1.0-1.2 (required)
#define DRJIT_VK_DEVICE_FUNCS(X)                                               \
    X(vkDestroyDevice)                                                         \
    X(vkDeviceWaitIdle)                                                        \
    X(vkGetDeviceQueue)                                                        \
    X(vkCreateCommandPool)                                                     \
    X(vkDestroyCommandPool)                                                    \
    X(vkResetCommandPool)                                                      \
    X(vkAllocateCommandBuffers)                                                \
    X(vkFreeCommandBuffers)                                                    \
    X(vkBeginCommandBuffer)                                                    \
    X(vkEndCommandBuffer)                                                      \
    X(vkResetCommandBuffer)                                                    \
    X(vkQueueSubmit)                                                           \
    X(vkQueueWaitIdle)                                                         \
    X(vkQueueBindSparse)                                                       \
    X(vkCreateSemaphore)                                                       \
    X(vkDestroySemaphore)                                                      \
    X(vkWaitSemaphores)                                                        \
    X(vkSignalSemaphore)                                                       \
    X(vkGetSemaphoreCounterValue)                                              \
    X(vkCreateBuffer)                                                          \
    X(vkDestroyBuffer)                                                         \
    X(vkGetBufferMemoryRequirements)                                           \
    X(vkGetBufferDeviceAddress)                                                \
    X(vkAllocateMemory)                                                        \
    X(vkFreeMemory)                                                            \
    X(vkBindBufferMemory)                                                      \
    X(vkMapMemory)                                                             \
    X(vkUnmapMemory)                                                           \
    X(vkFlushMappedMemoryRanges)                                               \
    X(vkInvalidateMappedMemoryRanges)                                          \
    X(vkCreateShaderModule)                                                    \
    X(vkDestroyShaderModule)                                                   \
    X(vkCreatePipelineLayout)                                                  \
    X(vkDestroyPipelineLayout)                                                 \
    X(vkCreateComputePipelines)                                                \
    X(vkDestroyPipeline)                                                       \
    X(vkCreatePipelineCache)                                                   \
    X(vkDestroyPipelineCache)                                                  \
    X(vkGetPipelineCacheData)                                                  \
    X(vkCmdBindPipeline)                                                       \
    X(vkCmdPushConstants)                                                      \
    X(vkCmdDispatch)                                                           \
    X(vkCmdPipelineBarrier)                                                    \
    X(vkCmdCopyBuffer)                                                         \
    X(vkCmdFillBuffer)                                                         \
    X(vkCmdUpdateBuffer)

/// Device-level functions of VK_KHR_acceleration_structure (optional)
#define DRJIT_VK_DEVICE_FUNCS_ACCEL(X)                                         \
    X(vkCreateAccelerationStructureKHR)                                        \
    X(vkDestroyAccelerationStructureKHR)                                       \
    X(vkGetAccelerationStructureBuildSizesKHR)                                 \
    X(vkGetAccelerationStructureDeviceAddressKHR)                              \
    X(vkCmdBuildAccelerationStructuresKHR)

/// Required SPIRV-Tools functions
#define DRJIT_SPV_FUNCS(X)                                                     \
    X(spvContextCreate)                                                        \
    X(spvContextDestroy)                                                       \
    X(spvTextToBinary)                                                         \
    X(spvBinaryDestroy)                                                        \
    X(spvDiagnosticDestroy)

/// Optional SPIRV-Tools functions (disassembly, validation, version)
#define DRJIT_SPV_FUNCS_OPT(X)                                                 \
    X(spvBinaryToText)                                                         \
    X(spvTextDestroy)                                                          \
    X(spvValidateBinary)                                                       \
    X(spvSoftwareVersionString)

#define DRJIT_VK_DECLARE(name) extern PFN_##name name;
extern PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr;
DRJIT_VK_GLOBAL_FUNCS(DRJIT_VK_DECLARE)
DRJIT_VK_INSTANCE_FUNCS(DRJIT_VK_DECLARE)
DRJIT_VK_DEVICE_FUNCS(DRJIT_VK_DECLARE)
DRJIT_VK_DEVICE_FUNCS_ACCEL(DRJIT_VK_DECLARE)
DRJIT_SPV_FUNCS(DRJIT_VK_DECLARE)
DRJIT_SPV_FUNCS_OPT(DRJIT_VK_DECLARE)
#undef DRJIT_VK_DECLARE

// ====================================================================
//                       Dr.Jit-internal interface
// ====================================================================

/// The Dr.Jit Vulkan instance (or \c nullptr if unavailable)
extern VkInstance jitc_vulkan_instance;

/// Instance-level API version supported by the Vulkan loader
extern uint32_t jitc_vulkan_loader_version;

/// Vulkan API version of 'jitc_vulkan_instance' (``VK_MAKE_API_VERSION``)
extern uint32_t jitc_vulkan_api_version;

/// Were the VK_KHR_acceleration_structure entry points found?
extern bool jitc_vulkan_has_accel_api;

/// SPIRV-Tools context targeting the Vulkan 1.2 environment (SPIR-V 1.5)
extern spv_context jitc_vulkan_spv_context;

/**
 * \brief Load the Vulkan loader & SPIRV-Tools, create the Dr.Jit Vulkan
 * instance, and resolve all Vulkan/SPIRV-Tools entry points.
 *
 * Returns \c false (after logging a warning) if any of these steps fail, in
 * which case all partially initialized state is released again. Calling the
 * function repeatedly is harmless.
 */
extern bool jitc_vulkan_api_init();

/// Release the Vulkan instance and unload the Vulkan loader & SPIRV-Tools
extern void jitc_vulkan_api_shutdown();

/// Look up a Vulkan (``vk*``) or SPIRV-Tools (``spv*``) function by name
extern void *jitc_vulkan_lookup(const char *name);

/// Return a human-readable name of a VkResult error code
extern const char *jitc_vulkan_error_name(VkResult result);

/// Raise an exception if a Vulkan API call failed
[[noreturn]] extern void jitc_vulkan_raise(VkResult result, const char *file,
                                           int line);

#define vk_check(expr)                                                         \
    do {                                                                       \
        VkResult vk_check_rv = (expr);                                         \
        if (vk_check_rv != VK_SUCCESS)                                         \
            jitc_vulkan_raise(vk_check_rv, __FILE__, __LINE__);                \
    } while (0)
