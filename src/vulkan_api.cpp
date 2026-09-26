/*
    src/vulkan_api.cpp -- Low-level interface to the Vulkan loader & SPIRV-Tools

    This file dynamically loads the Vulkan loader and the SPIRV-Tools shared
    library, creates the Vulkan instance used by Dr.Jit, and resolves all
    Vulkan and SPIRV-Tools entry points (see ``vulkan_api.h``).

    Copyright (c) 2026 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#include "vulkan_api.h"
#include "internal.h"
#include "log.h"
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#if defined(_WIN32)
#  include <windows.h>
#else
#  include <dlfcn.h>
#endif

// ====================================================================
//                  Function pointers & global state
// ====================================================================

#define DRJIT_VK_DEFINE(name) PFN_##name name = nullptr;
PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr = nullptr;
DRJIT_VK_GLOBAL_FUNCS(DRJIT_VK_DEFINE)
DRJIT_VK_INSTANCE_FUNCS(DRJIT_VK_DEFINE)
DRJIT_VK_DEVICE_FUNCS(DRJIT_VK_DEFINE)
DRJIT_VK_DEVICE_FUNCS_ACCEL(DRJIT_VK_DEFINE)
DRJIT_SPV_FUNCS(DRJIT_VK_DEFINE)
DRJIT_SPV_FUNCS_OPT(DRJIT_VK_DEFINE)
#undef DRJIT_VK_DEFINE

VkInstance jitc_vulkan_instance = nullptr;
uint32_t jitc_vulkan_loader_version = 0;
uint32_t jitc_vulkan_api_version = 0;
bool jitc_vulkan_has_accel_api = false;
spv_context jitc_vulkan_spv_context = nullptr;

/// Handles of the dynamically loaded Vulkan loader and SPIRV-Tools library
static void *jitc_vulkan_handle = nullptr;
static void *jitc_spirv_tools_handle = nullptr;

// ====================================================================
//                     Platform-specific library names
// ====================================================================

#if defined(_WIN32)
static const char *jitc_vulkan_fname = "vulkan-1.dll",
                  *jitc_vulkan_glob = nullptr,
                  *jitc_vulkan_alt[] = { nullptr },
                  *jitc_spirv_tools_fname = "SPIRV-Tools-shared.dll",
                  *jitc_spirv_tools_glob = nullptr,
                  *jitc_spirv_tools_alt[] = { nullptr },
                  *jitc_vulkan_sdk_lib_dir = "\\Bin\\";
#elif defined(__APPLE__)
static const char *jitc_vulkan_fname = "libvulkan.1.dylib",
                  *jitc_vulkan_glob = "{/usr/local,/opt/homebrew}/lib/libvulkan.1*.dylib",
                  // MoltenVK can also be used directly (without a loader)
                  *jitc_vulkan_alt[] = { "libMoltenVK.dylib", nullptr },
                  *jitc_spirv_tools_fname = "libSPIRV-Tools-shared.dylib",
                  *jitc_spirv_tools_glob = "{/usr/local,/opt/homebrew}/lib/libSPIRV-Tools-shared*.dylib",
                  *jitc_spirv_tools_alt[] = { nullptr },
                  *jitc_vulkan_sdk_lib_dir = "/lib/";
#else
static const char *jitc_vulkan_fname = "libvulkan.so.1",
                  *jitc_vulkan_glob = "/usr/{lib,lib64,lib/x86_64-linux-gnu,lib/aarch64-linux-gnu}/libvulkan.so.1*",
                  *jitc_vulkan_alt[] = { "libvulkan.so", nullptr },
                  *jitc_spirv_tools_fname = "libSPIRV-Tools-shared.so",
                  *jitc_spirv_tools_glob = "/usr/{lib,lib64,local/lib,lib/x86_64-linux-gnu,lib/aarch64-linux-gnu}/libSPIRV-Tools-shared.so*",
                  // Some distributions only ship the (shared) main library
                  *jitc_spirv_tools_alt[] = { "libSPIRV-Tools.so", nullptr },
                  *jitc_vulkan_sdk_lib_dir = "/lib/";
#endif

static void *jitc_vulkan_dlopen(const char *path) {
#if defined(_WIN32)
    return (void *) LoadLibraryA(path);
#else
    return dlopen(path, RTLD_LAZY);
#endif
}

static void jitc_vulkan_dlclose(void *handle) {
#if defined(_WIN32)
    FreeLibrary((HMODULE) handle);
#else
    dlclose(handle);
#endif
}

/**
 * \brief Load a shared library
 *
 * If the environment variable \c env_var is set, only the path it specifies
 * is considered. Otherwise, the function tries \c fname (default search
 * path), the glob pattern \c glob_pat (not on Windows), \c fname within the
 * Vulkan SDK (``VULKAN_SDK`` environment variable), and finally the
 * alternative names in the \c nullptr-terminated list \c alt.
 */
static void *jitc_vulkan_load_library(const char *fname, const char *glob_pat,
                                      const char *env_var,
                                      const char *const *alt) {
    const char *env_value = getenv(env_var);
    bool env_set = env_value && *env_value;

    void *handle =
        jitc_find_library(fname, glob_pat ? glob_pat : fname, env_var);
    if (handle || env_set)
        return handle;

    const char *sdk = getenv("VULKAN_SDK");
    if (sdk && *sdk) {
        std::string path = std::string(sdk) + jitc_vulkan_sdk_lib_dir + fname;
        handle = jitc_vulkan_dlopen(path.c_str());
    }

    for (size_t i = 0; alt[i] && !handle; ++i)
        handle = jitc_vulkan_dlopen(alt[i]);

    return handle;
}

// ====================================================================
//                      Initialization & shutdown
// ====================================================================

/// Release all state (instance, SPIRV-Tools context, libraries, pointers)
static void jitc_vulkan_api_release() {
    if (jitc_vulkan_spv_context) {
        spvContextDestroy(jitc_vulkan_spv_context);
        jitc_vulkan_spv_context = nullptr;
    }

    if (jitc_vulkan_instance) {
        if (vkDestroyInstance)
            vkDestroyInstance(jitc_vulkan_instance, nullptr);
        jitc_vulkan_instance = nullptr;
    }

    #define DRJIT_VK_RESET(name) name = nullptr;
    vkGetInstanceProcAddr = nullptr;
    DRJIT_VK_GLOBAL_FUNCS(DRJIT_VK_RESET)
    DRJIT_VK_INSTANCE_FUNCS(DRJIT_VK_RESET)
    DRJIT_VK_DEVICE_FUNCS(DRJIT_VK_RESET)
    DRJIT_VK_DEVICE_FUNCS_ACCEL(DRJIT_VK_RESET)
    DRJIT_SPV_FUNCS(DRJIT_VK_RESET)
    DRJIT_SPV_FUNCS_OPT(DRJIT_VK_RESET)
    #undef DRJIT_VK_RESET

    if (jitc_spirv_tools_handle) {
        jitc_vulkan_dlclose(jitc_spirv_tools_handle);
        jitc_spirv_tools_handle = nullptr;
    }

    if (jitc_vulkan_handle) {
        jitc_vulkan_dlclose(jitc_vulkan_handle);
        jitc_vulkan_handle = nullptr;
    }

    jitc_vulkan_loader_version = 0;
    jitc_vulkan_api_version = 0;
    jitc_vulkan_has_accel_api = false;
}

/// Log a warning, release all partially initialized state, and return false
#if defined(__GNUC__)
__attribute__((__format__(__printf__, 1, 2)))
#endif
static bool jitc_vulkan_api_fail(const char *fmt, ...) {
    char msg[1024];
    va_list args;
    va_start(args, fmt);
    vsnprintf(msg, sizeof(msg), fmt, args);
    va_end(args);

    jitc_log(Warn, "jit_vulkan_api_init(): %s -- disabling the Vulkan backend!",
             msg);
    jitc_vulkan_api_release();
    return false;
}

bool jitc_vulkan_api_init() {
    if (jitc_vulkan_instance)
        return true;

    // Discard leftovers of an earlier (failed) attempt, if any
    jitc_vulkan_api_release();

    // 1. Load the Vulkan loader
    jitc_vulkan_handle =
        jitc_vulkan_load_library(jitc_vulkan_fname, jitc_vulkan_glob,
                                 "DRJIT_LIBVULKAN_PATH", jitc_vulkan_alt);
    if (!jitc_vulkan_handle)
        return jitc_vulkan_api_fail(
            "could not load the Vulkan loader (\"%s\"). Please install a "
            "graphics driver with Vulkan 1.3 support, or set the "
            "DRJIT_LIBVULKAN_PATH environment variable to specify the path of "
            "the Vulkan loader", jitc_vulkan_fname);

    vkGetInstanceProcAddr = (PFN_vkGetInstanceProcAddr) dlsym(
        jitc_vulkan_handle, "vkGetInstanceProcAddr");
    if (!vkGetInstanceProcAddr)
        return jitc_vulkan_api_fail(
            "the Vulkan loader does not export \"vkGetInstanceProcAddr\"");

    const char *missing = nullptr;

    #define DRJIT_VK_LOAD_GLOBAL(name)                                         \
        name = (PFN_##name) vkGetInstanceProcAddr(nullptr, #name);             \
        if (!name && !missing)                                                 \
            missing = #name;

    DRJIT_VK_GLOBAL_FUNCS(DRJIT_VK_LOAD_GLOBAL)
    #undef DRJIT_VK_LOAD_GLOBAL

    // 'vkEnumerateInstanceVersion' was introduced in Vulkan 1.1
    if (!vkEnumerateInstanceVersion)
        return jitc_vulkan_api_fail(
            "the Vulkan loader only supports Vulkan 1.0, but Dr.Jit requires "
            "Vulkan 1.3 or newer. Please update your graphics driver and "
            "Vulkan loader");

    if (missing)
        return jitc_vulkan_api_fail(
            "could not resolve the Vulkan function \"%s\"", missing);

    uint32_t version = 0;
    VkResult rv = vkEnumerateInstanceVersion(&version);
    if (rv != VK_SUCCESS)
        return jitc_vulkan_api_fail("vkEnumerateInstanceVersion() failed: %s",
                                    jitc_vulkan_error_name(rv));

    uint32_t major = VK_API_VERSION_MAJOR(version),
             minor = VK_API_VERSION_MINOR(version),
             patch = VK_API_VERSION_PATCH(version);

    if (VK_API_VERSION_VARIANT(version) != 0 || major < 1 ||
        (major == 1 && minor < 3))
        return jitc_vulkan_api_fail(
            "the Vulkan loader only supports Vulkan %u.%u, but Dr.Jit requires "
            "Vulkan 1.3 or newer. Please update your graphics driver and "
            "Vulkan loader", major, minor);

    // 2. Load SPIRV-Tools, which assembles the generated SPIR-V code
    jitc_spirv_tools_handle = jitc_vulkan_load_library(
        jitc_spirv_tools_fname, jitc_spirv_tools_glob,
        "DRJIT_LIBSPIRV_TOOLS_PATH", jitc_spirv_tools_alt);
    if (!jitc_spirv_tools_handle)
        return jitc_vulkan_api_fail(
            "could not load the SPIRV-Tools library (\"%s\"), which the "
            "Vulkan backend needs to assemble SPIR-V code. Please install "
            "SPIRV-Tools (e.g., via the \"spirv-tools\" package of your "
            "distribution or the Vulkan SDK), or set the "
            "DRJIT_LIBSPIRV_TOOLS_PATH environment variable to specify its "
            "path", jitc_spirv_tools_fname);

    #define DRJIT_SPV_LOAD(name)                                               \
        name = (PFN_##name) dlsym(jitc_spirv_tools_handle, #name);             \
        if (!name && !missing)                                                 \
            missing = #name;
    #define DRJIT_SPV_LOAD_OPT(name)                                           \
        name = (PFN_##name) dlsym(jitc_spirv_tools_handle, #name);

    DRJIT_SPV_FUNCS(DRJIT_SPV_LOAD)
    DRJIT_SPV_FUNCS_OPT(DRJIT_SPV_LOAD_OPT)
    #undef DRJIT_SPV_LOAD
    #undef DRJIT_SPV_LOAD_OPT

    if (missing)
        return jitc_vulkan_api_fail(
            "could not find the function \"%s\" in the SPIRV-Tools library",
            missing);

    jitc_vulkan_spv_context = spvContextCreate(SPV_ENV_VULKAN_1_3);
    if (!jitc_vulkan_spv_context)
        return jitc_vulkan_api_fail(
            "spvContextCreate() failed. The SPIRV-Tools library is likely too "
            "old (support for the Vulkan 1.3 target environment is required)");

    // 3. Create the Vulkan instance
    std::vector<VkExtensionProperties> extensions;
    uint32_t n_extensions = 0;
    rv = vkEnumerateInstanceExtensionProperties(nullptr, &n_extensions,
                                                nullptr);
    if (rv == VK_SUCCESS && n_extensions > 0) {
        extensions.resize(n_extensions);
        rv = vkEnumerateInstanceExtensionProperties(nullptr, &n_extensions,
                                                    extensions.data());
        if (rv != VK_SUCCESS && rv != VK_INCOMPLETE)
            n_extensions = 0;
        extensions.resize(n_extensions);
    }

    bool has_portability_enumeration = false;
    for (const VkExtensionProperties &ext : extensions) {
        if (strcmp(ext.extensionName,
                   VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME) == 0)
            has_portability_enumeration = true;
    }

    uint32_t api_version = VK_API_VERSION_1_3;

    VkApplicationInfo app_info {};
    app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.pApplicationName = "Dr.Jit";
    app_info.pEngineName = "Dr.Jit";
    app_info.apiVersion = api_version;

    const char *enabled_extensions[1] = { };
    VkInstanceCreateInfo instance_info {};
    instance_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instance_info.pApplicationInfo = &app_info;
    if (has_portability_enumeration) {
        // Also enumerate non-conformant implementations such as MoltenVK
        enabled_extensions[instance_info.enabledExtensionCount++] =
            VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME;
        instance_info.flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
    }
    instance_info.ppEnabledExtensionNames = enabled_extensions;

    VkInstance instance = VK_NULL_HANDLE;
    rv = vkCreateInstance(&instance_info, nullptr, &instance);
    if (rv != VK_SUCCESS)
        return jitc_vulkan_api_fail(
            "vkCreateInstance() failed: %s%s", jitc_vulkan_error_name(rv),
            rv == VK_ERROR_INCOMPATIBLE_DRIVER
                ? " (no compatible Vulkan driver was found)" : "");
    jitc_vulkan_instance = instance;

    // 4. Resolve instance- and device-level functions. The latter dispatch to
    // the right driver for any VkDevice created from 'instance'.
    #define DRJIT_VK_LOAD(name)                                                \
        name = (PFN_##name) vkGetInstanceProcAddr(instance, #name);            \
        if (!name && !missing)                                                 \
            missing = #name;

    DRJIT_VK_INSTANCE_FUNCS(DRJIT_VK_LOAD)
    DRJIT_VK_DEVICE_FUNCS(DRJIT_VK_LOAD)
    #undef DRJIT_VK_LOAD

    if (missing)
        return jitc_vulkan_api_fail(
            "could not resolve the Vulkan function \"%s\"", missing);

    // VK_KHR_acceleration_structure is optional (needed for ray tracing)
    bool has_accel_api = true;

    #define DRJIT_VK_LOAD_OPT(name)                                            \
        name = (PFN_##name) vkGetInstanceProcAddr(instance, #name);            \
        if (!name)                                                             \
            has_accel_api = false;

    DRJIT_VK_DEVICE_FUNCS_ACCEL(DRJIT_VK_LOAD_OPT)
    #undef DRJIT_VK_LOAD_OPT

    if (!has_accel_api) {
        #define DRJIT_VK_RESET(name) name = nullptr;
        DRJIT_VK_DEVICE_FUNCS_ACCEL(DRJIT_VK_RESET)
        #undef DRJIT_VK_RESET
    }

    jitc_vulkan_loader_version = version;
    jitc_vulkan_api_version = api_version;
    jitc_vulkan_has_accel_api = has_accel_api;

    jitc_log(Info,
             "jit_vulkan_api_init(): loaded the Vulkan loader (version "
             "%u.%u.%u, targeting Vulkan %u.%u) and SPIRV-Tools (%s).",
             major, minor, patch, VK_API_VERSION_MAJOR(api_version),
             VK_API_VERSION_MINOR(api_version),
             spvSoftwareVersionString ? spvSoftwareVersionString()
                                      : "unknown version");

    return true;
}

void jitc_vulkan_api_shutdown() {
    if (!jitc_vulkan_handle && !jitc_spirv_tools_handle)
        return;

    jitc_log(Info, "jit_vulkan_api_shutdown()");
    jitc_vulkan_api_release();
}

void *jitc_vulkan_lookup(const char *name) {
    if (!name)
        jitc_raise("jit_vulkan_lookup(): 'name' must not be NULL!");

    if (!jitc_vulkan_instance)
        jitc_raise("jit_vulkan_lookup(): the Vulkan backend has not been "
                   "initialized. Make sure to call jit_init() with the "
                   "JitBackend::Vulkan flag first.");

    void *ptr = nullptr;
    if (strncmp(name, "spv", 3) == 0) {
        ptr = (void *) dlsym(jitc_spirv_tools_handle, name);
    } else if (strncmp(name, "vk", 2) == 0) {
        ptr = (void *) vkGetInstanceProcAddr(jitc_vulkan_instance, name);

        // Global commands (e.g. 'vkEnumerateInstanceVersion') can only be
        // resolved without an instance
        if (!ptr)
            ptr = (void *) vkGetInstanceProcAddr(nullptr, name);
    }

    if (!ptr)
        jitc_raise("jit_vulkan_lookup(): function \"%s\" not found!", name);

    return ptr;
}

// ====================================================================
//                           Error handling
// ====================================================================

const char *jitc_vulkan_error_name(VkResult result) {
    switch (result) {
        case VK_SUCCESS: return "VK_SUCCESS";
        case VK_NOT_READY: return "VK_NOT_READY";
        case VK_TIMEOUT: return "VK_TIMEOUT";
        case VK_EVENT_SET: return "VK_EVENT_SET";
        case VK_EVENT_RESET: return "VK_EVENT_RESET";
        case VK_INCOMPLETE: return "VK_INCOMPLETE";
        case VK_ERROR_OUT_OF_HOST_MEMORY: return "VK_ERROR_OUT_OF_HOST_MEMORY";
        case VK_ERROR_OUT_OF_DEVICE_MEMORY: return "VK_ERROR_OUT_OF_DEVICE_MEMORY";
        case VK_ERROR_INITIALIZATION_FAILED: return "VK_ERROR_INITIALIZATION_FAILED";
        case VK_ERROR_DEVICE_LOST: return "VK_ERROR_DEVICE_LOST";
        case VK_ERROR_MEMORY_MAP_FAILED: return "VK_ERROR_MEMORY_MAP_FAILED";
        case VK_ERROR_LAYER_NOT_PRESENT: return "VK_ERROR_LAYER_NOT_PRESENT";
        case VK_ERROR_EXTENSION_NOT_PRESENT: return "VK_ERROR_EXTENSION_NOT_PRESENT";
        case VK_ERROR_FEATURE_NOT_PRESENT: return "VK_ERROR_FEATURE_NOT_PRESENT";
        case VK_ERROR_INCOMPATIBLE_DRIVER: return "VK_ERROR_INCOMPATIBLE_DRIVER";
        case VK_ERROR_TOO_MANY_OBJECTS: return "VK_ERROR_TOO_MANY_OBJECTS";
        case VK_ERROR_FORMAT_NOT_SUPPORTED: return "VK_ERROR_FORMAT_NOT_SUPPORTED";
        case VK_ERROR_FRAGMENTED_POOL: return "VK_ERROR_FRAGMENTED_POOL";
        case VK_ERROR_UNKNOWN: return "VK_ERROR_UNKNOWN";
        case VK_ERROR_OUT_OF_POOL_MEMORY: return "VK_ERROR_OUT_OF_POOL_MEMORY";
        case VK_ERROR_INVALID_EXTERNAL_HANDLE: return "VK_ERROR_INVALID_EXTERNAL_HANDLE";
        case VK_ERROR_FRAGMENTATION: return "VK_ERROR_FRAGMENTATION";
        case VK_ERROR_INVALID_OPAQUE_CAPTURE_ADDRESS: return "VK_ERROR_INVALID_OPAQUE_CAPTURE_ADDRESS";
        case VK_PIPELINE_COMPILE_REQUIRED: return "VK_PIPELINE_COMPILE_REQUIRED";
        case VK_ERROR_VALIDATION_FAILED_EXT: return "VK_ERROR_VALIDATION_FAILED_EXT";
        default: return "unknown VkResult";
    }
}

void jitc_vulkan_raise(VkResult result, const char *file, int line) {
    jitc_raise("vk_check(): Vulkan API error %i (%s) in %s:%i.", (int) result,
               jitc_vulkan_error_name(result), file, line);
}
