/*
    tests/vulkan_api.cpp -- Tests of the dynamic Vulkan & SPIRV-Tools loader

    Checks that jit_init() with the JitBackend::Vulkan flag loads the Vulkan
    loader and SPIRV-Tools, creates a Vulkan instance, and resolves entry
    points (which also exercises the declarations in 'src/vulkan_api.h'
    against a real driver), that the libraries can be unloaded and reloaded,
    and that failures only disable the Vulkan backend.

    The test is skipped when Vulkan or SPIRV-Tools are unavailable, unless the
    environment variable DRJIT_TEST_REQUIRE_VULKAN=1 is set. Pass '-v' for
    verbose log output.

    Copyright (c) 2026 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#include <drjit-core/jit.h>
#include "vulkan_api.h" // Private header, only used for the API types
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <string>
#include <vector>

static int checks_run = 0;

#define CHECK(cond)                                                            \
    do {                                                                       \
        checks_run++;                                                          \
        if (!(cond)) {                                                         \
            fprintf(stderr, "Check failed: %s (%s:%i)\n", #cond, __FILE__,     \
                    __LINE__);                                                 \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

static const uint32_t BackendLLVM   = 1u << (uint32_t) JitBackend::LLVM,
                      BackendVulkan = 1u << (uint32_t) JitBackend::Vulkan;

static bool have_llvm = false;

/// Does the system have a Vulkan device that is compatible with Dr.Jit?
static bool have_device = false;

/// Does 'func()' raise an exception?
template <typename Func> static bool raises(Func func) {
    try {
        func();
    } catch (const std::exception &) {
        return true;
    }
    return false;
}

/// Look up a Vulkan/SPIRV-Tools function via the public Dr.Jit API
template <typename T> static T lookup(const char *name) {
    T result = (T) jit_vulkan_lookup(name);
    CHECK(result != nullptr);
    return result;
}

static bool is_pow2(uint64_t value) {
    return value != 0 && (value & (value - 1)) == 0;
}

// ----------------------------------------------------------------------
//                     Environment & log helpers
// ----------------------------------------------------------------------

static void set_env(const char *name, const char *value) {
#if defined(_WIN32)
    _putenv_s(name, value ? value : "");
#else
    if (value)
        setenv(name, value, 1);
    else
        unsetenv(name);
#endif
}

/// Restores an environment variable upon destruction
struct EnvGuard {
    std::string name, value;
    bool was_set;

    EnvGuard(const char *name) : name(name) {
        const char *v = getenv(name);
        was_set = v != nullptr;
        if (was_set)
            value = v;
    }

    ~EnvGuard() { set_env(name.c_str(), was_set ? value.c_str() : nullptr); }
};

static std::string captured_log;

static void log_callback(LogLevel, const char *msg) {
    captured_log += msg;
    captured_log += '\n';
}

/// Capture log messages (level >= Warn) instead of printing them
struct LogCapture {
    LogLevel prev_stderr;

    LogCapture() : prev_stderr(jit_log_level_stderr()) {
        captured_log.clear();
        jit_set_log_level_stderr(LogLevel::Error);
        jit_set_log_level_callback(LogLevel::Warn, log_callback);
    }

    ~LogCapture() {
        jit_set_log_level_callback(LogLevel::Disable, nullptr);
        jit_set_log_level_stderr(prev_stderr);
    }

    bool contains(const char *s) const {
        return captured_log.find(s) != std::string::npos;
    }
};

// ----------------------------------------------------------------------
//                            Test cases
// ----------------------------------------------------------------------

/// Check that the Vulkan libraries are not loaded
static void check_unloaded() {
    CHECK(jit_vulkan_instance() == nullptr);
    int major = -1, minor = -1, patch = -1;
    jit_vulkan_version(&major, &minor, &patch);
    CHECK(major == 0 && minor == 0 && patch == 0);
    CHECK(raises([] { jit_vulkan_lookup("vkCreateDevice"); }));
    CHECK(!jit_has_backend(JitBackend::Vulkan));
    CHECK(jit_vulkan_device_handle() == nullptr);
}

/// Check that the Vulkan libraries are loaded
static void check_loaded() {
    CHECK(jit_vulkan_instance() != nullptr);
    int major = 0, minor = 0, patch = -1;
    jit_vulkan_version(&major, &minor, &patch);
    CHECK(major == 1 && minor >= 3 && patch >= 0);
    CHECK(jit_vulkan_lookup("vkCreateDevice") != nullptr);
    CHECK(jit_vulkan_lookup("spvTextToBinary") != nullptr);

    // The backend is available if the system has a compatible device
    CHECK((bool) jit_has_backend(JitBackend::Vulkan) == have_device);
    CHECK((jit_vulkan_device_handle() != nullptr) == have_device);
}

/// Check that the LLVM backend is unaffected by what happens with Vulkan
static void check_llvm() {
    CHECK((bool) jit_has_backend(JitBackend::LLVM) == have_llvm);
    if (!have_llvm)
        return;

    uint32_t a = jit_var_counter(JitBackend::LLVM, 16),
             b = jit_var_add(a, a);
    jit_var_eval(b);
    uint32_t value = 0;
    jit_var_read(b, 7, &value);
    CHECK(value == 14);
    jit_var_dec_ref(a);
    jit_var_dec_ref(b);
}

static void test01_lookup() {
    // Global, instance-level, device-level, and SPIRV-Tools functions
    const char *names[] = {
        "vkGetInstanceProcAddr", "vkEnumerateInstanceVersion",
        "vkEnumeratePhysicalDevices", "vkGetPhysicalDeviceProperties2",
        "vkCreateDevice", "vkQueueSubmit", "vkCmdDispatch",
        "vkGetBufferDeviceAddress", "vkWaitSemaphores", "spvContextCreate",
        "spvTextToBinary", "spvBinaryToText", "spvValidateBinary"
    };
    for (const char *name : names)
        CHECK(jit_vulkan_lookup(name) != nullptr);

    CHECK(raises([] { jit_vulkan_lookup("vkThisFunctionDoesNotExist"); }));
    CHECK(raises([] { jit_vulkan_lookup("spvThisFunctionDoesNotExist"); }));
    CHECK(raises([] { jit_vulkan_lookup("cuInit"); }));
    CHECK(raises([] { jit_vulkan_lookup(nullptr); }));

    int major = 0, minor = 0, patch = 0;
    jit_vulkan_version(&major, &minor, &patch);
    jit_vulkan_version(nullptr, nullptr, nullptr); // must not crash

    // The public API agrees with a direct query of the loader
    auto EnumerateInstanceVersion =
        lookup<PFN_vkEnumerateInstanceVersion>("vkEnumerateInstanceVersion");
    uint32_t version = 0;
    CHECK(EnumerateInstanceVersion(&version) == VK_SUCCESS);
    CHECK((int) VK_API_VERSION_MAJOR(version) == major &&
          (int) VK_API_VERSION_MINOR(version) == minor &&
          (int) VK_API_VERSION_PATCH(version) == patch);
    printf("  Vulkan loader: version %i.%i.%i\n", major, minor, patch);
}

/// Query device properties & features. This checks the struct declarations
/// of 'vulkan_api.h' against a real driver (e.g. via spec-mandated limits).
static void test02_physical_devices(bool require) {
    VkInstance instance = (VkInstance) jit_vulkan_instance();

    auto EnumeratePhysicalDevices =
        lookup<PFN_vkEnumeratePhysicalDevices>("vkEnumeratePhysicalDevices");
    auto GetPhysicalDeviceProperties =
        lookup<PFN_vkGetPhysicalDeviceProperties>("vkGetPhysicalDeviceProperties");
    auto GetPhysicalDeviceProperties2 =
        lookup<PFN_vkGetPhysicalDeviceProperties2>("vkGetPhysicalDeviceProperties2");
    auto GetPhysicalDeviceFeatures2 =
        lookup<PFN_vkGetPhysicalDeviceFeatures2>("vkGetPhysicalDeviceFeatures2");
    auto GetPhysicalDeviceQueueFamilyProperties =
        lookup<PFN_vkGetPhysicalDeviceQueueFamilyProperties>(
            "vkGetPhysicalDeviceQueueFamilyProperties");
    auto GetPhysicalDeviceMemoryProperties =
        lookup<PFN_vkGetPhysicalDeviceMemoryProperties>(
            "vkGetPhysicalDeviceMemoryProperties");
    auto EnumerateDeviceExtensionProperties =
        lookup<PFN_vkEnumerateDeviceExtensionProperties>(
            "vkEnumerateDeviceExtensionProperties");

    uint32_t count = 0;
    CHECK(EnumeratePhysicalDevices(instance, &count, nullptr) == VK_SUCCESS);
    std::vector<VkPhysicalDevice> devices(count);
    if (count > 0)
        CHECK(EnumeratePhysicalDevices(instance, &count, devices.data()) ==
              VK_SUCCESS);
    printf("  %u physical device(s)\n", count);
    if (require)
        CHECK(count > 0);

    // The instance requires Vulkan 1.3
    int major = 0, minor = 0, patch = 0;
    jit_vulkan_version(&major, &minor, &patch);
    CHECK(major > 1 || (major == 1 && minor >= 3));

    for (VkPhysicalDevice dev : devices) {
        VkPhysicalDeviceProperties props;
        memset(&props, 0, sizeof(props));
        GetPhysicalDeviceProperties(dev, &props);
        uint32_t version = props.apiVersion;
        bool vk13 = version >= VK_API_VERSION_1_3;

        // Spec-mandated minimum limits (these fields span the whole struct)
        const VkPhysicalDeviceLimits &l = props.limits;
        CHECK(l.maxStorageBufferRange >= (1u << 27));
        CHECK(l.maxPushConstantsSize >= 128);
        CHECK(l.maxMemoryAllocationCount >= 4096);
        CHECK(l.maxBoundDescriptorSets >= 4);
        CHECK(l.maxComputeSharedMemorySize >= 16384);
        for (int i = 0; i < 3; ++i)
            CHECK(l.maxComputeWorkGroupCount[i] >= 65535);
        CHECK(l.maxComputeWorkGroupInvocations >= 128);
        CHECK(l.maxComputeWorkGroupSize[0] >= 128 &&
              l.maxComputeWorkGroupSize[1] >= 128 &&
              l.maxComputeWorkGroupSize[2] >= 64);
        CHECK(is_pow2(l.minStorageBufferOffsetAlignment) &&
              l.minStorageBufferOffsetAlignment <= 256);
        CHECK(l.discreteQueuePriorities >= 2);
        CHECK(is_pow2(l.nonCoherentAtomSize) && l.nonCoherentAtomSize <= 256);

        char driver_name[VK_MAX_DRIVER_NAME_SIZE] = "?";
        uint32_t subgroup_size = 0;

        if (vk13) {
            VkPhysicalDeviceVulkan13Properties p13 {};
            p13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_PROPERTIES;
            VkPhysicalDeviceVulkan12Properties p12 {};
            p12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_PROPERTIES;
            p12.pNext = &p13;
            VkPhysicalDeviceVulkan11Properties p11 {};
            p11.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_PROPERTIES;
            p11.pNext = &p12;
            VkPhysicalDeviceProperties2 p2 {};
            p2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
            p2.pNext = &p11;
            GetPhysicalDeviceProperties2(dev, &p2);

            CHECK(p2.pNext == &p11 && p11.pNext == &p12 && p12.pNext == &p13 && !p13.pNext);
            CHECK(p2.properties.apiVersion == props.apiVersion &&
                  p2.properties.vendorID == props.vendorID &&
                  p2.properties.deviceID == props.deviceID &&
                  strcmp(p2.properties.deviceName, props.deviceName) == 0 &&
                  p2.properties.limits.nonCoherentAtomSize ==
                      l.nonCoherentAtomSize);
            CHECK(is_pow2(p11.subgroupSize) && p11.subgroupSize <= 128);
            CHECK(p11.maxMemoryAllocationSize >= (1u << 30));
            CHECK(p12.driverID != 0 && p12.driverName[0] != '\0');
            CHECK(p12.maxTimelineSemaphoreValueDifference >= 0x7FFFFFFFull);
            CHECK(p13.maxBufferSize >= (1u << 30));

            memcpy(driver_name, p12.driverName, sizeof(driver_name));
            subgroup_size = p11.subgroupSize;

            // Features that Vulkan 1.3 devices must support
            VkPhysicalDeviceVulkan13Features f13 {};
            f13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
            VkPhysicalDeviceVulkan12Features f12 {};
            f12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
            f12.pNext = &f13;
            VkPhysicalDeviceVulkan11Features f11 {};
            f11.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
            f11.pNext = &f12;
            VkPhysicalDeviceFeatures2 f2 {};
            f2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
            f2.pNext = &f11;
            GetPhysicalDeviceFeatures2(dev, &f2);

            CHECK(f2.features.robustBufferAccess == VK_TRUE);
            CHECK(f12.timelineSemaphore == VK_TRUE &&
                  f12.uniformBufferStandardLayout == VK_TRUE &&
                  f12.hostQueryReset == VK_TRUE);
            CHECK(f12.bufferDeviceAddress == VK_TRUE &&
                  f12.vulkanMemoryModel == VK_TRUE &&
                  f13.synchronization2 == VK_TRUE &&
                  f13.dynamicRendering == VK_TRUE &&
                  f13.maintenance4 == VK_TRUE);
        }

        uint32_t n_families = 0;
        GetPhysicalDeviceQueueFamilyProperties(dev, &n_families, nullptr);
        std::vector<VkQueueFamilyProperties> families(n_families);
        GetPhysicalDeviceQueueFamilyProperties(dev, &n_families,
                                               families.data());
        CHECK(n_families > 0);
        bool has_compute = false;
        for (const VkQueueFamilyProperties &f : families)
            has_compute |= (f.queueFlags & VK_QUEUE_COMPUTE_BIT) &&
                           f.queueCount > 0;

        VkPhysicalDeviceMemoryProperties mem;
        memset(&mem, 0, sizeof(mem));
        GetPhysicalDeviceMemoryProperties(dev, &mem);
        CHECK(mem.memoryTypeCount >= 1 &&
              mem.memoryTypeCount <= VK_MAX_MEMORY_TYPES);
        CHECK(mem.memoryHeapCount >= 1 &&
              mem.memoryHeapCount <= VK_MAX_MEMORY_HEAPS);
        bool host_coherent = false, device_local = false;
        for (uint32_t i = 0; i < mem.memoryTypeCount; ++i) {
            VkMemoryPropertyFlags f = mem.memoryTypes[i].propertyFlags;
            CHECK(mem.memoryTypes[i].heapIndex < mem.memoryHeapCount);
            host_coherent |= (f & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) &&
                             (f & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
            device_local |= (f & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0;
        }
        CHECK(host_coherent && device_local); // Guaranteed by the spec
        for (uint32_t i = 0; i < mem.memoryHeapCount; ++i)
            CHECK(mem.memoryHeaps[i].size > 0);

        uint32_t n_ext = 0;
        CHECK(EnumerateDeviceExtensionProperties(dev, nullptr, &n_ext,
                                                 nullptr) == VK_SUCCESS);
        std::vector<VkExtensionProperties> ext(n_ext);
        CHECK(EnumerateDeviceExtensionProperties(dev, nullptr, &n_ext,
                                                 ext.data()) == VK_SUCCESS);
        bool has_accel = false, has_ray_query = false;
        for (const VkExtensionProperties &e : ext) {
            has_accel |= strcmp(e.extensionName,
                                VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME) == 0;
            has_ray_query |=
                strcmp(e.extensionName, VK_KHR_RAY_QUERY_EXTENSION_NAME) == 0;
        }

        printf("  - \"%s\" (Vulkan %u.%u.%u, driver \"%s\", subgroup size %u, "
               "%u extensions, compute queue: %s, ray queries: %s)\n",
               props.deviceName, VK_API_VERSION_MAJOR(version),
               VK_API_VERSION_MINOR(version), VK_API_VERSION_PATCH(version),
               driver_name, subgroup_size, n_ext, has_compute ? "yes" : "no",
               (has_accel && has_ray_query) ? "yes" : "no");
    }
}

/// Minimal compute kernel using the features that the Vulkan backend relies
/// on: SPIR-V 1.5, buffer device addresses (passed via push constants)
static const char *kernel_src = R"(
               OpCapability Shader
               OpCapability PhysicalStorageBufferAddresses
               OpMemoryModel PhysicalStorageBuffer64 GLSL450
               OpEntryPoint GLCompute %main "main" %gid %pc
               OpExecutionMode %main LocalSize 64 1 1
               OpDecorate %gid BuiltIn GlobalInvocationId
               OpDecorate %Params Block
               OpMemberDecorate %Params 0 Offset 0
               OpDecorate %ptr_buf ArrayStride 4
       %void = OpTypeVoid
       %func = OpTypeFunction %void
       %uint = OpTypeInt 32 0
     %v3uint = OpTypeVector %uint 3
  %ptr_in_v3 = OpTypePointer Input %v3uint
        %gid = OpVariable %ptr_in_v3 Input
    %ptr_buf = OpTypePointer PhysicalStorageBuffer %uint
     %Params = OpTypeStruct %ptr_buf
  %ptr_pc_st = OpTypePointer PushConstant %Params
         %pc = OpVariable %ptr_pc_st PushConstant
 %ptr_pc_buf = OpTypePointer PushConstant %ptr_buf
     %uint_0 = OpConstant %uint 0
       %main = OpFunction %void None %func
      %entry = OpLabel
         %id = OpLoad %v3uint %gid
          %x = OpCompositeExtract %uint %id 0
         %pp = OpAccessChain %ptr_pc_buf %pc %uint_0
        %buf = OpLoad %ptr_buf %pp
       %elem = OpPtrAccessChain %ptr_buf %buf %x
               OpStore %elem %x Aligned 4
               OpReturn
               OpFunctionEnd
)";

static void test03_spirv_tools() {
    auto ContextCreate = lookup<PFN_spvContextCreate>("spvContextCreate");
    auto ContextDestroy = lookup<PFN_spvContextDestroy>("spvContextDestroy");
    auto TextToBinary = lookup<PFN_spvTextToBinary>("spvTextToBinary");
    auto BinaryDestroy = lookup<PFN_spvBinaryDestroy>("spvBinaryDestroy");
    auto BinaryToText = lookup<PFN_spvBinaryToText>("spvBinaryToText");
    auto TextDestroy = lookup<PFN_spvTextDestroy>("spvTextDestroy");
    auto ValidateBinary = lookup<PFN_spvValidateBinary>("spvValidateBinary");
    auto DiagnosticDestroy =
        lookup<PFN_spvDiagnosticDestroy>("spvDiagnosticDestroy");

    spv_context ctx = ContextCreate(SPV_ENV_VULKAN_1_3);
    CHECK(ctx != nullptr);

    // Assemble, validate, and disassemble a valid kernel
    spv_binary binary = nullptr;
    spv_diagnostic diag = nullptr;
    CHECK(TextToBinary(ctx, kernel_src, strlen(kernel_src), &binary, &diag) ==
          SPV_SUCCESS);
    CHECK(binary && !diag && binary->wordCount > 5);
    CHECK(binary->code[0] == 0x07230203u); // SPIR-V magic number
    CHECK(binary->code[1] == 0x00010600u); // SPIR-V 1.6 (Vulkan 1.3)
    CHECK(ValidateBinary(ctx, binary->code, binary->wordCount, &diag) ==
          SPV_SUCCESS);
    CHECK(!diag);

    spv_text text = nullptr;
    CHECK(BinaryToText(ctx, binary->code, binary->wordCount,
                       SPV_BINARY_TO_TEXT_OPTION_NONE, &text,
                       &diag) == SPV_SUCCESS);
    CHECK(text && text->length > 0 &&
          strstr(text->str, "OpEntryPoint GLCompute") &&
          strstr(text->str, "PhysicalStorageBuffer64"));
    TextDestroy(text);
    BinaryDestroy(binary);

    // Syntax errors are reported via a diagnostic
    const char *bad_syntax = "OpCapability Shader\nOpThisIsNotAnInstruction\n";
    binary = nullptr;
    CHECK(TextToBinary(ctx, bad_syntax, strlen(bad_syntax), &binary, &diag) !=
          SPV_SUCCESS);
    CHECK(diag && diag->error && strlen(diag->error) > 0 &&
          diag->position.line == 1);
    printf("  SPIRV-Tools syntax error (expected): %s\n", diag->error);
    DiagnosticDestroy(diag);
    diag = nullptr;
    if (binary)
        BinaryDestroy(binary);

    // SPIR-V >= 1.4 requires listing all global variables in OpEntryPoint
    std::string invalid = kernel_src;
    size_t pos = invalid.find("%gid %pc");
    CHECK(pos != std::string::npos);
    invalid.replace(pos, 8, "%gid");
    binary = nullptr;
    CHECK(TextToBinary(ctx, invalid.c_str(), invalid.size(), &binary, &diag) ==
          SPV_SUCCESS);
    CHECK(ValidateBinary(ctx, binary->code, binary->wordCount, &diag) !=
          SPV_SUCCESS);
    CHECK(diag && diag->error && strstr(diag->error, "interface"));
    printf("  SPIRV-Tools validation error (expected): %s\n", diag->error);
    DiagnosticDestroy(diag);
    BinaryDestroy(binary);

    ContextDestroy(ctx);
}

static void test04_init_shutdown_cycles() {
    void *instance = jit_vulkan_instance();

    // Calling jit_init() again is harmless
    jit_init(BackendLLVM | BackendVulkan);
    CHECK(jit_vulkan_instance() == instance);

    // A light shutdown keeps the libraries (and the instance) loaded
    jit_shutdown(1);
    CHECK(jit_vulkan_instance() == instance);
    jit_init(BackendLLVM | BackendVulkan);
    CHECK(jit_vulkan_instance() == instance);
    check_loaded();
    check_llvm();

    // A full shutdown unloads everything, and it can be loaded again
    for (int i = 0; i < 3; ++i) {
        jit_shutdown(0);
        check_unloaded();
        jit_init(BackendLLVM | BackendVulkan);
        check_loaded();
        check_llvm();
    }

    // Loading only the Vulkan backend works as well
    jit_shutdown(0);
    jit_init(BackendVulkan);
    check_loaded();
    jit_shutdown(0);
    check_unloaded();
    jit_init(BackendLLVM | BackendVulkan);
    check_loaded();
}

/// Try to initialize the Vulkan backend, which is expected to fail
static void expect_failure(const char *expected_msg) {
    jit_shutdown(0);
    {
        LogCapture capture;
        jit_init(BackendLLVM | BackendVulkan);
        CHECK(capture.contains("disabling the Vulkan backend"));
        CHECK(capture.contains(expected_msg));
    }
    check_unloaded(); // No partially initialized state is left behind
    check_llvm();
}

static void test05_failure_modes() {
    EnvGuard guard_vk("DRJIT_LIBVULKAN_PATH"),
             guard_spv("DRJIT_LIBSPIRV_TOOLS_PATH");

    // Missing SPIRV-Tools library
    set_env("DRJIT_LIBSPIRV_TOOLS_PATH", "/nonexistent/libSPIRV-Tools-shared.so");
    expect_failure("could not load the SPIRV-Tools library");
    set_env("DRJIT_LIBSPIRV_TOOLS_PATH", guard_spv.was_set ? guard_spv.value.c_str() : nullptr);

    // Missing Vulkan loader
    set_env("DRJIT_LIBVULKAN_PATH", "/nonexistent/libvulkan.so.1");
    expect_failure("could not load the Vulkan loader");
    set_env("DRJIT_LIBVULKAN_PATH", guard_vk.was_set ? guard_vk.value.c_str() : nullptr);

#if defined(__linux__)
    // Libraries that exist but are neither the Vulkan loader nor SPIRV-Tools
    set_env("DRJIT_LIBVULKAN_PATH", "libm.so.6");
    expect_failure("does not export \"vkGetInstanceProcAddr\"");
    set_env("DRJIT_LIBVULKAN_PATH", guard_vk.was_set ? guard_vk.value.c_str() : nullptr);

    set_env("DRJIT_LIBSPIRV_TOOLS_PATH", "libm.so.6");
    expect_failure("in the SPIRV-Tools library");
    set_env("DRJIT_LIBSPIRV_TOOLS_PATH", guard_spv.was_set ? guard_spv.value.c_str() : nullptr);
#endif

    // Everything works again once the environment is fixed
    jit_shutdown(0);
    jit_init(BackendLLVM | BackendVulkan);
    check_loaded();
    check_llvm();
}

int main(int argc, char **argv) {
    bool verbose = argc > 1 && strcmp(argv[1], "-v") == 0;
    jit_set_log_level_stderr(verbose ? LogLevel::Info : LogLevel::Warn);

    const char *require_env = getenv("DRJIT_TEST_REQUIRE_VULKAN");
    bool require = require_env && strcmp(require_env, "1") == 0;

    // The Vulkan backend is opt-in: it is not loaded by default
    jit_init(BackendLLVM);
    have_llvm = jit_has_backend(JitBackend::LLVM);
    check_unloaded();

    jit_init(BackendLLVM | BackendVulkan);
    if (!jit_vulkan_instance()) {
        check_unloaded();
        check_llvm();
        jit_shutdown(0);
        if (require) {
            fprintf(stderr, "test_vulkan_api: FAILED (Vulkan or SPIRV-Tools "
                            "are unavailable, see above)\n");
            return EXIT_FAILURE;
        }
        printf("test_vulkan_api: skipped (Vulkan or SPIRV-Tools are "
               "unavailable, see above).\n");
        return EXIT_SUCCESS;
    }
    have_device = jit_has_backend(JitBackend::Vulkan);
    printf("  Vulkan device: %s\n",
           have_device ? jit_vulkan_device_name() : "(none)");
    if (require)
        CHECK(have_device);
    check_loaded();
    check_llvm();

    printf("test01_lookup\n");
    test01_lookup();
    printf("test02_physical_devices\n");
    test02_physical_devices(require);
    printf("test03_spirv_tools\n");
    test03_spirv_tools();
    printf("test04_init_shutdown_cycles\n");
    test04_init_shutdown_cycles();
    printf("test05_failure_modes\n");
    test05_failure_modes();

    jit_shutdown(0);
    check_unloaded();

    printf("test_vulkan_api: all %i checks passed.\n", checks_run);
    return EXIT_SUCCESS;
}
