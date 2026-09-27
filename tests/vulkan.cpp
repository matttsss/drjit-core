/*
    tests/vulkan.cpp -- Tests of the Vulkan backend: device selection, memory
    allocation, transfers, synchronization, host callbacks, SPIR-V 1.6 codegen,
    runtime pipeline compilation, and compute kernel execution

    These tests are skipped when no compatible Vulkan device is found, unless
    the environment variable DRJIT_TEST_REQUIRE_VULKAN=1 is set. Set
    DRJIT_VULKAN_DEVICE to the index of a physical device to run them on a
    specific device.

    Copyright (c) 2026 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#include "test.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

static const uint32_t BackendVulkan = 1u << (uint32_t) JitBackend::Vulkan;

// ====================================================================
//                              Helpers
// ====================================================================

/// Does 'func()' raise an exception whose message contains 'msg'?
template <typename Func> static bool raises(Func func, const char *msg) {
    try {
        func();
    } catch (const std::exception &e) {
        if (strstr(e.what(), msg))
            return true;
        fprintf(stderr, "Unexpected exception message: %s\n", e.what());
    }
    return false;
}

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

/// Generate deterministic pseudorandom bytes
static std::vector<uint8_t> random_bytes(size_t size, uint32_t seed) {
    std::vector<uint8_t> result(size);
    uint64_t state = 0x853c49e6748fea9bull ^ (seed * 0x9e3779b97f4a7c15ull);
    for (size_t i = 0; i < size; i += 8) {
        state = state * 6364136223846793005ull + 1442695040888963407ull;
        uint64_t value = state ^ (state >> 29);
        memcpy(result.data() + i, &value, std::min((size_t) 8, size - i));
    }
    return result;
}

/// Kinds of memory involved in transfers
enum class Kind { Host, Device, Shared };

/// Memory allocation of a given kind (RAII)
struct Alloc {
    Kind kind;
    size_t size;
    uint8_t *ptr;

    Alloc(Kind kind, size_t size) : kind(kind), size(size) {
        if (kind == Kind::Host)
            ptr = (uint8_t *) malloc(size);
        else
            ptr = (uint8_t *) jit_malloc(JitBackend::Vulkan, size,
                                         kind == Kind::Shared);
        jit_assert(ptr);
    }

    ~Alloc() {
        if (kind == Kind::Host)
            free(ptr);
        else
            jit_free(ptr);
    }

    Alloc(const Alloc &) = delete;
    Alloc &operator=(const Alloc &) = delete;
};

/// Synchronously copy host memory to any kind of memory
static void upload(void *dst, const void *src, size_t size) {
    jit_memcpy(JitBackend::Vulkan, dst, src, size);
}

/// Synchronously copy any kind of memory to host memory
static std::vector<uint8_t> download(const void *src, size_t size) {
    std::vector<uint8_t> result(size);
    jit_memcpy(JitBackend::Vulkan, result.data(), src, size);
    return result;
}

/// Reference implementation of jit_memset_async()
static void memset_ref(uint8_t *ptr, size_t count, uint32_t isize,
                       const void *value) {
    for (size_t i = 0; i < count; ++i)
        memcpy(ptr + i * isize, value, isize);
}

/// Quick end-to-end check of the Vulkan device (uploads, device copies,
/// fills, and downloads)
static void check_round_trip() {
    const size_t size = 100000;
    std::vector<uint8_t> ref = random_bytes(size, 1234);
    Alloc a(Kind::Device, size), b(Kind::Device, size),
          s(Kind::Shared, size);

    uint16_t value = 0xabcd;
    jit_memcpy_async(JitBackend::Vulkan, a.ptr, ref.data(), size);
    jit_memcpy_async(JitBackend::Vulkan, b.ptr, a.ptr, size);
    jit_memset_async(JitBackend::Vulkan, b.ptr + 2, 1000, 2, &value);
    jit_memcpy_async(JitBackend::Vulkan, s.ptr, b.ptr, size);
    memset_ref(ref.data() + 2, 1000, 2, &value);

    jit_sync_thread();
    jit_assert(memcmp(s.ptr, ref.data(), size) == 0);
}

/// Leading members of VkPhysicalDeviceProperties (padded to cover the
/// remaining ones), which avoids a dependency on the Vulkan headers
struct alignas(8) PhysicalDeviceProperties {
    uint32_t apiVersion, driverVersion, vendorID, deviceID, deviceType;
    char deviceName[256];
    uint8_t padding[2048];
};

using PFN_EnumeratePhysicalDevices = int32_t (*)(void *, uint32_t *, void **);
using PFN_GetPhysicalDeviceProperties = void (*)(void *, void *);
using PFN_GetPhysicalDeviceProperties2 = void (*)(void *, void *);

struct PhysicalDeviceVulkan13Properties {
    uint32_t sType = 54;
    void *pNext = nullptr;
    uint8_t padding[192] { };
    uint64_t maxBufferSize = 0;
};

struct PhysicalDeviceProperties2 {
    uint32_t sType = 1000059001;
    void *pNext = nullptr;
    PhysicalDeviceProperties properties { };
};

/// Enumerate the physical devices of the Vulkan instance of Dr.Jit
static std::vector<void *> physical_devices() {
    auto enumerate = (PFN_EnumeratePhysicalDevices) jit_vulkan_lookup(
        "vkEnumeratePhysicalDevices");
    uint32_t count = 0;
    jit_assert(enumerate(jit_vulkan_instance(), &count, nullptr) == 0);
    std::vector<void *> result(count);
    jit_assert(enumerate(jit_vulkan_instance(), &count, result.data()) == 0);
    result.resize(count);
    return result;
}

// ====================================================================
//                               Tests
// ====================================================================

TEST_VULKAN(01_device) {
    jit_assert(jit_has_backend(Backend));
    jit_assert(jit_vulkan_instance());
    jit_assert(jit_vulkan_device_handle());
    jit_assert(jit_vulkan_physical_device());
    jit_assert(jit_vulkan_queue());
    jit_assert(jit_vulkan_queue_family() >= 0);

    const char *name = jit_vulkan_device_name();
    jit_assert(name && strlen(name) > 0);

    // The device index refers to the order of vkEnumeratePhysicalDevices()
    int index = jit_vulkan_device_index();
    std::vector<void *> devices = physical_devices();
    jit_assert(index >= 0 && index < (int) devices.size());
    jit_assert(devices[index] == jit_vulkan_physical_device());

    const char *env = getenv("DRJIT_VULKAN_DEVICE");
    if (env && *env)
        jit_assert(index == atoi(env));

    auto get_properties = (PFN_GetPhysicalDeviceProperties) jit_vulkan_lookup(
        "vkGetPhysicalDeviceProperties");
    PhysicalDeviceProperties props;
    get_properties(devices[index], &props);
    jit_assert(strcmp(props.deviceName, name) == 0);
    jit_assert(props.apiVersion >= ((1u << 22) | (3u << 12))); // >= 1.3

    jit_log(LogLevel::Info, "Using Vulkan device %i (\"%s\")", index, name);
}

TEST_VULKAN(02_malloc) {
    jit_assert(jit_malloc(Backend, 0, false) == nullptr);
    jit_free(nullptr);

    for (int shared = 0; shared < 2; ++shared) {
        uint8_t *p = (uint8_t *) jit_malloc(Backend, 1024, shared);
        jit_assert(p);
        jit_assert(jit_malloc_device(p) == 0);

        // Each allocation is backed by a dedicated buffer
        size_t offset = 1;
        void *buffer = jit_vulkan_lookup_buffer(p, &offset);
        jit_assert(buffer && offset == 0);
        jit_assert(jit_vulkan_lookup_buffer(p + 1023, &offset) == buffer &&
                   offset == 1023);

        // Device-local memory is represented by its device address
        uint64_t address = jit_vulkan_device_address(p);
        jit_assert(address != 0);
        jit_assert(jit_vulkan_device_address(p + 123) == address + 123);
        if (!shared)
            jit_assert(address == (uint64_t) (uintptr_t) p);

        // Shared memory is directly accessible by the host
        if (shared) {
            memset(p, 0x5a, 1024);
            jit_assert(p[0] == 0x5a && p[1023] == 0x5a);
        }

        // Released memory is cached and reused
        jit_sync_thread();
        jit_free(p);
        uint8_t *p2 = (uint8_t *) jit_malloc(Backend, 1024, shared);
        jit_assert(p2 == p && jit_vulkan_lookup_buffer(p2) == buffer);
        jit_free(p2);
    }

    jit_assert(jit_malloc_watermark(Backend) >= 1024);

    // Flushing the cache releases the buffers
    uint8_t *p = (uint8_t *) jit_malloc(Backend, 4096, false);
    jit_free(p);
    jit_assert(jit_vulkan_lookup_buffer(p));
    jit_flush_malloc_cache();
    jit_assert(!jit_vulkan_lookup_buffer(p));
    jit_assert(jit_vulkan_device_address(p) == 0);

    // Host memory is not part of a Vulkan allocation
    int host_value = 0;
    size_t offset = 1;
    jit_assert(!jit_vulkan_lookup_buffer(&host_value, &offset) && offset == 0);
    jit_assert(jit_vulkan_device_address(&host_value) == 0);
    jit_assert(!jit_vulkan_lookup_buffer(nullptr));
    jit_assert(jit_vulkan_device_address(nullptr) == 0);

    // Large allocations (256 MiB)
    for (int shared = 0; shared < 2; ++shared) {
        size_t size = (size_t) 256 << 20;
        uint8_t *q = (uint8_t *) jit_malloc(Backend, size, shared);
        jit_assert(jit_vulkan_lookup_buffer(q + size - 1, &offset) &&
                   offset == size - 1);
        jit_free(q);
    }
    jit_flush_malloc_cache();

    // Large allocation (1 GiB)
    {
        size_t size = (size_t) 1 << 30;
        uint8_t *q = (uint8_t *) jit_malloc(Backend, size, false);
        jit_assert(q != nullptr);
        jit_assert(jit_vulkan_lookup_buffer(q + size - 1, &offset) &&
                   offset == size - 1);
        jit_free(q);
        jit_flush_malloc_cache();
    }

    // Large allocation (3 GiB requested -> 4 GiB power-of-two buffer).
    // On devices where maxMemoryAllocationSize < 4 GiB and maxBufferSize >= 4 GiB
    // (such as NVIDIA, where maxMemoryAllocationSize is ~3.998 GiB and
    // maxBufferSize is 1 TiB), this verifies allocation beyond maxMemoryAllocationSize
    // via direct allocation or sparse binding.
    {
        auto get_properties2 = (PFN_GetPhysicalDeviceProperties2) jit_vulkan_lookup(
            "vkGetPhysicalDeviceProperties2");
        PhysicalDeviceVulkan13Properties props13;
        PhysicalDeviceProperties2 props2;
        props2.pNext = &props13;
        get_properties2(jit_vulkan_physical_device(), &props2);

        size_t size = (size_t) 3 << 30; // rounds up to 4 GiB
        size_t rounded_size = (size_t) 4 << 30;

        if (rounded_size <= props13.maxBufferSize) {
            uint8_t *q = (uint8_t *) jit_malloc(Backend, size, false);
            jit_assert(q != nullptr);
            jit_assert(jit_vulkan_device_address(q) != 0);
            jit_assert(jit_vulkan_lookup_buffer(q + rounded_size - 1, &offset) &&
                       offset == rounded_size - 1);
            jit_free(q);
            jit_flush_malloc_cache();
        } else {
            jit_assert(raises([&] { jit_malloc(Backend, size, false); },
                              "exceeds the maximum allocation size"));
        }
    }

    // Shared allocation of 3 GiB (rounds up to 4 GiB) exceeds maxMemoryAllocationSize
    // on NVIDIA (~3.998 GiB) and llvmpipe (2.0 GiB) and cannot be sparse.
    jit_assert(raises([&] { jit_malloc(Backend, (size_t) 3 << 30, true); },
                      "exceeds the maximum allocation size"));

    // Allocations exceeding the maximum buffer size raise an exception
    for (int shared = 0; shared < 2; ++shared)
        jit_assert(raises([&] { jit_malloc(Backend, (size_t) 1 << 50, shared); },
                          "exceeds the maximum allocation size"));
}

/// Copy between all combinations of memory kinds, sizes, and alignments
static void check_memcpy(bool async) {
    const size_t sizes[] = { 1,     3,     4,     13,     64,
                             1000,  16384, 16389, 65536,  65540,
                             100003, ((size_t) 2 << 20) + 12 };
    const size_t offsets[][2] = { { 0, 0 }, { 4, 12 }, { 1, 6 }, { 7, 3 } };
    const Kind kinds[] = { Kind::Host, Kind::Device, Kind::Shared };
    const size_t pad = 16;
    uint32_t seed = 0;

    for (Kind src_kind : kinds) {
        for (Kind dst_kind : kinds) {
            for (size_t size : sizes) {
                for (const auto &[src_offset, dst_offset] : offsets) {
                    Alloc src(src_kind, src_offset + size + pad),
                          dst(dst_kind, dst_offset + size + pad);

                    std::vector<uint8_t> src_data = random_bytes(src.size, seed++),
                                         dst_data = random_bytes(dst.size, seed++);
                    upload(src.ptr, src_data.data(), src.size);
                    upload(dst.ptr, dst_data.data(), dst.size);

                    if (async) {
                        jit_memcpy_async(JitBackend::Vulkan, dst.ptr + dst_offset,
                                         src.ptr + src_offset, size);

                        // Copies from host memory capture the data right away
                        if (src_kind == Kind::Host)
                            memset(src.ptr, 0, src.size);

                        // Copies to host memory are complete upon return
                        if (dst_kind != Kind::Host)
                            jit_sync_thread();
                    } else {
                        jit_memcpy(JitBackend::Vulkan, dst.ptr + dst_offset,
                                   src.ptr + src_offset, size);
                    }

                    memcpy(dst_data.data() + dst_offset,
                           src_data.data() + src_offset, size);
                    jit_assert(download(dst.ptr, dst.size) == dst_data);
                }
            }
        }
    }
}

TEST_VULKAN(03_memcpy) { check_memcpy(false); }

TEST_VULKAN(04_memcpy_async) { check_memcpy(true); }

TEST_VULKAN(05_memcpy_large) {
    const size_t size = (size_t) 64 << 20;
    std::vector<uint8_t> ref = random_bytes(size, 5);
    Alloc a(Kind::Device, size), b(Kind::Device, size),
          s(Kind::Shared, size);

    jit_memcpy_async(Backend, a.ptr, ref.data(), size);
    jit_memcpy_async(Backend, b.ptr, a.ptr, size);
    jit_memcpy_async(Backend, s.ptr, b.ptr, size);
    jit_sync_thread();
    jit_assert(memcmp(s.ptr, ref.data(), size) == 0);
    jit_assert(download(b.ptr, size) == ref);
}

TEST_VULKAN(06_hazards) {
    const size_t size = (size_t) 1 << 20;
    std::vector<uint8_t> d0 = random_bytes(size, 60),
                         d1 = random_bytes(size, 61),
                         ref;
    Alloc a(Kind::Device, size), b(Kind::Device, size),
          c(Kind::Device, size), s(Kind::Shared, size);

    // Read after write: chain of dependent copies
    jit_memcpy_async(Backend, a.ptr, d0.data(), size);
    jit_memcpy_async(Backend, b.ptr, a.ptr, size);
    jit_memcpy_async(Backend, c.ptr, b.ptr, size);
    jit_memcpy_async(Backend, s.ptr, c.ptr, size);
    jit_sync_thread();
    jit_assert(memcmp(s.ptr, d0.data(), size) == 0);

    // Write after read: overwrite the source of an earlier copy
    jit_memcpy_async(Backend, c.ptr, a.ptr, size);
    jit_memcpy_async(Backend, a.ptr, d1.data(), size);
    jit_memcpy_async(Backend, s.ptr, c.ptr, size);
    jit_sync_thread();
    jit_assert(memcmp(s.ptr, d0.data(), size) == 0);
    jit_assert(download(a.ptr, size) == d1);

    // Write after write, including partially overlapping writes
    uint32_t v0 = 0x11111111u, v1 = 0x22222222u;
    jit_memset_async(Backend, c.ptr, (uint32_t) (size / 4), 4, &v0);
    jit_memcpy_async(Backend, c.ptr + 1000, d0.data(), 5000);
    jit_memset_async(Backend, c.ptr + 2000, 1000, 4, &v1);
    jit_memcpy_async(Backend, s.ptr, c.ptr, size);
    ref.resize(size);
    memset_ref(ref.data(), size / 4, 4, &v0);
    memcpy(ref.data() + 1000, d0.data(), 5000);
    memset_ref(ref.data() + 2000, 1000, 4, &v1);
    jit_sync_thread();
    jit_assert(memcmp(s.ptr, ref.data(), size) == 0);

    // Host writes to shared memory are visible to subsequent commands
    memset(s.ptr, 0x33, size);
    jit_memcpy_async(Backend, a.ptr, s.ptr, size);
    jit_assert(download(a.ptr, size) == std::vector<uint8_t>(size, 0x33));

    // Copies between disjoint parts of the same allocation
    jit_memcpy_async(Backend, b.ptr, d1.data(), size);
    jit_memcpy_async(Backend, b.ptr + size / 2, b.ptr, size / 2);
    ref = d1;
    memcpy(ref.data() + size / 2, d1.data(), size / 2);
    jit_assert(download(b.ptr, size) == ref);

    // Many small dependent commands (which exceed the number of commands per
    // command buffer and the number of tracked memory ranges)
    const uint32_t n = 5000;
    for (uint32_t i = 0; i < n; ++i) {
        uint32_t k = i % 256;
        jit_memcpy_async(Backend, a.ptr + 4 * k, &i, 4);
        jit_memcpy_async(Backend, b.ptr + 4 * k, a.ptr + 4 * k, 4);
    }
    std::vector<uint8_t> out = download(b.ptr, 1024);
    for (uint32_t k = 0; k < 256; ++k) {
        uint32_t value;
        memcpy(&value, out.data() + 4 * k, 4);
        jit_assert(value == k + 256 * ((n - 1 - k) / 256));
    }
}

TEST_VULKAN(07_memset) {
    const uint8_t v1 = 0xa5;
    const uint16_t v2 = 0x1234;
    const uint32_t v4 = 0xdeadbeefu;
    const uint64_t v8a = 0x0102030401020304ull, // (4-byte period)
                   v8b = 0x0123456789abcdefull;
    struct Pattern { uint32_t isize; const void *value; };
    const Pattern patterns[] = { { 1, &v1 }, { 2, &v2 }, { 4, &v4 },
                                 { 8, &v8a }, { 8, &v8b } };
    const size_t counts[] = { 1, 2, 3, 5, 17, 1000, 2049, 8193, 100000 };
    const size_t offsets[] = { 0, 1, 2, 3, 4, 6, 8 };
    const size_t pad = 16;
    uint32_t seed = 0;

    // Fill part of an allocation and check that the rest is unchanged
    for (Kind kind : { Kind::Device, Kind::Shared }) {
        for (const Pattern &p : patterns) {
            for (size_t count : counts) {
                for (size_t offset : offsets) {
                    Alloc a(kind, offset + count * p.isize + pad);
                    std::vector<uint8_t> ref = random_bytes(a.size, seed++);
                    upload(a.ptr, ref.data(), a.size);

                    jit_memset_async(Backend, a.ptr + offset, (uint32_t) count,
                                     p.isize, p.value);
                    memset_ref(ref.data() + offset, count, p.isize, p.value);
                    jit_assert(download(a.ptr, a.size) == ref);
                }
            }
        }
    }

    // Large fills (64 MiB)
    const size_t size = (size_t) 64 << 20;
    Alloc a(Kind::Device, size);
    std::vector<uint8_t> ref(size);
    for (const Pattern &p : { patterns[2], patterns[4] }) {
        jit_memset_async(Backend, a.ptr, (uint32_t) (size / p.isize), p.isize,
                         p.value);
        memset_ref(ref.data(), size / p.isize, p.isize, p.value);
        jit_assert(download(a.ptr, size) == ref);
    }

    // Empty fills don't do anything
    jit_memset_async(Backend, a.ptr, 0, 1, &v1);
    jit_assert(download(a.ptr, 16) == std::vector<uint8_t>(ref.begin(), ref.begin() + 16));
}

/// Check jit_var_read() and jit_var_write() (which uses jit_poke())
template <template <class> class Array, typename T>
static void check_read_write(uint32_t n) {
    std::vector<T> data(n);
    for (uint32_t i = 0; i < n; ++i)
        data[i] = (T) (3 * i + 1);

    Array<T> x = Array<T>::copy(data.data(), n);
    jit_assert(jit_var_state(x.index()) == VarState::Evaluated);
    for (uint32_t i = 0; i < n; ++i)
        jit_assert(x.read(i) == data[i]);

    for (uint32_t i = 0; i < n; i += 2) {
        data[i] = (T) (100 + i);
        x.write(i, data[i]);
    }

    // jit_var_write() copies arrays that are also referenced elsewhere
    Array<T> y = x;
    y.write(1, (T) 42);
    jit_assert(x.index() != y.index());

    for (uint32_t i = 0; i < n; ++i) {
        jit_assert(x.read(i) == data[i]);
        jit_assert(y.read(i) == (i == 1 ? (T) 42 : data[i]));
    }
}

TEST_VULKAN(08_var_read_write) {
    // (the small types exercise unaligned writes)
    check_read_write<Array, uint8_t>(37);
    check_read_write<Array, int16_t>(21);
    check_read_write<Array, uint32_t>(13);
    check_read_write<Array, float>(13);
    check_read_write<Array, uint64_t>(9);
    check_read_write<Array, double>(9);

    UInt32 x(1, 2, 3);
    jit_assert(strcmp(x.str(), "[1, 2, 3]") == 0);
}

TEST_VULKAN(09_literals) {
    // Evaluated literals are initialized using jit_memset_async()
    float f = 1.25f;
    Float x = Float::steal(
        jit_var_literal(Backend, VarType::Float32, &f, 1000, 1));
    jit_assert(jit_var_state(x.index()) == VarState::Evaluated);
    jit_assert(x.read(0) == 1.25f && x.read(999) == 1.25f);

    // .. and so are literals whose contents are accessed
    uint64_t value = 0x0123456789abcdefull;
    Array<uint64_t> y = Array<uint64_t>::steal(
        jit_var_literal(Backend, VarType::UInt64, &value, 100, 0));
    jit_assert(jit_var_state(y.index()) == VarState::Literal);
    const uint64_t *ptr = y.data();
    jit_assert(jit_var_state(y.index()) == VarState::Evaluated);
    std::vector<uint64_t> out(100);
    jit_memcpy(Backend, out.data(), ptr, 100 * sizeof(uint64_t));
    for (uint64_t v : out)
        jit_assert(v == value);

    UInt32 z = 7;
    z.resize(10);
    z.make_opaque();
    jit_assert(jit_var_state(z.index()) == VarState::Evaluated);
    jit_assert(strcmp(z.str(), "[7, 7, 7, 7, 7, 7, 7, 7, 7, 7]") == 0);
}

TEST_VULKAN(10_copy_migrate) {
    const uint32_t n = 1000;
    std::vector<uint32_t> data(n);
    for (uint32_t i = 0; i < n; ++i)
        data[i] = i * i;

    auto check = [&](const UInt32 &v) {
        jit_assert(v.size() == n);
        std::vector<uint32_t> out(n);
        jit_memcpy(Backend, out.data(), v.data(), n * sizeof(uint32_t));
        jit_assert(out == data);
    };

    UInt32 a = UInt32::copy(data.data(), n);
    check(a);

    // Device-to-device copies
    check(UInt32::steal(
        jit_var_mem_copy(Backend, VarType::UInt32, a.data(), n, 0)));
    UInt32 b = UInt32::steal(jit_var_copy(a.index()));
    jit_assert(b.index() != a.index());
    check(b);
    check(UInt32::steal(jit_var_migrate(a.index(), Backend)));

    // Migration to host-accessible (shared) memory
    UInt32 c = UInt32::steal(jit_var_migrate(a.index(), JitBackend::None));
    const uint32_t *c_ptr = c.data();
    jit_sync_thread();
    jit_assert(memcmp(c_ptr, data.data(), n * sizeof(uint32_t)) == 0);

    // Mapping a Vulkan allocation
    void *ptr = jit_malloc(Backend, n * sizeof(uint32_t), false);
    upload(ptr, data.data(), n * sizeof(uint32_t));
    check(UInt32::map(ptr, n, true));

    // Mapping host memory, which jit_var_migrate() copies to the device
    std::vector<uint32_t> host = data;
    UInt32 d = UInt32::map(host.data(), n, false);
    UInt32 e = UInt32::steal(jit_var_migrate(d.index(), Backend));
    host.assign(n, 0); // (the migration already captured the data)
    jit_assert(jit_vulkan_lookup_buffer(e.data()));
    check(e);

    // jit_malloc_migrate(): host -> device -> device -> host
    std::vector<uint8_t> ref = random_bytes(4096, 10);
    void *p0 = jit_malloc(JitBackend::None, 4096);
    memcpy(p0, ref.data(), 4096);
    void *p1 = jit_malloc_migrate(p0, Backend, 1), // (releases p0)
         *p2 = jit_malloc_migrate(p1, Backend, 0),
         *p3 = jit_malloc_migrate(p2, JitBackend::None, 1); // (releases p2)
    jit_assert(jit_vulkan_device_address(p1) == (uint64_t) (uintptr_t) p1);
    jit_sync_thread();
    jit_assert(memcmp(p3, ref.data(), 4096) == 0);
    jit_assert(download(p1, 4096) == ref);

    // Migrations within the same backend and device are no-ops if move=1
    jit_assert(jit_malloc_migrate(p1, Backend, 1) == p1);
    jit_free(p1);
    jit_free(p3);
}

TEST_VULKAN(11_sync) {
    const size_t size = (size_t) 4 << 20;
    Alloc d(Kind::Device, size), s(Kind::Shared, size);
    const uint32_t *p = (const uint32_t *) s.ptr;

    void (*syncs[])() = { jit_sync_thread, jit_sync_device,
                          jit_sync_all_devices,
                          [] { jit_flush_thread(); jit_sync_thread(); } };

    for (uint32_t i = 0; i < 4; ++i) {
        uint32_t value = 0x01020304u * (i + 1);
        jit_memset_async(Backend, d.ptr, (uint32_t) (size / 4), 4, &value);
        jit_memcpy_async(Backend, s.ptr, d.ptr, size);
        syncs[i]();
        for (size_t j = 0; j < size / 4; j += 1021)
            jit_assert(p[j] == value);
        jit_assert(p[size / 4 - 1] == value);
    }

    // jit_flush_thread() submits the work without waiting for it
    uint32_t value = 0xfeedf00du;
    jit_memset_async(Backend, d.ptr, (uint32_t) (size / 4), 4, &value);
    jit_memcpy_async(Backend, s.ptr, d.ptr, size);
    jit_flush_thread();
    auto start = std::chrono::steady_clock::now();
    while (((const volatile uint32_t *) p)[size / 4 - 1] != value) {
        jit_assert(std::chrono::steady_clock::now() - start <
                   std::chrono::seconds(30));
        std::this_thread::yield();
    }
    jit_sync_thread();
    jit_assert(p[0] == value);

    // Synchronizing without pending work is harmless
    jit_sync_thread();
    jit_sync_device();
    jit_sync_all_devices();
    jit_flush_thread();
    jit_eval();
}

TEST_VULKAN(12_shared_reuse) {
    /* jit_malloc(shared=true) must return memory that the backend no longer
       accesses, since the host may write to it right away */
    const size_t size = (size_t) 16 << 20;
    std::vector<uint8_t> ref = random_bytes(size, 12);
    Alloc d(Kind::Device, size);

    uint8_t *s = (uint8_t *) jit_malloc(Backend, size, true);
    memcpy(s, ref.data(), size);
    jit_memcpy_async(Backend, d.ptr, s, size); // (not yet submitted)
    jit_free(s);

    uint8_t *s2 = (uint8_t *) jit_malloc(Backend, size, true);
    memset(s2, 0, size);
    jit_assert(download(d.ptr, size) == ref);
    jit_free(s2);
}

struct CallbackData {
    std::vector<int> *log;
    int id;
    const uint32_t *check_ptr;
    uint32_t check_value;
    bool check_ok;
};

static void callback_log(void *payload) {
    CallbackData *d = (CallbackData *) payload;
    if (d->check_ptr)
        d->check_ok = *d->check_ptr == d->check_value;
    d->log->push_back(d->id);
}

static void callback_throw(void *) {
    throw std::runtime_error("callback_throw()");
}

static void callback_free(void *ptr) { jit_free(ptr); }

static void callback_memset(void *ptr) {
    uint32_t value = 0xccccccccu;
    jit_memset_async(JitBackend::Vulkan, ptr, 256, 4, &value);
}

TEST_VULKAN(13_host_callbacks) {
    const size_t size = (size_t) 1 << 20;
    Alloc d(Kind::Device, size), s(Kind::Shared, size);
    const uint32_t *p = (const uint32_t *) s.ptr;
    std::vector<int> log;

    // The callback runs right away when no work is pending
    CallbackData c0 { &log, 0, nullptr, 0, false };
    jit_sync_thread();
    jit_enqueue_host_func(Backend, callback_log, &c0);
    jit_assert(log == std::vector<int>({ 0 }));

    // Callbacks run in order once the preceding work has finished
    uint32_t v1 = 0xaaaaaaaau, v2 = 0xbbbbbbbbu;
    memset(s.ptr, 0, size);
    CallbackData c1 { &log, 1, p, v1, false },
                 c2 { &log, 2, p + size / 8, v2, false };

    jit_memset_async(Backend, d.ptr, (uint32_t) (size / 4), 4, &v1);
    jit_memcpy_async(Backend, s.ptr, d.ptr, size / 2);
    jit_enqueue_host_func(Backend, callback_log, &c1);
    jit_flush_thread();
    jit_memset_async(Backend, d.ptr, (uint32_t) (size / 4), 4, &v2);
    jit_memcpy_async(Backend, s.ptr + size / 2, d.ptr, size / 2);
    jit_flush_thread();
    jit_enqueue_host_func(Backend, callback_log, &c2);
    jit_sync_thread();
    jit_assert(log == std::vector<int>({ 0, 1, 2 }));
    jit_assert(c1.check_ok && c2.check_ok);

    // Callbacks can call Dr.Jit functions (they run without holding its lock)
    void *tmp = jit_malloc(Backend, 1024, true);
    jit_memcpy_async(Backend, d.ptr, tmp, 1024);
    jit_enqueue_host_func(Backend, callback_free, tmp);
    jit_sync_thread();

    // jit_sync_all_devices() also waits for work enqueued by callbacks
    jit_memset_async(Backend, d.ptr, 256, 4, &v1);
    jit_enqueue_host_func(Backend, callback_memset, s.ptr);
    jit_sync_all_devices();
    jit_assert(p[0] == 0xccccccccu && p[255] == 0xccccccccu);

    // Exceptions raised by callbacks propagate to the synchronizing thread.
    // The remaining callbacks run during the next synchronization.
    log.clear();
    CallbackData c3 { &log, 3, nullptr, 0, false };
    jit_memset_async(Backend, d.ptr, 256, 4, &v1);
    jit_enqueue_host_func(Backend, callback_throw, nullptr);
    jit_enqueue_host_func(Backend, callback_log, &c3);
    jit_assert(raises([] { jit_sync_thread(); }, "callback_throw()"));
    jit_assert(log.empty());
    jit_sync_thread();
    jit_assert(log == std::vector<int>({ 3 }));
}

TEST_VULKAN(14_threads) {
    const int n_threads = 4, n_iterations = 100;
    std::atomic<int> failures { 0 }, callbacks { 0 };
    std::vector<std::thread> threads;

    for (int t = 0; t < n_threads; ++t) {
        threads.emplace_back([t, &failures, &callbacks] {
            try {
                for (int i = 0; i < n_iterations; ++i) {
                    size_t size = 1000 + 4099 * (size_t) ((t * 31 + i * 17) % 64);
                    std::vector<uint8_t> ref =
                        random_bytes(size, (uint32_t) (t * 1000 + i));

                    Alloc a(Kind::Device, size), b(Kind::Device, size),
                          s(Kind::Shared, size);
                    jit_memcpy_async(JitBackend::Vulkan, a.ptr, ref.data(), size);
                    jit_memcpy_async(JitBackend::Vulkan, b.ptr, a.ptr, size);
                    jit_memcpy_async(JitBackend::Vulkan, s.ptr, b.ptr, size);
                    uint8_t value = (uint8_t) t;
                    jit_memset_async(JitBackend::Vulkan, a.ptr, (uint32_t) size,
                                     1, &value);
                    jit_enqueue_host_func(
                        JitBackend::Vulkan,
                        [](void *p) { ((std::atomic<int> *) p)->fetch_add(1); },
                        &callbacks);

                    if (download(b.ptr, size) != ref ||
                        download(a.ptr, size) != std::vector<uint8_t>(size, value))
                        failures++;

                    jit_sync_thread();
                    if (memcmp(s.ptr, ref.data(), size) != 0)
                        failures++;
                }
            } catch (const std::exception &e) {
                fprintf(stderr, "Thread %i: %s\n", t, e.what());
                failures++;
            }
        });
    }

    for (std::thread &t : threads)
        t.join();

    jit_sync_all_devices();
    jit_assert(failures == 0);
    jit_assert(callbacks == n_threads * n_iterations);
}

TEST_VULKAN(15_errors) {
    // Unimplemented kernel operations (e.g. cooperative vectors) raise a clear error
    {
        uint32_t idx = jit_var_counter(Backend, 10);
        uint32_t in[2] = { idx, idx };
        uint32_t cv = jit_coop_vec_pack(2, in);
        uint32_t out[2] = { 0, 0 };
        jit_coop_vec_unpack(cv, 2, out);
        jit_var_schedule(out[0]);
        jit_assert(raises([&] { jit_eval(); }, "cannot compile kernels"));
        jit_assert(jit_var_state(out[0]) == VarState::Unevaluated);
        jit_eval(); // (nothing left to do)
        jit_var_dec_ref(out[0]);
        jit_var_dec_ref(out[1]);
        jit_var_dec_ref(cv);
        jit_var_dec_ref(idx);
    }

    // Events and frozen functions are unsupported
    jit_assert(raises([&] { jit_event_create(Backend, 0); },
                      "not supported by the Vulkan backend"));
    jit_assert(raises([&] { jit_freeze_start(Backend, nullptr, 0); },
                      "not supported by the Vulkan backend"));

    // Operations that are not implemented yet or receive invalid arguments
    Alloc a(Kind::Device, 1024), b(Kind::Device, 1024);
    jit_assert(raises([&] { jit_batched_gemm(Backend, VarType::Float32, 0, 0,
                                             4, 4, 4, nullptr, a.ptr, a.ptr,
                                             b.ptr); },
                      "not supported by the Vulkan backend"));
    jit_assert(raises([&] { jit_block_reduce(Backend, VarType::UInt32,
                                             ReduceOp::Add, 16, 32, a.ptr,
                                             b.ptr); },
                      "invalid block size"));

    // Invalid memory operations
    uint8_t host[64] { };
    uint32_t value = 0;
    jit_assert(raises([&] { jit_memcpy(Backend, host, a.ptr + 1000, 64); },
                      "exceeds the bounds"));
    jit_assert(raises([&] { jit_memcpy_async(Backend, a.ptr + 1000, host, 64); },
                      "exceeds the bounds"));
    jit_assert(raises([&] { jit_memcpy_async(Backend, a.ptr + 100, a.ptr, 200); },
                      "overlap"));
    jit_assert(raises([&] { jit_memset_async(Backend, host, 16, 4, &value); },
                      "is not a Vulkan allocation"));
    jit_assert(raises([&] { jit_memset_async(Backend, a.ptr, 1000, 4, &value); },
                      "exceeds the bounds"));
    jit_assert(raises([&] { jit_memset_async(Backend, a.ptr, 16, 3, &value); },
                      "invalid element size"));

    // The backend remains functional
    check_round_trip();
}

TEST_VULKAN(16_shutdown) {
    // The device persists across light shutdowns
    void *device = jit_vulkan_device_handle();
    jit_shutdown(1);
    jit_assert(!jit_has_backend(Backend));
    jit_init(BackendVulkan);
    jit_assert(jit_has_backend(Backend));
    jit_assert(jit_vulkan_device_handle() == device);
    check_round_trip();

    // A full shutdown releases the device, including leaked allocations
    uint32_t value = 123;
    void *leaked_device = jit_malloc(Backend, 4096, false),
         *leaked_shared = jit_malloc(Backend, 4096, true);
    jit_memset_async(Backend, leaked_device, 1024, 4, &value);
    jit_memset_async(Backend, leaked_shared, 1024, 4, &value);

    int leak_warnings = jit_leak_warnings();
    jit_set_leak_warnings(0);
    jit_shutdown(0);
    jit_set_leak_warnings(leak_warnings);

    jit_assert(!jit_has_backend(Backend));
    jit_assert(!jit_vulkan_instance() && !jit_vulkan_device_handle() &&
               !jit_vulkan_physical_device() && !jit_vulkan_queue() &&
               !jit_vulkan_device_name());
    jit_assert(jit_vulkan_queue_family() == -1 &&
               jit_vulkan_device_index() == -1);
    jit_assert(!jit_vulkan_lookup_buffer(leaked_device));
    jit_assert(raises([&] { jit_malloc(Backend, 16, false); },
                      "has not been initialized"));

    jit_init(BackendVulkan);
    jit_assert(jit_has_backend(Backend) && jit_vulkan_device_handle());

    // Releasing allocations from before the full shutdown is harmless
    jit_free(leaked_device);
    jit_free(leaked_shared);
    check_round_trip();
}

TEST_VULKAN(17_device_selection) {
    const char *env = getenv("DRJIT_VULKAN_DEVICE");
    std::string env_value = env ? env : "";
    int default_index = jit_vulkan_device_index();
    size_t n_devices = physical_devices().size();

    // (suppress warnings about incompatible devices and invalid values)
    scoped_set_log_level guard(LogLevel::Error);

    // Select each device via DRJIT_VULKAN_DEVICE
    int n_compatible = 0;
    for (size_t i = 0; i < n_devices; ++i) {
        set_env("DRJIT_VULKAN_DEVICE", std::to_string(i).c_str());
        jit_shutdown(0);
        jit_init(BackendVulkan);
        if (!jit_has_backend(Backend)) {
            jit_log(LogLevel::Info, "Vulkan device %zu is incompatible.", i);
            continue;
        }
        jit_assert(jit_vulkan_device_index() == (int) i);
        jit_log(LogLevel::Info, "Testing Vulkan device %zu (\"%s\")", i,
                jit_vulkan_device_name());
        check_round_trip();
        n_compatible++;
    }
    jit_assert(n_compatible > 0);

    // Invalid values disable the Vulkan backend
    std::string out_of_range = std::to_string(n_devices);
    for (const char *value : { "-1", "abc", "1x", out_of_range.c_str() }) {
        set_env("DRJIT_VULKAN_DEVICE", value);
        jit_shutdown(0);
        jit_init(BackendVulkan);
        jit_assert(!jit_has_backend(Backend) && !jit_vulkan_device_handle());
        jit_assert(raises([&] { jit_malloc(Backend, 16, false); },
                          "has not been initialized"));
    }

    // Restore the original configuration
    set_env("DRJIT_VULKAN_DEVICE", env ? env_value.c_str() : nullptr);
    jit_shutdown(0);
    jit_init(BackendVulkan);
    jit_assert(jit_has_backend(Backend));
    jit_assert(jit_vulkan_device_index() == default_index);
}

TEST_VULKAN(18_eval_memory_and_atomics) {
    // 1. Gather and Scatter (unmasked & masked) across Int8, UInt16, Int32, Float32, Float64, Bool
    {
        const uint32_t n = 16;
        uint32_t idx_host[n], sidx_host[n];
        uint8_t mask_host[n], bool_src[n];
        int8_t i8_src[n];
        uint16_t u16_src[n];
        int32_t i32_src[n];
        float f32_src[n];
        double f64_src[n];

        for (uint32_t i = 0; i < n; ++i) {
            idx_host[i]  = (i * 5 + 3) % n;
            sidx_host[i] = (i * 7 + 1) % n; // bijection on 0..15
            mask_host[i] = (uint8_t) ((i % 3) != 0);
            bool_src[i]  = (uint8_t) (i & 1);
            i8_src[i]    = (int8_t) (i * 3 - 20);
            u16_src[i]   = (uint16_t) (i * 100 + 7);
            i32_src[i]   = (int32_t) (i * 1000 - 5000);
            f32_src[i]   = (float) i * 1.5f - 4.0f;
            f64_src[i]   = (double) i * 2.25 - 10.0;
        }

        uint32_t idx  = jit_var_mem_copy(Backend, VarType::UInt32, idx_host, n, 0);
        uint32_t sidx = jit_var_mem_copy(Backend, VarType::UInt32, sidx_host, n, 0);
        uint32_t mask = jit_var_mem_copy(Backend, VarType::Bool, mask_host, n, 0);

        auto check_gather_scatter = [&](VarType vt, const void *src_ptr, size_t esize) {
            uint32_t src = jit_var_mem_copy(Backend, vt, src_ptr, n, 0);
            uint32_t g   = jit_var_gather(src, idx, mask);

            std::vector<uint8_t> zeros_buf(n * esize, 0);
            uint32_t dst = jit_var_mem_copy(Backend, vt, zeros_buf.data(), n, 0);
            uint32_t out_dst = jit_var_scatter(dst, src, sidx, mask,
                                               ReduceOp::Identity, ReduceMode::Auto);
            jit_var_dec_ref(dst);

            jit_var_schedule(g);
            jit_var_schedule(out_dst);
            jit_eval();

            const uint8_t *src_bytes = (const uint8_t *) src_ptr;
            for (uint32_t i = 0; i < n; ++i) {
                uint8_t got_g[8] { }, exp_g[8] { };
                jit_var_read(g, i, got_g);
                if (mask_host[i])
                    memcpy(exp_g, src_bytes + idx_host[i] * esize, esize);
                jit_assert(memcmp(got_g, exp_g, esize) == 0);
            }

            std::vector<uint8_t> exp_dst(n * esize, 0);
            for (uint32_t i = 0; i < n; ++i) {
                if (mask_host[i])
                    memcpy(exp_dst.data() + sidx_host[i] * esize,
                           src_bytes + i * esize, esize);
            }
            for (uint32_t i = 0; i < n; ++i) {
                uint8_t got_d[8] { };
                jit_var_read(out_dst, i, got_d);
                jit_assert(memcmp(got_d, exp_dst.data() + i * esize, esize) == 0);
            }

            jit_var_dec_ref(out_dst);
            jit_var_dec_ref(g);
            jit_var_dec_ref(src);
        };

        check_gather_scatter(VarType::Bool,    bool_src, 1);
        check_gather_scatter(VarType::Int8,    i8_src,   1);
        check_gather_scatter(VarType::UInt16,  u16_src,  2);
        check_gather_scatter(VarType::Int32,   i32_src,  4);
        check_gather_scatter(VarType::Float32, f32_src,  4);
        check_gather_scatter(VarType::Float64, f64_src,  8);

        jit_var_dec_ref(mask);
        jit_var_dec_ref(sidx);
        jit_var_dec_ref(idx);
    }

    // 2. PacketGather and PacketScatter (packet width 4, masked)
    {
        int prev_pkt = jit_flag(JitFlag::PacketOps);
        jit_set_flag(JitFlag::PacketOps, 1);

        const uint32_t n = 8, pkt = 4;
        float src_host[n * pkt];
        uint32_t idx_host[n];
        uint8_t mask_host[n];
        for (uint32_t i = 0; i < n * pkt; ++i)
            src_host[i] = (float) (i + 1) * 0.5f;
        for (uint32_t i = 0; i < n; ++i) {
            idx_host[i]  = (i * 3 + 1) % n; // bijection on 0..7
            mask_host[i] = (uint8_t) ((i & 1) == 0);
        }

        uint32_t src  = jit_var_mem_copy(Backend, VarType::Float32, src_host, n * pkt, 0);
        uint32_t idx  = jit_var_mem_copy(Backend, VarType::UInt32, idx_host, n, 0);
        uint32_t mask = jit_var_mem_copy(Backend, VarType::Bool, mask_host, n, 0);

        uint32_t pg_outs[pkt] { };
        jit_var_gather_packet(pkt, src, idx, mask, pg_outs);

        std::vector<float> zero_dst(n * pkt, 0.f);
        uint32_t dst = jit_var_mem_copy(Backend, VarType::Float32, zero_dst.data(), n * pkt, 0);
        uint32_t out_dst = jit_var_scatter_packet(pkt, dst, pg_outs, idx, mask,
                                                  ReduceOp::Identity, ReduceMode::Auto);
        jit_var_dec_ref(dst);

        for (uint32_t c = 0; c < pkt; ++c)
            jit_var_schedule(pg_outs[c]);
        jit_var_schedule(out_dst);
        jit_eval();

        for (uint32_t i = 0; i < n; ++i) {
            for (uint32_t c = 0; c < pkt; ++c) {
                float got = 0.f;
                jit_var_read(pg_outs[c], i, &got);
                float exp = mask_host[i] ? src_host[idx_host[i] * pkt + c] : 0.f;
                jit_assert(got == exp);
            }
        }

        for (uint32_t i = 0; i < n; ++i) {
            for (uint32_t c = 0; c < pkt; ++c) {
                float got = 0.f;
                jit_var_read(out_dst, idx_host[i] * pkt + c, &got);
                float exp = mask_host[i] ? src_host[idx_host[i] * pkt + c] : 0.f;
                jit_assert(got == exp);
            }
        }

        jit_var_dec_ref(out_dst);
        for (uint32_t c = 0; c < pkt; ++c)
            jit_var_dec_ref(pg_outs[c]);
        jit_var_dec_ref(mask);
        jit_var_dec_ref(idx);
        jit_var_dec_ref(src);
        jit_set_flag(JitFlag::PacketOps, prev_pkt);
    }

    // 3. ScatterReduce (Add, Min, Max, And, Or; Direct & Local modes; Int32, Int64, Float32, Float64)
    {
        const uint32_t n = 128, bins = 4;
        uint32_t idx_colliding[n], idx_uniform[n];
        uint8_t mask_host[n];
        int32_t i32_vals[n];
        int64_t i64_vals[n];
        float f32_vals[n];
        double f64_vals[n];

        for (uint32_t i = 0; i < n; ++i) {
            idx_colliding[i] = i % bins;
            idx_uniform[i]   = 0;
            mask_host[i]     = (uint8_t) ((i % 5) != 0);
            i32_vals[i]      = (int32_t) (i * 7 - 400);
            i64_vals[i]      = (int64_t) (i * 13 - 800);
            f32_vals[i]      = (float) ((int) i - 64) * 1.25f;
            f64_vals[i]      = (double) ((int) i - 64) * 0.75;
        }

        uint32_t idx_c = jit_var_mem_copy(Backend, VarType::UInt32, idx_colliding, n, 0);
        uint32_t idx_u = jit_var_mem_copy(Backend, VarType::UInt32, idx_uniform, n, 0);
        uint32_t mask  = jit_var_mem_copy(Backend, VarType::Bool, mask_host, n, 0);

        for (ReduceMode mode : { ReduceMode::Direct, ReduceMode::Local }) {
            for (int uniform_idx = 0; uniform_idx < 2; ++uniform_idx) {
                uint32_t cur_idx = uniform_idx ? idx_u : idx_c;
                const uint32_t *host_idx = uniform_idx ? idx_uniform : idx_colliding;

                // Int32 Add, Min, Max, And, Or
                for (ReduceOp op : { ReduceOp::Add, ReduceOp::Min, ReduceOp::Max,
                                     ReduceOp::And, ReduceOp::Or }) {
                    int32_t init_val = (op == ReduceOp::Min) ? 1000000
                                     : (op == ReduceOp::Max) ? -1000000
                                     : (op == ReduceOp::And) ? -1 : 0;
                    std::vector<int32_t> init_bins(bins, init_val);
                    std::vector<int32_t> exp_bins = init_bins;
                    for (uint32_t i = 0; i < n; ++i) {
                        if (!mask_host[i])
                            continue;
                        uint32_t b = host_idx[i];
                        if (op == ReduceOp::Add) exp_bins[b] += i32_vals[i];
                        if (op == ReduceOp::Min) exp_bins[b] = std::min(exp_bins[b], i32_vals[i]);
                        if (op == ReduceOp::Max) exp_bins[b] = std::max(exp_bins[b], i32_vals[i]);
                        if (op == ReduceOp::And) exp_bins[b] &= i32_vals[i];
                        if (op == ReduceOp::Or)  exp_bins[b] |= i32_vals[i];
                    }

                    uint32_t dst = jit_var_mem_copy(Backend, VarType::Int32, init_bins.data(), bins, 0);
                    uint32_t val = jit_var_mem_copy(Backend, VarType::Int32, i32_vals, n, 0);
                    uint32_t out = jit_var_scatter(dst, val, cur_idx, mask, op, mode);
                    jit_var_dec_ref(dst);
                    jit_var_dec_ref(val);
                    jit_var_eval(out);

                    for (uint32_t b = 0; b < bins; ++b) {
                        int32_t got = 0;
                        jit_var_read(out, b, &got);
                        jit_assert(got == exp_bins[b]);
                    }
                    jit_var_dec_ref(out);

                    if (jit_can_scatter_reduce(Backend, VarType::Int64, op)) {
                        int64_t init_val64 = (op == ReduceOp::Min) ? 1000000LL
                                           : (op == ReduceOp::Max) ? -1000000LL
                                           : (op == ReduceOp::And) ? -1LL : 0LL;
                        std::vector<int64_t> init_bins64(bins, init_val64);
                        std::vector<int64_t> exp_bins64 = init_bins64;
                        for (uint32_t i = 0; i < n; ++i) {
                            if (!mask_host[i])
                                continue;
                            uint32_t b = host_idx[i];
                            if (op == ReduceOp::Add) exp_bins64[b] += i64_vals[i];
                            if (op == ReduceOp::Min) exp_bins64[b] = std::min(exp_bins64[b], i64_vals[i]);
                            if (op == ReduceOp::Max) exp_bins64[b] = std::max(exp_bins64[b], i64_vals[i]);
                            if (op == ReduceOp::And) exp_bins64[b] &= i64_vals[i];
                            if (op == ReduceOp::Or)  exp_bins64[b] |= i64_vals[i];
                        }

                        uint32_t dst64 = jit_var_mem_copy(Backend, VarType::Int64, init_bins64.data(), bins, 0);
                        uint32_t val64 = jit_var_mem_copy(Backend, VarType::Int64, i64_vals, n, 0);
                        uint32_t out64 = jit_var_scatter(dst64, val64, cur_idx, mask, op, mode);
                        jit_var_dec_ref(dst64);
                        jit_var_dec_ref(val64);
                        jit_var_eval(out64);

                        for (uint32_t b = 0; b < bins; ++b) {
                            int64_t got = 0;
                            jit_var_read(out64, b, &got);
                            jit_assert(got == exp_bins64[b]);
                        }
                        jit_var_dec_ref(out64);
                    }
                }

                // Float32 & Float64 Add, Min, Max (tests negative float atomics & CAS fallback)
                for (ReduceOp op : { ReduceOp::Add, ReduceOp::Min, ReduceOp::Max }) {
                    float init_f32 = (op == ReduceOp::Min) ? 1e6f
                                   : (op == ReduceOp::Max) ? -1e6f : 0.f;
                    std::vector<float> init_bins_f32(bins, init_f32), exp_bins_f32 = init_bins_f32;
                    for (uint32_t i = 0; i < n; ++i) {
                        if (!mask_host[i])
                            continue;
                        uint32_t b = host_idx[i];
                        if (op == ReduceOp::Add) exp_bins_f32[b] += f32_vals[i];
                        if (op == ReduceOp::Min) exp_bins_f32[b] = std::min(exp_bins_f32[b], f32_vals[i]);
                        if (op == ReduceOp::Max) exp_bins_f32[b] = std::max(exp_bins_f32[b], f32_vals[i]);
                    }

                    uint32_t dst32 = jit_var_mem_copy(Backend, VarType::Float32, init_bins_f32.data(), bins, 0);
                    uint32_t val32 = jit_var_mem_copy(Backend, VarType::Float32, f32_vals, n, 0);
                    uint32_t out32 = jit_var_scatter(dst32, val32, cur_idx, mask, op, mode);
                    jit_var_dec_ref(dst32);
                    jit_var_dec_ref(val32);
                    jit_var_eval(out32);

                    for (uint32_t b = 0; b < bins; ++b) {
                        float got = 0.f;
                        jit_var_read(out32, b, &got);
                        jit_assert(got == exp_bins_f32[b]);
                    }
                    jit_var_dec_ref(out32);

                    if (jit_can_scatter_reduce(Backend, VarType::Float64, op)) {
                        double init_f64 = (op == ReduceOp::Min) ? 1e6
                                        : (op == ReduceOp::Max) ? -1e6 : 0.0;
                        std::vector<double> init_bins_f64(bins, init_f64), exp_bins_f64 = init_bins_f64;
                        for (uint32_t i = 0; i < n; ++i) {
                            if (!mask_host[i])
                                continue;
                            uint32_t b = host_idx[i];
                            if (op == ReduceOp::Add) exp_bins_f64[b] += f64_vals[i];
                            if (op == ReduceOp::Min) exp_bins_f64[b] = std::min(exp_bins_f64[b], f64_vals[i]);
                            if (op == ReduceOp::Max) exp_bins_f64[b] = std::max(exp_bins_f64[b], f64_vals[i]);
                        }

                        uint32_t dst64 = jit_var_mem_copy(Backend, VarType::Float64, init_bins_f64.data(), bins, 0);
                        uint32_t val64 = jit_var_mem_copy(Backend, VarType::Float64, f64_vals, n, 0);
                        uint32_t out64 = jit_var_scatter(dst64, val64, cur_idx, mask, op, mode);
                        jit_var_dec_ref(dst64);
                        jit_var_dec_ref(val64);
                        jit_var_eval(out64);

                        for (uint32_t b = 0; b < bins; ++b) {
                            double got = 0.0;
                            jit_var_read(out64, b, &got);
                            jit_assert(got == exp_bins_f64[b]);
                        }
                        jit_var_dec_ref(out64);
                    }
                }
            }
        }

        jit_var_dec_ref(mask);
        jit_var_dec_ref(idx_u);
        jit_var_dec_ref(idx_c);
    }

    // 4. ScatterInc (masked compaction of 1024 lanes)
    {
        constexpr uint32_t n = 1024;
        UInt32 counter(0);
        UInt32 index = arange<UInt32>(n);
        UInt32 out   = zeros<UInt32>(n);
        Mask active  = eq(index & UInt32(1), UInt32(0));
        UInt32 offset = scatter_inc(counter, UInt32(0), active);
        scatter(out, index, offset, active);
        out.eval();

        jit_assert(counter.read(0) == n / 2);
        std::vector<uint32_t> out_cpu(n);
        jit_memcpy(Backend, out_cpu.data(), out.data(), n * sizeof(uint32_t));
        std::sort(out_cpu.begin(), out_cpu.begin() + n / 2);
        for (uint32_t i = 0; i < n / 2; ++i)
            jit_assert(out_cpu[i] == i * 2);
        for (uint32_t i = n / 2; i < n; ++i)
            jit_assert(out_cpu[i] == 0);
    }

    // 5. ScatterExch and ScatterCAS (Int32 and Int64)
    {
        const uint32_t n = 16;
        int32_t dst_init[n], exch_in[n], cas_cmp[n], cas_new[n];
        uint32_t idx_host[n];
        uint8_t mask_host[n];
        for (uint32_t i = 0; i < n; ++i) {
            dst_init[i]  = (int32_t) (i + 100);
            exch_in[i]   = (int32_t) (i + 500);
            cas_cmp[i]   = (i % 2 == 0) ? (int32_t) (i + 500) : -999;
            cas_new[i]   = (int32_t) (i + 900);
            idx_host[i]  = i;
            mask_host[i] = (uint8_t) ((i % 4) != 0);
        }

        uint32_t dst  = jit_var_mem_copy(Backend, VarType::Int32, dst_init, n, 0);
        uint32_t vin  = jit_var_mem_copy(Backend, VarType::Int32, exch_in, n, 0);
        uint32_t idx  = jit_var_mem_copy(Backend, VarType::UInt32, idx_host, n, 0);
        uint32_t mask = jit_var_mem_copy(Backend, VarType::Bool, mask_host, n, 0);

        uint32_t old_exch = jit_var_scatter_exch(&dst, vin, idx, mask);
        jit_var_schedule(old_exch);
        jit_var_schedule(dst);
        jit_eval();

        for (uint32_t i = 0; i < n; ++i) {
            int32_t got_old = 0, got_dst = 0;
            jit_var_read(old_exch, i, &got_old);
            jit_var_read(dst, i, &got_dst);
            jit_assert(got_old == (mask_host[i] ? dst_init[i] : 0));
            jit_assert(got_dst == (mask_host[i] ? exch_in[i] : dst_init[i]));
        }

        uint32_t vcmp = jit_var_mem_copy(Backend, VarType::Int32, cas_cmp, n, 0);
        uint32_t vnew = jit_var_mem_copy(Backend, VarType::Int32, cas_new, n, 0);
        uint32_t cas_old = 0, cas_ok = 0;
        jit_var_scatter_cas(&dst, vcmp, vnew, idx, mask, &cas_old, &cas_ok);
        jit_var_schedule(cas_old);
        jit_var_schedule(cas_ok);
        jit_var_schedule(dst);
        jit_eval();

        for (uint32_t i = 0; i < n; ++i) {
            int32_t prev_val = mask_host[i] ? exch_in[i] : dst_init[i];
            bool should_swap = mask_host[i] && (prev_val == cas_cmp[i]);
            int32_t got_old = 0, got_dst = 0;
            uint8_t got_ok = 0;
            jit_var_read(cas_old, i, &got_old);
            jit_var_read(cas_ok, i, &got_ok);
            jit_var_read(dst, i, &got_dst);
            jit_assert(got_old == (mask_host[i] ? prev_val : 0));
            jit_assert((bool) got_ok == should_swap);
            jit_assert(got_dst == (should_swap ? cas_new[i] : prev_val));
        }

        for (uint32_t r : { cas_old, cas_ok, vcmp, vnew, old_exch, mask, idx, vin, dst })
            jit_var_dec_ref(r);
    }

    // 6. Debug BoundsCheck: out-of-bounds Gather and Scatter safely mask off and warn
    {
        int prev_dbg = jit_flag(JitFlag::Debug);
        LogLevel prev_cb = jit_log_level_callback();
        jit_set_flag(JitFlag::Debug, 1);
        scoped_set_log_level suppress_stderr(LogLevel::Error);

        static std::string s_bounds_log;
        jit_set_log_level_callback(LogLevel::Warn, [](LogLevel, const char *msg) {
            if (msg) {
                s_bounds_log += msg;
                s_bounds_log += '\n';
            }
        });

        uint32_t src_host[4] = { 10, 20, 30, 40 };
        uint32_t idx_host[4] = { 0, 2, 99, 3 }; // lane 2 is out of bounds
        int32_t sidx_host[4] = { 1, -5, 2, 0 }; // lane 1 is negative out of bounds
        uint32_t src  = jit_var_mem_copy(Backend, VarType::UInt32, src_host, 4, 0);
        uint32_t idx  = jit_var_mem_copy(Backend, VarType::UInt32, idx_host, 4, 0);
        uint32_t sidx = jit_var_mem_copy(Backend, VarType::Int32, sidx_host, 4, 0);
        uint8_t m_one = 1;
        uint32_t mask = jit_var_literal(Backend, VarType::Bool, &m_one, 4, 0);

        s_bounds_log.clear();
        uint32_t g = jit_var_gather(src, idx, mask);
        jit_var_eval(g);
        jit_assert(s_bounds_log.find("out-of-bounds") != std::string::npos);

        uint32_t expected_g[4] = { 10, 30, 0, 40 };
        for (uint32_t i = 0; i < 4; ++i) {
            uint32_t got = 999;
            jit_var_read(g, i, &got);
            jit_assert(got == expected_g[i]);
        }

        s_bounds_log.clear();
        uint32_t dst_init[4] = { 1, 2, 3, 4 };
        uint32_t dst = jit_var_mem_copy(Backend, VarType::UInt32, dst_init, 4, 0);
        uint32_t out_dst = jit_var_scatter(dst, src, sidx, mask,
                                           ReduceOp::Identity, ReduceMode::Auto);
        jit_var_dec_ref(dst);
        jit_var_eval(out_dst);
        jit_assert(s_bounds_log.find("out-of-bounds") != std::string::npos);

        // Lanes 0 -> dst[1]=10, lane 1 (-5) -> masked off, lane 2 -> dst[2]=30, lane 3 -> dst[0]=40
        uint32_t expected_d[4] = { 40, 10, 30, 4 };
        for (uint32_t i = 0; i < 4; ++i) {
            uint32_t got = 999;
            jit_var_read(out_dst, i, &got);
            jit_assert(got == expected_d[i]);
        }

        for (uint32_t r : { out_dst, g, mask, sidx, idx, src })
            jit_var_dec_ref(r);
        jit_set_log_level_callback(prev_cb, nullptr);
        jit_set_flag(JitFlag::Debug, prev_dbg);
    }
}

TEST_VULKAN(19_kernel_cache_and_history) {
    int prev_kh = jit_flag(JitFlag::KernelHistory);
    jit_set_flag(JitFlag::KernelHistory, 1);
    jit_flush_kernel_cache();

    // Use a unique literal constant derived from high-resolution time so the
    // first launch is guaranteed to be a cold compile (not on disk yet).
    uint32_t unique_c = (uint32_t) std::chrono::steady_clock::now()
                            .time_since_epoch().count() | 1u;

    auto run_kernel = [&](uint32_t base_offset, KernelHistory **out_hist) {
        uint32_t host_in[16];
        for (uint32_t i = 0; i < 16; ++i)
            host_in[i] = base_offset + i;
        uint32_t x = jit_var_mem_copy(Backend, VarType::UInt32, host_in, 16, 0);
        uint32_t c = jit_var_literal(Backend, VarType::UInt32, &unique_c, 16, 0);
        uint32_t y = jit_var_add(x, c);
        uint64_t start = jit_kernel_history_begin();
        jit_var_eval(y);
        *out_hist = jit_kernel_history_end(start);

        for (uint32_t i = 0; i < 16; ++i) {
            uint32_t got = 0;
            jit_var_read(y, i, &got);
            jit_assert(got == base_offset + i + unique_c);
        }
        jit_var_dec_ref(y);
        jit_var_dec_ref(c);
        jit_var_dec_ref(x);
    };

    // 1. Cold compile: neither in-memory nor on-disk cache hit
    KernelHistory *h1 = nullptr;
    run_kernel(100, &h1);
    jit_assert(jit_kernel_history_size(h1) == 1);
    jit_assert(jit_kernel_history_query(h1, 0, KernelHistoryField::Backend) ==
               (uint64_t) JitBackend::Vulkan);
    jit_assert(jit_kernel_history_query(h1, 0, KernelHistoryField::Size) == 16);
    jit_assert(jit_kernel_history_query(h1, 0, KernelHistoryField::CacheHit) == 0);
    jit_assert(jit_kernel_history_query(h1, 0, KernelHistoryField::CacheDisk) == 0);
    jit_assert(jit_kernel_history_query(h1, 0, KernelHistoryField::OperationCount) > 0);
    std::string spv_src = jit_kernel_history_source(h1, 0);
    jit_assert(spv_src.find("OpCapability Shader") != std::string::npos);
    jit_kernel_history_free(h1);

    // 2. Second launch of identical kernel graph: in-memory cache hit
    KernelHistory *h2 = nullptr;
    run_kernel(200, &h2);
    jit_assert(jit_kernel_history_size(h2) == 1);
    jit_assert(jit_kernel_history_query(h2, 0, KernelHistoryField::CacheHit) == 1);
    jit_assert(jit_kernel_history_query(h2, 0, KernelHistoryField::CacheDisk) == 0);
    jit_kernel_history_free(h2);

    // 3. Flush in-memory kernel & unit caches, then relaunch: disk cache hit
    jit_flush_kernel_cache();
    KernelHistory *h3 = nullptr;
    run_kernel(300, &h3);
    jit_assert(jit_kernel_history_size(h3) == 1);
    jit_assert(jit_kernel_history_query(h3, 0, KernelHistoryField::CacheHit) == 1);
    jit_assert(jit_kernel_history_query(h3, 0, KernelHistoryField::CacheDisk) == 1);
    jit_kernel_history_free(h3);

    jit_set_flag(JitFlag::KernelHistory, prev_kh);
}
