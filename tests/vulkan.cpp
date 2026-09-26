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
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
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

    // Operations that require kernels
    Alloc a(Kind::Device, 1024), b(Kind::Device, 1024);
    jit_assert(raises([&] { jit_compress(Backend, a.ptr, 16, (uint32_t *) b.ptr); },
                      "not supported by the Vulkan backend"));
    jit_assert(raises([&] { jit_block_reduce(Backend, VarType::UInt32,
                                             ReduceOp::Add, 16, 16, a.ptr,
                                             b.ptr); },
                      "not supported by the Vulkan backend"));

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

// ====================================================================
//                   SPIR-V 1.6 Assembly & Validation
// ====================================================================

struct SpvBinary {
    uint32_t *code;
    size_t wordCount;
};

struct SpvPosition {
    size_t line, column, index;
};

struct SpvDiagnostic {
    SpvPosition position;
    char *error;
    bool isTextSource;
};

using PFN_SpvContextCreate     = void *(*)(int);
using PFN_SpvContextDestroy    = void (*)(void *);
using PFN_SpvTextToBinary      = int (*)(const void *, const char *, size_t,
                                         SpvBinary **, SpvDiagnostic **);
using PFN_SpvBinaryDestroy     = void (*)(SpvBinary *);
using PFN_SpvValidateBinary    = int (*)(const void *, const uint32_t *, size_t,
                                         SpvDiagnostic **);
using PFN_SpvDiagnosticDestroy = void (*)(SpvDiagnostic *);

static std::string g_captured_spirv;

static void capture_spirv_cb(LogLevel, const char *msg) {
    if (msg && strstr(msg, "OpCapability Shader"))
        g_captured_spirv = msg;
}

/// Schedule 'var_idx', run jit_eval() with PrintIR enabled to capture the
/// assembled SPIR-V module, and validate it using SPIRV-Tools (Vulkan 1.3 /
/// SPIR-V 1.6).
static std::string assemble_and_validate_spirv(uint32_t var_idx = 0) {
    g_captured_spirv.clear();

    LogLevel prev_stderr   = jit_log_level_stderr(),
             prev_callback = jit_log_level_callback();
    int prev_print_ir      = jit_flag(JitFlag::PrintIR);

    jit_set_log_level_stderr(LogLevel::Error);
    jit_set_log_level_callback(LogLevel::Warn, capture_spirv_cb);
    jit_set_flag(JitFlag::PrintIR, 1);

    if (var_idx)
        jit_var_schedule(var_idx);
    jit_eval();
    jit_sync_thread();

    jit_set_flag(JitFlag::PrintIR, prev_print_ir);
    jit_set_log_level_callback(prev_callback, nullptr);
    jit_set_log_level_stderr(prev_stderr);

    jit_assert(!g_captured_spirv.empty());

    // Check module structure and patched 32-hex entry point name
    jit_assert(g_captured_spirv.find("OpCapability Shader\n") != std::string::npos);
    jit_assert(g_captured_spirv.find("OpCapability Int64\n") != std::string::npos);
    jit_assert(g_captured_spirv.find("OpCapability PhysicalStorageBufferAddresses\n") != std::string::npos);
    jit_assert(g_captured_spirv.find("OpMemoryModel PhysicalStorageBuffer64 GLSL450\n") != std::string::npos);
    jit_assert(g_captured_spirv.find('^') == std::string::npos);

    size_t ep_pos = g_captured_spirv.find("OpEntryPoint GLCompute %main \"drjit_");
    jit_assert(ep_pos != std::string::npos);
    const char *hash_str = g_captured_spirv.c_str() + ep_pos + 36;
    for (int i = 0; i < 32; ++i) {
        char c = hash_str[i];
        jit_assert((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'));
    }
    jit_assert(hash_str[32] == '"');

    // Assemble and validate with SPIRV-Tools targeting SPV_ENV_VULKAN_1_3 (25)
    auto ContextCreate = (PFN_SpvContextCreate) jit_vulkan_lookup("spvContextCreate");
    auto ContextDestroy = (PFN_SpvContextDestroy) jit_vulkan_lookup("spvContextDestroy");
    auto TextToBinary = (PFN_SpvTextToBinary) jit_vulkan_lookup("spvTextToBinary");
    auto BinaryDestroy = (PFN_SpvBinaryDestroy) jit_vulkan_lookup("spvBinaryDestroy");
    auto ValidateBinary = (PFN_SpvValidateBinary) jit_vulkan_lookup("spvValidateBinary");
    auto DiagnosticDestroy = (PFN_SpvDiagnosticDestroy) jit_vulkan_lookup("spvDiagnosticDestroy");

    void *ctx = ContextCreate(25 /* SPV_ENV_VULKAN_1_3 */);
    jit_assert(ctx != nullptr);

    SpvBinary *binary = nullptr;
    SpvDiagnostic *diag = nullptr;
    int rv = TextToBinary(ctx, g_captured_spirv.c_str(),
                          g_captured_spirv.size(), &binary, &diag);
    if (rv != 0 || !binary) {
        fprintf(stderr, "spvTextToBinary failed (rv=%d): %s\n--- SPIR-V ---\n%s\n",
                rv, diag && diag->error ? diag->error : "?",
                g_captured_spirv.c_str());
        if (diag)
            DiagnosticDestroy(diag);
        ContextDestroy(ctx);
        jit_assert(false);
    }

    jit_assert(binary->wordCount > 5);
    jit_assert(binary->code[0] == 0x07230203u); // SPIR-V magic
    jit_assert(binary->code[1] == 0x00010600u); // SPIR-V 1.6

    rv = ValidateBinary(ctx, binary->code, binary->wordCount, &diag);
    if (rv != 0) {
        fprintf(stderr, "spvValidateBinary failed (rv=%d): %s\n--- SPIR-V ---\n%s\n",
                rv, diag && diag->error ? diag->error : "?",
                g_captured_spirv.c_str());
        if (diag)
            DiagnosticDestroy(diag);
        BinaryDestroy(binary);
        ContextDestroy(ctx);
        jit_assert(false);
    }

    BinaryDestroy(binary);
    ContextDestroy(ctx);
    return g_captured_spirv;
}

TEST_VULKAN(18_spirv_assemble) {
    const uint32_t n = 16;

    // 1. Counter kernel (UInt32 output only, no optional capabilities)
    {
        uint32_t c = jit_var_counter(Backend, n);
        std::string spv = assemble_and_validate_spirv(c);
        jit_assert(spv.find("OpCapability Int8\n") == std::string::npos);
        jit_assert(spv.find("OpCapability StorageBuffer8BitAccess\n") == std::string::npos);
        jit_assert(spv.find("OpCapability Int16\n") == std::string::npos);
        jit_assert(spv.find("OpCapability StorageBuffer16BitAccess\n") == std::string::npos);
        jit_assert(spv.find("OpCapability Float16\n") == std::string::npos);
        jit_assert(spv.find("OpCapability Float64\n") == std::string::npos);
        jit_assert(spv.find("OpCopyObject %u32 %idx\n") != std::string::npos);
        jit_assert(spv.find("%params = OpVariable %ptr_pc_Params PushConstant\n") != std::string::npos);
        jit_var_dec_ref(c);
    }

    // 2. Test all 12 scalar types: scalar input (size == 1), vector input
    //    (size == n), literal, undefined, bitcast/resize, select, and output
    int prev_cp = jit_flag(JitFlag::ConstantPropagation);
    jit_set_flag(JitFlag::ConstantPropagation, 0);

    const VarType types[] = {
        VarType::Bool,    VarType::Int8,    VarType::UInt8,   VarType::Int16,
        VarType::UInt16,  VarType::Int32,   VarType::UInt32,  VarType::Int64,
        VarType::UInt64,  VarType::Float16, VarType::Float32, VarType::Float64
    };

    uint8_t mask_data[n];
    for (uint32_t i = 0; i < n; ++i)
        mask_data[i] = (uint8_t) (i & 1);
    uint32_t mask_vec = jit_var_mem_copy(Backend, VarType::Bool,
                                         mask_data, n, 0);
    uint8_t mask_scalar_val = 1;
    uint32_t mask_scalar = jit_var_mem_copy(Backend, VarType::Bool,
                                            &mask_scalar_val, 1, 0);

    for (VarType vt : types) {
        size_t esize = 1;
        switch (vt) {
            case VarType::Bool:
            case VarType::Int8:
            case VarType::UInt8:   esize = 1; break;
            case VarType::Int16:
            case VarType::UInt16:
            case VarType::Float16: esize = 2; break;
            case VarType::Int32:
            case VarType::UInt32:
            case VarType::Float32: esize = 4; break;
            case VarType::Int64:
            case VarType::UInt64:
            case VarType::Float64: esize = 8; break;
            default: break;
        }

        std::vector<uint8_t> vec_bytes(n * esize, 1),
                             scalar_bytes(esize, 1);
        uint32_t in_vec = jit_var_mem_copy(Backend, vt,
                                           vec_bytes.data(), n, 0);
        uint32_t in_sca = jit_var_mem_copy(Backend, vt,
                                           scalar_bytes.data(), 1, 0);

        uint64_t lit_bits = 1;
        if (vt == VarType::Float16)
            lit_bits = 0x3c00u; // 1.0h
        else if (vt == VarType::Float32)
            lit_bits = 0x3f800000u; // 1.0f
        else if (vt == VarType::Float64)
            lit_bits = 0x3ff0000000000000ull; // 1.0
        uint32_t lit = jit_var_literal(Backend, vt, &lit_bits, n, 0);
        uint32_t undef = jit_var_undefined(Backend, vt, n);
        uint32_t resized = jit_var_resize(in_sca, n);

        uint32_t sel1 = jit_var_select(mask_vec, in_vec, lit);
        uint32_t sel2 = jit_var_select(mask_scalar, undef, resized);
        uint32_t out  = jit_var_select(mask_vec, sel1, sel2);

        std::string spv = assemble_and_validate_spirv(out);

        if (vt == VarType::Bool) {
            jit_assert(spv.find("OpINotEqual %bool") != std::string::npos);
            jit_assert(spv.find("OpSelect %u8") != std::string::npos);
            jit_assert(spv.find("OpConstantTrue %bool") != std::string::npos);
            jit_assert(spv.find("OpConstantNull %bool") != std::string::npos);
        } else if (vt == VarType::Int8 || vt == VarType::UInt8) {
            jit_assert(spv.find("OpCapability Int8\n") != std::string::npos);
            jit_assert(spv.find("OpCapability StorageBuffer8BitAccess\n") != std::string::npos);
        } else if (vt == VarType::Int16 || vt == VarType::UInt16) {
            jit_assert(spv.find("OpCapability Int16\n") != std::string::npos);
            jit_assert(spv.find("OpCapability StorageBuffer16BitAccess\n") != std::string::npos);
        } else if (vt == VarType::Float16) {
            jit_assert(spv.find("OpCapability Float16\n") != std::string::npos);
            jit_assert(spv.find("OpCapability StorageBuffer16BitAccess\n") != std::string::npos);
            jit_assert(spv.find("!0x3c00\n") != std::string::npos);
        } else if (vt == VarType::Float32) {
            jit_assert(spv.find("!0x3f800000\n") != std::string::npos);
        } else if (vt == VarType::Float64) {
            jit_assert(spv.find("OpCapability Float64\n") != std::string::npos);
            jit_assert(spv.find("!0x0 !0x3ff00000\n") != std::string::npos);
        }

        jit_var_dec_ref(out);
        jit_var_dec_ref(sel2);
        jit_var_dec_ref(sel1);
        jit_var_dec_ref(resized);
        jit_var_dec_ref(undef);
        jit_var_dec_ref(lit);
        jit_var_dec_ref(in_sca);
        jit_var_dec_ref(in_vec);
    }

    // 3. Reinterpret-cast (OpBitcast) across same-width types & special float literals
    {
        uint32_t f32_nan_bits = 0x7fc00001u,
                 f32_inf_bits = 0xff800000u; // -Inf
        uint64_t f64_neg0_bits = 0x8000000000000000ull;

        uint32_t f32_nan  = jit_var_literal(Backend, VarType::Float32, &f32_nan_bits, n, 0),
                 f32_inf  = jit_var_literal(Backend, VarType::Float32, &f32_inf_bits, n, 0),
                 f64_neg0 = jit_var_literal(Backend, VarType::Float64, &f64_neg0_bits, n, 0);

        uint32_t u32_nan  = jit_var_cast(f32_nan, VarType::UInt32, 1),
                 i32_inf  = jit_var_cast(f32_inf, VarType::Int32, 1),
                 u32_inf  = jit_var_cast(i32_inf, VarType::UInt32, 1),
                 u64_neg0 = jit_var_cast(f64_neg0, VarType::UInt64, 1);

        uint32_t sel_u32 = jit_var_select(mask_vec, u32_nan, u32_inf);
        std::string spv1 = assemble_and_validate_spirv(sel_u32);
        jit_assert(spv1.find("OpBitcast %u32") != std::string::npos);
        jit_assert(spv1.find("OpBitcast %i32") != std::string::npos);

        std::string spv2 = assemble_and_validate_spirv(u64_neg0);
        jit_assert(spv2.find("OpBitcast %u64") != std::string::npos);

        jit_var_dec_ref(sel_u32);
        jit_var_dec_ref(u64_neg0);
        jit_var_dec_ref(u32_inf);
        jit_var_dec_ref(i32_inf);
        jit_var_dec_ref(u32_nan);
        jit_var_dec_ref(f64_neg0);
        jit_var_dec_ref(f32_inf);
        jit_var_dec_ref(f32_nan);
    }

    // 4. Pointer literals (VarType::Pointer) loaded from %params
    {
        Alloc buf1(Kind::Device, 64), buf2(Kind::Device, 64);
        uint32_t ptr1 = jit_var_pointer(Backend, buf1.ptr, mask_vec, 0),
                 ptr2 = jit_var_pointer(Backend, buf2.ptr, mask_vec, 0),
                 ptr_sel = jit_var_select(mask_vec, ptr1, ptr2);
        std::string spv = assemble_and_validate_spirv(ptr_sel);
        jit_assert(spv.find("OpSelect %u64") != std::string::npos);
        jit_var_dec_ref(ptr_sel);
        jit_var_dec_ref(ptr2);
        jit_var_dec_ref(ptr1);
    }

    // 5. PhysicalStorageBuffer parameter fallback when n_params * 8 > maxPushConstantsSize
    {
        const uint32_t n_inputs = 40;
        std::vector<uint32_t> inputs(n_inputs);
        uint32_t val = 42;
        for (uint32_t i = 0; i < n_inputs; ++i)
            inputs[i] = jit_var_mem_copy(Backend,
                                         VarType::UInt32, &val, 1, 0);

        uint32_t acc = jit_var_resize(inputs[0], n);
        for (uint32_t i = 1; i < n_inputs; ++i) {
            uint32_t next = jit_var_select(mask_vec, acc, inputs[i]);
            jit_var_dec_ref(acc);
            acc = next;
        }

        std::string spv = assemble_and_validate_spirv(acc);
        jit_assert(spv.find("%pc = OpVariable %ptr_pc_PushConstants PushConstant\n") !=
                   std::string::npos);
        jit_assert(spv.find("%params = OpLoad %ptr_psb_Params %params_ptr_ptr\n") !=
                   std::string::npos);

        jit_var_dec_ref(acc);
        for (uint32_t in : inputs)
            jit_var_dec_ref(in);
    }

    jit_set_flag(JitFlag::ConstantPropagation, prev_cp);
    jit_var_dec_ref(mask_scalar);
    jit_var_dec_ref(mask_vec);
}

TEST_VULKAN(19_spirv_ops) {
    const uint32_t n = 16;
    int prev_cp = jit_flag(JitFlag::ConstantPropagation);
    int prev_fm = jit_flag(JitFlag::FastMath);
    jit_set_flag(JitFlag::ConstantPropagation, 0);

    uint8_t m0_data[n], m1_data[n];
    for (uint32_t i = 0; i < n; ++i) {
        m0_data[i] = (uint8_t) (i & 1);
        m1_data[i] = (uint8_t) ((i >> 1) & 1);
    }
    uint32_t m0 = jit_var_mem_copy(Backend, VarType::Bool, m0_data, n, 0),
             m1 = jit_var_mem_copy(Backend, VarType::Bool, m1_data, n, 0);

    auto make_input = [&](VarType vt, uint8_t seed) -> uint32_t {
        size_t esize = 1;
        switch (vt) {
            case VarType::Bool:
            case VarType::Int8:
            case VarType::UInt8:   esize = 1; break;
            case VarType::Int16:
            case VarType::UInt16:
            case VarType::Float16: esize = 2; break;
            case VarType::Int32:
            case VarType::UInt32:
            case VarType::Float32: esize = 4; break;
            case VarType::Int64:
            case VarType::UInt64:
            case VarType::Float64: esize = 8; break;
            default: break;
        }
        std::vector<uint8_t> bytes(n * esize, seed);
        return jit_var_mem_copy(Backend, vt, bytes.data(), n, 0);
    };

    // 1. Boolean logical and comparison operations
    {
        uint32_t r_not = jit_var_not(m0),
                 r_and = jit_var_and(m0, m1),
                 r_or  = jit_var_or(m0, m1),
                 r_xor = jit_var_xor(m0, m1),
                 r_eq  = jit_var_eq(m0, m1),
                 r_neq = jit_var_neq(m0, m1),
                 r_lt  = jit_var_lt(m0, m1),
                 r_le  = jit_var_le(m0, m1),
                 r_gt  = jit_var_gt(m0, m1),
                 r_ge  = jit_var_ge(m0, m1);

        uint32_t acc = jit_var_xor(r_not, r_and);
        for (uint32_t v : { r_or, r_xor, r_eq, r_neq, r_lt, r_le, r_gt, r_ge }) {
            uint32_t next = jit_var_xor(acc, v);
            jit_var_dec_ref(acc);
            acc = next;
        }

        std::string spv = assemble_and_validate_spirv(acc);
        jit_assert(spv.find("OpLogicalNot %bool") != std::string::npos);
        jit_assert(spv.find("OpLogicalAnd %bool") != std::string::npos);
        jit_assert(spv.find("OpLogicalOr %bool") != std::string::npos);
        jit_assert(spv.find("OpLogicalEqual %bool") != std::string::npos);
        jit_assert(spv.find("OpLogicalNotEqual %bool") != std::string::npos);

        jit_var_dec_ref(acc);
        for (uint32_t v : { r_not, r_and, r_or, r_xor, r_eq, r_neq, r_lt, r_le, r_gt, r_ge })
            jit_var_dec_ref(v);
    }

    // 2. Integer arithmetic, bitwise, bit-manipulation, and comparisons across all 8 integer types
    const VarType int_types[] = {
        VarType::Int8,  VarType::UInt8,  VarType::Int16, VarType::UInt16,
        VarType::Int32, VarType::UInt32, VarType::Int64, VarType::UInt64
    };

    for (VarType vt : int_types) {
        uint32_t a = make_input(vt, 1),
                 b = make_input(vt, 2),
                 c = make_input(vt, 3);

        uint32_t r_neg  = jit_var_neg(a),
                 r_abs  = jit_var_abs(a),
                 r_not  = jit_var_not(a),
                 r_add  = jit_var_add(a, b),
                 r_sub  = jit_var_sub(a, b),
                 r_mul  = jit_var_mul(a, b),
                 r_div  = jit_var_div(a, b),
                 r_mod  = jit_var_mod(a, b),
                 r_mhi  = jit_var_mul_hi(a, b),
                 r_fma  = jit_var_fma(a, b, c),
                 r_min  = jit_var_min(a, b),
                 r_max  = jit_var_max(a, b),
                 r_and  = jit_var_and(a, b),
                 r_or   = jit_var_or(a, b),
                 r_xor  = jit_var_xor(a, b),
                 r_andm = jit_var_and(a, m0),
                 r_orm  = jit_var_or(a, m0),
                 r_shl  = jit_var_shl(a, b),
                 r_shr  = jit_var_shr(a, b),
                 r_popc = jit_var_popc(a),
                 r_brev = jit_var_brev(a),
                 r_clz  = jit_var_clz(a),
                 r_ctz  = jit_var_ctz(a);

        uint32_t r_eq  = jit_var_eq(a, b),
                 r_neq = jit_var_neq(a, b),
                 r_lt  = jit_var_lt(a, b),
                 r_le  = jit_var_le(a, b),
                 r_gt  = jit_var_gt(a, b),
                 r_ge  = jit_var_ge(a, b);

        uint32_t cmp_m = jit_var_xor(r_eq, r_neq);
        for (uint32_t m : { r_lt, r_le, r_gt, r_ge }) {
            uint32_t next = jit_var_xor(cmp_m, m);
            jit_var_dec_ref(cmp_m);
            cmp_m = next;
        }

        uint32_t acc = jit_var_select(cmp_m, r_neg, r_abs);
        for (uint32_t v : { r_not, r_add, r_sub, r_mul, r_div, r_mod, r_mhi,
                            r_fma, r_min, r_max, r_and, r_or, r_xor, r_andm,
                            r_orm, r_shl, r_shr, r_popc, r_brev, r_clz, r_ctz }) {
            uint32_t next = jit_var_add(acc, v);
            jit_var_dec_ref(acc);
            acc = next;
        }

        std::string spv = assemble_and_validate_spirv(acc);
        jit_assert(spv.find("OpSNegate ") != std::string::npos);
        jit_assert(spv.find("OpIAdd ") != std::string::npos);
        jit_assert(spv.find("OpISub ") != std::string::npos);
        jit_assert(spv.find("OpIMul ") != std::string::npos);
        jit_assert(spv.find("OpBitCount ") != std::string::npos);
        jit_assert(spv.find("OpBitReverse ") != std::string::npos);
        jit_assert(spv.find("FindUMsb ") != std::string::npos);
        jit_assert(spv.find("FindILsb ") != std::string::npos);

        if (vt == VarType::Int32 || vt == VarType::UInt32) {
            uint32_t wide = jit_var_mul_wide(a, b);
            std::string spv_w = assemble_and_validate_spirv(wide);
            jit_assert(spv_w.find(vt == VarType::UInt32 ? "OpUConvert %u64"
                                                        : "OpSConvert %i64") !=
                       std::string::npos);
            jit_var_dec_ref(wide);
        }

        jit_var_dec_ref(acc);
        jit_var_dec_ref(cmp_m);
        for (uint32_t v : { r_eq, r_neq, r_lt, r_le, r_gt, r_ge, r_neg, r_abs,
                            r_not, r_add, r_sub, r_mul, r_div, r_mod, r_mhi,
                            r_fma, r_min, r_max, r_and, r_or, r_xor, r_andm,
                            r_orm, r_shl, r_shr, r_popc, r_brev, r_clz, r_ctz,
                            c, b, a })
            jit_var_dec_ref(v);
    }

    // 3. Floating-point arithmetic, math, bitwise, and comparisons across Float16, Float32, Float64
    const VarType float_types[] = {
        VarType::Float16, VarType::Float32, VarType::Float64
    };

    for (int fm = 0; fm <= 1; ++fm) {
        jit_set_flag(JitFlag::FastMath, fm);
        for (VarType vt : float_types) {
            uint32_t a = make_input(vt, 0x3c),
                     b = make_input(vt, 0x40),
                     c = make_input(vt, 0x42);

            uint32_t r_neg   = jit_var_neg(a),
                     r_abs   = jit_var_abs(a),
                     r_not   = jit_var_not(a),
                     r_sqrt  = jit_var_sqrt(a),
                     r_rcp   = jit_var_rcp(a),
                     r_rsqrt = jit_var_rsqrt(a),
                     r_ceil  = jit_var_ceil(a),
                     r_floor = jit_var_floor(a),
                     r_round = jit_var_round(a),
                     r_trunc = jit_var_trunc(a),
                     r_sin   = jit_var_sin_intrinsic(a),
                     r_cos   = jit_var_cos_intrinsic(a),
                     r_exp2  = jit_var_exp2_intrinsic(a),
                     r_log2  = jit_var_log2_intrinsic(a),
                     r_tanh  = jit_var_tanh_intrinsic(a),
                     r_add   = jit_var_add(a, b),
                     r_sub   = jit_var_sub(a, b),
                     r_mul   = jit_var_mul(a, b),
                     r_div   = jit_var_div(a, b),
                     r_fma   = jit_var_fma(a, b, c),
                     r_min   = jit_var_min(a, b),
                     r_max   = jit_var_max(a, b),
                     r_fmin  = jit_var_fmin(a, b),
                     r_fmax  = jit_var_fmax(a, b),
                     r_csgn  = jit_var_copysign(a, b),
                     r_and   = jit_var_and(a, b),
                     r_or    = jit_var_or(a, b),
                     r_xor   = jit_var_xor(a, b),
                     r_andm  = jit_var_and(a, m0),
                     r_orm   = jit_var_or(a, m0);

            uint32_t r_eq  = jit_var_eq(a, b),
                     r_neq = jit_var_neq(a, b),
                     r_lt  = jit_var_lt(a, b),
                     r_le  = jit_var_le(a, b),
                     r_gt  = jit_var_gt(a, b),
                     r_ge  = jit_var_ge(a, b);

            uint32_t cmp_m = jit_var_xor(r_eq, r_neq);
            for (uint32_t m : { r_lt, r_le, r_gt, r_ge }) {
                uint32_t next = jit_var_xor(cmp_m, m);
                jit_var_dec_ref(cmp_m);
                cmp_m = next;
            }

            uint32_t acc = jit_var_select(cmp_m, r_neg, r_abs);
            for (uint32_t v : { r_not, r_sqrt, r_rcp, r_rsqrt, r_ceil, r_floor,
                                r_round, r_trunc, r_sin, r_cos, r_exp2, r_log2,
                                r_tanh, r_add, r_sub, r_mul, r_div, r_fma,
                                r_min, r_max, r_fmin, r_fmax, r_csgn, r_and,
                                r_or, r_xor, r_andm, r_orm }) {
                uint32_t next = jit_var_add(acc, v);
                jit_var_dec_ref(acc);
                acc = next;
            }

            std::string spv = assemble_and_validate_spirv(acc);
            jit_assert(spv.find("OpFNegate ") != std::string::npos);
            jit_assert(spv.find("OpFAdd ") != std::string::npos);
            jit_assert(spv.find("OpFSub ") != std::string::npos);
            jit_assert(spv.find("OpFMul ") != std::string::npos);
            jit_assert(spv.find("OpFDiv ") != std::string::npos);
            jit_assert(spv.find("OpIsNan %bool") != std::string::npos);
            jit_assert(spv.find("RoundEven ") != std::string::npos);
            if (vt == VarType::Float32 && fm == 1)
                jit_assert(spv.find("InverseSqrt ") != std::string::npos);

            jit_var_dec_ref(acc);
            jit_var_dec_ref(cmp_m);
            for (uint32_t v : { r_eq, r_neq, r_lt, r_le, r_gt, r_ge, r_neg,
                                r_abs, r_not, r_sqrt, r_rcp, r_rsqrt, r_ceil,
                                r_floor, r_round, r_trunc, r_sin, r_cos, r_exp2,
                                r_log2, r_tanh, r_add, r_sub, r_mul, r_div,
                                r_fma, r_min, r_max, r_fmin, r_fmax, r_csgn,
                                r_and, r_or, r_xor, r_andm, r_orm, c, b, a })
                jit_var_dec_ref(v);
        }
    }

    // 4. Fused float-to-int rounding (ceil2int, floor2int, round2int, trunc2int)
    jit_set_flag(JitFlag::ConstantPropagation, 1);
    for (VarType fvt : { VarType::Float32, VarType::Float64 }) {
        uint32_t fa = make_input(fvt, 0x40);
        for (VarType ivt : int_types) {
            uint32_t c_rnd = jit_var_ceil(fa),
                     f_rnd = jit_var_floor(fa),
                     r_rnd = jit_var_round(fa),
                     t_rnd = jit_var_trunc(fa);
            uint32_t ci = jit_var_cast(c_rnd, ivt, 0),
                     fi = jit_var_cast(f_rnd, ivt, 0),
                     ri = jit_var_cast(r_rnd, ivt, 0),
                     ti = jit_var_cast(t_rnd, ivt, 0);
            uint32_t s0 = jit_var_add(ci, fi),
                     s1 = jit_var_add(ri, ti),
                     tot = jit_var_add(s0, s1);

            std::string spv = assemble_and_validate_spirv(tot);
            jit_assert(spv.find("Ceil ") != std::string::npos);
            jit_assert(spv.find("Floor ") != std::string::npos);
            jit_assert(spv.find("RoundEven ") != std::string::npos);

            for (uint32_t v : { tot, s1, s0, ti, ri, fi, ci, t_rnd, r_rnd, f_rnd, c_rnd })
                jit_var_dec_ref(v);
        }
        jit_var_dec_ref(fa);
    }
    jit_set_flag(JitFlag::ConstantPropagation, 0);

    // 5. All pairwise value casts across all 12 scalar types
    const VarType all_types[] = {
        VarType::Bool,    VarType::Int8,    VarType::UInt8,   VarType::Int16,
        VarType::UInt16,  VarType::Int32,   VarType::UInt32,  VarType::Int64,
        VarType::UInt64,  VarType::Float16, VarType::Float32, VarType::Float64
    };

    std::vector<uint32_t> src_vars;
    for (VarType svt : all_types)
        src_vars.push_back(make_input(svt, 1));

    for (VarType dvt : all_types) {
        uint32_t acc = 0;
        for (size_t i = 0; i < 12; ++i) {
            uint32_t casted = jit_var_cast(src_vars[i], dvt, 0);
            if (acc == 0) {
                acc = casted;
            } else {
                uint32_t next = (dvt == VarType::Bool)
                                    ? jit_var_xor(acc, casted)
                                    : jit_var_add(acc, casted);
                jit_var_dec_ref(acc);
                jit_var_dec_ref(casted);
                acc = next;
            }
        }
        assemble_and_validate_spirv(acc);
        jit_var_dec_ref(acc);
    }

    for (uint32_t sv : src_vars)
        jit_var_dec_ref(sv);

    jit_set_flag(JitFlag::FastMath, prev_fm);
    jit_set_flag(JitFlag::ConstantPropagation, prev_cp);
    jit_var_dec_ref(m1);
    jit_var_dec_ref(m0);
}

TEST_VULKAN(20_spirv_memory_ops) {
    const uint32_t n = 16;
    int prev_cp  = jit_flag(JitFlag::ConstantPropagation);
    int prev_pkt = jit_flag(JitFlag::PacketOps);
    int prev_dbg = jit_flag(JitFlag::Debug);
    jit_set_flag(JitFlag::ConstantPropagation, 0);
    jit_set_flag(JitFlag::PacketOps, 1);

    auto make_buf = [&](VarType vt, uint32_t count, uint8_t seed) -> uint32_t {
        size_t esize = 1;
        switch (vt) {
            case VarType::Bool:
            case VarType::Int8:
            case VarType::UInt8:   esize = 1; break;
            case VarType::Int16:
            case VarType::UInt16:
            case VarType::Float16: esize = 2; break;
            case VarType::Int32:
            case VarType::UInt32:
            case VarType::Float32: esize = 4; break;
            case VarType::Int64:
            case VarType::UInt64:
            case VarType::Float64: esize = 8; break;
            default: break;
        }
        std::vector<uint8_t> bytes(count * esize, seed);
        return jit_var_mem_copy(Backend, vt, bytes.data(), count, 0);
    };

    uint8_t mask_data[n];
    uint32_t idx_data[n], idx2_data[n];
    int32_t sidx_data[n];
    for (uint32_t i = 0; i < n; ++i) {
        mask_data[i] = (uint8_t) (i & 1);
        idx_data[i]  = (n - 1 - i);
        idx2_data[i] = (i * 3) % n;
        sidx_data[i] = (int32_t) ((n - 1 - i));
    }
    uint32_t mask_vec = jit_var_mem_copy(Backend, VarType::Bool, mask_data, n, 0);
    uint8_t mask_true_val = 1;
    uint32_t mask_true = jit_var_literal(Backend, VarType::Bool, &mask_true_val, n, 0);
    uint32_t idx_vec  = jit_var_mem_copy(Backend, VarType::UInt32, idx_data, n, 0),
             idx2_vec = jit_var_mem_copy(Backend, VarType::UInt32, idx2_data, n, 0),
             sidx_vec = jit_var_mem_copy(Backend, VarType::Int32, sidx_data, n, 0);

    const VarType all_types[] = {
        VarType::Bool,    VarType::Int8,    VarType::UInt8,   VarType::Int16,
        VarType::UInt16,  VarType::Int32,   VarType::UInt32,  VarType::Int64,
        VarType::UInt64,  VarType::Float16, VarType::Float32, VarType::Float64
    };

    // 1. Gather across all 12 scalar types (chained unmasked + masked in same kernel)
    for (VarType vt : all_types) {
        uint32_t src = make_buf(vt, n * 2, 1);
        uint32_t g1  = jit_var_gather(src, idx_vec, mask_true);
        uint32_t g2  = jit_var_gather(src, idx2_vec, mask_vec);
        uint32_t out = (vt == VarType::Bool) ? jit_var_xor(g1, g2)
                                             : jit_var_add(g1, g2);

        std::string spv = assemble_and_validate_spirv(out);
        jit_assert(spv.find("OpConvertUToPtr ") != std::string::npos);
        jit_assert(spv.find("OpPtrAccessChain ") != std::string::npos);
        jit_assert(spv.find("OpLoad ") != std::string::npos);
        jit_assert(spv.find("OpPhi ") != std::string::npos);

        jit_var_dec_ref(out);
        jit_var_dec_ref(g2);
        jit_var_dec_ref(g1);
        jit_var_dec_ref(src);
    }

    // 2. PacketGather + Extract for packet sizes 2, 4, 8 across all 12 scalar types
    for (size_t pkt_n : { (size_t) 2, (size_t) 4, (size_t) 8 }) {
        for (VarType vt : all_types) {
            uint32_t src = make_buf(vt, (uint32_t) (n * pkt_n), 2);
            uint32_t outs[8] = { };
            jit_var_gather_packet(pkt_n, src, idx_vec, mask_vec, outs);

            jit_var_inc_ref(outs[0]);
            uint32_t acc = outs[0];
            for (size_t k = 1; k < pkt_n; ++k) {
                uint32_t next = (vt == VarType::Bool) ? jit_var_xor(acc, outs[k])
                                                      : jit_var_add(acc, outs[k]);
                jit_var_dec_ref(acc);
                acc = next;
            }

            std::string spv = assemble_and_validate_spirv(acc);
            jit_assert(spv.find("OpTypeVector ") != std::string::npos);
            jit_assert(spv.find("OpCompositeExtract ") != std::string::npos);
            jit_assert(spv.find("OpPhi ") != std::string::npos);

            jit_var_dec_ref(acc);
            for (size_t k = 0; k < pkt_n; ++k)
                jit_var_dec_ref(outs[k]);
            jit_var_dec_ref(src);
        }
    }

    // 3. Scatter (ReduceOp::Identity) across all 12 scalar types
    for (VarType vt : all_types) {
        uint32_t dst = make_buf(vt, n * 2, 0),
                 val = make_buf(vt, n, 3);
        uint32_t out_dst = jit_var_scatter(dst, val, idx_vec, mask_vec,
                                           ReduceOp::Identity, ReduceMode::Auto);
        jit_var_dec_ref(dst);

        std::string spv = assemble_and_validate_spirv();
        jit_assert(spv.find("OpConvertUToPtr ") != std::string::npos);
        jit_assert(spv.find("OpPtrAccessChain ") != std::string::npos);
        jit_assert(spv.find("OpStore ") != std::string::npos);

        jit_var_dec_ref(val);
        jit_var_dec_ref(out_dst);
    }

    // 4. PacketScatter for packet sizes 2, 4, 8 across all 12 scalar types
    for (size_t pkt_n : { (size_t) 2, (size_t) 4, (size_t) 8 }) {
        for (VarType vt : all_types) {
            uint32_t dst = make_buf(vt, (uint32_t) (n * pkt_n), 0);
            uint32_t vals[8] = { };
            for (size_t k = 0; k < pkt_n; ++k)
                vals[k] = make_buf(vt, n, (uint8_t) (k + 1));

            uint32_t out_dst = jit_var_scatter_packet(
                pkt_n, dst, vals, idx_vec, mask_vec,
                ReduceOp::Identity, ReduceMode::Auto);
            jit_var_dec_ref(dst);

            std::string spv = assemble_and_validate_spirv();
            jit_assert(spv.find("OpCompositeConstruct ") != std::string::npos);
            jit_assert(spv.find("OpShiftRightLogical %u32 ") != std::string::npos);
            jit_assert(spv.find("OpStore ") != std::string::npos);

            for (size_t k = 0; k < pkt_n; ++k)
                jit_var_dec_ref(vals[k]);
            jit_var_dec_ref(out_dst);
        }
    }

    // 5. ScatterReduce capability queries and SPIR-V validation
    for (VarType unsupported_vt : { VarType::Bool, VarType::Int8, VarType::UInt8,
                                    VarType::Int16, VarType::UInt16, VarType::Float16 }) {
        jit_assert(jit_can_scatter_reduce(Backend, unsupported_vt, ReduceOp::Identity) == 1);
        jit_assert(jit_can_scatter_reduce(Backend, unsupported_vt, ReduceOp::Add) == 0);
        jit_assert(jit_can_scatter_reduce(Backend, unsupported_vt, ReduceOp::Min) == 0);
    }
    jit_assert(jit_can_scatter_reduce(Backend, VarType::UInt32, ReduceOp::Mul) == 0);
    jit_assert(jit_can_scatter_reduce(Backend, VarType::Float32, ReduceOp::And) == 0);
    jit_assert(jit_can_scatter_reduce(Backend, VarType::Float32, ReduceOp::Or) == 0);

    const ReduceOp int_red_ops[] = {
        ReduceOp::Add, ReduceOp::Min, ReduceOp::Max, ReduceOp::And, ReduceOp::Or
    };
    const ReduceMode red_modes[] = {
        ReduceMode::Direct, ReduceMode::Local, ReduceMode::NoConflicts
    };

    for (VarType vt : { VarType::Int32, VarType::UInt32, VarType::Int64, VarType::UInt64 }) {
        for (ReduceOp op : int_red_ops) {
            if (!jit_can_scatter_reduce(Backend, vt, op))
                continue;
            for (ReduceMode mode : red_modes) {
                uint32_t dst = make_buf(vt, n * 2, 0),
                         val = make_buf(vt, n, 5);
                uint32_t out_dst = jit_var_scatter(dst, val, idx_vec, mask_vec, op, mode);
                jit_var_dec_ref(dst);

                std::string spv = assemble_and_validate_spirv();
                if (mode == ReduceMode::NoConflicts) {
                    jit_assert(spv.find("OpLoad ") != std::string::npos);
                    jit_assert(spv.find("OpStore ") != std::string::npos);
                } else {
                    jit_assert(spv.find("OpAtomic") != std::string::npos);
                    if (vt == VarType::Int64 || vt == VarType::UInt64)
                        jit_assert(spv.find("OpCapability Int64Atomics\n") != std::string::npos);
                    if (mode == ReduceMode::Local &&
                        (vt == VarType::Int32 || vt == VarType::UInt32)) {
                        jit_assert(spv.find("OpCapability GroupNonUniformBallot\n") != std::string::npos);
                        jit_assert(spv.find("OpCapability GroupNonUniformArithmetic\n") != std::string::npos);
                        jit_assert(spv.find("OpGroupNonUniformBallot ") != std::string::npos);
                        jit_assert(spv.find("OpGroupNonUniformBroadcastFirst ") != std::string::npos);
                        jit_assert(spv.find("OpGroupNonUniformAll ") != std::string::npos);
                    }
                }

                jit_var_dec_ref(val);
                jit_var_dec_ref(out_dst);
            }
        }
    }

    const ReduceOp float_red_ops[] = {
        ReduceOp::Add, ReduceOp::Min, ReduceOp::Max
    };
    for (VarType vt : { VarType::Float32, VarType::Float64 }) {
        for (ReduceOp op : float_red_ops) {
            if (!jit_can_scatter_reduce(Backend, vt, op))
                continue;
            for (ReduceMode mode : red_modes) {
                uint32_t dst = make_buf(vt, n * 2, 0),
                         val = make_buf(vt, n, 0x3f);
                uint32_t out_dst = jit_var_scatter(dst, val, idx_vec, mask_vec, op, mode);
                jit_var_dec_ref(dst);

                std::string spv = assemble_and_validate_spirv();
                if (mode == ReduceMode::NoConflicts) {
                    jit_assert(spv.find("OpLoad ") != std::string::npos);
                    jit_assert(spv.find("OpStore ") != std::string::npos);
                } else if (op == ReduceOp::Add) {
                    bool has_ext = spv.find("OpAtomicFAddEXT ") != std::string::npos;
                    bool has_cas = spv.find("OpAtomicCompareExchange ") != std::string::npos;
                    jit_assert(has_ext || has_cas);
                    if (has_ext) {
                        jit_assert(spv.find("OpExtension \"SPV_EXT_shader_atomic_float_add\"\n") !=
                                   std::string::npos);
                        jit_assert(spv.find(vt == VarType::Float32
                                                ? "OpCapability AtomicFloat32AddEXT\n"
                                                : "OpCapability AtomicFloat64AddEXT\n") !=
                                   std::string::npos);
                    } else {
                        jit_assert(spv.find("OpLoopMerge ") != std::string::npos);
                    }
                } else {
                    jit_assert(spv.find("OpSGreaterThanEqual %bool ") != std::string::npos);
                    jit_assert(spv.find(op == ReduceOp::Min ? "OpAtomicSMin " : "OpAtomicSMax ") !=
                               std::string::npos);
                    jit_assert(spv.find(op == ReduceOp::Min ? "OpAtomicUMax " : "OpAtomicUMin ") !=
                               std::string::npos);
                }

                if (mode == ReduceMode::Local && vt == VarType::Float32) {
                    jit_assert(spv.find("OpGroupNonUniformF") != std::string::npos);
                }

                jit_var_dec_ref(val);
                jit_var_dec_ref(out_dst);
            }
        }
    }

    // 6. ScatterInc, ScatterExch, and ScatterCAS (including chained ops in same kernel)
    {
        uint32_t dst1 = make_buf(VarType::UInt32, n * 2, 0),
                 dst2 = make_buf(VarType::UInt32, n * 2, 0);
        uint32_t inc1 = jit_var_scatter_inc(&dst1, idx_vec, mask_vec);
        uint32_t inc2 = jit_var_scatter_inc(&dst2, idx2_vec, mask_true);
        uint32_t tot  = jit_var_add(inc1, inc2);

        std::string spv = assemble_and_validate_spirv(tot);
        jit_assert(spv.find("OpAtomicIAdd %u32 ") != std::string::npos);
        jit_assert(spv.find("OpPhi %u32 ") != std::string::npos);

        jit_var_dec_ref(tot);
        jit_var_dec_ref(inc2);
        jit_var_dec_ref(inc1);
        jit_var_dec_ref(dst2);
        jit_var_dec_ref(dst1);
    }

    for (VarType vt : { VarType::Int32, VarType::UInt32, VarType::Float32,
                        VarType::Int64, VarType::UInt64, VarType::Float64 }) {
        // ScatterExch
        {
            uint32_t dst = make_buf(vt, n * 2, 1),
                     val = make_buf(vt, n, 2);
            uint32_t old_val = jit_var_scatter_exch(&dst, val, idx_vec, mask_vec);

            std::string spv = assemble_and_validate_spirv(old_val);
            jit_assert(spv.find("OpAtomicExchange ") != std::string::npos);
            jit_assert(spv.find("OpPhi ") != std::string::npos);

            jit_var_dec_ref(old_val);
            jit_var_dec_ref(val);
            jit_var_dec_ref(dst);
        }

        // ScatterCAS
        {
            uint32_t dst = make_buf(vt, n * 2, 1),
                     cmp = make_buf(vt, n, 1),
                     val = make_buf(vt, n, 2),
                     old_val = 0,
                     success = 0;
            jit_var_scatter_cas(&dst, cmp, val, idx_vec, mask_vec, &old_val, &success);
            uint32_t combined = jit_var_select(success, old_val, val);

            std::string spv = assemble_and_validate_spirv(combined);
            jit_assert(spv.find("OpAtomicCompareExchange ") != std::string::npos);
            jit_assert(spv.find("OpIEqual %bool ") != std::string::npos);
            jit_assert(spv.find("OpPhi %bool ") != std::string::npos);

            jit_var_dec_ref(combined);
            jit_var_dec_ref(success);
            jit_var_dec_ref(old_val);
            jit_var_dec_ref(val);
            jit_var_dec_ref(cmp);
            jit_var_dec_ref(dst);
        }
    }

    // 7. BoundsCheck (JitFlag::Debug enabled) across Gather (unsigned + signed index),
    //    PacketGather, Scatter, PacketScatter, ScatterReduce, ScatterInc, ScatterExch, ScatterCAS
    jit_set_flag(JitFlag::Debug, 1);
    {
        uint32_t src = make_buf(VarType::UInt32, n * 4, 1);
        uint32_t g_u = jit_var_gather(src, idx_vec, mask_vec);
        uint32_t g_s = jit_var_gather(src, sidx_vec, mask_vec);
        uint32_t pg_outs[4] = { };
        jit_var_gather_packet(4, src, idx_vec, mask_vec, pg_outs);

        uint32_t sum1 = jit_var_add(g_u, g_s),
                 sum2 = jit_var_add(pg_outs[0], pg_outs[3]),
                 tot  = jit_var_add(sum1, sum2);

        std::string spv = assemble_and_validate_spirv(tot);
        jit_assert(spv.find("OpULessThan %bool ") != std::string::npos);
        jit_assert(spv.find("%bchk_err_") != std::string::npos);
        jit_assert(spv.find("%bchk_done_") != std::string::npos);

        jit_var_dec_ref(tot);
        jit_var_dec_ref(sum2);
        jit_var_dec_ref(sum1);
        for (int k = 0; k < 4; ++k)
            jit_var_dec_ref(pg_outs[k]);
        jit_var_dec_ref(g_s);
        jit_var_dec_ref(g_u);
        jit_var_dec_ref(src);
    }
    {
        uint32_t dst1 = make_buf(VarType::UInt32, n * 4, 0),
                 dst2 = make_buf(VarType::UInt32, n * 4, 0),
                 dst3 = make_buf(VarType::UInt32, n * 4, 0),
                 dst4 = make_buf(VarType::UInt32, n * 4, 0),
                 dst5 = make_buf(VarType::UInt32, n * 4, 0),
                 dst6 = make_buf(VarType::UInt32, n * 4, 0),
                 val  = make_buf(VarType::UInt32, n, 7),
                 vals[2] = { val, val };
        uint32_t out_dst1 = jit_var_scatter(dst1, val, sidx_vec, mask_vec, ReduceOp::Identity, ReduceMode::Auto);
        jit_var_dec_ref(dst1);
        uint32_t out_dst2 = jit_var_scatter_packet(2, dst2, vals, idx_vec, mask_vec, ReduceOp::Identity, ReduceMode::Auto);
        jit_var_dec_ref(dst2);
        uint32_t out_dst3 = jit_var_scatter(dst3, val, idx_vec, mask_vec, ReduceOp::Add, ReduceMode::Direct);
        jit_var_dec_ref(dst3);
        uint32_t inc_v  = jit_var_scatter_inc(&dst4, idx_vec, mask_vec);
        uint32_t exch_v = jit_var_scatter_exch(&dst5, val, idx_vec, mask_vec);
        uint32_t cas_old = 0, cas_ok = 0;
        jit_var_scatter_cas(&dst6, val, val, idx_vec, mask_vec, &cas_old, &cas_ok);
        uint32_t s0  = jit_var_add(inc_v, exch_v),
                 tot = jit_var_select(cas_ok, s0, cas_old);

        std::string spv = assemble_and_validate_spirv(tot);
        jit_assert(spv.find("%bchk_err_") != std::string::npos);
        jit_assert(spv.find("%bchk_done_") != std::string::npos);

        jit_var_dec_ref(tot);
        jit_var_dec_ref(s0);
        jit_var_dec_ref(cas_ok);
        jit_var_dec_ref(cas_old);
        jit_var_dec_ref(exch_v);
        jit_var_dec_ref(inc_v);
        jit_var_dec_ref(val);
        jit_var_dec_ref(dst6);
        jit_var_dec_ref(dst5);
        jit_var_dec_ref(dst4);
        jit_var_dec_ref(out_dst3);
        jit_var_dec_ref(out_dst2);
        jit_var_dec_ref(out_dst1);
    }

    jit_set_flag(JitFlag::Debug, prev_dbg);
    jit_set_flag(JitFlag::PacketOps, prev_pkt);
    jit_set_flag(JitFlag::ConstantPropagation, prev_cp);

    jit_var_dec_ref(sidx_vec);
    jit_var_dec_ref(idx2_vec);
    jit_var_dec_ref(idx_vec);
    jit_var_dec_ref(mask_true);
    jit_var_dec_ref(mask_vec);
}

// ====================================================================
//  End-to-end compute execution & numerical verification (Step 2.4)
// ====================================================================

template <typename T> bool vulkan_test_const_prop() {
    using Value = typename T::Value;
    constexpr JitBackend Backend = T::Backend;
    constexpr bool IsFloat  = drjit::detail::is_floating_point_v<Value>;
    constexpr bool IsSigned = drjit::detail::is_signed_v<Value>;
    constexpr bool IsMask   = std::is_same<Value, bool>::value;
    constexpr bool IsInt    = !IsFloat && !IsMask;
    constexpr size_t Size   = IsMask ? 2 : 10;

    bool fast_math = jit_flags() & (uint32_t) JitFlag::FastMath;
    jit_set_flag(JitFlag::FastMath, false);

    bool fail = false;
    Value values[Size];
    for (uint32_t i = 0; i < Size; ++i) {
        values[i] = (Value) ((IsMask || !IsSigned) ? (int) i : ((int) i - 4));
        if constexpr (IsFloat) {
            if (values[i] < Value(-1) || values[i] > Value(1))
                values[i] = (Value) (1.1f * values[i]);
        }
    }

    uint8_t *shared_read = (uint8_t *) jit_malloc(Backend, 64 * 1024, true);
    size_t read_cursor = 0;

    struct StagedRead {
        uint32_t var_id;
        size_t offset;
        size_t count;
        size_t elem_size;
        bool is_scalar;
        bool is_lit;
    };
    std::vector<StagedRead> staged_reads;

    auto reset_reads = [&]() {
        read_cursor = 0;
        staged_reads.clear();
    };

    auto stage_read = [&](uint32_t var_id, size_t count, size_t elem_size) -> size_t {
        size_t offset = (read_cursor + 7) & ~size_t(7);
        read_cursor = offset + count * elem_size;
        StagedRead sr { var_id, offset, count, elem_size,
                        jit_var_size(var_id) == 1,
                        jit_var_state(var_id) == VarState::Literal };
        staged_reads.push_back(sr);
        return offset;
    };

    auto flush_reads = [&]() {
        jit_eval();
        for (StagedRead &sr : staged_reads) {
            sr.is_scalar = jit_var_size(sr.var_id) == 1;
            sr.is_lit    = jit_var_state(sr.var_id) == VarState::Literal;
            if (!sr.is_lit) {
                void *ptr = nullptr;
                uint32_t ref_id = jit_var_data(sr.var_id, &ptr);
                size_t copy_n = sr.is_scalar ? 1 : sr.count;
                jit_memcpy_async(Backend, shared_read + sr.offset, ptr,
                                 copy_n * sr.elem_size);
                jit_var_dec_ref(ref_id);
            }
        }
        jit_sync_thread();
        for (const StagedRead &sr : staged_reads) {
            if (sr.is_lit) {
                uint8_t scalar[8] { };
                jit_var_read(sr.var_id, 0, scalar);
                for (size_t i = 0; i < sr.count; ++i)
                    memcpy(shared_read + sr.offset + i * sr.elem_size,
                           scalar, sr.elem_size);
            } else if (sr.is_scalar) {
                uint8_t scalar[8] { };
                memcpy(scalar, shared_read + sr.offset, sr.elem_size);
                for (size_t i = 1; i < sr.count; ++i)
                    memcpy(shared_read + sr.offset + i * sr.elem_size,
                           scalar, sr.elem_size);
            }
        }
    };

    uint32_t lit_in[Size] { };
    for (uint32_t i = 0; i < Size; ++i)
        lit_in[i] = jit_var_literal(Backend, T::Type, &values[i], 1, 0);

    // 1. Unary operations
    struct UnaryBatch {
        JitOp op;
        Value ref[Size];
        uint32_t gpu_out;
        size_t read_off;
    };
    std::vector<UnaryBatch> unary_batches;
    reset_reads();

    for (JitOp op :
         { JitOp::Not, JitOp::Neg, JitOp::Abs, JitOp::Sqrt, JitOp::Rcp,
           JitOp::Rsqrt, JitOp::Ceil, JitOp::Floor, JitOp::Round,
           JitOp::Trunc, JitOp::Popc, JitOp::Clz, JitOp::Ctz, JitOp::Brev }) {
        if ((op == JitOp::Popc || op == JitOp::Clz || op == JitOp::Ctz ||
             op == JitOp::Brev) && !IsInt)
            continue;
        else if (op == JitOp::Not && !IsMask && !IsInt)
            continue;
        else if ((op == JitOp::Sqrt || op == JitOp::Rcp ||
                  op == JitOp::Rsqrt || op == JitOp::Ceil ||
                  op == JitOp::Floor || op == JitOp::Round ||
                  op == JitOp::Trunc) && !IsFloat)
            continue;
        else if ((op == JitOp::Neg || op == JitOp::Abs) && IsMask)
            continue;

        UnaryBatch ub { };
        ub.op = op;
        Value safe_vals[Size] { };
        for (uint32_t i = 0; i < Size; ++i) {
            bool skip = (op == JitOp::Rcp && values[i] == Value(0));
            safe_vals[i] = skip ? Value(1) : values[i];
            if (!skip) {
                uint32_t l_id = jit_var_op(op, &lit_in[i]);
                jit_assert(jit_var_state(l_id) == VarState::Literal);
                jit_var_read(l_id, 0, &ub.ref[i]);
                jit_var_dec_ref(l_id);
            }
        }

        uint32_t gpu_in = jit_var_mem_copy(Backend, T::Type, safe_vals, Size, 1);
        ub.gpu_out = jit_var_op(op, &gpu_in);
        jit_var_schedule(ub.gpu_out);
        jit_var_dec_ref(gpu_in);
        ub.read_off = stage_read(ub.gpu_out, Size, sizeof(Value));
        unary_batches.push_back(ub);
    }
    flush_reads();

    for (const UnaryBatch &ub : unary_batches) {
        const Value *got = (const Value *) (shared_read + ub.read_off);
        for (uint32_t i = 0; i < Size; ++i) {
            if (ub.op == JitOp::Rcp && values[i] == Value(0))
                continue;
            Value value = got[i], r = ub.ref[i];
            if (memcmp(&value, &r, sizeof(Value)) != 0) {
                if (std::isnan((double) value) && std::isnan((double) r))
                    continue;
                if ((ub.op == JitOp::Sqrt || ub.op == JitOp::Rcp ||
                     ub.op == JitOp::Rsqrt) &&
                    std::abs(double(value) - double(r)) < 1e-5)
                    continue;
                if (double(value) == 0.0 && double(r) == 0.0)
                    continue;
                fprintf(stderr, "Unary mismatch (type=%s, op=%d, a=%g): got %g vs ref %g\n",
                        typeid(Value).name(), (int) ub.op, double(values[i]),
                        double(value), double(r));
                fail = true;
            }
        }
        jit_var_dec_ref(ub.gpu_out);
    }

    // 2. Binary operations: tests (lit,lit), (eval,lit), (lit,eval), and (eval,eval)
    struct BinBatch {
        JitOp op;
        bool is_cmp;
        Value lhs_safe[Size];
        Value rhs_safe[Size];
        Value ref_val[Size][Size];
        bool  ref_cmp[Size][Size];
        uint32_t out_el[Size];
        uint32_t out_le[Size];
        uint32_t out_ee[Size];
        size_t off_el[Size];
        size_t off_le[Size];
        size_t off_ee[Size];
    };
    std::vector<BinBatch> bin_batches;
    reset_reads();

    for (JitOp op :
         { JitOp::Add, JitOp::Sub, JitOp::Mul, JitOp::Div,
           JitOp::Mod, JitOp::Min, JitOp::Max, JitOp::And, JitOp::Or,
           JitOp::Xor, JitOp::Shl, JitOp::Shr, JitOp::Eq, JitOp::Neq,
           JitOp::Lt, JitOp::Le, JitOp::Gt, JitOp::Ge, JitOp::MulHi }) {
        if (op == JitOp::Mod && IsFloat)
            continue;
        if ((IsFloat || IsMask) && (op == JitOp::Shl || op == JitOp::Shr))
            continue;
        if (IsMask && !(op == JitOp::Or || op == JitOp::And || op == JitOp::Xor))
            continue;
        if ((!IsInt || sizeof(Value) < 4) && op == JitOp::MulHi)
            continue;

        BinBatch b { };
        b.op = op;
        b.is_cmp = (op == JitOp::Eq || op == JitOp::Neq || op == JitOp::Lt ||
                    op == JitOp::Le || op == JitOp::Gt  || op == JitOp::Ge);
        size_t out_esize = b.is_cmp ? sizeof(bool) : sizeof(Value);

        uint32_t lit_lhs_safe[Size] { }, lit_rhs_safe[Size] { }, gpu_rhs_scalar[Size] { };
        for (uint32_t i = 0; i < Size; ++i) {
            bool bad_lhs = IsSigned && op == JitOp::Shl && values[i] < Value(0);
            bool bad_rhs = ((op == JitOp::Div || op == JitOp::Mod) && values[i] == Value(0)) ||
                           ((op == JitOp::Shr || op == JitOp::Shl) &&
                            (values[i] < Value(0) || (size_t) values[i] >= sizeof(Value) * 8));
            b.lhs_safe[i] = bad_lhs ? Value(0) : values[i];
            b.rhs_safe[i] = bad_rhs ? Value(1) : values[i];
            lit_lhs_safe[i]   = jit_var_literal(Backend, T::Type, &b.lhs_safe[i], 1, 0);
            lit_rhs_safe[i]   = jit_var_literal(Backend, T::Type, &b.rhs_safe[i], 1, 0);
            gpu_rhs_scalar[i] = jit_var_literal(Backend, T::Type, &b.rhs_safe[i], 1, 1);
        }

        uint32_t gpu_lhs_vec = jit_var_mem_copy(Backend, T::Type, b.lhs_safe, Size, 1);
        uint32_t gpu_rhs_vec = jit_var_mem_copy(Backend, T::Type, b.rhs_safe, Size, 1);

        for (uint32_t i = 0; i < Size; ++i) {
            for (uint32_t k = 0; k < Size; ++k) {
                uint32_t deps[2] = { lit_lhs_safe[i], lit_rhs_safe[k] };
                uint32_t l_id = jit_var_op(op, deps);
                jit_assert(jit_var_state(l_id) == VarState::Literal);
                if (b.is_cmp)
                    jit_var_read(l_id, 0, &b.ref_cmp[i][k]);
                else
                    jit_var_read(l_id, 0, &b.ref_val[i][k]);
                jit_var_dec_ref(l_id);
            }
        }

        for (uint32_t k = 0; k < Size; ++k) {
            uint32_t deps_el[2] = { gpu_lhs_vec, lit_rhs_safe[k] };
            uint32_t deps_le[2] = { lit_lhs_safe[k], gpu_rhs_vec };
            uint32_t deps_ee[2] = { gpu_lhs_vec, gpu_rhs_scalar[k] };
            b.out_el[k] = jit_var_op(op, deps_el);
            b.out_le[k] = jit_var_op(op, deps_le);
            b.out_ee[k] = jit_var_op(op, deps_ee);
            jit_var_schedule(b.out_el[k]);
            jit_var_schedule(b.out_le[k]);
            jit_var_schedule(b.out_ee[k]);
            b.off_el[k] = stage_read(b.out_el[k], Size, out_esize);
            b.off_le[k] = stage_read(b.out_le[k], Size, out_esize);
            b.off_ee[k] = stage_read(b.out_ee[k], Size, out_esize);
            jit_var_dec_ref(lit_lhs_safe[k]);
            jit_var_dec_ref(lit_rhs_safe[k]);
            jit_var_dec_ref(gpu_rhs_scalar[k]);
        }
        jit_var_dec_ref(gpu_lhs_vec);
        jit_var_dec_ref(gpu_rhs_vec);
        bin_batches.push_back(b);
    }
    flush_reads();

    for (const BinBatch &b : bin_batches) {
        auto check_vec = [&](size_t off, uint32_t fixed_idx, bool fixed_is_rhs, const char *mode) {
            if (b.is_cmp) {
                const bool *got = (const bool *) (shared_read + off);
                for (uint32_t m = 0; m < Size; ++m) {
                    uint32_t i = fixed_is_rhs ? m : fixed_idx;
                    uint32_t k = fixed_is_rhs ? fixed_idx : m;
                    if (got[m] != b.ref_cmp[i][k]) {
                        fprintf(stderr, "Cmp mismatch (%s, type=%s, op=%d, a=%g, b=%g): got %d vs ref %d\n",
                                mode, typeid(Value).name(), (int) b.op,
                                double(b.lhs_safe[i]), double(b.rhs_safe[k]),
                                (int) got[m], (int) b.ref_cmp[i][k]);
                        fail = true;
                    }
                }
            } else {
                const Value *got = (const Value *) (shared_read + off);
                for (uint32_t m = 0; m < Size; ++m) {
                    uint32_t i = fixed_is_rhs ? m : fixed_idx;
                    uint32_t k = fixed_is_rhs ? fixed_idx : m;
                    Value value = got[m], r = b.ref_val[i][k];
                    if (memcmp(&value, &r, sizeof(Value)) != 0) {
                        if (b.op == JitOp::Div &&
                            std::abs(double(value) - double(r)) < 1e-5)
                            continue;
                        if (double(value) == 0.0 && double(r) == 0.0)
                            continue;
                        fprintf(stderr, "Binary mismatch (%s, type=%s, op=%d, a=%g, b=%g): got %g vs ref %g\n",
                                mode, typeid(Value).name(), (int) b.op,
                                double(b.lhs_safe[i]), double(b.rhs_safe[k]),
                                double(value), double(r));
                        fail = true;
                    }
                }
            }
        };

        for (uint32_t k = 0; k < Size; ++k) {
            check_vec(b.off_el[k], k, true,  "eval,lit");
            check_vec(b.off_le[k], k, false, "lit,eval");
            check_vec(b.off_ee[k], k, true,  "eval,eval");
            jit_var_dec_ref(b.out_el[k]);
            jit_var_dec_ref(b.out_le[k]);
            jit_var_dec_ref(b.out_ee[k]);
        }
    }

    // 3. Ternary operations (Fma & Select)
    constexpr uint32_t Small = IsMask ? 2 : 4;
    Value svals[Small];
    uint32_t lit_s[Small] { }, gpu_s[Small] { };
    for (uint32_t i = 0; i < Small; ++i) {
        svals[i] = (Value) ((IsMask || !IsSigned) ? (int) i : ((int) i - 2));
        if constexpr (IsFloat) {
            if (svals[i] < Value(-1) || svals[i] > Value(1))
                svals[i] = (Value) (1.1f * svals[i]);
        }
        lit_s[i] = jit_var_literal(Backend, T::Type, &svals[i], 1, 0);
        gpu_s[i] = jit_var_literal(Backend, T::Type, &svals[i], 1, 1);
    }
    uint32_t gpu_s_vec = jit_var_mem_copy(Backend, T::Type, svals, Small, 1);
    uint32_t in_b[4] { };
    for (uint32_t i = 0; i < 4; ++i) {
        bool b = i & 1;
        in_b[i] = jit_var_literal(Backend, VarType::Bool, &b, 1, i >= 2);
    }

    for (JitOp op : { JitOp::Fma, JitOp::Select }) {
        if (op == JitOp::Fma && IsMask)
            continue;

        uint32_t n_i = (op == JitOp::Select) ? 2 : Small;
        Value ref_t[4][Small][Small] { };
        for (uint32_t i = 0; i < n_i; ++i) {
            for (uint32_t j = 0; j < Small; ++j) {
                for (uint32_t k = 0; k < Small; ++k) {
                    uint32_t deps[3] = { op == JitOp::Select ? in_b[i] : lit_s[i],
                                         lit_s[j], lit_s[k] };
                    uint32_t l_id = jit_var_op(op, deps);
                    jit_assert(jit_var_state(l_id) == VarState::Literal);
                    jit_var_read(l_id, 0, &ref_t[i][j][k]);
                    jit_var_dec_ref(l_id);
                }
            }
        }

        reset_reads();
        uint32_t out_t[4][Small] { };
        size_t off_t[4][Small] { };
        for (uint32_t i = 0; i < n_i; ++i) {
            for (uint32_t j = 0; j < Small; ++j) {
                uint32_t a0 = (op == JitOp::Select) ? in_b[i + 2] : gpu_s[i];
                uint32_t deps[3] = { a0, gpu_s[j], gpu_s_vec };
                out_t[i][j] = jit_var_op(op, deps);
                jit_var_schedule(out_t[i][j]);
                off_t[i][j] = stage_read(out_t[i][j], Small, sizeof(Value));
            }
        }
        flush_reads();

        for (uint32_t i = 0; i < n_i; ++i) {
            for (uint32_t j = 0; j < Small; ++j) {
                const Value *got = (const Value *) (shared_read + off_t[i][j]);
                for (uint32_t k = 0; k < Small; ++k) {
                    Value value = got[k], r = ref_t[i][j][k];
                    if (memcmp(&value, &r, sizeof(Value)) != 0) {
                        if (op == JitOp::Fma &&
                            std::abs(double(value) - double(r)) < 1e-5)
                            continue;
                        fprintf(stderr, "Ternary mismatch (type=%s, op=%d): got %g vs ref %g\n",
                                typeid(Value).name(), (int) op, double(value), double(r));
                        fail = true;
                    }
                }
                jit_var_dec_ref(out_t[i][j]);
            }
        }
    }

    for (uint32_t i = 0; i < 4; ++i)
        jit_var_dec_ref(in_b[i]);
    jit_var_dec_ref(gpu_s_vec);
    for (uint32_t i = 0; i < Small; ++i) {
        jit_var_dec_ref(lit_s[i]);
        jit_var_dec_ref(gpu_s[i]);
    }
    for (uint32_t i = 0; i < Size; ++i)
        jit_var_dec_ref(lit_in[i]);
    jit_free(shared_read);

    jit_set_flag(JitFlag::FastMath, fast_math);
    return fail;
}

TEST_VULKAN(21_eval_arithmetic_and_casts) {
    // 1. Counter, scalar/vector load-store, masks, and >32-param PSB fallback
    {
        UInt32 ctr = arange<UInt32>(64);
        UInt32 y = ctr * UInt32(3) + UInt32(7);
        y.eval();
        for (uint32_t i = 0; i < 64; ++i)
            jit_assert(y.read(i) == i * 3 + 7);

        Mask even = eq(ctr & UInt32(1), UInt32(0));
        Mask odd  = ~even;
        even.eval();
        odd.eval();
        for (uint32_t i = 0; i < 64; ++i) {
            jit_assert(even.read(i) == ((i & 1) == 0));
            jit_assert(odd.read(i)  == ((i & 1) == 1));
        }

        // > 32 input buffers (triggers PhysicalStorageBuffer parameter block)
        const int n_inputs = 40;
        const uint32_t width = 64;
        std::vector<uint32_t> inputs(n_inputs);
        for (int k = 0; k < n_inputs; ++k) {
            std::vector<uint32_t> host(width);
            for (uint32_t i = 0; i < width; ++i)
                host[i] = (uint32_t) k * 100u + i;
            inputs[k] = jit_var_mem_copy(Backend, VarType::UInt32,
                                         host.data(), width, 1);
        }
        uint32_t acc = jit_var_inc_ref(inputs[0]);
        for (int k = 1; k < n_inputs; ++k) {
            uint32_t next = jit_var_add(acc, inputs[k]);
            jit_var_dec_ref(acc);
            acc = next;
        }
        void *acc_ptr = nullptr;
        uint32_t acc_ref = jit_var_data(acc, &acc_ptr);
        std::vector<uint32_t> got_acc(width);
        jit_memcpy(Backend, got_acc.data(), acc_ptr, width * sizeof(uint32_t));
        jit_var_dec_ref(acc_ref);
        for (uint32_t i = 0; i < width; ++i) {
            uint32_t expected = 100u * (uint32_t) (n_inputs * (n_inputs - 1) / 2) +
                                (uint32_t) n_inputs * i;
            jit_assert(got_acc[i] == expected);
        }
        jit_var_dec_ref(acc);
        for (int k = 0; k < n_inputs; ++k)
            jit_var_dec_ref(inputs[k]);
    }

    // 2. Constant propagation vs. GPU evaluation across all types & ops
    bool fail = false;
    fail |= vulkan_test_const_prop<VulkanArray<float>>();
    fail |= vulkan_test_const_prop<VulkanArray<drjit::half>>();
    fail |= vulkan_test_const_prop<VulkanArray<double>>();
    fail |= vulkan_test_const_prop<VulkanArray<int8_t>>();
    fail |= vulkan_test_const_prop<VulkanArray<uint8_t>>();
    fail |= vulkan_test_const_prop<VulkanArray<int16_t>>();
    fail |= vulkan_test_const_prop<VulkanArray<uint16_t>>();
    fail |= vulkan_test_const_prop<VulkanArray<int32_t>>();
    fail |= vulkan_test_const_prop<VulkanArray<uint32_t>>();
    fail |= vulkan_test_const_prop<VulkanArray<int64_t>>();
    fail |= vulkan_test_const_prop<VulkanArray<uint64_t>>();
    fail |= vulkan_test_const_prop<VulkanArray<bool>>();
    jit_assert(!fail);

    // 3. Special floating-point ops: Min/Max NaN propagation, FMin/FMax,
    //    Copysign, MulHi (8/16-bit), MulWide, and fused float-to-int rounding
    {
        int prev_cp = jit_flag(JitFlag::ConstantPropagation);
        jit_set_flag(JitFlag::ConstantPropagation, 0);

        float nan_f = std::numeric_limits<float>::quiet_NaN();
        float val_f = 3.5f, neg_f = -2.25f;
        uint32_t v_nan = jit_var_literal(Backend, VarType::Float32, &nan_f, 4, 1);
        uint32_t v_val = jit_var_literal(Backend, VarType::Float32, &val_f, 4, 1);
        uint32_t v_neg = jit_var_literal(Backend, VarType::Float32, &neg_f, 4, 1);

        // Min/Max propagate quiet NaN
        uint32_t r_min_nan = jit_var_min(v_val, v_nan);
        uint32_t r_max_nan = jit_var_max(v_nan, v_val);
        // FMin/FMax select the non-NaN operand
        uint32_t r_fmin = jit_var_fmin(v_val, v_nan);
        uint32_t r_fmax = jit_var_fmax(v_nan, v_neg);

        // Fused float-to-int rounding (requires ConstantPropagation = 1)
        jit_set_flag(JitFlag::ConstantPropagation, 1);
        uint32_t t_ceil    = jit_var_ceil(v_val),
                 t_floor   = jit_var_floor(v_neg),
                 t_trunc   = jit_var_trunc(v_neg),
                 t_round   = jit_var_round(v_val);
        uint32_t r_ceil_i  = jit_var_cast(t_ceil,  VarType::Int32, 0),
                 r_floor_i = jit_var_cast(t_floor, VarType::Int32, 0),
                 r_trunc_i = jit_var_cast(t_trunc, VarType::Int32, 0),
                 r_round_i = jit_var_cast(t_round, VarType::Int32, 0);
        for (uint32_t t : { t_ceil, t_floor, t_trunc, t_round })
            jit_var_dec_ref(t);
        jit_set_flag(JitFlag::ConstantPropagation, 0);

        for (uint32_t r : { r_min_nan, r_max_nan, r_fmin, r_fmax,
                            r_ceil_i, r_floor_i, r_trunc_i, r_round_i })
            jit_var_schedule(r);
        jit_eval();

        float out_f = 0.f;
        jit_var_read(r_min_nan, 0, &out_f); jit_assert(std::isnan(out_f));
        jit_var_read(r_max_nan, 0, &out_f); jit_assert(std::isnan(out_f));
        jit_var_read(r_fmin, 0, &out_f);    jit_assert(out_f == 3.5f);
        jit_var_read(r_fmax, 0, &out_f);    jit_assert(out_f == -2.25f);

        int32_t out_i = 0;
        jit_var_read(r_ceil_i, 0, &out_i);  jit_assert(out_i == 4);
        jit_var_read(r_floor_i, 0, &out_i); jit_assert(out_i == -3);
        jit_var_read(r_trunc_i, 0, &out_i); jit_assert(out_i == -2);
        jit_var_read(r_round_i, 0, &out_i); jit_assert(out_i == 4);

        for (uint32_t r : { r_min_nan, r_max_nan, r_fmin, r_fmax,
                            r_ceil_i, r_floor_i, r_trunc_i, r_round_i,
                            v_nan, v_val, v_neg })
            jit_var_dec_ref(r);

        // MulHi for 8-bit and 16-bit integers, plus MulWide (32-bit -> 64-bit)
        int8_t  i8a = -100, i8b = 50;
        uint8_t u8a = 200,  u8b = 100;
        int16_t i16a = -20000, i16b = 15000;
        uint16_t u16a = 50000, u16b = 40000;
        int32_t sa = -100000, sb = 300000;
        uint32_t ua = 4000000000u, ub = 3u;

        uint32_t vi8a  = jit_var_literal(Backend, VarType::Int8,   &i8a,  4, 1);
        uint32_t vi8b  = jit_var_literal(Backend, VarType::Int8,   &i8b,  4, 1);
        uint32_t vu8a  = jit_var_literal(Backend, VarType::UInt8,  &u8a,  4, 1);
        uint32_t vu8b  = jit_var_literal(Backend, VarType::UInt8,  &u8b,  4, 1);
        uint32_t vi16a = jit_var_literal(Backend, VarType::Int16,  &i16a, 4, 1);
        uint32_t vi16b = jit_var_literal(Backend, VarType::Int16,  &i16b, 4, 1);
        uint32_t vu16a = jit_var_literal(Backend, VarType::UInt16, &u16a, 4, 1);
        uint32_t vu16b = jit_var_literal(Backend, VarType::UInt16, &u16b, 4, 1);
        uint32_t vsa   = jit_var_literal(Backend, VarType::Int32,  &sa,   4, 1);
        uint32_t vsb   = jit_var_literal(Backend, VarType::Int32,  &sb,   4, 1);
        uint32_t vua   = jit_var_literal(Backend, VarType::UInt32, &ua,   4, 1);
        uint32_t vub   = jit_var_literal(Backend, VarType::UInt32, &ub,   4, 1);

        uint32_t d_i8[2]  = { vi8a, vi8b },   d_u8[2]  = { vu8a, vu8b };
        uint32_t d_i16[2] = { vi16a, vi16b }, d_u16[2] = { vu16a, vu16b };
        uint32_t s_deps[2] = { vsa, vsb },    u_deps[2] = { vua, vub };

        uint32_t r_mh_i8  = jit_var_op(JitOp::MulHi,   d_i8);
        uint32_t r_mh_u8  = jit_var_op(JitOp::MulHi,   d_u8);
        uint32_t r_mh_i16 = jit_var_op(JitOp::MulHi,   d_i16);
        uint32_t r_mh_u16 = jit_var_op(JitOp::MulHi,   d_u16);
        uint32_t r_swide  = jit_var_op(JitOp::MulWide, s_deps);
        uint32_t r_uwide  = jit_var_op(JitOp::MulWide, u_deps);

        for (uint32_t r : { r_mh_i8, r_mh_u8, r_mh_i16, r_mh_u16, r_swide, r_uwide })
            jit_var_schedule(r);
        jit_eval();

        int8_t got_i8 = 0;     uint8_t got_u8 = 0;
        int16_t got_i16 = 0;   uint16_t got_u16 = 0;
        int64_t got_s = 0;     uint64_t got_u = 0;
        jit_var_read(r_mh_i8,  0, &got_i8);
        jit_var_read(r_mh_u8,  0, &got_u8);
        jit_var_read(r_mh_i16, 0, &got_i16);
        jit_var_read(r_mh_u16, 0, &got_u16);
        jit_var_read(r_swide,  0, &got_s);
        jit_var_read(r_uwide,  0, &got_u);

        jit_assert(got_i8  == (int8_t)  (((int16_t) i8a * (int16_t) i8b) >> 8));
        jit_assert(got_u8  == (uint8_t) (((uint16_t) u8a * (uint16_t) u8b) >> 8));
        jit_assert(got_i16 == (int16_t) (((int32_t) i16a * (int32_t) i16b) >> 16));
        jit_assert(got_u16 == (uint16_t) (((uint32_t) u16a * (uint32_t) u16b) >> 16));
        jit_assert(got_s   == (int64_t) sa * (int64_t) sb);
        jit_assert(got_u   == (uint64_t) ua * (uint64_t) ub);

        for (uint32_t r : { r_mh_i8, r_mh_u8, r_mh_i16, r_mh_u16, r_swide, r_uwide,
                            vi8a, vi8b, vu8a, vu8b, vi16a, vi16b, vu16a, vu16b,
                            vsa, vsb, vua, vub })
            jit_var_dec_ref(r);

        jit_set_flag(JitFlag::ConstantPropagation, prev_cp);
    }

    // 4. Pairwise Cast and Bitcast across all 12 scalar types
    {
        VarType types[] = {
            VarType::Float16, VarType::Float32, VarType::Float64,
            VarType::Int8,    VarType::UInt8,   VarType::Int16,
            VarType::UInt16,  VarType::Int32,   VarType::UInt32,
            VarType::Int64,   VarType::UInt64,  VarType::Bool
        };
        size_t type_sizes[(int) VarType::Count] = {
            0, 1, 0, 1, 1, 2, 2, 4, 4, 8, 8, 8, 0, 2, 4, 8
        };

        struct CastJob {
            uint32_t gpu_d;
            size_t bytes;
            size_t read_off;
            uint8_t ref_buf[80];
        };

        uint8_t *cast_shared = (uint8_t *) jit_malloc(Backend, 32 * 1024, true);

        for (VarType source_type : types) {
            std::vector<CastJob> jobs;
            size_t cast_off = 0;
            for (int reinterpret = 0; reinterpret < 2; ++reinterpret) {
                for (VarType target_type : types) {
                    bool source_bool = source_type == VarType::Bool,
                         target_bool = target_type == VarType::Bool;
                    bool test_sign =
                        source_type != VarType::UInt8 &&
                        source_type != VarType::UInt16 &&
                        source_type != VarType::UInt32 &&
                        source_type != VarType::UInt64 &&
                        !source_bool &&
                        target_type != VarType::UInt8 &&
                        target_type != VarType::UInt16 &&
                        target_type != VarType::UInt32 &&
                        target_type != VarType::UInt64 &&
                        !target_bool;

                    if (reinterpret &&
                        (source_bool ? 0 : type_sizes[(int) source_type]) !=
                        (target_bool ? 0 : type_sizes[(int) target_type]))
                        continue;

                    int size = source_bool ? 2 : 10;
                    size_t src_esize = type_sizes[(int) source_type];
                    size_t dst_esize = type_sizes[(int) target_type];
                    uint8_t src_buf[80] { };
                    CastJob job { };
                    job.bytes = size * dst_esize;
                    job.read_off = (cast_off + 7) & ~size_t(7);
                    cast_off = job.read_off + job.bytes;

                    for (int i = 0; i < size; ++i) {
                        int64_t value = i;
                        if (test_sign)
                            value -= 4;
                        if (source_type == VarType::Float32) {
                            float f = (float) value;
                            if (std::abs(f) > 2.f)
                                f += f * .1f;
                            value = 0;
                            memcpy(&value, &f, sizeof(float));
                        } else if (source_type == VarType::Float64) {
                            double d = (double) value;
                            if (std::abs(d) > 2.0)
                                d += d * .1;
                            memcpy(&value, &d, sizeof(double));
                        } else if (source_type == VarType::Float16) {
                            drjit::half h((float) value);
                            if (std::abs((float) h) > 2.f)
                                h = drjit::half((float) h * 1.1f);
                            value = 0;
                            memcpy(&value, &h, sizeof(drjit::half));
                        }

                        memcpy(src_buf + i * src_esize, &value, src_esize);
                        uint32_t lit_s = jit_var_literal(Backend, source_type, &value, 1, 0);
                        uint32_t lit_d = jit_var_cast(lit_s, target_type, reinterpret);
                        jit_var_read(lit_d, 0, job.ref_buf + i * dst_esize);
                        jit_var_dec_ref(lit_d);
                        jit_var_dec_ref(lit_s);
                    }

                    uint32_t gpu_s = jit_var_mem_copy(Backend, source_type, src_buf, size, 1);
                    job.gpu_d = jit_var_cast(gpu_s, target_type, reinterpret);
                    jit_var_schedule(job.gpu_d);
                    jit_var_dec_ref(gpu_s);
                    jobs.push_back(job);
                }
            }
            jit_eval();

            for (const CastJob &job : jobs) {
                void *dst_ptr = nullptr;
                uint32_t dst_ref = jit_var_data(job.gpu_d, &dst_ptr);
                jit_memcpy_async(Backend, cast_shared + job.read_off, dst_ptr, job.bytes);
                jit_var_dec_ref(dst_ref);
            }
            jit_sync_thread();

            for (const CastJob &job : jobs) {
                jit_assert(memcmp(cast_shared + job.read_off, job.ref_buf, job.bytes) == 0);
                jit_var_dec_ref(job.gpu_d);
            }
        }
        jit_free(cast_shared);
    }
}

TEST_VULKAN(22_eval_memory_and_atomics) {
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

TEST_VULKAN(23_kernel_cache_and_history) {
    int prev_kh = jit_flag(JitFlag::KernelHistory);
    jit_set_flag(JitFlag::KernelHistory, 1);
    jit_flush_kernel_cache();

    // Use a unique literal constant derived from high-resolution time so the
    // first launch is guaranteed to be a cold compile (not on disk yet).
    uint32_t unique_c = (uint32_t) std::chrono::steady_clock::now()
                            .time_since_epoch().count() | 1u;

    auto run_kernel = [&](uint32_t base_offset, KernelHistory **out_hist) {
        uint64_t start = jit_kernel_history_begin();
        uint32_t host_in[16];
        for (uint32_t i = 0; i < 16; ++i)
            host_in[i] = base_offset + i;
        uint32_t x = jit_var_mem_copy(Backend, VarType::UInt32, host_in, 16, 0);
        uint32_t c = jit_var_literal(Backend, VarType::UInt32, &unique_c, 16, 0);
        uint32_t y = jit_var_add(x, c);
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

namespace {

/// RAII helper to temporarily record symbolic computation
struct ScopedRecord {
    ScopedRecord(JitBackend backend, const char *name = nullptr,
                 bool new_scope = false)
        : backend(backend) {
        checkpoint = jit_record_begin(backend, name);
        if (new_scope)
            scope = jit_new_scope(backend);
    }

    ~ScopedRecord() {
        jit_record_end(backend, checkpoint, cleanup);
    }

    void disarm() { cleanup = false; }

    JitBackend backend;
    uint32_t checkpoint = 0, scope = 0;
    bool cleanup = true;
};

struct VarRef {
    VarRef() : index(0) { }
    ~VarRef() { jit_var_dec_ref(index); }
    VarRef(VarRef &&r) : index(r.index) { r.index = 0; }
    VarRef &operator=(VarRef &&r) {
        jit_var_dec_ref(index);
        index = r.index;
        r.index = 0;
        return *this;
    }
    VarRef(const VarRef &) = delete;
    VarRef &operator=(const VarRef &) = delete;
    operator uint32_t() const { return index; }

    static VarRef steal(uint32_t idx) {
        VarRef r;
        r.index = idx;
        return r;
    }

    uint32_t index = 0;
};

template <typename Value> struct LocalArr {
    using UInt32 = typename Value::template ReplaceValue<uint32_t>;
    using Bool = typename Value::template ReplaceValue<bool>;
    using Scalar = typename Value::Value;

    LocalArr() : m_index(0) { }
    LocalArr(LocalArr &&a) : m_index(a.m_index) { a.m_index = 0; }
    LocalArr(const LocalArr &a) : m_index(jit_var_inc_ref(a.m_index)) { }

    LocalArr &operator=(const LocalArr &) = delete;
    LocalArr &operator=(LocalArr &&a) {
        jit_var_dec_ref(m_index);
        m_index = a.m_index;
        a.m_index = 0;
        return *this;
    }

    LocalArr(size_t length) {
        m_index = jit_array_create(Value::Backend, Value::Type, 1, length);
    }

    LocalArr(size_t length, Value value) {
        uint32_t tmp = jit_array_create(Value::Backend, Value::Type, 1, length);
        m_index = jit_array_init(tmp, value.index());
        jit_var_dec_ref(tmp);
    }

    ~LocalArr() {
        jit_var_dec_ref(m_index);
    }

    size_t length() const {
        return jit_array_length(m_index);
    }

    void eval() {
        jit_var_eval(m_index);
    }

    void schedule() {
        jit_var_schedule(m_index);
    }

    void write(const UInt32 &offset, const Value &value, const Bool &mask = true) {
        uint32_t new_index =
            jit_array_write(m_index, offset.index(), value.index(), mask.index());
        jit_var_dec_ref(m_index);
        m_index = new_index;
    }

    Value read(const UInt32 &offset, const Bool &mask = true) const {
        return Value::steal(jit_array_read(m_index, offset.index(), mask.index()));
    }

    static LocalArr steal(uint32_t idx) {
        LocalArr a;
        a.m_index = idx;
        return a;
    }

    uint32_t m_index;
};

} // namespace

TEST_VULKAN(24_symbolic_conditionals) {
    // 1. Multi-output symbolic conditional with active and invariant outputs
    {
        UInt32 idx = arange<UInt32>(8);
        Float  x   = Float(idx) * 2.f;
        Mask   c   = idx < 4u;

        Float  out_f;
        UInt32 out_u;
        Float  out_inv;
        {
            ScopedRecord rec(Backend, "CondBasic");
            Mask cond_t = Mask::steal(jit_var_mask_apply(c.index(), 8));
            Mask cond_f = Mask::steal(jit_var_mask_apply((!c).index(), 8));
            VarRef cond = VarRef::steal(
                jit_var_cond_start("CondBasic", false, cond_t.index(), cond_f.index()));

            // True branch: idx < 4
            jit_var_mask_push(Backend, cond_t.index());
            Float  ft   = x + 10.f;
            UInt32 ut   = idx + 100u;
            Float  invt = Float(42.f);
            jit_var_mask_pop(Backend);
            uint32_t rv_t[3] = { ft.index(), ut.index(), invt.index() };
            VarRef lbl_t = VarRef::steal(jit_var_cond_append(cond, rv_t, 3));

            // False branch: idx >= 4
            jit_var_mask_push(Backend, cond_f.index());
            Float  ff   = x * 3.f;
            UInt32 uf   = idx + 200u;
            Float  invf = Float(42.f);
            jit_var_mask_pop(Backend);
            uint32_t rv_f[3] = { ff.index(), uf.index(), invf.index() };
            VarRef lbl_f = VarRef::steal(jit_var_cond_append(cond, rv_f, 3));

            uint32_t rv_out[3] = { 0, 0, 0 };
            jit_var_cond_end(cond, rv_out);
            out_f   = Float::steal(rv_out[0]);
            out_u   = UInt32::steal(rv_out[1]);
            out_inv = Float::steal(rv_out[2]);
            rec.disarm();
        }

        out_f.schedule();
        out_u.schedule();
        out_inv.schedule();
        std::string spv = assemble_and_validate_spirv();
        jit_assert(spv.find("OpSelectionMerge %cond_merge_r") != std::string::npos);

        for (uint32_t i = 0; i < 8; ++i) {
            float exp_f = (i < 4) ? (i * 2.f + 10.f) : (i * 2.f * 3.f);
            uint32_t exp_u = (i < 4) ? (i + 100u) : (i + 200u);
            jit_assert(out_f.read(i) == exp_f);
            jit_assert(out_u.read(i) == exp_u);
            jit_assert(out_inv.read(i) == 42.f);
        }
    }

    // 2. Conditional with side effects (scatter in true/false branches) and
    //    nested conditionals + gathers inside branches (tests current_block_label)
    {
        UInt32 idx = arange<UInt32>(8);
        UInt32 tbl = arange<UInt32>(8) * 10u + 1u;
        tbl.eval();

        UInt32 dst = zeros<UInt32>(8);
        Mask c_outer = eq(idx & UInt32(1), UInt32(0)); // even lanes
        Mask c_inner = idx < 4u;                       // first 4 lanes

        UInt32 res;
        {
            ScopedRecord rec(Backend, "CondNested");
            Mask outer_t = Mask::steal(jit_var_mask_apply(c_outer.index(), 8));
            Mask outer_f = Mask::steal(jit_var_mask_apply((!c_outer).index(), 8));
            VarRef cond_outer = VarRef::steal(
                jit_var_cond_start("Outer", false, outer_t.index(), outer_f.index()));

            // Outer true branch (even lanes): nested conditional + gather + scatter
            jit_var_mask_push(Backend, outer_t.index());
            UInt32 rt;
            {
                Mask inner_t = Mask::steal(jit_var_mask_apply(c_inner.index(), 8));
                Mask inner_f = Mask::steal(jit_var_mask_apply((!c_inner).index(), 8));
                VarRef cond_inner = VarRef::steal(
                    jit_var_cond_start("Inner", true, inner_t.index(), inner_f.index()));

                jit_var_mask_push(Backend, inner_t.index());
                UInt32 g_t = gather<UInt32>(tbl, idx);
                scatter(dst, g_t + 1000u, idx);
                jit_var_mask_pop(Backend);
                uint32_t rv_it[1] = { g_t.index() };
                VarRef lbl_it = VarRef::steal(jit_var_cond_append(cond_inner, rv_it, 1));

                jit_var_mask_push(Backend, inner_f.index());
                UInt32 g_f = gather<UInt32>(tbl, UInt32(7) - idx);
                scatter(dst, g_f + 2000u, idx);
                jit_var_mask_pop(Backend);
                uint32_t rv_if[1] = { g_f.index() };
                VarRef lbl_if = VarRef::steal(jit_var_cond_append(cond_inner, rv_if, 1));

                uint32_t rv_iout[1] = { 0 };
                jit_var_cond_end(cond_inner, rv_iout);
                rt = UInt32::steal(rv_iout[0]);
            }
            jit_var_mask_pop(Backend);
            uint32_t rv_ot[1] = { rt.index() };
            VarRef lbl_ot = VarRef::steal(jit_var_cond_append(cond_outer, rv_ot, 1));

            // Outer false branch (odd lanes): gather + scatter
            jit_var_mask_push(Backend, outer_f.index());
            UInt32 rf = gather<UInt32>(tbl, idx) + 500u;
            scatter(dst, rf, idx);
            jit_var_mask_pop(Backend);
            uint32_t rv_of[1] = { rf.index() };
            VarRef lbl_of = VarRef::steal(jit_var_cond_append(cond_outer, rv_of, 1));

            uint32_t rv_oout[1] = { 0 };
            jit_var_cond_end(cond_outer, rv_oout);
            res = UInt32::steal(rv_oout[0]);
            rec.disarm();
        }

        res.schedule();
        dst.schedule();
        assemble_and_validate_spirv();

        for (uint32_t i = 0; i < 8; ++i) {
            uint32_t tbl_i = i * 10u + 1u;
            uint32_t tbl_rev = (7u - i) * 10u + 1u;
            if ((i & 1) == 0) {
                uint32_t exp_res = (i < 4) ? tbl_i : tbl_rev;
                uint32_t exp_dst = (i < 4) ? (tbl_i + 1000u) : (tbl_rev + 2000u);
                jit_assert(res.read(i) == exp_res);
                jit_assert(dst.read(i) == exp_dst);
            } else {
                uint32_t exp_res = tbl_i + 500u;
                jit_assert(res.read(i) == exp_res);
                jit_assert(dst.read(i) == exp_res);
            }
        }
    }
}

TEST_VULKAN(25_symbolic_loops) {
    // 1. Simple symbolic loop (OptimizeLoops off & on)
    for (uint32_t i = 0; i < 2; ++i) {
        jit_set_flag(JitFlag::OptimizeLoops, i == 1);

        for (uint32_t j = 0; j < 2; ++j) {
            UInt32 x = arange<UInt32>(10);
            Float  y = zeros<Float>(1);
            Float  z = 1;
            {
                ScopedRecord record_guard(Backend);
                uint32_t var_indices[3] = { x.index(), y.index(), z.index() };

                VarRef loop = VarRef::steal(
                    jit_var_loop_start("MyLoop", false, 3, var_indices));

                UInt32 xl = UInt32::steal(var_indices[0]);
                Float  yl = Float::steal(var_indices[1]);
                Float  zl = Float::steal(var_indices[2]);

                do {
                    Mask active_i = xl < 5;
                    Mask active = Mask::steal(jit_var_mask_apply(
                        active_i.index(), (uint32_t) jit_var_size(active_i.index())));

                    VarRef loop_cond = VarRef::steal(
                        jit_var_loop_cond(loop, active.index()));

                    yl += Float(xl);
                    xl += 1;
                    zl += 1;

                    var_indices[0] = xl.index();
                    var_indices[1] = yl.index();
                    var_indices[2] = zl.index();

                    int rv = jit_var_loop_end(loop, loop_cond, var_indices,
                                              record_guard.checkpoint);
                    if (rv) {
                        x = UInt32::steal(var_indices[0]);
                        y = Float::steal(var_indices[1]);
                        z = Float::steal(var_indices[2]);
                        record_guard.disarm();
                        break;
                    }
                } while (true);
            }

            if (j == 0) {
                jit_var_schedule(z.index());
                jit_var_schedule(y.index());
                jit_var_schedule(x.index());
                if (i == 0) {
                    std::string spv = assemble_and_validate_spirv();
                    jit_assert(spv.find("OpLoopMerge %loop_exit_r") != std::string::npos);
                }
            }

            jit_assert(strcmp(x.str(), "[5, 5, 5, 5, 5, 5, 6, 7, 8, 9]") == 0);
            jit_assert(strcmp(y.str(), "[10, 10, 9, 7, 4, 0, 0, 0, 0, 0]") == 0);
            jit_assert(strcmp(z.str(), "[6, 5, 4, 3, 2, 1, 1, 1, 1, 1]") == 0);
        }
    }

    // 2. Loop with scatter_reduce side effects (referencing and not referencing loop vars)
    for (uint32_t i = 0; i < 2; ++i) {
        jit_set_flag(JitFlag::OptimizeLoops, i == 1);

        UInt32 x = arange<UInt32>(10);
        Float  y = zeros<Float>(1);
        UInt32 target1 = zeros<UInt32>(11);
        UInt32 target2 = zeros<UInt32>(11);
        {
            uint32_t var_indices[2] = { x.index(), y.index() };
            ScopedRecord record_guard(Backend);

            VarRef loop = VarRef::steal(
                jit_var_loop_start("SideEffectLoop", false, 2, var_indices));

            UInt32 xl = UInt32::steal(var_indices[0]);
            Float  yl = Float::steal(var_indices[1]);

            do {
                Mask active_i = xl < 5;
                Mask active = Mask::steal(jit_var_mask_apply(
                    active_i.index(), (uint32_t) jit_var_size(active_i.index())));

                VarRef loop_cond = VarRef::steal(
                    jit_var_loop_cond(loop, active.index()));

                scatter_reduce(ReduceOp::Add, target1, UInt32(1), xl, active);
                scatter_reduce(ReduceOp::Add, target2, UInt32(2), UInt32(2), active);
                yl += Float(xl);
                xl += 1;

                var_indices[0] = xl.index();
                var_indices[1] = yl.index();

                int rv = jit_var_loop_end(loop, loop_cond, var_indices,
                                          record_guard.checkpoint);
                if (rv) {
                    x = UInt32::steal(var_indices[0]);
                    y = Float::steal(var_indices[1]);
                    record_guard.disarm();
                    break;
                }
            } while (true);
        }

        jit_assert(strcmp(x.str(), "[5, 5, 5, 5, 5, 5, 6, 7, 8, 9]") == 0);
        jit_assert(strcmp(y.str(), "[10, 10, 9, 7, 4, 0, 0, 0, 0, 0]") == 0);
        jit_assert(strcmp(target1.str(), "[1, 2, 3, 4, 5, 0, 0, 0, 0, 0, 0]") == 0);
        jit_assert(strcmp(target2.str(), "[0, 0, 30, 0, 0, 0, 0, 0, 0, 0, 0]") == 0);
    }

    // 3. Loop invariant optimization (04_optimize_invariant)
    for (uint32_t i = 0; i < 2; ++i) {
        jit_set_flag(JitFlag::OptimizeLoops, i == 1);

        UInt32 j = 0,
               v1 = 123, v1_orig = v1,
               v2 = opaque<UInt32>(123), v2_orig = v2,
               v3 = 124, v3_orig = v3,
               v4 = 125, v4_orig = v4,
               v5 = 1,   v5_orig = v5,
               v6 = 0,   v6_orig = v6;

        int count = 0;
        {
            uint32_t var_indices[7] = {
                j.index(), v1.index(), v2.index(), v3.index(),
                v4.index(), v5.index(), v6.index()
            };
            ScopedRecord record_guard(Backend);

            VarRef loop = VarRef::steal(
                jit_var_loop_start("InvariantLoop", false, 7, var_indices));

            UInt32 jl  = UInt32::steal(var_indices[0]),
                   v1l = UInt32::steal(var_indices[1]),
                   v2l = UInt32::steal(var_indices[2]),
                   v3l = UInt32::steal(var_indices[3]),
                   v4l = UInt32::steal(var_indices[4]),
                   v5l = UInt32::steal(var_indices[5]),
                   v6l = UInt32::steal(var_indices[6]);
            do {
                Mask active_i = jl < 10;
                Mask active = Mask::steal(jit_var_mask_apply(
                    active_i.index(), (uint32_t) jit_var_size(active_i.index())));

                VarRef loop_cond = VarRef::steal(
                    jit_var_loop_cond(loop, active.index()));

                jl += 1;
                (void) v1l;
                (void) v2l;
                v3l = 124;
                v4l = 100;
                (void) v5l;
                v6l += v5l;
                ++count;

                var_indices[0] = jl.index();
                var_indices[1] = v1l.index();
                var_indices[2] = v2l.index();
                var_indices[3] = v3l.index();
                var_indices[4] = v4l.index();
                var_indices[5] = v5l.index();
                var_indices[6] = v6l.index();
                int rv = jit_var_loop_end(loop, loop_cond, var_indices,
                                          record_guard.checkpoint);
                if (!rv) {
                    jl  = UInt32::steal(var_indices[0]);
                    v1l = UInt32::steal(var_indices[1]);
                    v2l = UInt32::steal(var_indices[2]);
                    v3l = UInt32::steal(var_indices[3]);
                    v4l = UInt32::steal(var_indices[4]);
                    v5l = UInt32::steal(var_indices[5]);
                    v6l = UInt32::steal(var_indices[6]);
                } else {
                    j  = UInt32::steal(var_indices[0]);
                    v1 = UInt32::steal(var_indices[1]);
                    v2 = UInt32::steal(var_indices[2]);
                    v3 = UInt32::steal(var_indices[3]);
                    v4 = UInt32::steal(var_indices[4]);
                    v5 = UInt32::steal(var_indices[5]);
                    v6 = UInt32::steal(var_indices[6]);
                    record_guard.disarm();
                    break;
                }
            } while (true);
        }

        jit_assert(count == (i == 0 ? 1 : 2));
        if (i == 1) {
            jit_assert(jit_var_state(v1.index()) == VarState::Literal && v1.index() == v1_orig.index());
            jit_assert(jit_var_state(v2.index()) != VarState::Literal && v2.index() == v2_orig.index());
            jit_assert(jit_var_state(v3.index()) == VarState::Literal && v3.index() == v3_orig.index());
            jit_assert(jit_var_state(v4.index()) != VarState::Literal && v4.index() != v4_orig.index());
            jit_assert(jit_var_state(v5.index()) == VarState::Literal && v5.index() == v5_orig.index());
            jit_assert(jit_var_state(v6.index()) != VarState::Literal && v6.index() != v6_orig.index());
        }

        v4.schedule();
        v6.schedule();
        jit_eval();
        jit_assert(v4.read(0) == 100);
        jit_assert(v6.read(0) == 10);
    }

    // 4. Nested Collatz loop (05_collatz)
    auto collatz = [](const char *name, UInt32 value, bool symbolic) -> UInt32 {
        UInt32 counter = 0;
        {
            uint32_t var_indices[2] = { value.index(), counter.index() };
            ScopedRecord record_guard(Backend);
            VarRef loop = VarRef::steal(
                jit_var_loop_start(name, symbolic, 2, var_indices));

            UInt32 valuel   = UInt32::steal(var_indices[0]),
                   counterl = UInt32::steal(var_indices[1]);
            do {
                Mask active_i = neq(valuel, 1);
                Mask active = Mask::steal(jit_var_mask_apply(
                    active_i.index(), (uint32_t) jit_var_size(active_i.index())));

                VarRef loop_cond = VarRef::steal(
                    jit_var_loop_cond(loop, active.index()));

                Mask is_even = eq(valuel & UInt32(1), 0);
                valuel = select(is_even, valuel / 2, valuel * 3 + 1);
                counterl += 1;

                var_indices[0] = valuel.index();
                var_indices[1] = counterl.index();

                int rv = jit_var_loop_end(loop, loop_cond, var_indices,
                                          record_guard.checkpoint);
                if (rv) {
                    value   = UInt32::steal(var_indices[0]);
                    counter = UInt32::steal(var_indices[1]);
                    record_guard.disarm();
                    break;
                }
            } while (true);
        }
        return counter;
    };

    for (uint32_t i = 0; i < 2; ++i) {
        jit_set_flag(JitFlag::LoopOptimize, i == 1);
        for (uint32_t j = 0; j < 2; ++j) {
            UInt32 buf = full<UInt32>(1000, 11);
            if (j == 0) {
                UInt32 k = 1;
                {
                    uint32_t var_indices[1] = { k.index() };
                    ScopedRecord record_guard(Backend);
                    VarRef loop = VarRef::steal(
                        jit_var_loop_start("Outer", false, 1, var_indices));
                    UInt32 kl = UInt32::steal(var_indices[0]);

                    do {
                        Mask active_i = kl <= 10;
                        Mask active = Mask::steal(jit_var_mask_apply(
                            active_i.index(), (uint32_t) jit_var_size(active_i.index())));
                        VarRef loop_cond = VarRef::steal(
                            jit_var_loop_cond(loop, active.index()));

                        scatter(buf, collatz("Inner", kl, true), kl - 1, active);
                        kl += 1;
                        var_indices[0] = kl.index();

                        int rv = jit_var_loop_end(loop, loop_cond, var_indices,
                                                  record_guard.checkpoint);
                        if (rv) {
                            k = UInt32::steal(var_indices[0]);
                            record_guard.disarm();
                            break;
                        }
                    } while (true);
                }
            } else {
                for (uint32_t k = 1; k <= 10; ++k) {
                    char tmpname[20];
                    snprintf(tmpname, sizeof(tmpname), "Inner [%u]", k);
                    scatter(buf, collatz(tmpname, k, false), UInt32(k - 1));
                }
            }
            jit_assert(strcmp(buf.str(), "[0, 1, 7, 2, 5, 8, 16, 3, 19, 6, 1000]") == 0);
        }
    }

    // 5. Nested loop writing to the same variable (06_nested_write)
    auto inner_loop = [](UInt32 k) -> UInt32 {
        uint32_t var_indices[1] = { k.index() };
        ScopedRecord record_guard(Backend);
        VarRef loop = VarRef::steal(
            jit_var_loop_start("Inner", true, 1, var_indices));
        UInt32 kl = UInt32::steal(var_indices[0]);

        do {
            Mask active_i = neq(kl % 3, 0);
            Mask active = Mask::steal(jit_var_mask_apply(
                active_i.index(), (uint32_t) jit_var_size(active_i.index())));
            VarRef loop_cond = VarRef::steal(
                jit_var_loop_cond(loop, active.index()));

            kl += 1;
            var_indices[0] = kl.index();
            int rv = jit_var_loop_end(loop, loop_cond, var_indices,
                                      record_guard.checkpoint);
            if (rv) {
                k = UInt32::steal(var_indices[0]);
                record_guard.disarm();
                break;
            }
        } while (true);
        return k;
    };

    for (uint32_t i = 0; i < 2; ++i) {
        jit_set_flag(JitFlag::LoopOptimize, i == 1);
        UInt32 k = arange<UInt32>(10) * 12;
        {
            uint32_t var_indices[1] = { k.index() };
            ScopedRecord record_guard(Backend);
            VarRef loop = VarRef::steal(
                jit_var_loop_start("Outer", false, 1, var_indices));
            UInt32 kl = UInt32::steal(var_indices[0]);

            do {
                Mask active_i = neq(kl % 7, 0);
                Mask active = Mask::steal(jit_var_mask_apply(
                    active_i.index(), (uint32_t) jit_var_size(active_i.index())));
                VarRef loop_cond = VarRef::steal(
                    jit_var_loop_cond(loop, active.index()));

                kl = inner_loop(kl);
                kl += 1;
                var_indices[0] = kl.index();

                int rv = jit_var_loop_end(loop, loop_cond, var_indices,
                                          record_guard.checkpoint);
                if (rv) {
                    k = UInt32::steal(var_indices[0]);
                    record_guard.disarm();
                    break;
                }
            } while (true);
        }
        jit_assert(strcmp(k.str(), "[0, 28, 28, 49, 49, 70, 91, 84, 112, 112]") == 0);
    }

    // 6. Multiple scatters before, inside, and after a loop (09_test_multiple_scatter_loop)
    {
        jit_set_flag(JitFlag::KernelHistory, true);
        jit_set_flag(JitFlag::LoopOptimize, false);

        UInt32 i = zeros<UInt32>(10);
        UInt32 target = zeros<UInt32>(10);
        jit_kernel_history_clear();

        scatter(target, UInt32(1), UInt32(0), true);
        {
            uint32_t var_indices[1] = { i.index() };
            ScopedRecord record_guard(Backend);
            VarRef loop = VarRef::steal(
                jit_var_loop_start("MultiScatterLoop", false, 1, var_indices));
            UInt32 il = UInt32::steal(var_indices[0]);

            do {
                Mask active_i = il < 1;
                Mask active = Mask::steal(jit_var_mask_apply(
                    active_i.index(), (uint32_t) jit_var_size(active_i.index())));
                VarRef loop_cond = VarRef::steal(
                    jit_var_loop_cond(loop, active.index()));

                scatter(target, UInt32(1), UInt32(1), active);
                scatter(target, UInt32(1), UInt32(2), active);
                scatter(target, UInt32(1), UInt32(3), active);
                il += 1;
                var_indices[0] = il.index();

                int rv = jit_var_loop_end(loop, loop_cond, var_indices,
                                          record_guard.checkpoint);
                if (rv) {
                    i = UInt32::steal(var_indices[0]);
                    record_guard.disarm();
                    break;
                }
            } while (true);
        }
        scatter(target, UInt32(1), UInt32(4), true);

        jit_assert(strcmp(target.str(), "[1, 1, 1, 1, 1, 0, 0, 0, 0, 0]") == 0);

        size_t kernel_launches = 0;
        {
            KernelHistory *h = jit_kernel_history_view(0);
            for (size_t k = 0; k < jit_kernel_history_size(h); ++k) {
                KernelType type = (KernelType) (uint32_t) jit_kernel_history_query(
                    h, k, KernelHistoryField::Type);
                if (type == KernelType::JIT)
                    kernel_launches++;
            }
            jit_kernel_history_free(h);
        }
        jit_kernel_history_clear();
        jit_assert(kernel_launches == 3);
        jit_set_flag(JitFlag::KernelHistory, false);
        jit_set_flag(JitFlag::LoopOptimize, true);
    }
}

TEST_VULKAN(26_local_arrays) {
    // 1. Literal index read/write
    {
        LocalArr<Float> x(10);
        for (int i = 0; i < 10; ++i)
            x.write(i, i);
        Float result = 0;
        for (int i = 0; i < 10; ++i)
            result += x.read(i);
        jit_assert(result.read(0) == 45);
        jit_assert(x.length() == 10);
    }

    // 2. Opaque dynamic index read/write + LVN check
    {
        LocalArr<Float> x(10);
        UInt32 opaque_0 = opaque<UInt32>(0);
        for (int i = 0; i < 10; ++i)
            x.write(UInt32(i) + opaque_0, i);
        Float result = 0;
        for (int i = 0; i < 10; ++i) {
            Float a = x.read(UInt32(i) + opaque_0);
            Float b = x.read(UInt32(i) + opaque_0);
            jit_assert(a.index() == b.index());
            result += a;
        }
        jit_assert(result.read(0) == 45);
    }

    // 3. Literal & dynamic index masked read/write
    {
        UInt32 opaque_0 = opaque<UInt32>(0);
        UInt32 opaque_1 = opaque<UInt32>(1);

        LocalArr<Float> x(10);
        for (int i = 0; i < 10; ++i)
            x.write(i, i);
        Float r1 = 0, r2 = 0;
        for (int i = 0; i < 10; ++i) {
            r1 += x.read(i, Mask(UInt32(i) & opaque_1));
            r2 += x.read(UInt32(i) + opaque_0, Mask(UInt32(i) & opaque_1));
        }
        jit_assert(r1.read(0) == 25);
        jit_assert(r2.read(0) == 25);

        LocalArr<Float> y(10, 0);
        LocalArr<Float> z(10, 0);
        for (int i = 0; i < 10; ++i) {
            y.write(i, i, Mask(UInt32(i) & opaque_1));
            z.write(UInt32(i) + opaque_0, Float(i), Mask(UInt32(i) & opaque_1));
        }
        Float ry = 0, rz = 0;
        for (int i = 0; i < 10; ++i) {
            ry += y.read(i);
            rz += z.read(UInt32(i) + opaque_0);
        }
        jit_assert(ry.read(0) == 25);
        jit_assert(rz.read(0) == 25);
    }

    // 4. Conflict copy-on-write (05_conflict) and eval conflict (09_write_eval_conflict)
    {
        LocalArr<Float> x(10);
        for (int i = 0; i < 10; ++i)
            x.write(i, i);

        LocalArr<Float> y(x);
        for (int i = 0; i < 10; ++i)
            y.write(i, UInt32(i) + 10);

        Float result_1 = 0, result_2 = 0;
        for (int i = 0; i < 10; ++i) {
            result_1 += x.read(UInt32(i));
            result_2 += y.read(UInt32(i));
        }
        result_1.schedule();
        result_2.schedule();
        assemble_and_validate_spirv();
        jit_assert(result_1.read(0) == 45 && result_2.read(0) == 145);

        LocalArr<Float> a(2, 1);
        LocalArr<Float> b(a);
        a.write(0, 2);
        a.write(1, 3);
        a.schedule();
        b.schedule();
        assemble_and_validate_spirv();

        Float aa = a.read(0) + a.read(1);
        Float bb = b.read(0) + b.read(1);
        aa.schedule();
        bb.schedule();
        assemble_and_validate_spirv();
        jit_assert(aa.read(0) == 5 && bb.read(0) == 2);
    }

    // 5. Evaluate multi-lane array to global memory and read back (06_eval_array)
    {
        LocalArr<Float> x(10);
        Float offset = arange<Float>(3);
        for (int i = 0; i < 10; ++i)
            x.write(i, Float(i) + offset);
        x.eval();

        Float result = 0;
        for (int i = 0; i < 10; ++i)
            result += x.read(UInt32(i));
        jit_assert(result.read(0) == 45);
        jit_assert(result.read(1) == 55);
        jit_assert(result.read(2) == 65);
    }

    // 6. Per-lane complex modulo indexing (08_complex_indexing)
    {
        LocalArr<Float> x(10, 0);
        UInt32 index = arange<UInt32>(3);
        for (int i = 0; i < 10; ++i)
            x.write((index + UInt32(i)) % 10, Float(i), Mask(UInt32(i) & 1));
        Float r = 0;
        for (int i = 0; i < 10; ++i)
            r += x.read((index + UInt32(i)) % 10);
        jit_assert(r.read(0) == 25 && r.read(1) == 25 && r.read(2) == 25);
    }

    // 7. Boolean mask arrays (10_mask_simple, 11_mask_eval, 13_mask_conflict)
    {
        LocalArr<Mask> x(10, false);
        LocalArr<Mask> y(10, true);
        for (int i = 0; i < 10; ++i)
            x.write(i, Mask(i & 1));
        x.schedule();
        y.schedule();
        assemble_and_validate_spirv();

        UInt32 ctr_1 = 0, ctr_2 = 0;
        for (int i = 0; i < 10; ++i) {
            ctr_1 += UInt32(x.read(i));
            ctr_2 += UInt32(y.read(i));
        }
        ctr_1.schedule();
        ctr_2.schedule();
        assemble_and_validate_spirv();
        jit_assert(ctr_1.read(0) == 5 && ctr_2.read(0) == 10);

        LocalArr<Mask> m1(2, true);
        LocalArr<Mask> m2(m1);
        m1.write(0, false);
        UInt32 mc1 = UInt32(m1.read(0)) + UInt32(m1.read(1));
        UInt32 mc2 = UInt32(m2.read(0)) + UInt32(m2.read(1));
        mc1.schedule();
        mc2.schedule();
        jit_assert(mc1.read(0) == 1 && mc2.read(0) == 2);
    }

    // 8. ArraySelect (select between two local arrays)
    {
        UInt32 lane = arange<UInt32>(4);
        LocalArr<Float> a(4, 0.f);
        LocalArr<Float> b(4, 0.f);
        for (uint32_t i = 0; i < 4; ++i) {
            a.write(i, Float(lane) + Float(i * 10));
            b.write(i, Float(lane) + Float(i * 100));
        }
        Mask m = eq(lane & UInt32(1), UInt32(0));
        LocalArr<Float> s = LocalArr<Float>::steal(
            jit_var_select(m.index(), a.m_index, b.m_index));

        Float total = 0.f;
        for (uint32_t i = 0; i < 4; ++i)
            total += s.read(i);
        assemble_and_validate_spirv(total.index());

        for (uint32_t l = 0; l < 4; ++l) {
            float exp = 0.f;
            for (uint32_t i = 0; i < 4; ++i)
                exp += (l % 2 == 0) ? (l + i * 10.f) : (l + i * 100.f);
            jit_assert(total.read(l) == exp);
        }
    }

    // 9. Local array inside a symbolic loop (LoopPhi + ArrayPhi)
    {
        UInt32 lane = arange<UInt32>(4);
        LocalArr<UInt32> arr(4, 0u);
        UInt32 i = zeros<UInt32>(4);
        {
            uint32_t var_indices[2] = { i.index(), arr.m_index };
            ScopedRecord record_guard(Backend);
            VarRef loop = VarRef::steal(
                jit_var_loop_start("ArrayLoop", false, 2, var_indices));

            UInt32 il = UInt32::steal(var_indices[0]);
            LocalArr<UInt32> arrl = LocalArr<UInt32>::steal(var_indices[1]);

            do {
                Mask active_i = il < 4;
                Mask active = Mask::steal(jit_var_mask_apply(
                    active_i.index(), (uint32_t) jit_var_size(active_i.index())));
                VarRef loop_cond = VarRef::steal(
                    jit_var_loop_cond(loop, active.index()));

                arrl.write(il, (lane + 1u) * (il + 10u), active);
                il += 1;

                var_indices[0] = il.index();
                var_indices[1] = arrl.m_index;

                int rv = jit_var_loop_end(loop, loop_cond, var_indices,
                                          record_guard.checkpoint);
                if (rv) {
                    i   = UInt32::steal(var_indices[0]);
                    arr = LocalArr<UInt32>::steal(var_indices[1]);
                    record_guard.disarm();
                    break;
                } else {
                    il   = UInt32::steal(var_indices[0]);
                    arrl = LocalArr<UInt32>::steal(var_indices[1]);
                }
            } while (true);
        }

        UInt32 sum = 0u;
        for (uint32_t k = 0; k < 4; ++k)
            sum += arr.read(k);
        assemble_and_validate_spirv(sum.index());

        for (uint32_t l = 0; l < 4; ++l) {
            uint32_t exp = (l + 1u) * (10u + 11u + 12u + 13u);
            jit_assert(sum.read(l) == exp);
        }
    }

    // 10. tile and repeat
    {
        UInt32 x = tile(arange<UInt32>(3), 3);
        jit_assert(strcmp(x.str(), "[0, 1, 2, 0, 1, 2, 0, 1, 2]") == 0);
        UInt32 y = repeat(arange<UInt32>(3), 3);
        jit_assert(strcmp(y.str(), "[0, 0, 0, 1, 1, 1, 2, 2, 2]") == 0);
    }
}

