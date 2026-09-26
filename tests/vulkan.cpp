/*
    tests/vulkan.cpp -- Tests of the Vulkan backend: device selection, memory
    allocation, transfers, synchronization, and host callbacks

    The Vulkan backend cannot compile kernels yet, hence these tests exercise
    the operations that it implements using transfer commands. They are
    skipped when no compatible Vulkan device is found, unless the environment
    variable DRJIT_TEST_REQUIRE_VULKAN=1 is set. Set DRJIT_VULKAN_DEVICE to
    the index of a physical device to run them on a specific device.

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
    jit_assert(props.apiVersion >= ((1u << 22) | (2u << 12))); // >= 1.2

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

    // Allocations exceeding the device limits raise an exception
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
    // Kernels cannot be compiled yet
    {
        UInt32 a = UInt32::copy(std::vector<uint32_t>(10, 1).data(), 10);
        UInt32 b = a + a;
        jit_assert(raises([&] { b.eval(); }, "cannot compile kernels"));
        jit_assert(jit_var_state(b.index()) == VarState::Unevaluated);
        jit_eval(); // (nothing left to do)
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
