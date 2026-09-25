/*
    tests/backend_bits.cpp -- White-box checks of the packed bit fields that
    encode the JIT backend and variable type (``Variable``, ``VariableKey``,
    ``AllocInfo``). Adding the Vulkan backend (``JitBackend::Vulkan == 4``)
    required widening all backend fields to 3 bits.

    Copyright (c) 2026 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#include "internal.h" // Private Dr.Jit-Core header (see tests/CMakeLists.txt)
#include <cstdio>
#include <cstdlib>
#include <cstring>

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

/// Every (kind, backend, type, write_ptr, written) combination must survive a
/// round trip through the 'Variable' bit fields without disturbing the
/// neighboring flags, and 'VariableKey::packed' must have the documented
/// kind:7|backend:3|type:4|write_ptr:1|written:1 layout.
static void test_variable_bits() {
    const uint32_t kinds[] = { (uint32_t) VarKind::Invalid,
                               (uint32_t) VarKind::Evaluated,
                               (uint32_t) VarKind::Literal,
                               (uint32_t) VarKind::Add,
                               (uint32_t) VarKind::Count - 1 };

    for (uint32_t kind : kinds) {
        for (uint32_t backend = 0; backend < (uint32_t) JitBackend::Count; ++backend) {
            for (uint32_t type = 0; type < (uint32_t) VarType::Count; ++type) {
                for (uint32_t bits = 0; bits < 4; ++bits) {
                    uint32_t write_ptr = bits & 1, written = bits >> 1;

                    Variable v;
                    // Set all neighboring flags to detect overlapping fields
                    v.retain_data = v.symbolic = v.unaligned = v.optix =
                        v.side_effect = v.coop_vec = 1;
                    v.scratch = 0xFFFFFFFFu;
                    v.kind = kind;
                    v.backend = backend;
                    v.type = type;
                    v.write_ptr = write_ptr;
                    v.written = written;

                    CHECK(v.kind == kind);
                    CHECK(v.backend == backend);
                    CHECK(v.type == type);
                    CHECK(v.write_ptr == write_ptr);
                    CHECK(v.written == written);
                    CHECK(v.retain_data && v.symbolic && v.unaligned &&
                          v.optix && v.side_effect && v.coop_vec);
                    CHECK(v.scratch == 0xFFFFFFFFu);

                    // 'coop_vec' is set, so the array length ends up in the
                    // upper 16 bits of the packed key
                    v.array_length = 0xABCD;

                    VariableKey key(v);
                    CHECK(( key.packed        & 0x7f) == kind);
                    CHECK(((key.packed >>  7) & 0x7)  == backend);
                    CHECK(((key.packed >> 10) & 0xf)  == type); // see jitc_shutdown()
                    CHECK(((key.packed >> 14) & 0x1)  == write_ptr);
                    CHECK(((key.packed >> 15) & 0x1)  == written);
                    CHECK( (key.packed >> 16)         == 0xABCD);
                }
            }
        }
    }

    // Keys of variables that only differ in their backend must not compare equal
    Variable a, b;
    a.kind = b.kind = (uint32_t) VarKind::Add;
    a.type = b.type = (uint32_t) VarType::Float32;
    a.backend = (uint32_t) JitBackend::Metal;
    b.backend = (uint32_t) JitBackend::Vulkan;
    CHECK(!(VariableKey(a) == VariableKey(b)));
    b.backend = (uint32_t) JitBackend::Metal;
    CHECK(VariableKey(a) == VariableKey(b));
}

/// Round trip through the [size:52][shared:1][backend:3][device:8] encoding
static void test_alloc_info() {
    const size_t sizes[] = { 0, 1, 64, 4096, ((size_t) 1 << 40) + 3,
                             ((size_t) 1 << 52) - 1 };
    const int devices[] = { 0, 1, 7, 255 };

    for (size_t size : sizes) {
        for (uint32_t backend = 0; backend < (uint32_t) JitBackend::Count; ++backend) {
            for (int shared = 0; shared < 2; ++shared) {
                for (int device : devices) {
                    AllocInfo ai = alloc_info_encode(size, (JitBackend) backend,
                                                     (bool) shared, device);
                    auto [size_2, backend_2, shared_2, device_2] =
                        alloc_info_decode(ai);
                    CHECK(size_2 == size);
                    CHECK(backend_2 == (JitBackend) backend);
                    CHECK(shared_2 == (bool) shared);
                    CHECK(device_2 == device);
                }
            }
        }
    }
}

static void test_names_and_predicates() {
    CHECK(strcmp(jitc_backend_name(JitBackend::Vulkan), "Vulkan") == 0);
    CHECK(strcmp(jitc_backend_name(JitBackend::Metal), "Metal") == 0);
    CHECK(strcmp(jit_backend_name_v<JitBackend::Vulkan>, "vulkan") == 0);
    CHECK(strcmp(jit_backend_name_v<JitBackend::LLVM>, "llvm") == 0);
    CHECK((uint32_t) JitBackend::Vulkan == 4 && (uint32_t) JitBackend::Count == 5);

    CHECK(!jitc_is_gpu(JitBackend::LLVM));
    CHECK(!jitc_is_vulkan(JitBackend::LLVM));
    CHECK(!jitc_is_vulkan(JitBackend::CUDA));
#if defined(DRJIT_ENABLE_VULKAN)
    CHECK(jitc_is_vulkan(JitBackend::Vulkan));
    CHECK(jitc_is_gpu(JitBackend::Vulkan));
#else
    CHECK(!jitc_is_vulkan(JitBackend::Vulkan));
#endif
}

/// Public API: requesting the Vulkan backend must never raise or disturb the
/// other backends
static void test_public_api() {
    // Only show errors: a warning is expected if Vulkan is unavailable
    jit_set_log_level_stderr(LogLevel::Error);
    uint32_t backends = (1u << (uint32_t) JitBackend::LLVM) |
                        (1u << (uint32_t) JitBackend::Vulkan);

    for (int i = 0; i < 2; ++i) {
        jit_init(backends);
        int has_vulkan = jit_has_backend(JitBackend::Vulkan);
        CHECK(has_vulkan == 0 || has_vulkan == 1);

        if (jit_has_backend(JitBackend::LLVM)) {
            uint32_t index = jit_var_u32(JitBackend::LLVM, 1234);
            CHECK(jit_var_type(index) == VarType::UInt32);
            jit_var_dec_ref(index);
        }

        jit_shutdown(i); // light shutdown, then full shutdown
    }
}

int main(int, char **) {
    test_variable_bits();
    test_alloc_info();
    test_names_and_predicates();
    test_public_api();
    printf("Passed %i checks.\n", checks_run);
    return 0;
}
