/*
    src/vulkan_eval.h -- Vulkan SPIR-V code generation macros and helpers.

    Copyright (c) 2026 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#include "common.h"
#include "strbuf.h"

extern StringBuffer spv_header;
extern StringBuffer spv_annotations;
extern StringBuffer spv_types_consts;
extern StringBuffer spv_func_locals;

#define fmt(f, ...)                                                            \
    buffer.fmt_vulkan(count_args(__VA_ARGS__), fmt_strlen(f), f, ##__VA_ARGS__)

#define fmt_header(f, ...)                                                     \
    spv_header.fmt_vulkan(count_args(__VA_ARGS__), fmt_strlen(f), f,           \
                          ##__VA_ARGS__)

#define fmt_annot(f, ...)                                                      \
    spv_annotations.fmt_vulkan(count_args(__VA_ARGS__), fmt_strlen(f), f,      \
                               ##__VA_ARGS__)

#define fmt_tc(f, ...)                                                         \
    spv_types_consts.fmt_vulkan(count_args(__VA_ARGS__), fmt_strlen(f), f,     \
                                ##__VA_ARGS__)

#define fmt_local(f, ...)                                                      \
    spv_func_locals.fmt_vulkan(count_args(__VA_ARGS__), fmt_strlen(f), f,      \
                               ##__VA_ARGS__)

template <typename... Args> JIT_INLINE void put(Args &&...args) {
    buffer.put((Args &&) args...);
}
template <typename... Args> JIT_INLINE void put_header(Args &&...args) {
    spv_header.put((Args &&) args...);
}
template <typename... Args> JIT_INLINE void put_annot(Args &&...args) {
    spv_annotations.put((Args &&) args...);
}
template <typename... Args> JIT_INLINE void put_tc(Args &&...args) {
    spv_types_consts.put((Args &&) args...);
}
template <typename... Args> JIT_INLINE void put_local(Args &&...args) {
    spv_func_locals.put((Args &&) args...);
}

/// Ensure that the SPIR-V type corresponding to ``vt`` is declared in
/// ``spv_types_consts`` and that the active Vulkan device supports it.
extern void jitc_vulkan_require_type(VarType vt);

/// Ensure that a PhysicalStorageBuffer pointer type to ``vt`` (e.g.,
/// ``%ptr_psb_u32``) is declared and decorated with ``ArrayStride``, and return
/// its SPIR-V type identifier.
extern const char *jitc_vulkan_require_psb_ptr(VarType vt);

/// Ensure that a Function pointer type to ``vt`` (e.g., ``%ptr_fn_u32``) is
/// declared in ``spv_types_consts`` and return its SPIR-V type identifier.
extern const char *jitc_vulkan_require_fn_ptr(VarType vt);

/// Ensure that ``%c_u32_<value> = OpConstant %u32 <value>`` is declared in
/// ``spv_types_consts``.
extern void jitc_vulkan_require_u32_const(uint32_t value);

/// Ensure that ``%bool``, ``%u8``, ``%ptr_psb_u8``, ``%c_u8_0``, and
/// ``%c_u8_1`` are declared for loading/storing 8-bit boolean buffers.
extern void jitc_vulkan_require_bool_storage();

/// Ensure that ``%c_zero_<type> = OpConstantNull %<type>`` is declared in
/// ``spv_types_consts`` and return its SPIR-V constant identifier.
extern const char *jitc_vulkan_require_zero(VarType vt);

/// Request the ``GLSL.std.450`` extended instruction set in ``spv_header``.
extern void jitc_vulkan_require_glsl450();

