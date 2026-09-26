/**
 * This file contains the logic that assembles a textual SPIR-V 1.6 module from
 * a recorded Dr.Jit computation graph. It implements a multi-section template
 * engine using the 'fmt_vulkan' formatting routine.
 *
 * Its format interface supports the following format string characters:
 *
 *  Format  Input          Example result    Description
 * --------------------------------------------------------------------------
 *  $u      uint32_t      `1234`             Decimal number (32 bit)
 *  $Q      uint64_t      `0000000000001234` Hexadecimal number (64 bit, 0-padded)
 *  $s      const char *  `foo`              Zero-terminated string
 *  $t      Variable      `%f32`             SPIR-V variable type ID
 *  $b      Variable      `%u32`             SPIR-V binary view type ID
 *  $v      Variable      `%r1234`           SPIR-V SSA register ID
 *  $l      Variable      `0x1` / `!0x3f800000` Literal value for OpConstant
 *  $a      Variable      `4`                Byte alignment / size of variable
 *  $o      Variable      `2`                Index into ``params.args[]``
 * --------------------------------------------------------------------------
 */

#include "eval.h"
#include "internal.h"
#include "var.h"
#include "log.h"
#include "strbuf.h"
#include "vulkan.h"
#include "vulkan_eval.h"
#include "call.h"
#include <tsl/robin_set.h>
#include <algorithm>

// Multi-section SPIR-V staging buffers
StringBuffer spv_header       { 512 };
StringBuffer spv_annotations  { 512 };
StringBuffer spv_types_consts { 1024 };
StringBuffer spv_func_locals  { 256 };
static StringBuffer spv_body  { 1024 };

// Forward declaration
static void jitc_vulkan_render(Variable *v);

// ----------------------------------------------------------------------------
//  Per-kernel SPIR-V type, constant, and capability tracking
// ----------------------------------------------------------------------------

static const VulkanDevice *active_vulkan_dev = nullptr;

static bool emitted_types      [(size_t) VarType::Count];
static bool emitted_psb_ptrs   [(size_t) VarType::Count];
static bool emitted_fn_ptrs    [(size_t) VarType::Count];
static bool emitted_zero_consts[(size_t) VarType::Count];
static bool emitted_one_consts [(size_t) VarType::Count];
static bool emitted_ones_consts[(size_t) VarType::Count];
static bool emitted_nan_consts [(size_t) VarType::Count];
static bool emitted_sign_consts[(size_t) VarType::Count];
static bool emitted_mul_structs[(size_t) VarType::Count];

static tsl::robin_set<uint32_t> emitted_u32_consts;

static bool has_bool_storage_consts = false;
static bool need_glsl450 = false;

static bool cap_int8         = false;
static bool cap_storage_8bit = false;
static bool cap_int16        = false;
static bool cap_storage_16bit = false;
static bool cap_float16      = false;
static bool cap_float64      = false;

static const char *psb_ptr_names[(size_t) VarType::Count] = {
    "???",           "%ptr_psb_u8",  "???",           "%ptr_psb_i8",
    "%ptr_psb_u8",   "%ptr_psb_i16", "%ptr_psb_u16",  "%ptr_psb_i32",
    "%ptr_psb_u32",  "%ptr_psb_i64", "%ptr_psb_u64",  "%ptr_psb_u64",
    "???",           "%ptr_psb_f16", "%ptr_psb_f32",  "%ptr_psb_f64"
};

static const char *fn_ptr_names[(size_t) VarType::Count] = {
    "???",          "%ptr_fn_bool", "???",          "%ptr_fn_i8",
    "%ptr_fn_u8",   "%ptr_fn_i16",  "%ptr_fn_u16",  "%ptr_fn_i32",
    "%ptr_fn_u32",  "%ptr_fn_i64",  "%ptr_fn_u64",  "%ptr_fn_u64",
    "???",          "%ptr_fn_f16",  "%ptr_fn_f32",  "%ptr_fn_f64"
};

static const char *zero_const_names[(size_t) VarType::Count] = {
    "???",          "%c_zero_bool", "???",          "%c_zero_i8",
    "%c_zero_u8",   "%c_zero_i16",  "%c_zero_u16",  "%c_zero_i32",
    "%c_zero_u32",  "%c_zero_i64",  "%c_zero_u64",  "%c_zero_u64",
    "???",          "%c_zero_f16",  "%c_zero_f32",  "%c_zero_f64"
};

static const char *one_const_names[(size_t) VarType::Count] = {
    "???",          "%c_one_bool",  "???",          "%c_one_i8",
    "%c_one_u8",    "%c_one_i16",   "%c_one_u16",   "%c_one_i32",
    "%c_one_u32",   "%c_one_i64",   "%c_one_u64",   "%c_one_u64",
    "???",          "%c_one_f16",   "%c_one_f32",   "%c_one_f64"
};

static const char *ones_const_names[(size_t) VarType::Count] = {
    "???",          "%c_one_bool",  "???",          "%c_ones_i8",
    "%c_ones_u8",   "%c_ones_i16",  "%c_ones_u16",  "%c_ones_i32",
    "%c_ones_u32",  "%c_ones_i64",  "%c_ones_u64",  "%c_ones_u64",
    "???",          "???",          "???",          "???"
};

static const char *mul_struct_names[(size_t) VarType::Count] = {
    "???",              "???",              "???",              "%struct_mul_i8",
    "%struct_mul_u8",   "%struct_mul_i16",  "%struct_mul_u16",  "%struct_mul_i32",
    "%struct_mul_u32",  "%struct_mul_i64",  "%struct_mul_u64",  "%struct_mul_u64",
    "???",              "???",              "???",              "???"
};

void jitc_vulkan_assemble_reset() {
    spv_header.clear();
    spv_annotations.clear();
    spv_types_consts.clear();
    spv_func_locals.clear();
    spv_body.clear();

    memset(emitted_types, 0, sizeof(emitted_types));
    memset(emitted_psb_ptrs, 0, sizeof(emitted_psb_ptrs));
    memset(emitted_fn_ptrs, 0, sizeof(emitted_fn_ptrs));
    memset(emitted_zero_consts, 0, sizeof(emitted_zero_consts));
    memset(emitted_one_consts, 0, sizeof(emitted_one_consts));
    memset(emitted_ones_consts, 0, sizeof(emitted_ones_consts));
    memset(emitted_nan_consts, 0, sizeof(emitted_nan_consts));
    memset(emitted_sign_consts, 0, sizeof(emitted_sign_consts));
    memset(emitted_mul_structs, 0, sizeof(emitted_mul_structs));
    emitted_u32_consts.clear();

    has_bool_storage_consts = false;
    need_glsl450 = false;

    cap_int8          = false;
    cap_storage_8bit  = false;
    cap_int16         = false;
    cap_storage_16bit = false;
    cap_float16       = false;
    cap_float64       = false;
}

void jitc_vulkan_require_glsl450() {
    need_glsl450 = true;
}

void jitc_vulkan_require_type(VarType vt) {
    if (vt == VarType::Pointer)
        vt = VarType::UInt64;

    uint32_t idx = (uint32_t) vt;
    if (emitted_types[idx])
        return;

    const VulkanDevice &dev = *active_vulkan_dev;
    switch (vt) {
        case VarType::Void:
            put_tc("%void = OpTypeVoid\n"
                   "%fn_void = OpTypeFunction %void\n");
            break;

        case VarType::Bool:
            put_tc("%bool = OpTypeBool\n");
            break;

        case VarType::Int8:
            if (unlikely(!dev.shader_int8))
                jitc_raise("jit_eval(): the Vulkan device \"%s\" does not "
                           "support 8-bit integers (shaderInt8)!", dev.name);
            cap_int8 = true;
            put_tc("%i8 = OpTypeInt 8 1\n");
            break;

        case VarType::UInt8:
            if (unlikely(!dev.shader_int8))
                jitc_raise("jit_eval(): the Vulkan device \"%s\" does not "
                           "support 8-bit integers (shaderInt8)!", dev.name);
            cap_int8 = true;
            put_tc("%u8 = OpTypeInt 8 0\n");
            break;

        case VarType::Int16:
            if (unlikely(!dev.shader_int16))
                jitc_raise("jit_eval(): the Vulkan device \"%s\" does not "
                           "support 16-bit integers (shaderInt16)!", dev.name);
            cap_int16 = true;
            put_tc("%i16 = OpTypeInt 16 1\n");
            break;

        case VarType::UInt16:
            if (unlikely(!dev.shader_int16))
                jitc_raise("jit_eval(): the Vulkan device \"%s\" does not "
                           "support 16-bit integers (shaderInt16)!", dev.name);
            cap_int16 = true;
            put_tc("%u16 = OpTypeInt 16 0\n");
            break;

        case VarType::Int32:
            put_tc("%i32 = OpTypeInt 32 1\n");
            break;

        case VarType::UInt32:
            put_tc("%u32 = OpTypeInt 32 0\n");
            break;

        case VarType::Int64:
            put_tc("%i64 = OpTypeInt 64 1\n");
            break;

        case VarType::UInt64:
            put_tc("%u64 = OpTypeInt 64 0\n");
            break;

        case VarType::Float16:
            if (unlikely(!dev.shader_float16))
                jitc_raise("jit_eval(): the Vulkan device \"%s\" does not "
                           "support 16-bit floating-point arithmetic (shaderFloat16)!",
                           dev.name);
            cap_float16 = true;
            put_tc("%f16 = OpTypeFloat 16\n");
            break;

        case VarType::Float32:
            put_tc("%f32 = OpTypeFloat 32\n");
            break;

        case VarType::Float64:
            if (unlikely(!dev.shader_float64))
                jitc_raise("jit_eval(): the Vulkan device \"%s\" does not "
                           "support 64-bit floating-point arithmetic (shaderFloat64)!",
                           dev.name);
            cap_float64 = true;
            put_tc("%f64 = OpTypeFloat 64\n");
            break;

        default:
            jitc_fail("jitc_vulkan_require_type(): unsupported type %u!", idx);
    }

    emitted_types[idx] = true;
}

const char *jitc_vulkan_require_psb_ptr(VarType vt) {
    if (vt == VarType::Bool)
        vt = VarType::UInt8;
    else if (vt == VarType::Pointer)
        vt = VarType::UInt64;

    uint32_t idx = (uint32_t) vt;
    const char *ptr_name = psb_ptr_names[idx];
    if (emitted_psb_ptrs[idx])
        return ptr_name;

    jitc_vulkan_require_type(vt);

    const VulkanDevice &dev = *active_vulkan_dev;
    uint32_t elem_size = type_size[idx];
    if (elem_size == 1) {
        if (unlikely(!dev.storage_buffer_8bit))
            jitc_raise("jit_eval(): the Vulkan device \"%s\" does not "
                       "support 8-bit storage buffer access "
                       "(storageBuffer8BitAccess)!", dev.name);
        cap_storage_8bit = true;
    } else if (elem_size == 2) {
        if (unlikely(!dev.storage_buffer_16bit))
            jitc_raise("jit_eval(): the Vulkan device \"%s\" does not "
                       "support 16-bit storage buffer access "
                       "(storageBuffer16BitAccess)!", dev.name);
        cap_storage_16bit = true;
    }

    fmt_annot("OpDecorate $s ArrayStride $u\n", ptr_name, elem_size);
    fmt_tc("$s = OpTypePointer PhysicalStorageBuffer $s\n",
           ptr_name, type_name_vulkan[idx]);

    emitted_psb_ptrs[idx] = true;
    return ptr_name;
}

const char *jitc_vulkan_require_fn_ptr(VarType vt) {
    if (vt == VarType::Pointer)
        vt = VarType::UInt64;

    uint32_t idx = (uint32_t) vt;
    const char *ptr_name = fn_ptr_names[idx];
    if (emitted_fn_ptrs[idx])
        return ptr_name;

    jitc_vulkan_require_type(vt);
    fmt_tc("$s = OpTypePointer Function $s\n", ptr_name, type_name_vulkan[idx]);

    emitted_fn_ptrs[idx] = true;
    return ptr_name;
}

void jitc_vulkan_require_u32_const(uint32_t value) {
    jitc_vulkan_require_type(VarType::UInt32);
    if (emitted_u32_consts.insert(value).second)
        fmt_tc("%c_u32_$u = OpConstant %u32 $u\n", value, value);
}

void jitc_vulkan_require_bool_storage() {
    jitc_vulkan_require_type(VarType::Bool);
    jitc_vulkan_require_psb_ptr(VarType::UInt8);
    if (!has_bool_storage_consts) {
        put_tc("%c_u8_0 = OpConstant %u8 0\n"
               "%c_u8_1 = OpConstant %u8 1\n");
        has_bool_storage_consts = true;
    }
}

static VarType jitc_vulkan_uint_type(VarType vt) {
    switch (type_size[(uint32_t) vt]) {
        case 1: return VarType::UInt8;
        case 2: return VarType::UInt16;
        case 4: return VarType::UInt32;
        case 8: return VarType::UInt64;
        default:
            jitc_fail("jitc_vulkan_uint_type(): unsupported type %u!",
                      (uint32_t) vt);
    }
}

static VarType jitc_vulkan_sint_type(VarType vt) {
    switch (type_size[(uint32_t) vt]) {
        case 1: return VarType::Int8;
        case 2: return VarType::Int16;
        case 4: return VarType::Int32;
        case 8: return VarType::Int64;
        default:
            jitc_fail("jitc_vulkan_sint_type(): unsupported type %u!",
                      (uint32_t) vt);
    }
}

const char *jitc_vulkan_require_zero(VarType vt) {
    if (vt == VarType::Pointer)
        vt = VarType::UInt64;

    uint32_t idx = (uint32_t) vt;
    const char *name = zero_const_names[idx];
    if (emitted_zero_consts[idx])
        return name;

    jitc_vulkan_require_type(vt);
    fmt_tc("$s = OpConstantNull $s\n", name, type_name_vulkan[idx]);

    emitted_zero_consts[idx] = true;
    return name;
}

static const char *jitc_vulkan_require_one(VarType vt) {
    if (vt == VarType::Pointer)
        vt = VarType::UInt64;

    uint32_t idx = (uint32_t) vt;
    const char *name = one_const_names[idx];
    if (emitted_one_consts[idx])
        return name;

    jitc_vulkan_require_type(vt);
    switch (vt) {
        case VarType::Bool:
            fmt_tc("$s = OpConstantTrue %bool\n", name);
            break;
        case VarType::Float16:
            fmt_tc("$s = OpConstant %f16 !0x3c00\n", name);
            break;
        case VarType::Float32:
            fmt_tc("$s = OpConstant %f32 !0x3f800000\n", name);
            break;
        case VarType::Float64:
            fmt_tc("$s = OpConstant %f64 !0x0 !0x3ff00000\n", name);
            break;
        default:
            fmt_tc("$s = OpConstant $s 1\n", name, type_name_vulkan[idx]);
            break;
    }

    emitted_one_consts[idx] = true;
    return name;
}

static const char *jitc_vulkan_require_ones(VarType vt) {
    if (vt == VarType::Pointer)
        vt = VarType::UInt64;
    if (vt == VarType::Bool)
        return jitc_vulkan_require_one(VarType::Bool);

    uint32_t idx = (uint32_t) vt;
    const char *name = ones_const_names[idx];
    if (emitted_ones_consts[idx])
        return name;

    jitc_vulkan_require_type(vt);
    const char *val_str = nullptr;
    switch (type_size[idx]) {
        case 1: val_str = "0xff";               break;
        case 2: val_str = "0xffff";             break;
        case 4: val_str = "0xffffffff";         break;
        case 8: val_str = "0xffffffffffffffff"; break;
        default:
            jitc_fail("jitc_vulkan_require_ones(): unsupported type %u!", idx);
    }

    fmt_tc("$s = OpConstant $s $s\n", name, type_name_vulkan[idx], val_str);

    emitted_ones_consts[idx] = true;
    return name;
}

static const char *jitc_vulkan_require_nan(VarType vt) {
    uint32_t idx = (uint32_t) vt;
    const char *name = nullptr;
    const char *val_str = nullptr;
    switch (vt) {
        case VarType::Float16:
            name = "%c_nan_f16";
            val_str = "!0x7e00";
            break;
        case VarType::Float32:
            name = "%c_nan_f32";
            val_str = "!0x7fc00000";
            break;
        case VarType::Float64:
            name = "%c_nan_f64";
            val_str = "!0x0 !0x7ff80000";
            break;
        default:
            jitc_fail("jitc_vulkan_require_nan(): unsupported type %u!", idx);
    }

    if (emitted_nan_consts[idx])
        return name;

    jitc_vulkan_require_type(vt);
    fmt_tc("$s = OpConstant $s $s\n", name, type_name_vulkan[idx], val_str);

    emitted_nan_consts[idx] = true;
    return name;
}

static const char *jitc_vulkan_require_sign_mask(VarType vt) {
    VarType ut = jitc_vulkan_uint_type(vt);
    uint32_t idx = (uint32_t) ut;
    const char *name = nullptr;
    const char *val_str = nullptr;
    switch (ut) {
        case VarType::UInt16:
            name = "%c_sign_u16";
            val_str = "0x8000";
            break;
        case VarType::UInt32:
            name = "%c_sign_u32";
            val_str = "0x80000000";
            break;
        case VarType::UInt64:
            name = "%c_sign_u64";
            val_str = "0x8000000000000000";
            break;
        default:
            jitc_fail("jitc_vulkan_require_sign_mask(): unsupported type %u!", idx);
    }

    if (emitted_sign_consts[idx])
        return name;

    jitc_vulkan_require_type(ut);
    fmt_tc("$s = OpConstant $s $s\n", name, type_name_vulkan[idx], val_str);

    emitted_sign_consts[idx] = true;
    return name;
}

static const char *jitc_vulkan_require_mul_struct(VarType vt) {
    if (vt == VarType::Pointer)
        vt = VarType::UInt64;

    uint32_t idx = (uint32_t) vt;
    const char *name = mul_struct_names[idx];
    if (emitted_mul_structs[idx])
        return name;

    jitc_vulkan_require_type(vt);
    const char *tname = type_name_vulkan[idx];
    fmt_tc("$s = OpTypeStruct $s $s\n", name, tname, tname);

    emitted_mul_structs[idx] = true;
    return name;
}

// ----------------------------------------------------------------------------
//  Main kernel assembly routine
// ----------------------------------------------------------------------------

void jitc_vulkan_assemble(ThreadState *ts, ScheduledGroup group,
                          uint32_t /*n_regs*/, uint32_t n_params) {
    const VulkanDevice &dev = state.vulkan_devices[ts->device];
    active_vulkan_dev = &dev;

    jitc_vulkan_assemble_reset();

    uint32_t wg_size = std::min({ 256u,
                                  dev.max_compute_work_group_invocations,
                                  dev.max_compute_work_group_size[0] });
    if (unlikely(wg_size == 0))
        wg_size = 64u;

    bool params_in_pc =
        (size_t) n_params * sizeof(void *) <= dev.max_push_constants_size;
    uint32_t n_args = n_params > 1 ? n_params - 1 : 1;

    // 1. Baseline types, built-ins, and kernel parameter block
    jitc_vulkan_require_type(VarType::Void);
    jitc_vulkan_require_type(VarType::Bool);
    jitc_vulkan_require_type(VarType::UInt32);
    jitc_vulkan_require_type(VarType::UInt64);

    put_annot("OpDecorate %gl_GlobalInvocationID BuiltIn GlobalInvocationId\n"
              "OpDecorate %gl_NumWorkGroups BuiltIn NumWorkgroups\n"
              "OpDecorate %arr_u64 ArrayStride 8\n"
              "OpDecorate %Params Block\n"
              "OpMemberDecorate %Params 0 Offset 0\n"
              "OpMemberDecorate %Params 1 Offset 8\n");

    put_tc("%v3u32 = OpTypeVector %u32 3\n"
           "%ptr_in_v3u32 = OpTypePointer Input %v3u32\n"
           "%gl_GlobalInvocationID = OpVariable %ptr_in_v3u32 Input\n"
           "%gl_NumWorkGroups = OpVariable %ptr_in_v3u32 Input\n");

    jitc_vulkan_require_u32_const(0);
    jitc_vulkan_require_u32_const(1);
    jitc_vulkan_require_u32_const(wg_size);
    jitc_vulkan_require_u32_const(n_args);

    fmt_tc("%arr_u64 = OpTypeArray %u64 %c_u32_$u\n"
           "%Params = OpTypeStruct %u32 %arr_u64\n",
           n_args);

    const char *param_u64_ptr_type = nullptr;
    const char *param_u64_load_align = "";

    if (params_in_pc) {
        put_tc("%ptr_pc_Params = OpTypePointer PushConstant %Params\n"
               "%ptr_pc_u32 = OpTypePointer PushConstant %u32\n"
               "%ptr_pc_u64 = OpTypePointer PushConstant %u64\n"
               "%params = OpVariable %ptr_pc_Params PushConstant\n");
        param_u64_ptr_type = "%ptr_pc_u64";
    } else {
        put_annot("OpDecorate %PushConstants Block\n"
                  "OpMemberDecorate %PushConstants 0 Offset 0\n");
        jitc_vulkan_require_psb_ptr(VarType::UInt32);
        jitc_vulkan_require_psb_ptr(VarType::UInt64);
        put_tc("%ptr_psb_Params = OpTypePointer PhysicalStorageBuffer %Params\n"
               "%PushConstants = OpTypeStruct %ptr_psb_Params\n"
               "%ptr_pc_PushConstants = OpTypePointer PushConstant %PushConstants\n"
               "%ptr_pc_ptr_Params = OpTypePointer PushConstant %ptr_psb_Params\n"
               "%pc = OpVariable %ptr_pc_PushConstants PushConstant\n");
        param_u64_ptr_type = "%ptr_psb_u64";
        param_u64_load_align = " Aligned 8";
    }

    // 2. Load the call-data base pointer if indirect calls / getters are used
    if (call_buffer.base_v) {
        uint32_t arg_idx = call_buffer.base_param_index - 1;
        jitc_vulkan_require_u32_const(arg_idx);
        fmt("%p_r$u = OpAccessChain $s %params %c_u32_1 %c_u32_$u\n"
            "%r$u = OpLoad %u64 %p_r$u$s\n",
            call_buffer.base_reg, param_u64_ptr_type, arg_idx,
            call_buffer.base_reg, call_buffer.base_reg, param_u64_load_align);
    }

    // 3. Render every variable in the schedule
    for (uint32_t gi = group.start; gi != group.end; ++gi) {
        uint32_t index = schedule[gi].index;
        Variable *v = jitc_var(index);
        ParamType ptype = (ParamType) v->param_type;
        VarType vt = (VarType) v->type;

        if (likely(ptype == ParamType::Input)) {
            if (v->is_literal()) {
                if (vt == VarType::Pointer) {
                    if (v->resource_kind() != ResourceKind::Buffer)
                        jitc_raise("jitc_vulkan_assemble(): resource handles "
                                   "are not supported yet (the Vulkan backend "
                                   "cannot compile kernels with this operation "
                                   "yet)!");

                    uint32_t arg_idx =
                        v->param_offset / (uint32_t) sizeof(void *) - 1;
                    jitc_vulkan_require_u32_const(arg_idx);
                    fmt("%p_r$u = OpAccessChain $s %params %c_u32_1 %c_u32_$o\n"
                        "$v = OpLoad %u64 %p_r$u$s\n",
                        v->reg_index, param_u64_ptr_type, v,
                        v, v->reg_index, param_u64_load_align);
                } else {
                    jitc_vulkan_render(v);
                }
                continue;
            }

            if (unlikely(v->is_array()))
                jitc_raise("jitc_vulkan_assemble(): variable arrays are not "
                           "supported yet (the Vulkan backend cannot compile "
                           "kernels with this operation yet)!");

            uint32_t arg_idx = v->param_offset / (uint32_t) sizeof(void *) - 1;
            jitc_vulkan_require_u32_const(arg_idx);

            fmt("%p_r$u = OpAccessChain $s %params %c_u32_1 %c_u32_$o\n"
                "%a_r$u = OpLoad %u64 %p_r$u$s\n",
                v->reg_index, param_u64_ptr_type, v,
                v->reg_index, v->reg_index, param_u64_load_align);

            if (vt == VarType::Bool) {
                jitc_vulkan_require_bool_storage();
                fmt("%b_r$u = OpConvertUToPtr %ptr_psb_u8 %a_r$u\n",
                    v->reg_index, v->reg_index);
                if (v->size > 1) {
                    fmt("%e_r$u = OpPtrAccessChain %ptr_psb_u8 %b_r$u %idx\n"
                        "%raw_r$u = OpLoad %u8 %e_r$u Aligned 1\n",
                        v->reg_index, v->reg_index,
                        v->reg_index, v->reg_index);
                } else {
                    fmt("%raw_r$u = OpLoad %u8 %b_r$u Aligned 1\n",
                        v->reg_index, v->reg_index);
                }
                fmt("$v = OpINotEqual %bool %raw_r$u %c_u8_0\n",
                    v, v->reg_index);
            } else {
                const char *ptr_type = jitc_vulkan_require_psb_ptr(vt);
                fmt("%b_r$u = OpConvertUToPtr $s %a_r$u\n",
                    v->reg_index, ptr_type, v->reg_index);
                if (v->size > 1) {
                    fmt("%e_r$u = OpPtrAccessChain $s %b_r$u %idx\n"
                        "$v = OpLoad $t %e_r$u Aligned $a\n",
                        v->reg_index, ptr_type, v->reg_index,
                        v, v, v->reg_index, v);
                } else {
                    fmt("$v = OpLoad $t %b_r$u Aligned $a\n",
                        v, v, v->reg_index, v);
                }
            }
            continue;
        }

        // Regular / output variable
        jitc_vulkan_render(v);

        if (ptype == ParamType::Output) {
            if (unlikely(v->is_array()))
                jitc_raise("jitc_vulkan_assemble(): variable arrays are not "
                           "supported yet (the Vulkan backend cannot compile "
                           "kernels with this operation yet)!");

            uint32_t arg_idx = v->param_offset / (uint32_t) sizeof(void *) - 1;
            jitc_vulkan_require_u32_const(arg_idx);

            fmt("%out_p_r$u = OpAccessChain $s %params %c_u32_1 %c_u32_$o\n"
                "%out_a_r$u = OpLoad %u64 %out_p_r$u$s\n",
                v->reg_index, param_u64_ptr_type, v,
                v->reg_index, v->reg_index, param_u64_load_align);

            if (vt == VarType::Bool) {
                jitc_vulkan_require_bool_storage();
                fmt("%st_val_r$u = OpSelect %u8 $v %c_u8_1 %c_u8_0\n"
                    "%out_b_r$u = OpConvertUToPtr %ptr_psb_u8 %out_a_r$u\n"
                    "%out_e_r$u = OpPtrAccessChain %ptr_psb_u8 %out_b_r$u %idx\n"
                    "OpStore %out_e_r$u %st_val_r$u Aligned 1\n",
                    v->reg_index, v,
                    v->reg_index, v->reg_index,
                    v->reg_index, v->reg_index,
                    v->reg_index, v->reg_index);
            } else {
                const char *ptr_type = jitc_vulkan_require_psb_ptr(vt);
                fmt("%out_b_r$u = OpConvertUToPtr $s %out_a_r$u\n"
                    "%out_e_r$u = OpPtrAccessChain $s %out_b_r$u %idx\n"
                    "OpStore %out_e_r$u $v Aligned $a\n",
                    v->reg_index, ptr_type, v->reg_index,
                    v->reg_index, ptr_type, v->reg_index,
                    v->reg_index, v, v);
            }
        }
    }

    // 4. Emit the SPIR-V module header with all required capabilities
    put_header("OpCapability Shader\n");
    if (cap_int8)
        put_header("OpCapability Int8\n");
    if (cap_storage_8bit)
        put_header("OpCapability StorageBuffer8BitAccess\n");
    if (cap_int16)
        put_header("OpCapability Int16\n");
    if (cap_storage_16bit)
        put_header("OpCapability StorageBuffer16BitAccess\n");
    if (cap_float16)
        put_header("OpCapability Float16\n");
    if (cap_float64)
        put_header("OpCapability Float64\n");
    put_header("OpCapability Int64\n"
               "OpCapability PhysicalStorageBufferAddresses\n");

    if (need_glsl450)
        put_header("%glsl450 = OpExtInstImport \"GLSL.std.450\"\n");

    put_header("OpMemoryModel PhysicalStorageBuffer64 GLSL450\n");
    fmt_header("OpEntryPoint GLCompute %main "
               "\"drjit_^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\" "
               "%gl_GlobalInvocationID %gl_NumWorkGroups $s\n"
               "OpExecutionMode %main LocalSize $u 1 1\n",
               params_in_pc ? "%params" : "%pc", wg_size);

    // 5. Concatenate all SPIR-V sections in order into 'buffer'
    spv_body.swap(buffer);
    buffer.clear();

    put(spv_header.get(), spv_header.size());
    put(spv_annotations.get(), spv_annotations.size());
    put(spv_types_consts.get(), spv_types_consts.size());

    put("%main = OpFunction %void None %fn_void\n"
        "%entry = OpLabel\n");

    if (spv_func_locals.size() > 0)
        put(spv_func_locals.get(), spv_func_locals.size());

    fmt("%gid_vec = OpLoad %v3u32 %gl_GlobalInvocationID\n"
        "%gid = OpCompositeExtract %u32 %gid_vec 0\n"
        "%nwg_vec = OpLoad %v3u32 %gl_NumWorkGroups\n"
        "%nwg = OpCompositeExtract %u32 %nwg_vec 0\n"
        "%stride = OpIMul %u32 %nwg %c_u32_$u\n",
        wg_size);

    if (params_in_pc) {
        put("%size_ptr = OpAccessChain %ptr_pc_u32 %params %c_u32_0\n"
            "%size = OpLoad %u32 %size_ptr\n");
    } else {
        put("%params_ptr_ptr = OpAccessChain %ptr_pc_ptr_Params %pc %c_u32_0\n"
            "%params = OpLoad %ptr_psb_Params %params_ptr_ptr\n"
            "%size_ptr = OpAccessChain %ptr_psb_u32 %params %c_u32_0\n"
            "%size = OpLoad %u32 %size_ptr Aligned 4\n");
    }

    put("OpBranch %loop_head\n"
        "%loop_head = OpLabel\n"
        "%idx = OpPhi %u32 %gid %entry %idx_next %loop_cont\n"
        "%in_bounds = OpULessThan %bool %idx %size\n"
        "OpLoopMerge %loop_exit %loop_cont None\n"
        "OpBranchConditional %in_bounds %loop_body %loop_exit\n"
        "%loop_body = OpLabel\n");

    put(spv_body.get(), spv_body.size());

    put("OpBranch %loop_cont\n"
        "%loop_cont = OpLabel\n"
        "%idx_next = OpIAdd %u32 %idx %stride\n"
        "OpBranch %loop_head\n"
        "%loop_exit = OpLabel\n"
        "OpReturn\n"
        "OpFunctionEnd\n");
}

static void jitc_vulkan_render_round(Variable *v, Variable *a0, const char *op) {
    jitc_vulkan_require_type((VarType) a0->type);
    if (jitc_is_float(v)) {
        jitc_vulkan_require_glsl450();
        fmt("$v = OpExtInst $t %glsl450 $s $v\n", v, v, op, a0);
    } else if ((VarKind) v->kind == VarKind::Trunc) {
        fmt("$v = $s $t $v\n", v,
            jitc_is_uint(v) ? "OpConvertFToU" : "OpConvertFToS", v, a0);
    } else {
        jitc_vulkan_require_glsl450();
        fmt("$v_0 = OpExtInst $t %glsl450 $s $v\n"
            "$v = $s $t $v_0\n",
            v, a0, op, a0,
            v, jitc_is_uint(v) ? "OpConvertFToU" : "OpConvertFToS", v, v);
    }
}

static void jitc_vulkan_render_trig_exp(Variable *v, Variable *a0,
                                        const char *op) {
    jitc_vulkan_require_glsl450();
    if (jitc_is_double(v)) {
        // GLSL.std.450 transcendental instructions only accept 16/32-bit floats
        jitc_vulkan_require_type(VarType::Float32);
        fmt("$v_0 = OpFConvert %f32 $v\n"
            "$v_1 = OpExtInst %f32 %glsl450 $s $v_0\n"
            "$v = OpFConvert $t $v_1\n",
            v, a0, v, op, v, v, v, v);
    } else {
        fmt("$v = OpExtInst $t %glsl450 $s $v\n", v, v, op, a0);
    }
}

static void jitc_vulkan_render(Variable *v) {
    Variable *a0 = v->dep[0] ? jitc_var(v->dep[0]) : nullptr,
             *a1 = v->dep[1] ? jitc_var(v->dep[1]) : nullptr,
             *a2 = v->dep[2] ? jitc_var(v->dep[2]) : nullptr;

    VarType vt = (VarType) v->type;
    VarKind kind = (VarKind) v->kind;

    if (vt != VarType::Void)
        jitc_vulkan_require_type(vt);

    switch (kind) {
        case VarKind::Nop:
            break;

        case VarKind::Undefined:
            fmt_tc("$v = OpConstantNull $t\n", v, v);
            break;

        case VarKind::Literal:
            if (vt == VarType::Bool) {
                if (v->literal)
                    fmt_tc("$v = OpConstantTrue $t\n", v, v);
                else
                    fmt_tc("$v = OpConstantFalse $t\n", v, v);
            } else {
                fmt_tc("$v = OpConstant $t $l\n", v, v, v);
            }
            break;

        case VarKind::Counter:
            fmt("$v = OpCopyObject $t %idx\n", v, v);
            break;

        case VarKind::Neg:
            fmt(jitc_is_float(v) ? "$v = OpFNegate $t $v\n"
                                 : "$v = OpSNegate $t $v\n",
                v, v, a0);
            break;

        case VarKind::Not:
            if (jitc_is_bool(v)) {
                fmt("$v = OpLogicalNot $t $v\n", v, v, a0);
            } else if (jitc_is_float(v)) {
                jitc_vulkan_require_type(jitc_vulkan_uint_type(vt));
                fmt("$v_0 = OpBitcast $b $v\n"
                    "$v_1 = OpNot $b $v_0\n"
                    "$v = OpBitcast $t $v_1\n",
                    v, v, a0, v, v, v, v, v, v);
            } else {
                fmt("$v = OpNot $t $v\n", v, v, a0);
            }
            break;

        case VarKind::Sqrt:
        case VarKind::SqrtApprox:
            jitc_vulkan_require_glsl450();
            fmt("$v = OpExtInst $t %glsl450 Sqrt $v\n", v, v, a0);
            break;

        case VarKind::Abs:
            if (jitc_is_float(v)) {
                jitc_vulkan_require_glsl450();
                fmt("$v = OpExtInst $t %glsl450 FAbs $v\n", v, v, a0);
            } else if (jitc_is_sint(v)) {
                jitc_vulkan_require_glsl450();
                fmt("$v = OpExtInst $t %glsl450 SAbs $v\n", v, v, a0);
            } else {
                fmt("$v = OpCopyObject $t $v\n", v, v, a0);
            }
            break;

        case VarKind::Add:
            fmt(jitc_is_float(v) ? "$v = OpFAdd $t $v $v\n"
                                 : "$v = OpIAdd $t $v $v\n",
                v, v, a0, a1);
            break;

        case VarKind::Sub:
            fmt(jitc_is_float(v) ? "$v = OpFSub $t $v $v\n"
                                 : "$v = OpISub $t $v $v\n",
                v, v, a0, a1);
            break;

        case VarKind::Mul:
            fmt(jitc_is_float(v) ? "$v = OpFMul $t $v $v\n"
                                 : "$v = OpIMul $t $v $v\n",
                v, v, a0, a1);
            break;

        case VarKind::Div:
        case VarKind::DivApprox:
            if (jitc_is_float(v))
                fmt("$v = OpFDiv $t $v $v\n", v, v, a0, a1);
            else if (jitc_is_uint(v))
                fmt("$v = OpUDiv $t $v $v\n", v, v, a0, a1);
            else
                fmt("$v = OpSDiv $t $v $v\n", v, v, a0, a1);
            break;

        case VarKind::Mod:
            fmt(jitc_is_uint(v) ? "$v = OpUMod $t $v $v\n"
                                : "$v = OpSRem $t $v $v\n",
                v, v, a0, a1);
            break;

        case VarKind::MulHi: {
            const char *st_name = jitc_vulkan_require_mul_struct(vt);
            const char *op = jitc_is_uint(v) ? "OpUMulExtended"
                                             : "OpSMulExtended";
            fmt("$v_0 = $s $s $v $v\n"
                "$v = OpCompositeExtract $t $v_0 1\n",
                v, op, st_name, a0, a1,
                v, v, v);
            break;
        }

        case VarKind::MulWide: {
            jitc_vulkan_require_type((VarType) a0->type);
            const char *op = jitc_is_uint(v) ? "OpUConvert" : "OpSConvert";
            fmt("$v_0 = $s $t $v\n"
                "$v_1 = $s $t $v\n"
                "$v = OpIMul $t $v_0 $v_1\n",
                v, op, v, a0,
                v, op, v, a1,
                v, v, v, v);
            break;
        }

        case VarKind::Fma:
            if (jitc_is_float(v)) {
                jitc_vulkan_require_glsl450();
                fmt("$v = OpExtInst $t %glsl450 Fma $v $v $v\n",
                    v, v, a0, a1, a2);
            } else {
                fmt("$v_0 = OpIMul $t $v $v\n"
                    "$v = OpIAdd $t $v_0 $v\n",
                    v, v, a0, a1,
                    v, v, v, a2);
            }
            break;

        case VarKind::Min:
        case VarKind::Max: {
            bool is_min = kind == VarKind::Min;
            jitc_vulkan_require_glsl450();
            if (jitc_is_float(v)) {
                const char *fn = is_min ? "FMin" : "FMax";
                const char *nan_c = jitc_vulkan_require_nan(vt);
                jitc_vulkan_require_type(VarType::Bool);
                fmt("$v_0 = OpExtInst $t %glsl450 $s $v $v\n"
                    "$v_1 = OpIsNan %bool $v\n"
                    "$v_2 = OpIsNan %bool $v\n"
                    "$v_3 = OpLogicalOr %bool $v_1 $v_2\n"
                    "$v = OpSelect $t $v_3 $s $v_0\n",
                    v, v, fn, a0, a1,
                    v, a0,
                    v, a1,
                    v, v, v,
                    v, v, v, nan_c, v);
            } else {
                const char *fn = jitc_is_uint(v) ? (is_min ? "UMin" : "UMax")
                                                 : (is_min ? "SMin" : "SMax");
                fmt("$v = OpExtInst $t %glsl450 $s $v $v\n",
                    v, v, fn, a0, a1);
            }
            break;
        }

        case VarKind::FMin:
        case VarKind::FMax: {
            jitc_vulkan_require_glsl450();
            const char *fn = kind == VarKind::FMin ? "NMin" : "NMax";
            fmt("$v = OpExtInst $t %glsl450 $s $v $v\n", v, v, fn, a0, a1);
            break;
        }

        case VarKind::Copysign: {
            const char *sign_mask = jitc_vulkan_require_sign_mask(vt);
            fmt("$v_0 = OpBitcast $b $v\n"
                "$v_1 = OpBitcast $b $v\n"
                "$v_2 = OpBitwiseXor $b $v_0 $v_1\n"
                "$v_3 = OpBitwiseAnd $b $v_2 $s\n"
                "$v_4 = OpBitwiseXor $b $v_0 $v_3\n"
                "$v = OpBitcast $t $v_4\n",
                v, v, a0,
                v, v, a1,
                v, v, v, v,
                v, v, v, sign_mask,
                v, v, v, v,
                v, v, v);
            break;
        }

        case VarKind::Ceil:
            jitc_vulkan_render_round(v, a0, "Ceil");
            break;

        case VarKind::Floor:
            jitc_vulkan_render_round(v, a0, "Floor");
            break;

        case VarKind::Round:
            jitc_vulkan_render_round(v, a0, "RoundEven");
            break;

        case VarKind::Trunc:
            jitc_vulkan_render_round(v, a0, "Trunc");
            break;

        case VarKind::Eq:
            jitc_vulkan_require_type((VarType) a0->type);
            if (jitc_is_float(a0))
                fmt("$v = OpFOrdEqual $t $v $v\n", v, v, a0, a1);
            else if (jitc_is_bool(a0))
                fmt("$v = OpLogicalEqual $t $v $v\n", v, v, a0, a1);
            else
                fmt("$v = OpIEqual $t $v $v\n", v, v, a0, a1);
            break;

        case VarKind::Neq:
            jitc_vulkan_require_type((VarType) a0->type);
            if (jitc_is_float(a0))
                fmt("$v = OpFOrdNotEqual $t $v $v\n", v, v, a0, a1);
            else if (jitc_is_bool(a0))
                fmt("$v = OpLogicalNotEqual $t $v $v\n", v, v, a0, a1);
            else
                fmt("$v = OpINotEqual $t $v $v\n", v, v, a0, a1);
            break;

        case VarKind::Lt:
            jitc_vulkan_require_type((VarType) a0->type);
            if (jitc_is_float(a0))
                fmt("$v = OpFOrdLessThan $t $v $v\n", v, v, a0, a1);
            else if (jitc_is_uint(a0))
                fmt("$v = OpULessThan $t $v $v\n", v, v, a0, a1);
            else if (jitc_is_sint(a0))
                fmt("$v = OpSLessThan $t $v $v\n", v, v, a0, a1);
            else
                fmt("$v_0 = OpLogicalNot $t $v\n"
                    "$v = OpLogicalAnd $t $v_0 $v\n",
                    v, v, a0, v, v, v, a1);
            break;

        case VarKind::Le:
            jitc_vulkan_require_type((VarType) a0->type);
            if (jitc_is_float(a0))
                fmt("$v = OpFOrdLessThanEqual $t $v $v\n", v, v, a0, a1);
            else if (jitc_is_uint(a0))
                fmt("$v = OpULessThanEqual $t $v $v\n", v, v, a0, a1);
            else if (jitc_is_sint(a0))
                fmt("$v = OpSLessThanEqual $t $v $v\n", v, v, a0, a1);
            else
                fmt("$v_0 = OpLogicalNot $t $v\n"
                    "$v = OpLogicalOr $t $v_0 $v\n",
                    v, v, a0, v, v, v, a1);
            break;

        case VarKind::Gt:
            jitc_vulkan_require_type((VarType) a0->type);
            if (jitc_is_float(a0))
                fmt("$v = OpFOrdGreaterThan $t $v $v\n", v, v, a0, a1);
            else if (jitc_is_uint(a0))
                fmt("$v = OpUGreaterThan $t $v $v\n", v, v, a0, a1);
            else if (jitc_is_sint(a0))
                fmt("$v = OpSGreaterThan $t $v $v\n", v, v, a0, a1);
            else
                fmt("$v_0 = OpLogicalNot $t $v\n"
                    "$v = OpLogicalAnd $t $v $v_0\n",
                    v, v, a1, v, v, a0, v);
            break;

        case VarKind::Ge:
            jitc_vulkan_require_type((VarType) a0->type);
            if (jitc_is_float(a0))
                fmt("$v = OpFOrdGreaterThanEqual $t $v $v\n", v, v, a0, a1);
            else if (jitc_is_uint(a0))
                fmt("$v = OpUGreaterThanEqual $t $v $v\n", v, v, a0, a1);
            else if (jitc_is_sint(a0))
                fmt("$v = OpSGreaterThanEqual $t $v $v\n", v, v, a0, a1);
            else
                fmt("$v_0 = OpLogicalNot $t $v\n"
                    "$v = OpLogicalOr $t $v $v_0\n",
                    v, v, a1, v, v, a0, v);
            break;

        case VarKind::Select:
            jitc_vulkan_require_type(VarType::Bool);
            fmt("$v = OpSelect $t $v $v $v\n", v, v, a0, a1, a2);
            break;

        case VarKind::Popc: {
            jitc_vulkan_require_type(VarType::UInt32);
            uint32_t sz = type_size[v->type];
            if (sz <= 2) {
                if (jitc_is_uint(v)) {
                    fmt("$v_0 = OpUConvert %u32 $v\n", v, a0);
                } else {
                    jitc_vulkan_require_type(jitc_vulkan_uint_type(vt));
                    fmt("$v_u = OpBitcast $b $v\n"
                        "$v_0 = OpUConvert %u32 $v_u\n",
                        v, v, a0, v, v);
                }
                fmt("$v_1 = OpBitCount %u32 $v_0\n"
                    "$v = $s $t $v_1\n",
                    v, v,
                    v, jitc_is_uint(v) ? "OpUConvert" : "OpSConvert", v, v);
            } else if (sz == 4) {
                fmt("$v = OpBitCount $t $v\n", v, v, a0);
            } else {
                jitc_vulkan_require_type(VarType::UInt64);
                jitc_vulkan_require_u32_const(32);
                if (jitc_is_uint(v))
                    fmt("$v_u = OpCopyObject %u64 $v\n", v, a0);
                else
                    fmt("$v_u = OpBitcast %u64 $v\n", v, a0);
                fmt("$v_lo = OpUConvert %u32 $v_u\n"
                    "$v_hi64 = OpShiftRightLogical %u64 $v_u %c_u32_32\n"
                    "$v_hi = OpUConvert %u32 $v_hi64\n"
                    "$v_clo = OpBitCount %u32 $v_lo\n"
                    "$v_chi = OpBitCount %u32 $v_hi\n"
                    "$v_c32 = OpIAdd %u32 $v_clo $v_chi\n",
                    v, v, v, v, v, v, v, v, v, v, v, v, v);
                if (jitc_is_uint(v))
                    fmt("$v = OpUConvert %u64 $v_c32\n", v, v);
                else
                    fmt("$v_c64 = OpUConvert %u64 $v_c32\n"
                        "$v = OpBitcast %i64 $v_c64\n",
                        v, v, v, v);
            }
            break;
        }

        case VarKind::Brev: {
            jitc_vulkan_require_type(VarType::UInt32);
            uint32_t sz = type_size[v->type];
            if (sz <= 2) {
                uint32_t shift = 32 - sz * 8;
                jitc_vulkan_require_u32_const(shift);
                if (jitc_is_uint(v)) {
                    fmt("$v_0 = OpUConvert %u32 $v\n", v, a0);
                } else {
                    jitc_vulkan_require_type(jitc_vulkan_uint_type(vt));
                    fmt("$v_u = OpBitcast $b $v\n"
                        "$v_0 = OpUConvert %u32 $v_u\n",
                        v, v, a0, v, v);
                }
                fmt("$v_1 = OpBitReverse %u32 $v_0\n"
                    "$v_2 = OpShiftRightLogical %u32 $v_1 %c_u32_$u\n"
                    "$v = $s $t $v_2\n",
                    v, v, v, v, shift,
                    v, jitc_is_uint(v) ? "OpUConvert" : "OpSConvert", v, v);
            } else if (sz == 4) {
                fmt("$v = OpBitReverse $t $v\n", v, v, a0);
            } else {
                jitc_vulkan_require_type(VarType::UInt64);
                jitc_vulkan_require_u32_const(32);
                if (jitc_is_uint(v))
                    fmt("$v_u = OpCopyObject %u64 $v\n", v, a0);
                else
                    fmt("$v_u = OpBitcast %u64 $v\n", v, a0);
                fmt("$v_lo = OpUConvert %u32 $v_u\n"
                    "$v_hi64 = OpShiftRightLogical %u64 $v_u %c_u32_32\n"
                    "$v_hi = OpUConvert %u32 $v_hi64\n"
                    "$v_rlo = OpBitReverse %u32 $v_lo\n"
                    "$v_rhi = OpBitReverse %u32 $v_hi\n"
                    "$v_rlo64 = OpUConvert %u64 $v_rlo\n"
                    "$v_rhi64 = OpUConvert %u64 $v_rhi\n"
                    "$v_rhi_s = OpShiftLeftLogical %u64 $v_rlo64 %c_u32_32\n",
                    v, v, v, v, v, v, v, v, v, v, v, v, v, v, v, v);
                if (jitc_is_uint(v))
                    fmt("$v = OpBitwiseOr %u64 $v_rhi_s $v_rhi64\n", v, v, v);
                else
                    fmt("$v_res = OpBitwiseOr %u64 $v_rhi_s $v_rhi64\n"
                        "$v = OpBitcast %i64 $v_res\n",
                        v, v, v, v, v);
            }
            break;
        }

        case VarKind::Clz: {
            jitc_vulkan_require_glsl450();
            jitc_vulkan_require_type(VarType::UInt32);
            uint32_t sz = type_size[v->type];
            if (sz <= 2) {
                uint32_t max_bit = sz * 8 - 1;
                jitc_vulkan_require_u32_const(max_bit);
                if (jitc_is_uint(v)) {
                    fmt("$v_0 = OpUConvert %u32 $v\n", v, a0);
                } else {
                    jitc_vulkan_require_type(jitc_vulkan_uint_type(vt));
                    fmt("$v_u = OpBitcast $b $v\n"
                        "$v_0 = OpUConvert %u32 $v_u\n",
                        v, v, a0, v, v);
                }
                fmt("$v_1 = OpExtInst %u32 %glsl450 FindUMsb $v_0\n"
                    "$v_2 = OpISub %u32 %c_u32_$u $v_1\n"
                    "$v = $s $t $v_2\n",
                    v, v, v, max_bit, v,
                    v, jitc_is_uint(v) ? "OpUConvert" : "OpSConvert", v, v);
            } else if (sz == 4) {
                jitc_vulkan_require_u32_const(31);
                if (jitc_is_uint(v)) {
                    fmt("$v_0 = OpExtInst %u32 %glsl450 FindUMsb $v\n"
                        "$v = OpISub %u32 %c_u32_31 $v_0\n",
                        v, a0, v, v);
                } else {
                    fmt("$v_0 = OpBitcast %u32 $v\n"
                        "$v_1 = OpExtInst %u32 %glsl450 FindUMsb $v_0\n"
                        "$v_2 = OpISub %u32 %c_u32_31 $v_1\n"
                        "$v = OpBitcast %i32 $v_2\n",
                        v, a0, v, v, v, v, v, v);
                }
            } else {
                jitc_vulkan_require_type(VarType::Bool);
                jitc_vulkan_require_type(VarType::UInt64);
                jitc_vulkan_require_u32_const(0);
                jitc_vulkan_require_u32_const(31);
                jitc_vulkan_require_u32_const(32);
                jitc_vulkan_require_u32_const(63);
                if (jitc_is_uint(v))
                    fmt("$v_u = OpCopyObject %u64 $v\n", v, a0);
                else
                    fmt("$v_u = OpBitcast %u64 $v\n", v, a0);
                fmt("$v_lo = OpUConvert %u32 $v_u\n"
                    "$v_hi64 = OpShiftRightLogical %u64 $v_u %c_u32_32\n"
                    "$v_hi = OpUConvert %u32 $v_hi64\n"
                    "$v_mhi = OpExtInst %u32 %glsl450 FindUMsb $v_hi\n"
                    "$v_mlo = OpExtInst %u32 %glsl450 FindUMsb $v_lo\n"
                    "$v_chi = OpISub %u32 %c_u32_31 $v_mhi\n"
                    "$v_clo = OpISub %u32 %c_u32_63 $v_mlo\n"
                    "$v_hnz = OpINotEqual %bool $v_hi %c_u32_0\n"
                    "$v_c32 = OpSelect %u32 $v_hnz $v_chi $v_clo\n",
                    v, v, v, v, v, v, v, v, v, v, v, v, v, v, v, v, v, v, v, v);
                if (jitc_is_uint(v))
                    fmt("$v = OpUConvert %u64 $v_c32\n", v, v);
                else
                    fmt("$v_c64 = OpUConvert %u64 $v_c32\n"
                        "$v = OpBitcast %i64 $v_c64\n",
                        v, v, v, v);
            }
            break;
        }

        case VarKind::Ctz: {
            jitc_vulkan_require_glsl450();
            jitc_vulkan_require_type(VarType::UInt32);
            uint32_t sz = type_size[v->type];
            if (sz <= 2) {
                uint32_t sentinel = 1u << (sz * 8);
                jitc_vulkan_require_u32_const(sentinel);
                if (jitc_is_uint(v)) {
                    fmt("$v_0 = OpUConvert %u32 $v\n", v, a0);
                } else {
                    jitc_vulkan_require_type(jitc_vulkan_uint_type(vt));
                    fmt("$v_u = OpBitcast $b $v\n"
                        "$v_0 = OpUConvert %u32 $v_u\n",
                        v, v, a0, v, v);
                }
                fmt("$v_1 = OpBitwiseOr %u32 $v_0 %c_u32_$u\n"
                    "$v_2 = OpExtInst %u32 %glsl450 FindILsb $v_1\n"
                    "$v = $s $t $v_2\n",
                    v, v, sentinel, v, v,
                    v, jitc_is_uint(v) ? "OpUConvert" : "OpSConvert", v, v);
            } else if (sz == 4) {
                jitc_vulkan_require_type(VarType::Bool);
                jitc_vulkan_require_u32_const(0);
                jitc_vulkan_require_u32_const(32);
                if (jitc_is_uint(v)) {
                    fmt("$v_0 = OpExtInst %u32 %glsl450 FindILsb $v\n"
                        "$v_1 = OpIEqual %bool $v %c_u32_0\n"
                        "$v = OpSelect %u32 $v_1 %c_u32_32 $v_0\n",
                        v, a0, v, a0, v, v, v);
                } else {
                    fmt("$v_0 = OpBitcast %u32 $v\n"
                        "$v_1 = OpExtInst %u32 %glsl450 FindILsb $v_0\n"
                        "$v_2 = OpIEqual %bool $v_0 %c_u32_0\n"
                        "$v_3 = OpSelect %u32 $v_2 %c_u32_32 $v_1\n"
                        "$v = OpBitcast %i32 $v_3\n",
                        v, a0, v, v, v, v, v, v, v, v, v);
                }
            } else {
                jitc_vulkan_require_type(VarType::Bool);
                jitc_vulkan_require_type(VarType::UInt64);
                jitc_vulkan_require_u32_const(0);
                jitc_vulkan_require_u32_const(32);
                jitc_vulkan_require_u32_const(64);
                if (jitc_is_uint(v))
                    fmt("$v_u = OpCopyObject %u64 $v\n", v, a0);
                else
                    fmt("$v_u = OpBitcast %u64 $v\n", v, a0);
                fmt("$v_lo = OpUConvert %u32 $v_u\n"
                    "$v_hi64 = OpShiftRightLogical %u64 $v_u %c_u32_32\n"
                    "$v_hi = OpUConvert %u32 $v_hi64\n"
                    "$v_llo = OpExtInst %u32 %glsl450 FindILsb $v_lo\n"
                    "$v_lhi = OpExtInst %u32 %glsl450 FindILsb $v_hi\n"
                    "$v_chi0 = OpIAdd %u32 $v_lhi %c_u32_32\n"
                    "$v_hz = OpIEqual %bool $v_hi %c_u32_0\n"
                    "$v_chi = OpSelect %u32 $v_hz %c_u32_64 $v_chi0\n"
                    "$v_lnz = OpINotEqual %bool $v_lo %c_u32_0\n"
                    "$v_c32 = OpSelect %u32 $v_lnz $v_llo $v_chi\n",
                    v, v, v, v, v, v, v, v, v, v,
                    v, v, v, v, v, v, v, v, v, v,
                    v, v, v);
                if (jitc_is_uint(v))
                    fmt("$v = OpUConvert %u64 $v_c32\n", v, v);
                else
                    fmt("$v_c64 = OpUConvert %u64 $v_c32\n"
                        "$v = OpBitcast %i64 $v_c64\n",
                        v, v, v, v);
            }
            break;
        }

        case VarKind::And:
            if (a0->type != a1->type) {
                jitc_vulkan_require_type(VarType::Bool);
                const char *zero_c = jitc_vulkan_require_zero(vt);
                fmt("$v = OpSelect $t $v $v $s\n", v, v, a1, a0, zero_c);
            } else if (jitc_is_bool(v)) {
                fmt("$v = OpLogicalAnd $t $v $v\n", v, v, a0, a1);
            } else if (jitc_is_float(v)) {
                jitc_vulkan_require_type(jitc_vulkan_uint_type(vt));
                fmt("$v_0 = OpBitcast $b $v\n"
                    "$v_1 = OpBitcast $b $v\n"
                    "$v_2 = OpBitwiseAnd $b $v_0 $v_1\n"
                    "$v = OpBitcast $t $v_2\n",
                    v, v, a0, v, v, a1, v, v, v, v, v, v, v);
            } else {
                fmt("$v = OpBitwiseAnd $t $v $v\n", v, v, a0, a1);
            }
            break;

        case VarKind::Or:
            if (a0->type != a1->type) {
                jitc_vulkan_require_type(VarType::Bool);
                if (jitc_is_float(v)) {
                    VarType ut = jitc_vulkan_uint_type(vt);
                    const char *ones_c = jitc_vulkan_require_ones(ut);
                    fmt("$v_0 = OpBitcast $b $v\n"
                        "$v_1 = OpSelect $b $v $s $v_0\n"
                        "$v = OpBitcast $t $v_1\n",
                        v, v, a0, v, v, a1, ones_c, v, v, v, v);
                } else {
                    const char *ones_c = jitc_vulkan_require_ones(vt);
                    fmt("$v = OpSelect $t $v $s $v\n", v, v, a1, ones_c, a0);
                }
            } else if (jitc_is_bool(v)) {
                fmt("$v = OpLogicalOr $t $v $v\n", v, v, a0, a1);
            } else if (jitc_is_float(v)) {
                jitc_vulkan_require_type(jitc_vulkan_uint_type(vt));
                fmt("$v_0 = OpBitcast $b $v\n"
                    "$v_1 = OpBitcast $b $v\n"
                    "$v_2 = OpBitwiseOr $b $v_0 $v_1\n"
                    "$v = OpBitcast $t $v_2\n",
                    v, v, a0, v, v, a1, v, v, v, v, v, v, v);
            } else {
                fmt("$v = OpBitwiseOr $t $v $v\n", v, v, a0, a1);
            }
            break;

        case VarKind::Xor:
            if (jitc_is_bool(v)) {
                fmt("$v = OpLogicalNotEqual $t $v $v\n", v, v, a0, a1);
            } else if (jitc_is_float(v)) {
                jitc_vulkan_require_type(jitc_vulkan_uint_type(vt));
                fmt("$v_0 = OpBitcast $b $v\n"
                    "$v_1 = OpBitcast $b $v\n"
                    "$v_2 = OpBitwiseXor $b $v_0 $v_1\n"
                    "$v = OpBitcast $t $v_2\n",
                    v, v, a0, v, v, a1, v, v, v, v, v, v, v);
            } else {
                fmt("$v = OpBitwiseXor $t $v $v\n", v, v, a0, a1);
            }
            break;

        case VarKind::Shl:
            fmt("$v = OpShiftLeftLogical $t $v $v\n", v, v, a0, a1);
            break;

        case VarKind::Shr:
            fmt(jitc_is_uint(v) ? "$v = OpShiftRightLogical $t $v $v\n"
                                : "$v = OpShiftRightArithmetic $t $v $v\n",
                v, v, a0, a1);
            break;

        case VarKind::Rcp:
        case VarKind::RcpApprox: {
            const char *one_c = jitc_vulkan_require_one(vt);
            fmt("$v = OpFDiv $t $s $v\n", v, v, one_c, a0);
            break;
        }

        case VarKind::RSqrtApprox:
            jitc_vulkan_require_glsl450();
            fmt("$v = OpExtInst $t %glsl450 InverseSqrt $v\n", v, v, a0);
            break;

        case VarKind::Sin:
            jitc_vulkan_render_trig_exp(v, a0, "Sin");
            break;

        case VarKind::Cos:
            jitc_vulkan_render_trig_exp(v, a0, "Cos");
            break;

        case VarKind::Exp2:
            jitc_vulkan_render_trig_exp(v, a0, "Exp2");
            break;

        case VarKind::Log2:
            jitc_vulkan_render_trig_exp(v, a0, "Log2");
            break;

        case VarKind::Tanh:
            jitc_vulkan_render_trig_exp(v, a0, "Tanh");
            break;

        case VarKind::Cast: {
            VarType src_vt = (VarType) a0->type;
            jitc_vulkan_require_type(src_vt);

            if (jitc_is_bool(v)) {
                const char *zero_c = jitc_vulkan_require_zero(src_vt);
                fmt(jitc_is_float(a0) ? "$v = OpFOrdNotEqual %bool $v $s\n"
                                      : "$v = OpINotEqual %bool $v $s\n",
                    v, a0, zero_c);
            } else if (jitc_is_bool(a0)) {
                const char *one_c  = jitc_vulkan_require_one(vt);
                const char *zero_c = jitc_vulkan_require_zero(vt);
                fmt("$v = OpSelect $t $v $s $s\n", v, v, a0, one_c, zero_c);
            } else if (jitc_is_float(v) && jitc_is_float(a0)) {
                fmt("$v = OpFConvert $t $v\n", v, v, a0);
            } else if (jitc_is_float(v) && !jitc_is_float(a0)) {
                fmt(jitc_is_uint(a0) ? "$v = OpConvertUToF $t $v\n"
                                     : "$v = OpConvertSToF $t $v\n",
                    v, v, a0);
            } else if (!jitc_is_float(v) && jitc_is_float(a0)) {
                fmt(jitc_is_uint(v) ? "$v = OpConvertFToU $t $v\n"
                                    : "$v = OpConvertFToS $t $v\n",
                    v, v, a0);
            } else if (type_size[v->type] < type_size[a0->type]) {
                fmt(jitc_is_uint(v) ? "$v = OpUConvert $t $v\n"
                                    : "$v = OpSConvert $t $v\n",
                    v, v, a0);
            } else if (type_size[v->type] > type_size[a0->type]) {
                if (jitc_is_uint(a0)) {
                    if (jitc_is_uint(v)) {
                        fmt("$v = OpUConvert $t $v\n", v, v, a0);
                    } else {
                        VarType ut = jitc_vulkan_uint_type(vt);
                        jitc_vulkan_require_type(ut);
                        fmt("$v_0 = OpUConvert $b $v\n"
                            "$v = OpBitcast $t $v_0\n",
                            v, v, a0, v, v, v);
                    }
                } else {
                    if (jitc_is_sint(v)) {
                        fmt("$v = OpSConvert $t $v\n", v, v, a0);
                    } else {
                        VarType st = jitc_vulkan_sint_type(vt);
                        jitc_vulkan_require_type(st);
                        fmt("$v_0 = OpSConvert $s $v\n"
                            "$v = OpBitcast $t $v_0\n",
                            v, type_name_vulkan[(uint32_t) st], a0, v, v, v);
                    }
                }
            } else {
                if (strcmp(type_name_vulkan[v->type],
                           type_name_vulkan[a0->type]) == 0)
                    fmt("$v = OpCopyObject $t $v\n", v, v, a0);
                else
                    fmt("$v = OpBitcast $t $v\n", v, v, a0);
            }
            break;
        }

        case VarKind::Bitcast:
            jitc_vulkan_require_type((VarType) a0->type);
            if (strcmp(type_name_vulkan[v->type],
                       type_name_vulkan[a0->type]) == 0)
                fmt("$v = OpCopyObject $t $v\n", v, v, a0);
            else
                fmt("$v = OpBitcast $t $v\n", v, v, a0);
            break;

        default:
            jitc_raise("jitc_vulkan_render(): unhandled variable kind \"%s\" "
                       "(the Vulkan backend cannot compile kernels with this "
                       "operation yet)!",
                       var_kind_name[v->kind]);
    }
}
