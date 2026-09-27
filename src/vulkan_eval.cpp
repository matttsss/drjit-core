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
#include "op.h"
#include "call.h"
#include "cond.h"
#include "loop.h"
#include "array.h"
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

enum EmittedFlag : uint16_t {
    EmittedType       = 1u << 0,
    EmittedPsbPtr     = 1u << 1,
    EmittedFnPtr      = 1u << 2,
    EmittedZero       = 1u << 3,
    EmittedOne        = 1u << 4,
    EmittedOnes       = 1u << 5,
    EmittedNan        = 1u << 6,
    EmittedSign       = 1u << 7,
    EmittedMulStruct  = 1u << 8,
    EmittedVec2Type   = 1u << 9,
    EmittedVec4Type   = 1u << 10,
    EmittedPsbVec2Ptr = 1u << 11,
    EmittedPsbVec4Ptr = 1u << 12,
    EmittedVec2Zero   = 1u << 13,
    EmittedVec4Zero   = 1u << 14,
};

enum SpvFlag : uint32_t {
    SpvCapInt8           = 1u << 0,
    SpvCapStorage8Bit    = 1u << 1,
    SpvCapInt16          = 1u << 2,
    SpvCapStorage16Bit   = 1u << 3,
    SpvCapFloat16        = 1u << 4,
    SpvCapFloat64        = 1u << 5,
    SpvCapInt64Atomics   = 1u << 6,
    SpvCapAtomicF32Add   = 1u << 7,
    SpvCapAtomicF64Add   = 1u << 8,
    SpvCapSubgroup       = 1u << 9,
    SpvNeedGlsl450       = 1u << 10,
    SpvBoolStorageConsts = 1u << 11,
};

static uint16_t emitted_flags[(size_t) VarType::Count];
static uint32_t spv_flags = 0;
static tsl::robin_set<uint32_t> emitted_u32_consts;
static tsl::robin_set<uint32_t> emitted_fn_arr_types;
static tsl::robin_set<XXH128_hash_t, XXH128Hasher, XXH128Eq> emitted_lit_consts;
static uint32_t spv_aux_id = 0;
static uint32_t spv_func_id = 0;

static char current_block_label[64];

static bool jitc_vulkan_mark_emitted(uint32_t idx, EmittedFlag flag) {
    bool already = (emitted_flags[idx] & flag) != 0;
    emitted_flags[idx] |= flag;
    return already;
}

static const char *jitc_vulkan_id(const char *prefix, uint32_t idx) {
    static char bufs[16][24];
    static uint32_t slot = 0;
    char *buf = bufs[slot++ & 15];
    snprintf(buf, sizeof(bufs[0]), "%s%s", prefix, type_name_vulkan[idx] + 1);
    return buf;
}

static void jitc_vulkan_set_label(const char *prefix, uint32_t reg_index) {
    snprintf(current_block_label, sizeof(current_block_label),
             "%%%s_r%u", prefix, reg_index);
}

void jitc_vulkan_assemble_reset() {
    spv_header.clear();
    spv_annotations.clear();
    spv_types_consts.clear();
    spv_func_locals.clear();
    spv_body.clear();

    memset(emitted_flags, 0, sizeof(emitted_flags));
    spv_flags = 0;
    emitted_u32_consts.clear();
    emitted_fn_arr_types.clear();
    emitted_lit_consts.clear();
    spv_aux_id = 0;
    spv_func_id = 0;

    snprintf(current_block_label, sizeof(current_block_label), "%%loop_body");
}

void jitc_vulkan_require_glsl450() {
    spv_flags |= SpvNeedGlsl450;
}

void jitc_vulkan_require_type(VarType vt) {
    if (vt == VarType::Pointer)
        vt = VarType::UInt64;

    uint32_t idx = (uint32_t) vt;
    if (jitc_vulkan_mark_emitted(idx, EmittedType))
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
        case VarType::UInt8:
            if (unlikely(!dev.shader_int8))
                jitc_raise("jit_eval(): the Vulkan device \"%s\" does not "
                           "support 8-bit integers (shaderInt8)!", dev.name);
            spv_flags |= SpvCapInt8;
            fmt_tc("$s = OpTypeInt 8 $u\n", type_name_vulkan[idx],
                   vt == VarType::Int8 ? 1u : 0u);
            break;

        case VarType::Int16:
        case VarType::UInt16:
            if (unlikely(!dev.shader_int16))
                jitc_raise("jit_eval(): the Vulkan device \"%s\" does not "
                           "support 16-bit integers (shaderInt16)!", dev.name);
            spv_flags |= SpvCapInt16;
            fmt_tc("$s = OpTypeInt 16 $u\n", type_name_vulkan[idx],
                   vt == VarType::Int16 ? 1u : 0u);
            break;

        case VarType::Int32:
        case VarType::UInt32:
            fmt_tc("$s = OpTypeInt 32 $u\n", type_name_vulkan[idx],
                   vt == VarType::Int32 ? 1u : 0u);
            break;

        case VarType::Int64:
        case VarType::UInt64:
            fmt_tc("$s = OpTypeInt 64 $u\n", type_name_vulkan[idx],
                   vt == VarType::Int64 ? 1u : 0u);
            break;

        case VarType::Float16:
            if (unlikely(!dev.shader_float16))
                jitc_raise("jit_eval(): the Vulkan device \"%s\" does not "
                           "support 16-bit floating-point arithmetic (shaderFloat16)!",
                           dev.name);
            spv_flags |= SpvCapFloat16;
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
            spv_flags |= SpvCapFloat64;
            put_tc("%f64 = OpTypeFloat 64\n");
            break;

        default:
            jitc_fail("jitc_vulkan_require_type(): unsupported type %u!", idx);
    }
}

const char *jitc_vulkan_require_psb_ptr(VarType vt) {
    if (vt == VarType::Bool)
        vt = VarType::UInt8;
    else if (vt == VarType::Pointer)
        vt = VarType::UInt64;

    uint32_t idx = (uint32_t) vt;
    const char *ptr_name = jitc_vulkan_id("%ptr_psb_", idx);
    if (jitc_vulkan_mark_emitted(idx, EmittedPsbPtr))
        return ptr_name;

    jitc_vulkan_require_type(vt);

    const VulkanDevice &dev = *active_vulkan_dev;
    uint32_t elem_size = type_size[idx];
    if (elem_size == 1) {
        if (unlikely(!dev.storage_buffer_8bit))
            jitc_raise("jit_eval(): the Vulkan device \"%s\" does not "
                       "support 8-bit storage buffer access "
                       "(storageBuffer8BitAccess)!", dev.name);
        spv_flags |= SpvCapStorage8Bit;
    } else if (elem_size == 2) {
        if (unlikely(!dev.storage_buffer_16bit))
            jitc_raise("jit_eval(): the Vulkan device \"%s\" does not "
                       "support 16-bit storage buffer access "
                       "(storageBuffer16BitAccess)!", dev.name);
        spv_flags |= SpvCapStorage16Bit;
    }

    fmt_annot("OpDecorate $s ArrayStride $u\n", ptr_name, elem_size);
    fmt_tc("$s = OpTypePointer PhysicalStorageBuffer $s\n",
           ptr_name, type_name_vulkan[idx]);
    return ptr_name;
}

const char *jitc_vulkan_require_fn_ptr(VarType vt) {
    if (vt == VarType::Pointer)
        vt = VarType::UInt64;

    uint32_t idx = (uint32_t) vt;
    const char *ptr_name = jitc_vulkan_id("%ptr_fn_", idx);
    if (jitc_vulkan_mark_emitted(idx, EmittedFnPtr))
        return ptr_name;

    jitc_vulkan_require_type(vt);
    fmt_tc("$s = OpTypePointer Function $s\n", ptr_name, type_name_vulkan[idx]);
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
    if (!(spv_flags & SpvBoolStorageConsts)) {
        put_tc("%c_u8_0 = OpConstant %u8 0\n"
               "%c_u8_1 = OpConstant %u8 1\n");
        spv_flags |= SpvBoolStorageConsts;
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
    const char *name = jitc_vulkan_id("%c_zero_", idx);
    if (jitc_vulkan_mark_emitted(idx, EmittedZero))
        return name;

    jitc_vulkan_require_type(vt);
    fmt_tc("$s = OpConstantNull $s\n", name, type_name_vulkan[idx]);
    return name;
}

static const char *jitc_vulkan_require_one(VarType vt) {
    if (vt == VarType::Pointer)
        vt = VarType::UInt64;

    uint32_t idx = (uint32_t) vt;
    const char *name = jitc_vulkan_id("%c_one_", idx);
    if (jitc_vulkan_mark_emitted(idx, EmittedOne))
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
    return name;
}

static void jitc_vulkan_render_literal(const Variable *v) {
    VarType vt = (VarType) v->type;
    if (vt == VarType::Pointer)
        vt = VarType::UInt64;
    jitc_vulkan_require_type(vt);
    if (vt == VarType::Bool) {
        const char *bc = v->literal ? jitc_vulkan_require_one(VarType::Bool)
                                    : jitc_vulkan_require_zero(VarType::Bool);
        fmt("$v = OpCopyObject $t $s\n", v, v, bc);
        return;
    }
    const char *short_t = type_name_vulkan[(uint32_t) vt] + 1;
    XXH128_hash_t key { v->literal, (uint64_t) vt };
    if (emitted_lit_consts.insert(key).second)
        fmt_tc("%c_lit_$s_$Q = OpConstant $t $l\n", short_t, v->literal, v, v);
    fmt("$v = OpCopyObject $t %c_lit_$s_$Q\n", v, v, short_t, v->literal);
}

static const char *jitc_vulkan_require_ones(VarType vt) {
    if (vt == VarType::Pointer)
        vt = VarType::UInt64;
    if (vt == VarType::Bool)
        return jitc_vulkan_require_one(VarType::Bool);

    uint32_t idx = (uint32_t) vt;
    const char *name = jitc_vulkan_id("%c_ones_", idx);
    if (jitc_vulkan_mark_emitted(idx, EmittedOnes))
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
    return name;
}

static const char *jitc_vulkan_require_nan(VarType vt) {
    uint32_t idx = (uint32_t) vt;
    const char *name = jitc_vulkan_id("%c_nan_", idx);
    const char *val_str = nullptr;
    switch (vt) {
        case VarType::Float16: val_str = "!0x7e00";            break;
        case VarType::Float32: val_str = "!0x7fc00000";        break;
        case VarType::Float64: val_str = "!0x0 !0x7ff80000";   break;
        default:
            jitc_fail("jitc_vulkan_require_nan(): unsupported type %u!", idx);
    }

    if (jitc_vulkan_mark_emitted(idx, EmittedNan))
        return name;

    jitc_vulkan_require_type(vt);
    fmt_tc("$s = OpConstant $s $s\n", name, type_name_vulkan[idx], val_str);
    return name;
}

static const char *jitc_vulkan_require_sign_mask(VarType vt) {
    VarType ut = jitc_vulkan_uint_type(vt);
    uint32_t idx = (uint32_t) ut;
    const char *name = jitc_vulkan_id("%c_sign_", idx);
    const char *val_str = nullptr;
    switch (ut) {
        case VarType::UInt16: val_str = "0x8000";             break;
        case VarType::UInt32: val_str = "0x80000000";         break;
        case VarType::UInt64: val_str = "0x8000000000000000"; break;
        default:
            jitc_fail("jitc_vulkan_require_sign_mask(): unsupported type %u!", idx);
    }

    if (jitc_vulkan_mark_emitted(idx, EmittedSign))
        return name;

    jitc_vulkan_require_type(ut);
    fmt_tc("$s = OpConstant $s $s\n", name, type_name_vulkan[idx], val_str);
    return name;
}

static const char *jitc_vulkan_require_mul_struct(VarType vt) {
    if (vt == VarType::Pointer)
        vt = VarType::UInt64;

    uint32_t idx = (uint32_t) vt;
    const char *name = jitc_vulkan_id("%struct_mul_", idx);
    if (jitc_vulkan_mark_emitted(idx, EmittedMulStruct))
        return name;

    jitc_vulkan_require_type(vt);
    const char *tname = type_name_vulkan[idx];
    fmt_tc("$s = OpTypeStruct $s $s\n", name, tname, tname);
    return name;
}

static const char *jitc_vulkan_require_vec_type(VarType vt, uint32_t lanes) {
    if (vt == VarType::Bool)
        vt = VarType::UInt8;
    else if (vt == VarType::Pointer)
        vt = VarType::UInt64;

    uint32_t idx = (uint32_t) vt;
    const char *name = jitc_vulkan_id((lanes == 4) ? "%v4_" : "%v2_", idx);
    if (jitc_vulkan_mark_emitted(idx, (lanes == 4) ? EmittedVec4Type : EmittedVec2Type))
        return name;

    jitc_vulkan_require_type(vt);
    fmt_tc("$s = OpTypeVector $s $u\n", name, type_name_vulkan[idx], lanes);
    return name;
}

static const char *jitc_vulkan_require_psb_vec_ptr(VarType vt, uint32_t lanes) {
    if (vt == VarType::Bool)
        vt = VarType::UInt8;
    else if (vt == VarType::Pointer)
        vt = VarType::UInt64;

    uint32_t idx = (uint32_t) vt;
    const char *ptr_name =
        jitc_vulkan_id((lanes == 4) ? "%ptr_psb_v4_" : "%ptr_psb_v2_", idx);
    if (jitc_vulkan_mark_emitted(idx, (lanes == 4) ? EmittedPsbVec4Ptr
                                                   : EmittedPsbVec2Ptr))
        return ptr_name;

    jitc_vulkan_require_psb_ptr(vt);
    const char *vec_name = jitc_vulkan_require_vec_type(vt, lanes);
    uint32_t stride = lanes * type_size[idx];

    fmt_annot("OpDecorate $s ArrayStride $u\n", ptr_name, stride);
    fmt_tc("$s = OpTypePointer PhysicalStorageBuffer $s\n", ptr_name, vec_name);
    return ptr_name;
}

static const char *jitc_vulkan_require_vec_zero(VarType vt, uint32_t lanes) {
    if (vt == VarType::Bool)
        vt = VarType::UInt8;
    else if (vt == VarType::Pointer)
        vt = VarType::UInt64;

    uint32_t idx = (uint32_t) vt;
    const char *name =
        jitc_vulkan_id((lanes == 4) ? "%c_zero_v4_" : "%c_zero_v2_", idx);
    if (jitc_vulkan_mark_emitted(idx, (lanes == 4) ? EmittedVec4Zero
                                                   : EmittedVec2Zero))
        return name;

    const char *vec_name = jitc_vulkan_require_vec_type(vt, lanes);
    fmt_tc("$s = OpConstantNull $s\n", name, vec_name);
    return name;
}

static void jitc_vulkan_require_int64_atomics() {
    const VulkanDevice &dev = *active_vulkan_dev;
    if (unlikely(!dev.shader_int64_atomics))
        jitc_raise("jit_eval(): the Vulkan device \"%s\" does not "
                   "support 64-bit buffer atomics "
                   "(shaderBufferInt64Atomics)!", dev.name);
    spv_flags |= SpvCapInt64Atomics;
}

static void jitc_vulkan_require_atomic_float_add(VarType vt) {
    spv_flags |= (vt == VarType::Float32) ? SpvCapAtomicF32Add
                                          : SpvCapAtomicF64Add;
}

static void jitc_vulkan_require_subgroup() {
    if (spv_flags & SpvCapSubgroup)
        return;
    jitc_vulkan_require_type(VarType::UInt32);
    put_annot("OpDecorate %gl_SubgroupLocalInvocationId BuiltIn "
              "SubgroupLocalInvocationId\n");
    put_tc("%ptr_in_u32 = OpTypePointer Input %u32\n"
           "%gl_SubgroupLocalInvocationId = OpVariable %ptr_in_u32 Input\n");
    spv_flags |= SpvCapSubgroup;
}

// ----------------------------------------------------------------------------
//  Local array code generation helpers
// ----------------------------------------------------------------------------

static void jitc_vulkan_require_fn_arr_type(VarType vt, uint32_t len) {
    if (vt == VarType::Pointer)
        vt = VarType::UInt64;
    uint32_t idx = (uint32_t) vt;
    uint32_t key = (idx << 16) | (len & 0xffffu);
    if (!emitted_fn_arr_types.insert(key).second)
        return;

    jitc_vulkan_require_type(vt);
    jitc_vulkan_require_fn_ptr(vt);
    jitc_vulkan_require_u32_const(len);
    const char *elem_type = type_name_vulkan[idx];
    const char *short_name = elem_type + 1;
    fmt_tc("%arr_$s_$u = OpTypeArray $s %c_u32_$u\n"
           "%ptr_fn_arr_$s_$u = OpTypePointer Function %arr_$s_$u\n",
           short_name, len, elem_type, len,
           short_name, len, short_name, len);
}

static void jitc_vulkan_declare_local_array(VarType vt, uint32_t len,
                                            uint32_t reg_index) {
    if (vt == VarType::Pointer)
        vt = VarType::UInt64;
    jitc_vulkan_require_fn_arr_type(vt, len);
    const char *short_name = type_name_vulkan[(uint32_t) vt] + 1;
    fmt_local("%arr_r$u = OpVariable %ptr_fn_arr_$s_$u Function\n",
              reg_index, short_name, len);
}

static void jitc_vulkan_render_array(Variable *v, Variable *pred) {
    if (pred && pred->array_state != (uint32_t) ArrayState::Conflicted) {
        v->reg_index = pred->reg_index;
        return;
    }
    jitc_vulkan_declare_local_array((VarType) v->type, v->array_length,
                                    v->reg_index);
}

static void jitc_vulkan_render_array_init(Variable *v, Variable *pred,
                                          Variable *value) {
    uint32_t id = ++spv_aux_id;
    v->reg_index = pred->reg_index;
    VarType vt = (VarType) v->type;
    uint32_t len = v->array_length;
    const char *fn_ptr = jitc_vulkan_require_fn_ptr(vt);
    jitc_vulkan_require_type(VarType::Bool);
    jitc_vulkan_require_u32_const(0);
    jitc_vulkan_require_u32_const(1);
    jitc_vulkan_require_u32_const(len);

    fmt("OpBranch %arr_init_hdr_r$u\n"
        "%arr_init_hdr_r$u = OpLabel\n"
        "%arr_init_i_r$u = OpPhi %u32 %c_u32_0 $s %arr_init_next_r$u %arr_init_body_r$u\n"
        "%arr_init_cmp_r$u = OpULessThan %bool %arr_init_i_r$u %c_u32_$u\n"
        "OpLoopMerge %arr_init_exit_r$u %arr_init_body_r$u None\n"
        "OpBranchConditional %arr_init_cmp_r$u %arr_init_body_r$u %arr_init_exit_r$u\n"
        "%arr_init_body_r$u = OpLabel\n"
        "%arr_init_ptr_r$u = OpAccessChain $s %arr_r$u %arr_init_i_r$u\n"
        "OpStore %arr_init_ptr_r$u $v\n"
        "%arr_init_next_r$u = OpIAdd %u32 %arr_init_i_r$u %c_u32_1\n"
        "OpBranch %arr_init_hdr_r$u\n"
        "%arr_init_exit_r$u = OpLabel\n",
        id,
        id,
        id, current_block_label, id, id,
        id, id, len,
        id, id,
        id, id, id,
        id,
        id, fn_ptr, v->reg_index, id,
        id, value,
        id, id,
        id,
        id);

    jitc_vulkan_set_label("arr_init_exit", id);
}

static void jitc_vulkan_render_array_read(Variable *v, Variable *source,
                                          Variable *mask, Variable *offset) {
    VarType vt = (VarType) v->type;
    const char *fn_ptr = jitc_vulkan_require_fn_ptr(vt);
    jitc_vulkan_require_type(vt);

    if (!offset)
        jitc_vulkan_require_u32_const((uint32_t) v->literal);

    if (mask->is_literal()) {
        if (offset) {
            fmt("$v_ptr = OpAccessChain $s %arr_r$u $v\n"
                "$v = OpLoad $t $v_ptr\n",
                v, fn_ptr, source->reg_index, offset,
                v, v, v);
        } else {
            fmt("$v_ptr = OpAccessChain $s %arr_r$u %c_u32_$u\n"
                "$v = OpLoad $t $v_ptr\n",
                v, fn_ptr, source->reg_index, (uint32_t) v->literal,
                v, v, v);
        }
    } else {
        jitc_vulkan_require_type(VarType::Bool);
        const char *zero_c = jitc_vulkan_require_zero(vt);
        if (offset) {
            fmt("OpSelectionMerge %arr_rd_done_r$u None\n"
                "OpBranchConditional $v %arr_rd_then_r$u %arr_rd_done_r$u\n"
                "%arr_rd_then_r$u = OpLabel\n"
                "$v_ptr = OpAccessChain $s %arr_r$u $v\n"
                "$v_val = OpLoad $t $v_ptr\n"
                "OpBranch %arr_rd_done_r$u\n"
                "%arr_rd_done_r$u = OpLabel\n"
                "$v = OpPhi $t $v_val %arr_rd_then_r$u $s $s\n",
                v->reg_index,
                mask, v->reg_index, v->reg_index,
                v->reg_index,
                v, fn_ptr, source->reg_index, offset,
                v, v, v,
                v->reg_index,
                v->reg_index,
                v, v, v, v->reg_index, zero_c, current_block_label);
        } else {
            fmt("OpSelectionMerge %arr_rd_done_r$u None\n"
                "OpBranchConditional $v %arr_rd_then_r$u %arr_rd_done_r$u\n"
                "%arr_rd_then_r$u = OpLabel\n"
                "$v_ptr = OpAccessChain $s %arr_r$u %c_u32_$u\n"
                "$v_val = OpLoad $t $v_ptr\n"
                "OpBranch %arr_rd_done_r$u\n"
                "%arr_rd_done_r$u = OpLabel\n"
                "$v = OpPhi $t $v_val %arr_rd_then_r$u $s $s\n",
                v->reg_index,
                mask, v->reg_index, v->reg_index,
                v->reg_index,
                v, fn_ptr, source->reg_index, (uint32_t) v->literal,
                v, v, v,
                v->reg_index,
                v->reg_index,
                v, v, v, v->reg_index, zero_c, current_block_label);
        }
        jitc_vulkan_set_label("arr_rd_done", v->reg_index);
    }
}

static void jitc_vulkan_render_array_write(Variable *v, Variable *target,
                                           Variable *value, Variable *mask,
                                           Variable *offset) {
    if (offset && offset->is_array())
        offset = nullptr;

    uint32_t id = ++spv_aux_id;
    bool copy = target->array_state == (uint32_t) ArrayState::Conflicted;
    uint32_t target_buffer = target->reg_index;
    VarType vt = (VarType) v->type;
    const char *fn_ptr = jitc_vulkan_require_fn_ptr(vt);
    jitc_vulkan_require_type(vt);

    if (copy) {
        target_buffer = jitc_array_buffer(v)->reg_index;
        uint32_t len = v->array_length;
        jitc_vulkan_require_type(VarType::Bool);
        jitc_vulkan_require_u32_const(0);
        jitc_vulkan_require_u32_const(1);
        jitc_vulkan_require_u32_const(len);

        fmt("OpBranch %arr_cp_hdr_r$u\n"
            "%arr_cp_hdr_r$u = OpLabel\n"
            "%arr_cp_i_r$u = OpPhi %u32 %c_u32_0 $s %arr_cp_next_r$u %arr_cp_body_r$u\n"
            "%arr_cp_cmp_r$u = OpULessThan %bool %arr_cp_i_r$u %c_u32_$u\n"
            "OpLoopMerge %arr_cp_exit_r$u %arr_cp_body_r$u None\n"
            "OpBranchConditional %arr_cp_cmp_r$u %arr_cp_body_r$u %arr_cp_exit_r$u\n"
            "%arr_cp_body_r$u = OpLabel\n"
            "%arr_cp_sp_r$u = OpAccessChain $s %arr_r$u %arr_cp_i_r$u\n"
            "%arr_cp_val_r$u = OpLoad $t %arr_cp_sp_r$u\n"
            "%arr_cp_dp_r$u = OpAccessChain $s %arr_r$u %arr_cp_i_r$u\n"
            "OpStore %arr_cp_dp_r$u %arr_cp_val_r$u\n"
            "%arr_cp_next_r$u = OpIAdd %u32 %arr_cp_i_r$u %c_u32_1\n"
            "OpBranch %arr_cp_hdr_r$u\n"
            "%arr_cp_exit_r$u = OpLabel\n",
            id,
            id,
            id, current_block_label, id, id,
            id, id, len,
            id, id,
            id, id, id,
            id,
            id, fn_ptr, target->reg_index, id,
            id, v, id,
            id, fn_ptr, target_buffer, id,
            id, id,
            id, id,
            id,
            id);

        jitc_vulkan_set_label("arr_cp_exit", id);
    }

    if (!offset)
        jitc_vulkan_require_u32_const((uint32_t) v->literal);

    if (mask->is_literal()) {
        if (offset) {
            fmt("%arr_wr_ptr_r$u = OpAccessChain $s %arr_r$u $v\n"
                "OpStore %arr_wr_ptr_r$u $v\n",
                id, fn_ptr, target_buffer, offset,
                id, value);
        } else {
            fmt("%arr_wr_ptr_r$u = OpAccessChain $s %arr_r$u %c_u32_$u\n"
                "OpStore %arr_wr_ptr_r$u $v\n",
                id, fn_ptr, target_buffer, (uint32_t) v->literal,
                id, value);
        }
    } else {
        jitc_vulkan_require_type(VarType::Bool);
        if (offset) {
            fmt("OpSelectionMerge %arr_wr_done_r$u None\n"
                "OpBranchConditional $v %arr_wr_then_r$u %arr_wr_done_r$u\n"
                "%arr_wr_then_r$u = OpLabel\n"
                "%arr_wr_ptr_r$u = OpAccessChain $s %arr_r$u $v\n"
                "OpStore %arr_wr_ptr_r$u $v\n"
                "OpBranch %arr_wr_done_r$u\n"
                "%arr_wr_done_r$u = OpLabel\n",
                id,
                mask, id, id,
                id,
                id, fn_ptr, target_buffer, offset,
                id, value,
                id,
                id);
        } else {
            fmt("OpSelectionMerge %arr_wr_done_r$u None\n"
                "OpBranchConditional $v %arr_wr_then_r$u %arr_wr_done_r$u\n"
                "%arr_wr_then_r$u = OpLabel\n"
                "%arr_wr_ptr_r$u = OpAccessChain $s %arr_r$u %c_u32_$u\n"
                "OpStore %arr_wr_ptr_r$u $v\n"
                "OpBranch %arr_wr_done_r$u\n"
                "%arr_wr_done_r$u = OpLabel\n",
                id,
                mask, id, id,
                id,
                id, fn_ptr, target_buffer, (uint32_t) v->literal,
                id, value,
                id,
                id);
        }
        jitc_vulkan_set_label("arr_wr_done", id);
    }

    v->reg_index = target_buffer;
}

static void jitc_vulkan_render_array_select(Variable *v, Variable *mask,
                                            Variable *t, Variable *f) {
    uint32_t id = ++spv_aux_id;
    uint32_t reg_index = jitc_array_buffer(v)->reg_index;
    VarType vt = (VarType) v->type;
    uint32_t len = f->array_length;
    const char *fn_ptr = jitc_vulkan_require_fn_ptr(vt);
    jitc_vulkan_require_type(vt);
    jitc_vulkan_require_type(VarType::Bool);
    jitc_vulkan_require_u32_const(0);
    jitc_vulkan_require_u32_const(1);
    jitc_vulkan_require_u32_const(len);

    fmt("OpBranch %arr_sel_hdr_r$u\n"
        "%arr_sel_hdr_r$u = OpLabel\n"
        "%arr_sel_i_r$u = OpPhi %u32 %c_u32_0 $s %arr_sel_next_r$u %arr_sel_body_r$u\n"
        "%arr_sel_cmp_r$u = OpULessThan %bool %arr_sel_i_r$u %c_u32_$u\n"
        "OpLoopMerge %arr_sel_exit_r$u %arr_sel_body_r$u None\n"
        "OpBranchConditional %arr_sel_cmp_r$u %arr_sel_body_r$u %arr_sel_exit_r$u\n"
        "%arr_sel_body_r$u = OpLabel\n"
        "%arr_sel_tp_r$u = OpAccessChain $s %arr_r$u %arr_sel_i_r$u\n"
        "%arr_sel_tv_r$u = OpLoad $t %arr_sel_tp_r$u\n"
        "%arr_sel_fp_r$u = OpAccessChain $s %arr_r$u %arr_sel_i_r$u\n"
        "%arr_sel_fv_r$u = OpLoad $t %arr_sel_fp_r$u\n"
        "%arr_sel_sv_r$u = OpSelect $t $v %arr_sel_tv_r$u %arr_sel_fv_r$u\n"
        "%arr_sel_dp_r$u = OpAccessChain $s %arr_r$u %arr_sel_i_r$u\n"
        "OpStore %arr_sel_dp_r$u %arr_sel_sv_r$u\n"
        "%arr_sel_next_r$u = OpIAdd %u32 %arr_sel_i_r$u %c_u32_1\n"
        "OpBranch %arr_sel_hdr_r$u\n"
        "%arr_sel_exit_r$u = OpLabel\n",
        id,
        id,
        id, current_block_label, id, id,
        id, id, len,
        id, id,
        id, id, id,
        id,
        id, fn_ptr, t->reg_index, id,
        id, v, id,
        id, fn_ptr, f->reg_index, id,
        id, v, id,
        id, v, mask, id, id,
        id, fn_ptr, reg_index, id,
        id, id,
        id, id,
        id,
        id);

    jitc_vulkan_set_label("arr_sel_exit", id);
    v->reg_index = reg_index;
}

static void jitc_vulkan_render_array_memcpy_in(const Variable *v) {
    uint32_t id = ++spv_aux_id;
    VarType vt = (VarType) v->type;
    uint32_t len = v->array_length;

    jitc_vulkan_declare_local_array(vt, len, v->reg_index);

    const char *fn_ptr = jitc_vulkan_require_fn_ptr(vt);
    const char *psb_ptr = nullptr;
    if (vt == VarType::Bool) {
        jitc_vulkan_require_bool_storage();
        psb_ptr = "%ptr_psb_u8";
    } else {
        psb_ptr = jitc_vulkan_require_psb_ptr(vt);
    }

    jitc_vulkan_require_type(VarType::Bool);
    jitc_vulkan_require_u32_const(0);
    jitc_vulkan_require_u32_const(1);
    jitc_vulkan_require_u32_const(len);

    fmt("%b_in_r$u = OpConvertUToPtr $s %a_r$u\n"
        "OpBranch %arr_in_hdr_r$u\n"
        "%arr_in_hdr_r$u = OpLabel\n"
        "%arr_in_i_r$u = OpPhi %u32 %c_u32_0 $s %arr_in_next_r$u %arr_in_body_r$u\n"
        "%arr_in_cmp_r$u = OpULessThan %bool %arr_in_i_r$u %c_u32_$u\n"
        "OpLoopMerge %arr_in_exit_r$u %arr_in_body_r$u None\n"
        "OpBranchConditional %arr_in_cmp_r$u %arr_in_body_r$u %arr_in_exit_r$u\n"
        "%arr_in_body_r$u = OpLabel\n",
        id, psb_ptr, v->reg_index,
        id,
        id,
        id, current_block_label, id, id,
        id, id, len,
        id, id,
        id, id, id,
        id);

    if (v->size > 1) {
        fmt("%arr_in_mul_r$u = OpIMul %u32 %arr_in_i_r$u %size\n"
            "%arr_in_gidx_r$u = OpIAdd %u32 %arr_in_mul_r$u %idx\n"
            "%arr_in_gp_r$u = OpPtrAccessChain $s %b_in_r$u %arr_in_gidx_r$u\n",
            id, id,
            id, id,
            id, psb_ptr, id, id);
    } else {
        fmt("%arr_in_gp_r$u = OpPtrAccessChain $s %b_in_r$u %arr_in_i_r$u\n",
            id, psb_ptr, id, id);
    }

    if (vt == VarType::Bool) {
        fmt("%arr_in_raw_r$u = OpLoad %u8 %arr_in_gp_r$u Aligned 1\n"
            "%arr_in_val_r$u = OpINotEqual %bool %arr_in_raw_r$u %c_u8_0\n",
            id, id,
            id, id);
    } else {
        fmt("%arr_in_val_r$u = OpLoad $t %arr_in_gp_r$u Aligned $a\n",
            id, v, id, v);
    }

    fmt("%arr_in_lp_r$u = OpAccessChain $s %arr_r$u %arr_in_i_r$u\n"
        "OpStore %arr_in_lp_r$u %arr_in_val_r$u\n"
        "%arr_in_next_r$u = OpIAdd %u32 %arr_in_i_r$u %c_u32_1\n"
        "OpBranch %arr_in_hdr_r$u\n"
        "%arr_in_exit_r$u = OpLabel\n",
        id, fn_ptr, v->reg_index, id,
        id, id,
        id, id,
        id,
        id);

    jitc_vulkan_set_label("arr_in_exit", id);
}

static void jitc_vulkan_render_array_memcpy_out(
    const Variable *v, const char *param_u64_ptr_type,
    const char *param_u64_load_align) {
    uint32_t id = ++spv_aux_id;
    VarType vt = (VarType) v->type;
    uint32_t len = v->array_length;

    const char *fn_ptr = jitc_vulkan_require_fn_ptr(vt);
    const char *psb_ptr = nullptr;
    if (vt == VarType::Bool) {
        jitc_vulkan_require_bool_storage();
        psb_ptr = "%ptr_psb_u8";
    } else {
        psb_ptr = jitc_vulkan_require_psb_ptr(vt);
    }

    jitc_vulkan_require_type(VarType::Bool);
    jitc_vulkan_require_u32_const(0);
    jitc_vulkan_require_u32_const(1);
    jitc_vulkan_require_u32_const(len);

    fmt("%out_p_r$u = OpAccessChain $s %params %c_u32_1 %c_u32_$o\n"
        "%out_a_r$u = OpLoad %u64 %out_p_r$u$s\n"
        "%out_b_r$u = OpConvertUToPtr $s %out_a_r$u\n"
        "OpBranch %arr_out_hdr_r$u\n"
        "%arr_out_hdr_r$u = OpLabel\n"
        "%arr_out_i_r$u = OpPhi %u32 %c_u32_0 $s %arr_out_next_r$u %arr_out_body_r$u\n"
        "%arr_out_cmp_r$u = OpULessThan %bool %arr_out_i_r$u %c_u32_$u\n"
        "OpLoopMerge %arr_out_exit_r$u %arr_out_body_r$u None\n"
        "OpBranchConditional %arr_out_cmp_r$u %arr_out_body_r$u %arr_out_exit_r$u\n"
        "%arr_out_body_r$u = OpLabel\n"
        "%arr_out_lp_r$u = OpAccessChain $s %arr_r$u %arr_out_i_r$u\n"
        "%arr_out_val_r$u = OpLoad $t %arr_out_lp_r$u\n"
        "%arr_out_mul_r$u = OpIMul %u32 %arr_out_i_r$u %size\n"
        "%arr_out_gidx_r$u = OpIAdd %u32 %arr_out_mul_r$u %idx\n"
        "%arr_out_gp_r$u = OpPtrAccessChain $s %out_b_r$u %arr_out_gidx_r$u\n",
        id, param_u64_ptr_type, v,
        id, id, param_u64_load_align,
        id, psb_ptr, id,
        id,
        id,
        id, current_block_label, id, id,
        id, id, len,
        id, id,
        id, id, id,
        id,
        id, fn_ptr, v->reg_index, id,
        id, v, id,
        id, id,
        id, id,
        id, psb_ptr, id, id);

    if (vt == VarType::Bool) {
        fmt("%arr_out_st_r$u = OpSelect %u8 %arr_out_val_r$u %c_u8_1 %c_u8_0\n"
            "OpStore %arr_out_gp_r$u %arr_out_st_r$u Aligned 1\n",
            id, id,
            id, id);
    } else {
        fmt("OpStore %arr_out_gp_r$u %arr_out_val_r$u Aligned $a\n",
            id, id, v);
    }

    fmt("%arr_out_next_r$u = OpIAdd %u32 %arr_out_i_r$u %c_u32_1\n"
        "OpBranch %arr_out_hdr_r$u\n"
        "%arr_out_exit_r$u = OpLabel\n",
        id, id,
        id,
        id);

    jitc_vulkan_set_label("arr_out_exit", id);
}

// ----------------------------------------------------------------------------
//  Main kernel assembly routine
// ----------------------------------------------------------------------------

static uint32_t active_vulkan_wg_size = 256;
static void jitc_vulkan_append_unit_funcs(const StringBuffer &unit_src);

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
    active_vulkan_wg_size = wg_size;

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
        fmt("%p_call_base = OpAccessChain $s %params %c_u32_1 %c_u32_$u\n"
            "%call_base = OpLoad %u64 %p_call_base$s\n",
            param_u64_ptr_type, arg_idx, param_u64_load_align);
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

            uint32_t arg_idx = v->param_offset / (uint32_t) sizeof(void *) - 1;
            jitc_vulkan_require_u32_const(arg_idx);

            fmt("%p_r$u = OpAccessChain $s %params %c_u32_1 %c_u32_$o\n"
                "%a_r$u = OpLoad %u64 %p_r$u$s\n",
                v->reg_index, param_u64_ptr_type, v,
                v->reg_index, v->reg_index, param_u64_load_align);

            if (unlikely(v->is_array())) {
                jitc_vulkan_render_array_memcpy_in(v);
                continue;
            }

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
            uint32_t arg_idx = v->param_offset / (uint32_t) sizeof(void *) - 1;
            jitc_vulkan_require_u32_const(arg_idx);

            if (unlikely(v->is_array())) {
                jitc_vulkan_render_array_memcpy_out(v, param_u64_ptr_type,
                                                    param_u64_load_align);
                continue;
            }

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
    struct CapEntry { uint32_t flag; const char *str; };
    static const CapEntry optional_caps[] = {
        { SpvCapInt8,         "OpCapability Int8\n" },
        { SpvCapStorage8Bit,  "OpCapability StorageBuffer8BitAccess\n" },
        { SpvCapInt16,        "OpCapability Int16\n" },
        { SpvCapStorage16Bit, "OpCapability StorageBuffer16BitAccess\n" },
        { SpvCapFloat16,      "OpCapability Float16\n" },
        { SpvCapFloat64,      "OpCapability Float64\n" },
        { SpvCapInt64Atomics, "OpCapability Int64Atomics\n" },
        { SpvCapAtomicF32Add, "OpCapability AtomicFloat32AddEXT\n" },
        { SpvCapAtomicF64Add, "OpCapability AtomicFloat64AddEXT\n" },
        { SpvCapSubgroup,     "OpCapability GroupNonUniform\n"
                              "OpCapability GroupNonUniformVote\n"
                              "OpCapability GroupNonUniformBallot\n"
                              "OpCapability GroupNonUniformArithmetic\n" },
    };

    put_header("OpCapability Shader\n"
               "OpCapability Int64\n"
               "OpCapability PhysicalStorageBufferAddresses\n");
    for (const CapEntry &c : optional_caps) {
        if (spv_flags & c.flag)
            fmt_header("$s", c.str);
    }

    if (spv_flags & (SpvCapAtomicF32Add | SpvCapAtomicF64Add))
        put_header("OpExtension \"SPV_EXT_shader_atomic_float_add\"\n");

    if (spv_flags & SpvNeedGlsl450)
        put_header("%glsl450 = OpExtInstImport \"GLSL.std.450\"\n");

    put_header("OpMemoryModel PhysicalStorageBuffer64 GLSL450\n");
    fmt_header("OpEntryPoint GLCompute %main "
               "\"drjit_^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\" "
               "%gl_GlobalInvocationID %gl_NumWorkGroups$s $s\n"
               "OpExecutionMode %main LocalSize $u 1 1\n",
               (spv_flags & SpvCapSubgroup) ? " %gl_SubgroupLocalInvocationId"
                                            : "",
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

    // 6. Merge direct callables (captured in unit_entry->src) and indirect
    //    callable units into this single SPIR-V module.
    jitc_unit_finalize(JitBackend::Vulkan);
    jitc_vulkan_append_unit_funcs(unit_entry->src);
    for (const CallableUnit &cu : callable_units)
        jitc_vulkan_append_unit_funcs(cu.unit->src);

    unit_entry->src.clear();
    callable_units.clear();
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
             *a2 = v->dep[2] ? jitc_var(v->dep[2]) : nullptr,
             *a3 = v->dep[3] ? jitc_var(v->dep[3]) : nullptr;

    VarType vt = (VarType) v->type;
    VarKind kind = (VarKind) v->kind;

    if (vt != VarType::Void)
        jitc_vulkan_require_type(vt);

    switch (kind) {
        case VarKind::Nop:
            break;

        case VarKind::Undefined: {
            const char *zero_c = jitc_vulkan_require_zero(vt);
            fmt("$v = OpCopyObject $t $s\n", v, v, zero_c);
            break;
        }

        case VarKind::Literal:
            jitc_vulkan_render_literal(v);
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
            if (jitc_is_half(v)) {
                jitc_vulkan_require_type(VarType::Float32);
                fmt("$v_0 = OpFConvert %f32 $v\n"
                    "$v_1 = OpExtInst %f32 %glsl450 Sqrt $v_0\n"
                    "$v = OpFConvert $t $v_1\n",
                    v, a0, v, v, v, v, v);
            } else {
                fmt("$v = OpExtInst $t %glsl450 Sqrt $v\n", v, v, a0);
            }
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
            if (jitc_is_half(v)) {
                jitc_vulkan_require_type(VarType::Float32);
                fmt("$v_0 = OpFConvert %f32 $v\n"
                    "$v_1 = OpFConvert %f32 $v\n"
                    "$v_2 = OpFDiv %f32 $v_0 $v_1\n"
                    "$v = OpFConvert $t $v_2\n",
                    v, a0, v, a1, v, v, v, v, v, v);
            } else if (jitc_is_float(v)) {
                fmt("$v = OpFDiv $t $v $v\n", v, v, a0, a1);
            } else if (jitc_is_uint(v)) {
                fmt("$v = OpUDiv $t $v $v\n", v, v, a0, a1);
            } else {
                fmt("$v = OpSDiv $t $v $v\n", v, v, a0, a1);
            }
            break;

        case VarKind::Mod:
            if (jitc_is_uint(v)) {
                fmt("$v = OpUMod $t $v $v\n", v, v, a0, a1);
            } else {
                // GLSL / Vulkan Shader capability leaves OpSRem undefined when
                // either operand is negative (NVIDIA lowers OpSRem as unsigned
                // modulo). Compute C99 signed remainder from OpSDiv: a - (a / b) * b.
                fmt("$v_q = OpSDiv $t $v $v\n"
                    "$v_p = OpIMul $t $v_q $v\n"
                    "$v = OpISub $t $v $v_p\n",
                    v, v, a0, a1,
                    v, v, v, a1,
                    v, v, a0, v);
            }
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
                fmt_annot("OpDecorate $v NoContraction\n", v);
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
        case VarKind::RcpApprox:
            if (jitc_is_half(v)) {
                jitc_vulkan_require_type(VarType::Float32);
                const char *one_c = jitc_vulkan_require_one(VarType::Float32);
                fmt("$v_0 = OpFConvert %f32 $v\n"
                    "$v_1 = OpFDiv %f32 $s $v_0\n"
                    "$v = OpFConvert $t $v_1\n",
                    v, a0, v, one_c, v, v, v, v);
            } else {
                const char *one_c = jitc_vulkan_require_one(vt);
                fmt("$v = OpFDiv $t $s $v\n", v, v, one_c, a0);
            }
            break;

        case VarKind::RSqrtApprox:
            jitc_vulkan_require_glsl450();
            if (jitc_is_half(v)) {
                jitc_vulkan_require_type(VarType::Float32);
                fmt("$v_0 = OpFConvert %f32 $v\n"
                    "$v_1 = OpExtInst %f32 %glsl450 InverseSqrt $v_0\n"
                    "$v = OpFConvert $t $v_1\n",
                    v, a0, v, v, v, v, v);
            } else {
                fmt("$v = OpExtInst $t %glsl450 InverseSqrt $v\n", v, v, a0);
            }
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

        case VarKind::BoundsCheck: {
            uint32_t array_size = (uint32_t) v->literal;
            jitc_vulkan_require_type(VarType::Bool);
            jitc_vulkan_require_type(VarType::UInt32);
            jitc_vulkan_require_u32_const(array_size);
            const char *ptr_type = jitc_vulkan_require_psb_ptr(VarType::UInt32);
            char idx_u32[32];
            if ((VarType) a0->type == VarType::UInt32) {
                snprintf(idx_u32, sizeof(idx_u32), "%%r%u", a0->reg_index);
            } else if ((VarType) a0->type == VarType::Int32) {
                fmt("$v_idx = OpBitcast %u32 $v\n", v, a0);
                snprintf(idx_u32, sizeof(idx_u32), "%%r%u_idx", v->reg_index);
            } else if (jitc_is_uint(a0)) {
                fmt("$v_idx = OpUConvert %u32 $v\n", v, a0);
                snprintf(idx_u32, sizeof(idx_u32), "%%r%u_idx", v->reg_index);
            } else {
                jitc_vulkan_require_type(VarType::Int32);
                fmt("$v_sidx = OpSConvert %i32 $v\n"
                    "$v_idx = OpBitcast %u32 $v_sidx\n",
                    v, a0, v, v);
                snprintf(idx_u32, sizeof(idx_u32), "%%r%u_idx", v->reg_index);
            }
            fmt("$v_inb = OpULessThan %bool $s %c_u32_$u\n"
                "$v = OpLogicalAnd %bool $v $v_inb\n"
                "$v_oob = OpLogicalNot %bool $v_inb\n"
                "$v_err = OpLogicalAnd %bool $v $v_oob\n"
                "OpSelectionMerge %bchk_done_r$u None\n"
                "OpBranchConditional $v_err %bchk_err_r$u %bchk_done_r$u\n"
                "%bchk_err_r$u = OpLabel\n"
                "$v_ptr = OpConvertUToPtr $s $v\n"
                "OpStore $v_ptr $s Aligned 4\n"
                "OpBranch %bchk_done_r$u\n"
                "%bchk_done_r$u = OpLabel\n",
                v, idx_u32, array_size,
                v, a1, v,
                v, v,
                v, a1, v,
                v->reg_index,
                v, v->reg_index, v->reg_index,
                v->reg_index,
                v, ptr_type, a2,
                v, idx_u32,
                v->reg_index,
                v->reg_index);
            jitc_vulkan_set_label("bchk_done", v->reg_index);
            break;
        }

        case VarKind::Gather: {
            jitc_vulkan_require_type(VarType::Bool);
            const char *zero_c = jitc_vulkan_require_zero(vt);
            if (vt == VarType::Bool) {
                jitc_vulkan_require_bool_storage();
                fmt("OpSelectionMerge %gather_done_r$u None\n"
                    "OpBranchConditional $v %gather_then_r$u %gather_done_r$u\n"
                    "%gather_then_r$u = OpLabel\n"
                    "$v_base = OpConvertUToPtr %ptr_psb_u8 $v\n"
                    "$v_ptr = OpPtrAccessChain %ptr_psb_u8 $v_base $v\n"
                    "$v_raw = OpLoad %u8 $v_ptr Aligned 1\n"
                    "$v_val = OpINotEqual %bool $v_raw %c_u8_0\n"
                    "OpBranch %gather_done_r$u\n"
                    "%gather_done_r$u = OpLabel\n"
                    "$v = OpPhi %bool $v_val %gather_then_r$u $s $s\n",
                    v->reg_index,
                    a2, v->reg_index, v->reg_index,
                    v->reg_index,
                    v, a0,
                    v, v, a1,
                    v, v,
                    v, v,
                    v->reg_index,
                    v->reg_index,
                    v, v, v->reg_index, zero_c, current_block_label);
            } else {
                const char *ptr_type = jitc_vulkan_require_psb_ptr(vt);
                fmt("OpSelectionMerge %gather_done_r$u None\n"
                    "OpBranchConditional $v %gather_then_r$u %gather_done_r$u\n"
                    "%gather_then_r$u = OpLabel\n"
                    "$v_base = OpConvertUToPtr $s $v\n"
                    "$v_ptr = OpPtrAccessChain $s $v_base $v\n"
                    "$v_val = OpLoad $t $v_ptr Aligned $a\n"
                    "OpBranch %gather_done_r$u\n"
                    "%gather_done_r$u = OpLabel\n"
                    "$v = OpPhi $t $v_val %gather_then_r$u $s $s\n",
                    v->reg_index,
                    a2, v->reg_index, v->reg_index,
                    v->reg_index,
                    v, ptr_type, a0,
                    v, ptr_type, v, a1,
                    v, v, v, v,
                    v->reg_index,
                    v->reg_index,
                    v, v, v, v->reg_index, zero_c, current_block_label);
            }
            jitc_vulkan_set_label("gather_done", v->reg_index);
            break;
        }

        case VarKind::PacketGather: {
            uint32_t count     = (uint32_t) v->literal,
                     vec_lanes = (count % 4 == 0) ? 4 : 2,
                     n_vecs    = count / vec_lanes,
                     vec_bytes = vec_lanes * type_size[v->type];

            jitc_vulkan_require_type(VarType::Bool);
            if (vt == VarType::Bool)
                jitc_vulkan_require_bool_storage();
            else
                jitc_vulkan_require_type(vt);

            const char *vec_type     = jitc_vulkan_require_vec_type(vt, vec_lanes);
            const char *vec_ptr_type = jitc_vulkan_require_psb_vec_ptr(vt, vec_lanes);
            const char *vec_zero     = jitc_vulkan_require_vec_zero(vt, vec_lanes);

            if (n_vecs > 1) {
                jitc_vulkan_require_u32_const(n_vecs);
                for (uint32_t c = 1; c < n_vecs; ++c)
                    jitc_vulkan_require_u32_const(c);
            }

            fmt("OpSelectionMerge %pgather_done_r$u None\n"
                "OpBranchConditional $v %pgather_then_r$u %pgather_done_r$u\n"
                "%pgather_then_r$u = OpLabel\n"
                "$v_base = OpConvertUToPtr $s $v\n",
                v->reg_index,
                a2, v->reg_index, v->reg_index,
                v->reg_index,
                v, vec_ptr_type, a0);

            for (uint32_t c = 0; c < n_vecs; ++c) {
                if (n_vecs == 1) {
                    fmt("$v_ptr_0 = OpPtrAccessChain $s $v_base $v\n",
                        v, vec_ptr_type, v, a1);
                } else if (c == 0) {
                    fmt("$v_vidx_0 = OpIMul %u32 $v %c_u32_$u\n"
                        "$v_ptr_0 = OpPtrAccessChain $s $v_base $v_vidx_0\n",
                        v, a1, n_vecs,
                        v, vec_ptr_type, v, v);
                } else {
                    fmt("$v_vidx_$u = OpIAdd %u32 $v_vidx_0 %c_u32_$u\n"
                        "$v_ptr_$u = OpPtrAccessChain $s $v_base $v_vidx_$u\n",
                        v, c, v, c,
                        v, c, vec_ptr_type, v, v, c);
                }
                fmt("$v_ld_$u = OpLoad $s $v_ptr_$u Aligned $u\n",
                    v, c, vec_type, v, c, vec_bytes);
            }

            fmt("OpBranch %pgather_done_r$u\n"
                "%pgather_done_r$u = OpLabel\n",
                v->reg_index, v->reg_index);

            for (uint32_t c = 0; c < n_vecs; ++c) {
                fmt("$v_vec_$u = OpPhi $s $v_ld_$u %pgather_then_r$u $s $s\n",
                    v, c, vec_type, v, c, v->reg_index, vec_zero,
                    current_block_label);
            }

            jitc_vulkan_set_label("pgather_done", v->reg_index);

            for (uint32_t i = 0; i < count; ++i) {
                uint32_t c = i / vec_lanes,
                         l = i % vec_lanes;
                if (vt == VarType::Bool) {
                    fmt("$v_raw_$u = OpCompositeExtract %u8 $v_vec_$u $u\n"
                        "$v_out_$u = OpINotEqual %bool $v_raw_$u %c_u8_0\n",
                        v, i, v, c, l,
                        v, i, v, i);
                } else {
                    fmt("$v_out_$u = OpCompositeExtract $t $v_vec_$u $u\n",
                        v, i, v, v, c, l);
                }
            }
            break;
        }

        case VarKind::Extract: {
            uint32_t sub_index = (uint32_t) v->literal;
            if ((VarKind) a0->kind == VarKind::PacketGather ||
                (VarKind) a0->kind == VarKind::ScatterCAS ||
                (VarKind) a0->kind == VarKind::TraceRay ||
                (VarKind) a0->kind == VarKind::TexLookup ||
                (VarKind) a0->kind == VarKind::TexLookupLod ||
                (VarKind) a0->kind == VarKind::TexLookupGrad ||
                (VarKind) a0->kind == VarKind::TexFetchBilerp) {
                fmt("$v = OpCopyObject $t $v_out_$u\n", v, v, a0, sub_index);
            } else {
                fmt("$v = OpCopyObject $t $v\n", v, v, a0);
            }
            break;
        }

        case VarKind::Scatter: {
            ReduceOp op     = (ReduceOp) (uint32_t) v->literal;
            ReduceMode mode = (ReduceMode) (uint32_t) (v->literal >> 32);
            VarType val_vt  = (VarType) a1->type;
            jitc_vulkan_require_type(VarType::Bool);

            if (op == ReduceOp::Identity) {
                fmt("OpSelectionMerge %scatter_done_r$u None\n"
                    "OpBranchConditional $v %scatter_then_r$u %scatter_done_r$u\n"
                    "%scatter_then_r$u = OpLabel\n",
                    v->reg_index,
                    a3, v->reg_index, v->reg_index,
                    v->reg_index);

                if (val_vt == VarType::Bool) {
                    jitc_vulkan_require_bool_storage();
                    fmt("$v_bval = OpSelect %u8 $v %c_u8_1 %c_u8_0\n"
                        "$v_base = OpConvertUToPtr %ptr_psb_u8 $v\n"
                        "$v_ptr = OpPtrAccessChain %ptr_psb_u8 $v_base $v\n"
                        "OpStore $v_ptr $v_bval Aligned 1\n",
                        v, a1,
                        v, a0,
                        v, v, a2,
                        v, v);
                } else {
                    const char *ptr_type = jitc_vulkan_require_psb_ptr(val_vt);
                    fmt("$v_base = OpConvertUToPtr $s $v\n"
                        "$v_ptr = OpPtrAccessChain $s $v_base $v\n"
                        "OpStore $v_ptr $v Aligned $a\n",
                        v, ptr_type, a0,
                        v, ptr_type, v, a2,
                        v, a1, a1);
                }

                fmt("OpBranch %scatter_done_r$u\n"
                    "%scatter_done_r$u = OpLabel\n",
                    v->reg_index, v->reg_index);
                jitc_vulkan_set_label("scatter_done", v->reg_index);
                break;
            }

            // ScatterReduce (op != ReduceOp::Identity)
            jitc_vulkan_require_type(val_vt);
            fmt("OpSelectionMerge %sred_done_r$u None\n"
                "OpBranchConditional $v %sred_then_r$u %sred_done_r$u\n"
                "%sred_then_r$u = OpLabel\n",
                v->reg_index,
                a3, v->reg_index, v->reg_index,
                v->reg_index);

            if (mode == ReduceMode::NoConflicts) {
                const char *ptr_type = jitc_vulkan_require_psb_ptr(val_vt);
                fmt("$v_base = OpConvertUToPtr $s $v\n"
                    "$v_ptr = OpPtrAccessChain $s $v_base $v\n"
                    "$v_old = OpLoad $t $v_ptr Aligned $a\n",
                    v, ptr_type, a0,
                    v, ptr_type, v, a2,
                    v, a1, v, a1);

                switch (op) {
                    case ReduceOp::Add:
                        fmt(jitc_is_float(a1) ? "$v_new = OpFAdd $t $v_old $v\n"
                                              : "$v_new = OpIAdd $t $v_old $v\n",
                            v, a1, v, a1);
                        break;
                    case ReduceOp::Min: {
                        jitc_vulkan_require_glsl450();
                        const char *fn = jitc_is_float(a1) ? "FMin"
                                       : (jitc_is_uint(a1) ? "UMin" : "SMin");
                        fmt("$v_new = OpExtInst $t %glsl450 $s $v_old $v\n",
                            v, a1, fn, v, a1);
                        break;
                    }
                    case ReduceOp::Max: {
                        jitc_vulkan_require_glsl450();
                        const char *fn = jitc_is_float(a1) ? "FMax"
                                       : (jitc_is_uint(a1) ? "UMax" : "SMax");
                        fmt("$v_new = OpExtInst $t %glsl450 $s $v_old $v\n",
                            v, a1, fn, v, a1);
                        break;
                    }
                    case ReduceOp::And:
                        fmt("$v_new = OpBitwiseAnd $t $v_old $v\n", v, a1, v, a1);
                        break;
                    case ReduceOp::Or:
                        fmt("$v_new = OpBitwiseOr $t $v_old $v\n", v, a1, v, a1);
                        break;
                    default:
                        jitc_fail("jitc_vulkan_render(): unsupported ReduceOp!");
                }

                fmt("OpStore $v_ptr $v_new Aligned $a\n"
                    "OpBranch %sred_done_r$u\n"
                    "%sred_done_r$u = OpLabel\n",
                    v, v, a1,
                    v->reg_index, v->reg_index);
                jitc_vulkan_set_label("sred_done", v->reg_index);
                break;
            }

            bool can_local = (val_vt == VarType::Int32 ||
                              val_vt == VarType::UInt32 ||
                              val_vt == VarType::Float32);
            bool use_local = (mode == ReduceMode::Local) && can_local;

            char val_reg[32];
            char entry_lbl[64];
            if (use_local) {
                jitc_vulkan_require_subgroup();
                jitc_vulkan_require_u32_const(3);
                const char *v4u32_name =
                    jitc_vulkan_require_vec_type(VarType::UInt32, 4);
                const char *one_bool = jitc_vulkan_require_one(VarType::Bool);
                const char *sg_op = nullptr;
                switch (op) {
                    case ReduceOp::Add:
                        sg_op = jitc_is_float(a1) ? "OpGroupNonUniformFAdd"
                                                  : "OpGroupNonUniformIAdd";
                        break;
                    case ReduceOp::Min:
                        sg_op = jitc_is_float(a1) ? "OpGroupNonUniformFMin"
                              : (jitc_is_uint(a1) ? "OpGroupNonUniformUMin"
                                                  : "OpGroupNonUniformSMin");
                        break;
                    case ReduceOp::Max:
                        sg_op = jitc_is_float(a1) ? "OpGroupNonUniformFMax"
                              : (jitc_is_uint(a1) ? "OpGroupNonUniformUMax"
                                                  : "OpGroupNonUniformSMax");
                        break;
                    case ReduceOp::And:
                        sg_op = "OpGroupNonUniformBitwiseAnd";
                        break;
                    case ReduceOp::Or:
                        sg_op = "OpGroupNonUniformBitwiseOr";
                        break;
                    default:
                        jitc_fail("jitc_vulkan_render(): unsupported ReduceOp!");
                }

                fmt("$v_ballot = OpGroupNonUniformBallot $s %c_u32_3 $s\n"
                    "$v_leader = OpGroupNonUniformBallotFindLSB %u32 %c_u32_3 $v_ballot\n"
                    "$v_fptr = OpGroupNonUniformBroadcastFirst %u64 %c_u32_3 $v\n"
                    "$v_same_ptr = OpIEqual %bool $v $v_fptr\n"
                    "$v_fidx = OpGroupNonUniformBroadcastFirst %u32 %c_u32_3 $v\n"
                    "$v_same_idx = OpIEqual %bool $v $v_fidx\n"
                    "$v_same = OpLogicalAnd %bool $v_same_ptr $v_same_idx\n"
                    "$v_all = OpGroupNonUniformAll %bool %c_u32_3 $v_same\n"
                    "$v_sgred = $s $t %c_u32_3 Reduce $v\n"
                    "$v_sg_id = OpLoad %u32 %gl_SubgroupLocalInvocationId\n"
                    "$v_isldr = OpIEqual %bool $v_sg_id $v_leader\n"
                    "$v_rval = OpSelect $t $v_all $v_sgred $v\n"
                    "$v_do = OpSelect %bool $v_all $v_isldr $s\n"
                    "OpSelectionMerge %sred_loc_done_r$u None\n"
                    "OpBranchConditional $v_do %sred_loc_do_r$u %sred_loc_done_r$u\n"
                    "%sred_loc_do_r$u = OpLabel\n",
                    v, v4u32_name, one_bool,
                    v, v,
                    v, a0,
                    v, a0, v,
                    v, a2,
                    v, a2, v,
                    v, v, v,
                    v, v,
                    v, sg_op, a1, a1,
                    v,
                    v, v, v,
                    v, a1, v, v, a1,
                    v, v, v, one_bool,
                    v->reg_index,
                    v, v->reg_index, v->reg_index,
                    v->reg_index);

                snprintf(val_reg, sizeof(val_reg), "%%r%u_rval", v->reg_index);
                snprintf(entry_lbl, sizeof(entry_lbl), "%%sred_loc_do_r%u",
                         v->reg_index);
            } else {
                snprintf(val_reg, sizeof(val_reg), "%%r%u", a1->reg_index);
                snprintf(entry_lbl, sizeof(entry_lbl), "%%sred_then_r%u",
                         v->reg_index);
            }

            jitc_vulkan_require_u32_const(0);
            jitc_vulkan_require_u32_const(1);
            const VulkanDevice &dev = *active_vulkan_dev;
            uint32_t tsize = type_size[(uint32_t) val_vt];

            if (!jitc_is_float(a1)) {
                if (tsize == 8)
                    jitc_vulkan_require_int64_atomics();
                const char *ptr_type = jitc_vulkan_require_psb_ptr(val_vt);
                const char *atomic_op = nullptr;
                switch (op) {
                    case ReduceOp::Add: atomic_op = "OpAtomicIAdd"; break;
                    case ReduceOp::Min:
                        atomic_op = jitc_is_uint(a1) ? "OpAtomicUMin" : "OpAtomicSMin";
                        break;
                    case ReduceOp::Max:
                        atomic_op = jitc_is_uint(a1) ? "OpAtomicUMax" : "OpAtomicSMax";
                        break;
                    case ReduceOp::And: atomic_op = "OpAtomicAnd"; break;
                    case ReduceOp::Or:  atomic_op = "OpAtomicOr";  break;
                    default:
                        jitc_fail("jitc_vulkan_render(): unsupported integer ReduceOp!");
                }
                fmt("$v_base = OpConvertUToPtr $s $v\n"
                    "$v_ptr = OpPtrAccessChain $s $v_base $v\n"
                    "$v_old = $s $t $v_ptr %c_u32_1 %c_u32_0 $s\n",
                    v, ptr_type, a0,
                    v, ptr_type, v, a2,
                    v, atomic_op, a1, v, val_reg);
            } else if (op == ReduceOp::Add) {
                bool has_hw_fadd =
                    (val_vt == VarType::Float32 && dev.shader_float32_atomic_add) ||
                    (val_vt == VarType::Float64 && dev.shader_float64_atomic_add);
                if (has_hw_fadd) {
                    jitc_vulkan_require_atomic_float_add(val_vt);
                    const char *ptr_type = jitc_vulkan_require_psb_ptr(val_vt);
                    fmt("$v_base = OpConvertUToPtr $s $v\n"
                        "$v_ptr = OpPtrAccessChain $s $v_base $v\n"
                        "$v_old = OpAtomicFAddEXT $t $v_ptr %c_u32_1 %c_u32_0 $s\n",
                        v, ptr_type, a0,
                        v, ptr_type, v, a2,
                        v, a1, v, val_reg);
                } else {
                    VarType ut = jitc_vulkan_uint_type(val_vt);
                    if (tsize == 8)
                        jitc_vulkan_require_int64_atomics();
                    const char *u_ptr_type = jitc_vulkan_require_psb_ptr(ut);
                    fmt("$v_base = OpConvertUToPtr $s $v\n"
                        "$v_ptr = OpPtrAccessChain $s $v_base $v\n"
                        "$v_init = OpLoad $b $v_ptr Aligned $a\n"
                        "OpBranch %cas_head_r$u\n"
                        "%cas_head_r$u = OpLabel\n"
                        "$v_exp = OpPhi $b $v_init $s $v_old %cas_cont_r$u\n"
                        "OpLoopMerge %cas_exit_r$u %cas_cont_r$u None\n"
                        "OpBranch %cas_body_r$u\n"
                        "%cas_body_r$u = OpLabel\n"
                        "$v_oldf = OpBitcast $t $v_exp\n"
                        "$v_newf = OpFAdd $t $v_oldf $s\n"
                        "$v_des = OpBitcast $b $v_newf\n"
                        "$v_old = OpAtomicCompareExchange $b $v_ptr %c_u32_1 %c_u32_0 %c_u32_0 $v_des $v_exp\n"
                        "$v_ok = OpIEqual %bool $v_old $v_exp\n"
                        "OpBranchConditional $v_ok %cas_exit_r$u %cas_cont_r$u\n"
                        "%cas_cont_r$u = OpLabel\n"
                        "OpBranch %cas_head_r$u\n"
                        "%cas_exit_r$u = OpLabel\n",
                        v, u_ptr_type, a0,
                        v, u_ptr_type, v, a2,
                        v, a1, v, a1,
                        v->reg_index,
                        v->reg_index,
                        v, a1, v, entry_lbl, v, v->reg_index,
                        v->reg_index, v->reg_index,
                        v->reg_index,
                        v->reg_index,
                        v, a1, v,
                        v, a1, v, val_reg,
                        v, a1, v,
                        v, a1, v, v, v,
                        v, v, v,
                        v, v->reg_index, v->reg_index,
                        v->reg_index,
                        v->reg_index,
                        v->reg_index);
                }
            } else if (op == ReduceOp::Min || op == ReduceOp::Max) {
                VarType st = jitc_vulkan_sint_type(val_vt);
                VarType ut = jitc_vulkan_uint_type(val_vt);
                if (tsize == 8)
                    jitc_vulkan_require_int64_atomics();
                const char *s_ptr_type = jitc_vulkan_require_psb_ptr(st);
                const char *u_ptr_type = jitc_vulkan_require_psb_ptr(ut);
                const char *zero_s = jitc_vulkan_require_zero(st);
                const char *sname = type_name_vulkan[(uint32_t) st];
                const char *uname = type_name_vulkan[(uint32_t) ut];
                const char *pos_op =
                    (op == ReduceOp::Min) ? "OpAtomicSMin" : "OpAtomicSMax";
                const char *neg_op =
                    (op == ReduceOp::Min) ? "OpAtomicUMax" : "OpAtomicUMin";

                fmt("$v_sval = OpBitcast $s $s\n"
                    "$v_pos = OpSGreaterThanEqual %bool $v_sval $s\n"
                    "OpSelectionMerge %fmm_done_r$u None\n"
                    "OpBranchConditional $v_pos %fmm_pos_r$u %fmm_neg_r$u\n"
                    "%fmm_pos_r$u = OpLabel\n"
                    "$v_sbase = OpConvertUToPtr $s $v\n"
                    "$v_sptr = OpPtrAccessChain $s $v_sbase $v\n"
                    "$v_sold = $s $s $v_sptr %c_u32_1 %c_u32_0 $v_sval\n"
                    "OpBranch %fmm_done_r$u\n"
                    "%fmm_neg_r$u = OpLabel\n"
                    "$v_uval = OpBitcast $s $s\n"
                    "$v_ubase = OpConvertUToPtr $s $v\n"
                    "$v_uptr = OpPtrAccessChain $s $v_ubase $v\n"
                    "$v_uold = $s $s $v_uptr %c_u32_1 %c_u32_0 $v_uval\n"
                    "OpBranch %fmm_done_r$u\n"
                    "%fmm_done_r$u = OpLabel\n",
                    v, sname, val_reg,
                    v, v, zero_s,
                    v->reg_index,
                    v, v->reg_index, v->reg_index,
                    v->reg_index,
                    v, s_ptr_type, a0,
                    v, s_ptr_type, v, a2,
                    v, pos_op, sname, v, v,
                    v->reg_index,
                    v->reg_index,
                    v, uname, val_reg,
                    v, u_ptr_type, a0,
                    v, u_ptr_type, v, a2,
                    v, neg_op, uname, v, v,
                    v->reg_index,
                    v->reg_index);
            } else {
                jitc_fail("jitc_vulkan_render(): unsupported float ReduceOp!");
            }

            if (use_local) {
                fmt("OpBranch %sred_loc_done_r$u\n"
                    "%sred_loc_done_r$u = OpLabel\n",
                    v->reg_index, v->reg_index);
            }

            fmt("OpBranch %sred_done_r$u\n"
                "%sred_done_r$u = OpLabel\n",
                v->reg_index, v->reg_index);
            jitc_vulkan_set_label("sred_done", v->reg_index);
            break;
        }

        case VarKind::PacketScatter: {
            PacketScatterData *psd = (PacketScatterData *) v->data;
            const std::vector<uint32_t> &values = psd->values;
            uint32_t count = (uint32_t) values.size();
            Variable *v0 = jitc_var(values[0]);
            VarType val_vt = (VarType) v0->type;

            uint32_t vec_lanes = (count % 4 == 0) ? 4 : 2,
                     n_vecs    = count / vec_lanes,
                     vec_bytes = vec_lanes * type_size[v0->type],
                     shift     = (vec_lanes == 4) ? 2 : 1;

            jitc_vulkan_require_type(VarType::Bool);
            jitc_vulkan_require_u32_const(shift);
            if (val_vt == VarType::Bool)
                jitc_vulkan_require_bool_storage();
            else
                jitc_vulkan_require_type(val_vt);

            const char *vec_type     = jitc_vulkan_require_vec_type(val_vt, vec_lanes);
            const char *vec_ptr_type = jitc_vulkan_require_psb_vec_ptr(val_vt, vec_lanes);

            for (uint32_t c = 1; c < n_vecs; ++c)
                jitc_vulkan_require_u32_const(c);

            fmt("OpSelectionMerge %pscatter_done_r$u None\n"
                "OpBranchConditional $v %pscatter_then_r$u %pscatter_done_r$u\n"
                "%pscatter_then_r$u = OpLabel\n"
                "$v_base = OpConvertUToPtr $s $v\n"
                "$v_vidx_0 = OpShiftRightLogical %u32 $v %c_u32_$u\n",
                v->reg_index,
                a2, v->reg_index, v->reg_index,
                v->reg_index,
                v, vec_ptr_type, a0,
                v, a1, shift);

            for (uint32_t c = 0; c < n_vecs; ++c) {
                if (val_vt == VarType::Bool) {
                    for (uint32_t l = 0; l < vec_lanes; ++l) {
                        uint32_t idx_val = c * vec_lanes + l;
                        fmt("$v_b_$u = OpSelect %u8 $v %c_u8_1 %c_u8_0\n",
                            v, idx_val, jitc_var(values[idx_val]));
                    }
                }

                fmt("$v_vec_$u = OpCompositeConstruct $s", v, c, vec_type);
                for (uint32_t l = 0; l < vec_lanes; ++l) {
                    uint32_t idx_val = c * vec_lanes + l;
                    if (val_vt == VarType::Bool)
                        fmt(" $v_b_$u", v, idx_val);
                    else
                        fmt(" $v", jitc_var(values[idx_val]));
                }
                put("\n");

                if (c == 0) {
                    fmt("$v_ptr_0 = OpPtrAccessChain $s $v_base $v_vidx_0\n",
                        v, vec_ptr_type, v, v);
                } else {
                    fmt("$v_vidx_$u = OpIAdd %u32 $v_vidx_0 %c_u32_$u\n"
                        "$v_ptr_$u = OpPtrAccessChain $s $v_base $v_vidx_$u\n",
                        v, c, v, c,
                        v, c, vec_ptr_type, v, v, c);
                }

                fmt("OpStore $v_ptr_$u $v_vec_$u Aligned $u\n",
                    v, c, v, c, vec_bytes);
            }

            fmt("OpBranch %pscatter_done_r$u\n"
                "%pscatter_done_r$u = OpLabel\n",
                v->reg_index, v->reg_index);
            jitc_vulkan_set_label("pscatter_done", v->reg_index);
            break;
        }

        case VarKind::ScatterInc: {
            jitc_vulkan_require_type(VarType::Bool);
            jitc_vulkan_require_u32_const(0);
            jitc_vulkan_require_u32_const(1);
            const char *ptr_type = jitc_vulkan_require_psb_ptr(VarType::UInt32);
            const char *zero_c   = jitc_vulkan_require_zero(VarType::UInt32);

            fmt("OpSelectionMerge %sinc_done_r$u None\n"
                "OpBranchConditional $v %sinc_then_r$u %sinc_done_r$u\n"
                "%sinc_then_r$u = OpLabel\n"
                "$v_base = OpConvertUToPtr $s $v\n"
                "$v_ptr = OpPtrAccessChain $s $v_base $v\n"
                "$v_old = OpAtomicIAdd %u32 $v_ptr %c_u32_1 %c_u32_0 %c_u32_1\n"
                "OpBranch %sinc_done_r$u\n"
                "%sinc_done_r$u = OpLabel\n"
                "$v = OpPhi %u32 $v_old %sinc_then_r$u $s $s\n",
                v->reg_index,
                a2, v->reg_index, v->reg_index,
                v->reg_index,
                v, ptr_type, a0,
                v, ptr_type, v, a1,
                v, v,
                v->reg_index,
                v->reg_index,
                v, v, v->reg_index, zero_c, current_block_label);

            jitc_vulkan_set_label("sinc_done", v->reg_index);
            v->consumed = 1;
            break;
        }

        case VarKind::ScatterExch: {
            VarType ut = jitc_vulkan_uint_type(vt);
            if (type_size[v->type] == 8)
                jitc_vulkan_require_int64_atomics();
            jitc_vulkan_require_type(VarType::Bool);
            jitc_vulkan_require_u32_const(0);
            jitc_vulkan_require_u32_const(1);

            const char *u_ptr_type = jitc_vulkan_require_psb_ptr(ut);
            const char *zero_c     = jitc_vulkan_require_zero(vt);
            bool same_type = strcmp(type_name_vulkan[v->type],
                                    type_name_vulkan[(uint32_t) ut]) == 0;
            const char *cast_op = same_type ? "OpCopyObject" : "OpBitcast";

            fmt("OpSelectionMerge %sexch_done_r$u None\n"
                "OpBranchConditional $v %sexch_then_r$u %sexch_done_r$u\n"
                "%sexch_then_r$u = OpLabel\n"
                "$v_uval = $s $b $v\n"
                "$v_base = OpConvertUToPtr $s $v\n"
                "$v_ptr = OpPtrAccessChain $s $v_base $v\n"
                "$v_uold = OpAtomicExchange $b $v_ptr %c_u32_1 %c_u32_0 $v_uval\n"
                "$v_old = $s $t $v_uold\n"
                "OpBranch %sexch_done_r$u\n"
                "%sexch_done_r$u = OpLabel\n"
                "$v = OpPhi $t $v_old %sexch_then_r$u $s $s\n",
                v->reg_index,
                a3, v->reg_index, v->reg_index,
                v->reg_index,
                v, cast_op, v, a1,
                v, u_ptr_type, a0,
                v, u_ptr_type, v, a2,
                v, v, v, v,
                v, cast_op, v, v,
                v->reg_index,
                v->reg_index,
                v, v, v, v->reg_index, zero_c, current_block_label);

            jitc_vulkan_set_label("sexch_done", v->reg_index);
            v->consumed = 1;
            break;
        }

        case VarKind::ScatterCAS: {
            ScatterCASDData *cas_data = (ScatterCASDData *) v->data;
            Variable *mask = jitc_var(cas_data->mask);
            VarType val_vt = (VarType) a1->type;
            VarType ut = jitc_vulkan_uint_type(val_vt);
            if (type_size[a1->type] == 8)
                jitc_vulkan_require_int64_atomics();
            jitc_vulkan_require_type(VarType::Bool);
            jitc_vulkan_require_type(val_vt);
            jitc_vulkan_require_u32_const(0);
            jitc_vulkan_require_u32_const(1);

            const char *u_ptr_type = jitc_vulkan_require_psb_ptr(ut);
            const char *zero_val   = jitc_vulkan_require_zero(val_vt);
            const char *zero_bool  = jitc_vulkan_require_zero(VarType::Bool);
            bool same_type = strcmp(type_name_vulkan[a1->type],
                                    type_name_vulkan[(uint32_t) ut]) == 0;
            const char *cast_op = same_type ? "OpCopyObject" : "OpBitcast";

            fmt("OpSelectionMerge %scas_done_r$u None\n"
                "OpBranchConditional $v %scas_then_r$u %scas_done_r$u\n"
                "%scas_then_r$u = OpLabel\n"
                "$v_ucmp = $s $b $v\n"
                "$v_uval = $s $b $v\n"
                "$v_base = OpConvertUToPtr $s $v\n"
                "$v_ptr = OpPtrAccessChain $s $v_base $v\n"
                "$v_uold = OpAtomicCompareExchange $b $v_ptr %c_u32_1 %c_u32_0 %c_u32_0 $v_uval $v_ucmp\n"
                "$v_ok = OpIEqual %bool $v_uold $v_ucmp\n"
                "$v_old = $s $t $v_uold\n"
                "OpBranch %scas_done_r$u\n"
                "%scas_done_r$u = OpLabel\n"
                "$v_out_0 = OpPhi $t $v_old %scas_then_r$u $s $s\n"
                "$v_out_1 = OpPhi %bool $v_ok %scas_then_r$u $s $s\n",
                v->reg_index,
                mask, v->reg_index, v->reg_index,
                v->reg_index,
                v, cast_op, a1, a1,
                v, cast_op, a1, a2,
                v, u_ptr_type, a0,
                v, u_ptr_type, v, a3,
                v, a1, v, v, v,
                v, v, v,
                v, cast_op, a1, v,
                v->reg_index,
                v->reg_index,
                v, a1, v, v->reg_index, zero_val, current_block_label,
                v, v, v->reg_index, zero_bool, current_block_label);

            jitc_vulkan_set_label("scas_done", v->reg_index);
            v->consumed = 1;
            break;
        }

        // ================ Structured Control Flow: Loops ================

        case VarKind::LoopStart: {
            const LoopData *ld = (const LoopData *) v->data;
            for (size_t i = 0; i < ld->size; ++i) {
                Variable *inner_in = jitc_var(ld->inner_in[i]),
                         *outer_in = jitc_var(ld->outer_in[i]);
                if (inner_in == outer_in || !inner_in->reg_index ||
                    inner_in->is_array())
                    continue;

                VarType ivt = (VarType) inner_in->type;
                const char *fn_ptr = jitc_vulkan_require_fn_ptr(ivt);
                fmt_local("%loop_var_r$u = OpVariable $s Function\n",
                          inner_in->reg_index, fn_ptr);
                if (outer_in && outer_in->reg_index) {
                    fmt("OpStore %loop_var_r$u $v\n",
                        inner_in->reg_index, outer_in);
                } else {
                    const char *zero_c = jitc_vulkan_require_zero(ivt);
                    fmt("OpStore %loop_var_r$u $s\n",
                        inner_in->reg_index, zero_c);
                }
            }

            fmt("OpBranch %loop_header_r$u\n"
                "%loop_header_r$u = OpLabel\n"
                "OpLoopMerge %loop_exit_r$u %loop_cont_r$u None\n"
                "OpBranch %loop_body_r$u\n"
                "%loop_body_r$u = OpLabel\n",
                v->reg_index,
                v->reg_index,
                v->reg_index, v->reg_index,
                v->reg_index,
                v->reg_index);
            jitc_vulkan_set_label("loop_body", v->reg_index);

            for (size_t i = 0; i < ld->size; ++i) {
                Variable *inner_in = jitc_var(ld->inner_in[i]),
                         *outer_in = jitc_var(ld->outer_in[i]);
                if (inner_in == outer_in || !inner_in->reg_index ||
                    inner_in->is_array())
                    continue;
                jitc_vulkan_require_type((VarType) inner_in->type);
                fmt("$v = OpLoad $t %loop_var_r$u\n",
                    inner_in, inner_in, inner_in->reg_index);
            }
            break;
        }

        case VarKind::LoopPhi:
            if (v->is_array())
                v->reg_index = a3->reg_index;
            break;

        case VarKind::LoopCond:
            jitc_vulkan_require_type(VarType::Bool);
            fmt("OpSelectionMerge %loop_cond_ok_r$u None\n"
                "OpBranchConditional $v %loop_cond_ok_r$u %loop_exit_r$u\n"
                "%loop_cond_ok_r$u = OpLabel\n",
                a0->reg_index,
                a1, a0->reg_index, a0->reg_index,
                a0->reg_index);
            jitc_vulkan_set_label("loop_cond_ok", a0->reg_index);
            break;

        case VarKind::LoopEnd: {
            const LoopData *ld = (const LoopData *) a0->data;
            fmt("OpBranch %loop_cont_r$u\n"
                "%loop_cont_r$u = OpLabel\n",
                a0->reg_index,
                a0->reg_index);
            for (size_t i = 0; i < ld->size; ++i) {
                Variable *in  = jitc_var(ld->inner_in[i]),
                         *out = jitc_var(ld->inner_out[i]);
                if (in == out || !in->reg_index || !out->reg_index ||
                    in->is_array())
                    continue;
                fmt("OpStore %loop_var_r$u $v\n", in->reg_index, out);
            }
            fmt("OpBranch %loop_header_r$u\n"
                "%loop_exit_r$u = OpLabel\n",
                a0->reg_index,
                a0->reg_index);
            jitc_vulkan_set_label("loop_exit", a0->reg_index);
            break;
        }

        case VarKind::LoopOutput: {
            const LoopData *ld = (const LoopData *) a0->data;
            size_t i = (size_t) v->literal;
            Variable *inner_in = jitc_var(ld->inner_in[i]),
                     *outer_in = jitc_var(ld->outer_in[i]);
            if (v->reg_index && inner_in && inner_in->reg_index) {
                jitc_vulkan_require_type((VarType) v->type);
                if (inner_in == outer_in)
                    fmt("$v = OpCopyObject $t $v\n", v, v, inner_in);
                else
                    fmt("$v = OpLoad $t %loop_var_r$u\n",
                        v, v, inner_in->reg_index);
            }
            break;
        }

        // ================ Structured Control Flow: Conditionals ================

        case VarKind::CondStart: {
            const CondData *cd = (const CondData *) v->data;
            for (size_t i = 0; i < cd->indices_out.size(); ++i) {
                Variable *vo = jitc_var(cd->indices_out[i]);
                if (jitc_cond_output_active(vo)) {
                    const char *fn_ptr =
                        jitc_vulkan_require_fn_ptr((VarType) vo->type);
                    fmt_local("%cond_out_r$u = OpVariable $s Function\n",
                              vo->reg_index, fn_ptr);
                }
            }
            jitc_vulkan_require_type(VarType::Bool);
            fmt("OpSelectionMerge %cond_merge_r$u None\n"
                "OpBranchConditional $v %cond_true_r$u %cond_false_r$u\n"
                "%cond_true_r$u = OpLabel\n",
                v->reg_index,
                a0, v->reg_index, v->reg_index,
                v->reg_index);
            jitc_vulkan_set_label("cond_true", v->reg_index);
            break;
        }

        case VarKind::CondMid: {
            const CondData *cd = (const CondData *) a0->data;
            for (size_t i = 0; i < cd->indices_out.size(); ++i) {
                Variable *vt_var = jitc_var(cd->indices_t[i]),
                         *vo     = jitc_var(cd->indices_out[i]);
                if (jitc_cond_output_active(vo)) {
                    if (vt_var && vt_var->reg_index) {
                        fmt("OpStore %cond_out_r$u $v\n",
                            vo->reg_index, vt_var);
                    } else {
                        const char *zero_c =
                            jitc_vulkan_require_zero((VarType) vo->type);
                        fmt("OpStore %cond_out_r$u $s\n",
                            vo->reg_index, zero_c);
                    }
                }
            }
            fmt("OpBranch %cond_merge_r$u\n"
                "%cond_false_r$u = OpLabel\n",
                a0->reg_index,
                a0->reg_index);
            jitc_vulkan_set_label("cond_false", a0->reg_index);
            break;
        }

        case VarKind::CondEnd: {
            const CondData *cd = (const CondData *) a0->data;
            for (size_t i = 0; i < cd->indices_out.size(); ++i) {
                Variable *vf_var = jitc_var(cd->indices_f[i]),
                         *vo     = jitc_var(cd->indices_out[i]);
                if (jitc_cond_output_active(vo)) {
                    if (vf_var && vf_var->reg_index) {
                        fmt("OpStore %cond_out_r$u $v\n",
                            vo->reg_index, vf_var);
                    } else {
                        const char *zero_c =
                            jitc_vulkan_require_zero((VarType) vo->type);
                        fmt("OpStore %cond_out_r$u $s\n",
                            vo->reg_index, zero_c);
                    }
                }
            }
            fmt("OpBranch %cond_merge_r$u\n"
                "%cond_merge_r$u = OpLabel\n",
                a0->reg_index,
                a0->reg_index);
            jitc_vulkan_set_label("cond_merge", a0->reg_index);

            for (size_t i = 0; i < cd->indices_out.size(); ++i) {
                Variable *vo = jitc_var(cd->indices_out[i]);
                if (jitc_cond_output_active(vo)) {
                    jitc_vulkan_require_type((VarType) vo->type);
                    fmt("$v = OpLoad $t %cond_out_r$u\n",
                        vo, vo, vo->reg_index);
                }
            }
            break;
        }

        case VarKind::CondOutput:
            break;

        // ================ Local Arrays ================

        case VarKind::Array:
            jitc_vulkan_render_array(v, a0);
            break;

        case VarKind::ArrayInit:
            jitc_vulkan_render_array_init(v, a0, a1);
            break;

        case VarKind::ArrayWrite:
            jitc_vulkan_render_array_write(v, a0, a1, a2, a3);
            break;

        case VarKind::ArrayRead:
            jitc_vulkan_render_array_read(v, a0, a1, a2);
            break;

        case VarKind::ArrayPhi:
            v->reg_index = a0->reg_index;
            break;

        case VarKind::ArraySelect:
            jitc_vulkan_render_array_select(v, a0, a1, a2);
            break;

        // ================ Polymorphic Calls ================

        case VarKind::Call:
            jitc_var_call_assemble((CallData *) v->data, v->reg_index,
                                   a0->reg_index, a1->reg_index);
            break;

        case VarKind::CallGetter:
            jitc_var_call_getter_assemble(v, a0, a1);
            break;

        case VarKind::CallOutput:
        case VarKind::CallInput:
            break;

        case VarKind::CallSelf:
            fmt("$v = OpCopyObject $t %self_r0\n", v, v);
            break;

        default:
            jitc_raise("jitc_vulkan_render(): unhandled variable kind \"%s\" "
                       "(the Vulkan backend cannot compile kernels with this "
                       "operation yet)!",
                       var_kind_name[v->kind]);
    }
}

// ----------------------------------------------------------------------------
//  Polymorphic call & getter SPIR-V code generation
// ----------------------------------------------------------------------------

static void jitc_vulkan_append_unit_funcs(const StringBuffer &unit_src) {
    const char *src = unit_src.get(), *end = src + unit_src.size();
    while (src < end) {
        if (*src == '\n') {
            src++;
            continue;
        }
        const char *fn_end = strstr(src, "OpFunctionEnd\n");
        if (!fn_end)
            break;
        fn_end += strlen("OpFunctionEnd\n");

        if (strncmp(src, "%func_unique_", 13) == 0) {
            XXH128_hash_t h = XXH128(src, 45, 0);
            if (!unit_entry->registered.insert(h).second) {
                src = fn_end;
                continue;
            }
        }

        uint32_t fn_id = ++spv_func_id;
        buffer.put('\n');
        for (const char *p = src; p < fn_end; ) {
            if (p + 2 < fn_end && (p[0] == '%' || p[0] == '_') &&
                p[1] == 'r' && p[2] >= '0' && p[2] <= '9') {
                buffer.put(p[0]);
                buffer.put('f');
                buffer.put_u32(fn_id);
                buffer.put("_r", 2);
                p += 2;
            } else {
                buffer.put(*p++);
            }
        }
        src = fn_end;
    }
}

static bool jitc_vulkan_call_has_out(const CallData *call) {
    for (uint32_t i = 0; i < call->n_out; ++i)
        if (call->out_offset[i] != (uint32_t) -1)
            return true;
    return false;
}

static void jitc_vulkan_put_ret_type(const CallData *call,
                                     StringBuffer &b = buffer) {
    if (!jitc_vulkan_call_has_out(call)) {
        b.put("%void");
        return;
    }
    b.put("%Tuple_");
    for (uint32_t i = 0; i < call->n_out; ++i)
        if (call->out_offset[i] != (uint32_t) -1)
            b.put(type_mangle[jitc_var(call->inner_out[i])->type]);
}

static void jitc_vulkan_put_zero_tuple(const CallData *call,
                                       StringBuffer &b = buffer) {
    b.put("%c_zero_Tuple_");
    for (uint32_t i = 0; i < call->n_out; ++i)
        if (call->out_offset[i] != (uint32_t) -1)
            b.put(type_mangle[jitc_var(call->inner_out[i])->type]);
}

static void jitc_vulkan_put_fn_type(const CallData *call,
                                    StringBuffer &b = buffer) {
    b.put("%fn_");
    if (!jitc_vulkan_call_has_out(call)) {
        b.put('v');
    } else {
        for (uint32_t i = 0; i < call->n_out; ++i)
            if (call->out_offset[i] != (uint32_t) -1)
                b.put(type_mangle[jitc_var(call->inner_out[i])->type]);
    }
    b.put('_');
    if (call->use_index)     b.put('i');
    if (call->use_self)      b.put('s');
    if (!call->slots.empty()) b.put('d');
    if (call->use_nested)    b.put('b');
    b.put('_');
    for (uint32_t i = 0; i < call->n_in; ++i)
        if (call->in_active[i])
            b.put(type_mangle[jitc_var(call->outer_in[i])->type]);
}

static bool jitc_vulkan_emit_ret_type(const CallData *call) {
    if (!jitc_vulkan_call_has_out(call)) {
        jitc_vulkan_require_type(VarType::Void);
        return false;
    }
    for (uint32_t i = 0; i < call->n_out; ++i)
        if (call->out_offset[i] != (uint32_t) -1)
            jitc_vulkan_require_type((VarType) jitc_var(call->inner_out[i])->type);

    size_t start = spv_types_consts.size();
    jitc_vulkan_put_ret_type(call, spv_types_consts);
    put_tc(" = OpTypeStruct");
    for (uint32_t i = 0; i < call->n_out; ++i)
        if (call->out_offset[i] != (uint32_t) -1)
            fmt_tc(" $t", jitc_var(call->inner_out[i]));
    put_tc("\n");
    jitc_vulkan_put_zero_tuple(call, spv_types_consts);
    put_tc(" = OpConstantNull ");
    jitc_vulkan_put_ret_type(call, spv_types_consts);
    put_tc("\n");

    XXH128_hash_t h = XXH128(spv_types_consts.get() + start,
                             spv_types_consts.size() - start, 0);
    if (!emitted_lit_consts.insert(h).second)
        spv_types_consts.rewind_to(start);
    return true;
}

static void jitc_vulkan_emit_fn_type(const CallData *call) {
    if (call->use_index || call->use_self)
        jitc_vulkan_require_type(VarType::UInt32);
    if (!call->slots.empty() || call->use_nested)
        jitc_vulkan_require_type(VarType::UInt64);
    for (uint32_t i = 0; i < call->n_in; ++i)
        if (call->in_active[i])
            jitc_vulkan_require_type((VarType) jitc_var(call->outer_in[i])->type);

    size_t start = spv_types_consts.size();
    jitc_vulkan_put_fn_type(call, spv_types_consts);
    put_tc(" = OpTypeFunction ");
    jitc_vulkan_put_ret_type(call, spv_types_consts);
    if (call->use_index)     put_tc(" %u32");
    if (call->use_self)      put_tc(" %u32");
    if (!call->slots.empty()) put_tc(" %u64");
    if (call->use_nested)    put_tc(" %u64");
    for (uint32_t i = 0; i < call->n_in; ++i)
        if (call->in_active[i])
            fmt_tc(" $t", jitc_var(call->outer_in[i]));
    put_tc("\n");

    XXH128_hash_t h = XXH128(spv_types_consts.get() + start,
                             spv_types_consts.size() - start, 0);
    if (!emitted_lit_consts.insert(h).second)
        spv_types_consts.rewind_to(start);
}

static void jitc_vulkan_render_func_body(const CallData *call, uint32_t inst) {
    jitc_call_bind_slots(call, inst);

    const CallData::InstanceLayout &call_layout = call->instance_layout[inst];
    uint32_t numeric_end = call_layout.coalesce_end;
    bool cd_coalesce = call_layout.coalesce_count >= 2;

    if (cd_coalesce) {
        uint32_t end = (numeric_end + 3u) & ~3u;
        uint32_t p = 0;
        while (p < end) {
            uint32_t w_bytes = jitc_call_pick_word_chunk_size(p, end - p, 16u);
            uint32_t w0 = p / 4;
            if (w_bytes == 16) {
                const char *vec_ptr =
                    jitc_vulkan_require_psb_vec_ptr(VarType::UInt32, 4);
                const char *vec_t =
                    jitc_vulkan_require_vec_type(VarType::UInt32, 4);
                jitc_vulkan_require_u32_const(p / 16);
                fmt("%cdp_r0_$u = OpConvertUToPtr $s %data_r0\n"
                    "%cdac_r0_$u = OpPtrAccessChain $s %cdp_r0_$u %c_u32_$u\n"
                    "%cdv_r0_$u = OpLoad $s %cdac_r0_$u Aligned 16\n"
                    "%cdw_r0_$u = OpCompositeExtract %u32 %cdv_r0_$u 0\n"
                    "%cdw_r0_$u = OpCompositeExtract %u32 %cdv_r0_$u 1\n"
                    "%cdw_r0_$u = OpCompositeExtract %u32 %cdv_r0_$u 2\n"
                    "%cdw_r0_$u = OpCompositeExtract %u32 %cdv_r0_$u 3\n",
                    w0, vec_ptr,
                    w0, vec_ptr, w0, p / 16,
                    w0, vec_t, w0,
                    w0, w0,
                    w0 + 1, w0,
                    w0 + 2, w0,
                    w0 + 3, w0);
            } else if (w_bytes == 8) {
                const char *vec_ptr =
                    jitc_vulkan_require_psb_vec_ptr(VarType::UInt32, 2);
                const char *vec_t =
                    jitc_vulkan_require_vec_type(VarType::UInt32, 2);
                jitc_vulkan_require_u32_const(p / 8);
                fmt("%cdp_r0_$u = OpConvertUToPtr $s %data_r0\n"
                    "%cdac_r0_$u = OpPtrAccessChain $s %cdp_r0_$u %c_u32_$u\n"
                    "%cdv_r0_$u = OpLoad $s %cdac_r0_$u Aligned 8\n"
                    "%cdw_r0_$u = OpCompositeExtract %u32 %cdv_r0_$u 0\n"
                    "%cdw_r0_$u = OpCompositeExtract %u32 %cdv_r0_$u 1\n",
                    w0, vec_ptr,
                    w0, vec_ptr, w0, p / 8,
                    w0, vec_t, w0,
                    w0, w0,
                    w0 + 1, w0);
            } else {
                const char *u32_ptr =
                    jitc_vulkan_require_psb_ptr(VarType::UInt32);
                jitc_vulkan_require_u32_const(w0);
                fmt("%cdp_r0_$u = OpConvertUToPtr $s %data_r0\n"
                    "%cdac_r0_$u = OpPtrAccessChain $s %cdp_r0_$u %c_u32_$u\n"
                    "%cdw_r0_$u = OpLoad %u32 %cdac_r0_$u Aligned 4\n",
                    w0, u32_ptr,
                    w0, u32_ptr, w0, w0,
                    w0, w0);
            }
            p += w_bytes;
        }
    }

    auto put_cd_extract = [&](const Variable *v, uint32_t off, VarType svt) {
        uint32_t s = type_size[(int) svt], j = off / 4;
        jitc_vulkan_require_type(svt);
        if (s == 8) {
            jitc_vulkan_require_u32_const(32);
            fmt("$v_lo = OpUConvert %u64 %cdw_r0_$u\n"
                "$v_hi0 = OpUConvert %u64 %cdw_r0_$u\n"
                "$v_hi = OpShiftLeftLogical %u64 $v_hi0 %c_u32_32\n",
                v, j,
                v, j + 1,
                v, v);
            if (svt == VarType::UInt64 || svt == VarType::Pointer) {
                fmt("$v = OpBitwiseOr %u64 $v_lo $v_hi\n", v, v, v);
            } else {
                fmt("$v_u64 = OpBitwiseOr %u64 $v_lo $v_hi\n"
                    "$v = OpBitcast $t $v_u64\n",
                    v, v, v,
                    v, v, v);
            }
        } else if (s == 4) {
            if (svt == VarType::UInt32)
                fmt("$v = OpCopyObject %u32 %cdw_r0_$u\n", v, j);
            else
                fmt("$v = OpBitcast $t %cdw_r0_$u\n", v, v, j);
        } else if (s == 2) {
            jitc_vulkan_require_type(VarType::UInt16);
            uint32_t shift = (off % 4) * 8;
            if (shift > 0) {
                jitc_vulkan_require_u32_const(shift);
                fmt("$v_sh = OpShiftRightLogical %u32 %cdw_r0_$u %c_u32_$u\n",
                    v, j, shift);
                if (svt == VarType::UInt16)
                    fmt("$v = OpUConvert %u16 $v_sh\n", v, v);
                else
                    fmt("$v_u16 = OpUConvert %u16 $v_sh\n"
                        "$v = OpBitcast $t $v_u16\n",
                        v, v,
                        v, v, v);
            } else {
                if (svt == VarType::UInt16)
                    fmt("$v = OpUConvert %u16 %cdw_r0_$u\n", v, j);
                else
                    fmt("$v_u16 = OpUConvert %u16 %cdw_r0_$u\n"
                        "$v = OpBitcast $t $v_u16\n",
                        v, j,
                        v, v, v);
            }
        } else {
            jitc_vulkan_require_type(VarType::UInt8);
            uint32_t shift = (off % 4) * 8;
            if (shift > 0) {
                jitc_vulkan_require_u32_const(shift);
                fmt("$v_sh = OpShiftRightLogical %u32 %cdw_r0_$u %c_u32_$u\n",
                    v, j, shift);
                if (svt == VarType::UInt8) {
                    fmt("$v = OpUConvert %u8 $v_sh\n", v, v);
                } else if (svt == VarType::Int8) {
                    fmt("$v_u8 = OpUConvert %u8 $v_sh\n"
                        "$v = OpBitcast %i8 $v_u8\n",
                        v, v,
                        v, v);
                } else {
                    jitc_vulkan_require_bool_storage();
                    fmt("$v_u8 = OpUConvert %u8 $v_sh\n"
                        "$v = OpINotEqual %bool $v_u8 %c_u8_0\n",
                        v, v,
                        v, v);
                }
            } else {
                if (svt == VarType::UInt8) {
                    fmt("$v = OpUConvert %u8 %cdw_r0_$u\n", v, j);
                } else if (svt == VarType::Int8) {
                    fmt("$v_u8 = OpUConvert %u8 %cdw_r0_$u\n"
                        "$v = OpBitcast %i8 $v_u8\n",
                        v, j,
                        v, v);
                } else {
                    jitc_vulkan_require_bool_storage();
                    fmt("$v_u8 = OpUConvert %u8 %cdw_r0_$u\n"
                        "$v = OpINotEqual %bool $v_u8 %c_u8_0\n",
                        v, j,
                        v, v);
                }
            }
        }
    };

    for (size_t i = 0; i < schedule.size(); ++i) {
        ScheduledVariable &sv = schedule[i];
        Variable *v = jitc_var(sv.index);
        VarType vt = (VarType) v->type;
        VarKind kind = (VarKind) v->kind;

        if (kind == VarKind::Counter) {
            jitc_vulkan_require_type(vt);
            fmt("$v = OpCopyObject $t %idx_r0\n", v, v);
        } else if (kind == VarKind::CallInput) {
            uint32_t in_i = 0;
            for (; in_i < call->n_in; ++in_i) {
                if (call->inner_in[in_i] == sv.index)
                    break;
            }
            jitc_vulkan_require_type(vt);
            fmt("$v = OpCopyObject $t %in_r0_$u\n", v, v, in_i);
        } else if (kind == VarKind::CallSelf) {
            jitc_vulkan_require_type(vt);
            fmt("$v = OpCopyObject $t %self_r0\n", v, v);
        } else if (v->is_evaluated() ||
                   (vt == VarType::Pointer && kind == VarKind::Literal)) {
            uint32_t offset = jitc_call_slot_rel_offset(call, inst, v, sv.index);
            if (vt == VarType::Pointer &&
                v->resource_kind() != ResourceKind::Buffer)
                jitc_raise("jitc_vulkan_assemble_func(): resource handles "
                           "are not supported yet!");

            const CallData::CaptureSlot &capture = call->slots[v->param_offset];
            if (cd_coalesce && capture.bucket.coalesceable()) {
                put_cd_extract(v, offset, vt);
            } else if (vt == VarType::Bool) {
                jitc_vulkan_require_bool_storage();
                jitc_vulkan_require_u32_const(offset);
                fmt("$v_bp = OpConvertUToPtr %ptr_psb_u8 %data_r0\n"
                    "$v_ptr = OpPtrAccessChain %ptr_psb_u8 $v_bp %c_u32_$u\n"
                    "$v_raw = OpLoad %u8 $v_ptr Aligned 1\n"
                    "$v = OpINotEqual %bool $v_raw %c_u8_0\n",
                    v,
                    v, v, offset,
                    v, v,
                    v, v);
            } else {
                uint32_t tsz = type_size[(int) vt];
                uint32_t elem_idx = offset / tsz;
                const char *ptr_type = jitc_vulkan_require_psb_ptr(vt);
                jitc_vulkan_require_type(vt);
                jitc_vulkan_require_u32_const(elem_idx);
                fmt("$v_bp = OpConvertUToPtr $s %data_r0\n"
                    "$v_ptr = OpPtrAccessChain $s $v_bp %c_u32_$u\n"
                    "$v = OpLoad $t $v_ptr Aligned $a\n",
                    v, ptr_type,
                    v, ptr_type, v, elem_idx,
                    v, v, v, v);
            }
        } else {
            jitc_vulkan_render(v);
        }
    }
}

void jitc_vulkan_assemble_func(const CallData *call, uint32_t inst) {
    bool has_out = jitc_vulkan_emit_ret_type(call);
    jitc_vulkan_emit_fn_type(call);

    fmt("%func$s_^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ = OpFunction ",
        call->n_inst == 1 ? "_unique" : "");
    jitc_vulkan_put_ret_type(call);
    put(" None ");
    jitc_vulkan_put_fn_type(call);
    put("\n");

    if (call->use_index)
        put("%idx_r0 = OpFunctionParameter %u32\n");
    if (call->use_self)
        put("%self_r0 = OpFunctionParameter %u32\n");
    if (!call->slots.empty())
        put("%data_r0 = OpFunctionParameter %u64\n");
    if (call->use_nested)
        put("%base_r0 = OpFunctionParameter %u64\n");
    for (uint32_t i = 0; i < call->n_in; ++i) {
        if (!call->in_active[i])
            continue;
        Variable *vo = jitc_var(call->outer_in[i]);
        fmt("%in_r0_$u = OpFunctionParameter $t\n", i, vo);
    }

    put("%entry_r0 = OpLabel\n");

    size_t body_offset = buffer.size();
    size_t locals_start = spv_func_locals.size();
    uint32_t saved_aux_id = spv_aux_id;
    spv_aux_id = 0;

    char saved_label[64];
    memcpy(saved_label, current_block_label, sizeof(saved_label));
    jitc_vulkan_set_label("entry", 0);

    jitc_vulkan_render_func_body(call, inst);

    if (has_out) {
        put("%ret_r0 = OpCompositeConstruct ");
        jitc_vulkan_put_ret_type(call);
        for (uint32_t i = 0; i < call->n_out; ++i) {
            if (call->out_offset[i] == (uint32_t) -1)
                continue;
            Variable *v_out = jitc_var(call->inner_out[inst * call->n_out + i]);
            fmt(" $v", v_out);
        }
        put("\nOpReturnValue %ret_r0\n");
    } else {
        put("OpReturn\n");
    }
    put("OpFunctionEnd\n");

    if (spv_func_locals.size() > locals_start) {
        size_t suffix_start = buffer.size();
        put(spv_func_locals.get() + locals_start,
            spv_func_locals.size() - locals_start);
        buffer.move_suffix(suffix_start, body_offset);
        spv_func_locals.rewind_to(locals_start);
    }

    spv_aux_id = saved_aux_id;
    memcpy(current_block_label, saved_label, sizeof(saved_label));
}

void jitc_var_call_getter_assemble_vulkan(Variable *v, const Variable *index,
                                          const Variable *mask) {
    GetterData *gd = (GetterData *) v->data;
    uint32_t header_offset = gd->header_offset;
    VarType vt = (VarType) v->type;
    uint32_t tsize = type_size[(int) vt];
    uint32_t base_elem = header_offset / tsize;

    const char *base_name = (callable_depth == 0) ? "%call_base" : "%base_r0";

    jitc_vulkan_require_u32_const(base_elem);
    jitc_vulkan_require_type(vt);

    bool is_unmasked = mask->is_literal() && mask->literal == 1;
    if (is_unmasked) {
        if (vt == VarType::Bool) {
            jitc_vulkan_require_bool_storage();
            fmt("$v_base = OpConvertUToPtr %ptr_psb_u8 $s\n"
                "$v_idx = OpIAdd %u32 %c_u32_$u $v\n"
                "$v_ptr = OpPtrAccessChain %ptr_psb_u8 $v_base $v_idx\n"
                "$v_raw = OpLoad %u8 $v_ptr Aligned 1\n"
                "$v = OpINotEqual %bool $v_raw %c_u8_0\n",
                v, base_name,
                v, base_elem, index,
                v, v, v,
                v, v,
                v, v);
        } else {
            const char *ptr_type = jitc_vulkan_require_psb_ptr(vt);
            fmt("$v_base = OpConvertUToPtr $s $s\n"
                "$v_idx = OpIAdd %u32 %c_u32_$u $v\n"
                "$v_ptr = OpPtrAccessChain $s $v_base $v_idx\n"
                "$v = OpLoad $t $v_ptr Aligned $a\n",
                v, ptr_type, base_name,
                v, base_elem, index,
                v, ptr_type, v, v,
                v, v, v, v);
        }
    } else {
        jitc_vulkan_require_type(VarType::Bool);
        const char *zero_c = jitc_vulkan_require_zero(vt);
        fmt("OpSelectionMerge %getter_done_r$u None\n"
            "OpBranchConditional $v %getter_then_r$u %getter_done_r$u\n"
            "%getter_then_r$u = OpLabel\n",
            v->reg_index,
            mask, v->reg_index, v->reg_index,
            v->reg_index);
        if (vt == VarType::Bool) {
            jitc_vulkan_require_bool_storage();
            fmt("$v_base = OpConvertUToPtr %ptr_psb_u8 $s\n"
                "$v_idx = OpIAdd %u32 %c_u32_$u $v\n"
                "$v_ptr = OpPtrAccessChain %ptr_psb_u8 $v_base $v_idx\n"
                "$v_raw = OpLoad %u8 $v_ptr Aligned 1\n"
                "$v_val = OpINotEqual %bool $v_raw %c_u8_0\n",
                v, base_name,
                v, base_elem, index,
                v, v, v,
                v, v,
                v, v);
        } else {
            const char *ptr_type = jitc_vulkan_require_psb_ptr(vt);
            fmt("$v_base = OpConvertUToPtr $s $s\n"
                "$v_idx = OpIAdd %u32 %c_u32_$u $v\n"
                "$v_ptr = OpPtrAccessChain $s $v_base $v_idx\n"
                "$v_val = OpLoad $t $v_ptr Aligned $a\n",
                v, ptr_type, base_name,
                v, base_elem, index,
                v, ptr_type, v, v,
                v, v, v, v);
        }
        fmt("OpBranch %getter_done_r$u\n"
            "%getter_done_r$u = OpLabel\n"
            "$v = OpPhi $t $v_val %getter_then_r$u $s $s\n",
            v->reg_index,
            v->reg_index,
            v, v, v, v->reg_index, zero_c, current_block_label);
        jitc_vulkan_set_label("getter_done", v->reg_index);
    }
}

void jitc_var_call_assemble_vulkan(CallData *call, uint32_t call_reg,
                                   uint32_t self_reg, uint32_t mask_reg) {
    bool has_out = jitc_vulkan_emit_ret_type(call);

    std::vector<std::pair<Variable *, uint32_t>> out_regs;
    out_regs.reserve(call->n_out);
    for (uint32_t i = 0, k = 0; i < call->n_out; ++i) {
        if (call->out_offset[i] == (uint32_t) -1)
            continue;
        Variable *v = jitc_var(call->outer_out[i]);
        if (v && v->reg_index && v->param_type != ParamType::Input) {
            jitc_vulkan_require_type((VarType) v->type);
            out_regs.emplace_back(v, k);
        }
        k++;
    }

    Variable *mask = jitc_var(jitc_var(call->id)->dep[1]);
    bool is_masked = !mask->is_literal() || mask->literal != 1;

    char before_call_label[64];
    memcpy(before_call_label, current_block_label, sizeof(before_call_label));

    if (is_masked) {
        jitc_vulkan_require_type(VarType::Bool);
        fmt("OpSelectionMerge %call_done_r$u None\n"
            "OpBranchConditional %r$u %call_then_r$u %call_done_r$u\n"
            "%call_then_r$u = OpLabel\n",
            call_reg,
            mask_reg, call_reg, call_reg,
            call_reg);
        jitc_vulkan_set_label("call_then", call_reg);
    }

    const char *base_name = (callable_depth == 0) ? "%call_base" : "%base_r0";
    const char *idx_name  = (callable_depth == 0) ? "%idx" : "%idx_r0";

    bool has_slots = !call->slots.empty();
    if (has_slots) {
        uint32_t table_base_idx =
            call->offset_base / (uint32_t) sizeof(uint64_t);
        const char *u64_ptr = jitc_vulkan_require_psb_ptr(VarType::UInt64);
        jitc_vulkan_require_u32_const(table_base_idx);
        jitc_vulkan_require_u32_const(32);
        fmt("%call_tbp_r$u = OpConvertUToPtr $s $s\n"
            "%call_tidx_r$u = OpIAdd %u32 %c_u32_$u %r$u\n"
            "%call_tep_r$u = OpPtrAccessChain $s %call_tbp_r$u %call_tidx_r$u\n"
            "%call_oe_r$u = OpLoad %u64 %call_tep_r$u Aligned 8\n"
            "%call_doff_r$u = OpShiftRightLogical %u64 %call_oe_r$u %c_u32_32\n"
            "%call_dptr_r$u = OpIAdd %u64 $s %call_doff_r$u\n",
            call_reg, u64_ptr, base_name,
            call_reg, table_base_idx, self_reg,
            call_reg, u64_ptr, call_reg, call_reg,
            call_reg, call_reg,
            call_reg, call_reg,
            call_reg, base_name, call_reg);
    }

    auto emit_call_args = [&]() {
        if (call->use_index)
            fmt(" $s", idx_name);
        if (call->use_self)
            fmt(" %r$u", self_reg);
        if (has_slots)
            fmt(" %call_dptr_r$u", call_reg);
        if (call->use_nested)
            fmt(" $s", base_name);
        for (uint32_t i = 0; i < call->n_in; ++i) {
            if (call->in_active[i])
                fmt(" $v", jitc_var(call->outer_in[i]));
        }
        put("\n");
    };

    if (call->n_inst == 1) {
        XXH128_hash_t h = call->inst_hash[0];
        fmt("%call_ret_r$u = OpFunctionCall ", call_reg);
        jitc_vulkan_put_ret_type(call);
        fmt(" %func_unique_$Q$Q", h.high64, h.low64);
        emit_call_args();
    } else {
        std::vector<XXH128_hash_t> unique_hashes;
        unique_hashes.reserve(call->n_inst);
        for (uint32_t i = 0; i < call->n_inst; ++i) {
            XXH128_hash_t h = call->inst_hash[i];
            bool seen = false;
            for (const XXH128_hash_t &uh : unique_hashes) {
                if (XXH128Eq()(uh, h)) {
                    seen = true;
                    break;
                }
            }
            if (!seen)
                unique_hashes.push_back(h);
        }

        fmt("OpSelectionMerge %call_sw_merge_r$u None\n"
            "OpSwitch %r$u %call_def_r$u",
            call_reg, self_reg, call_reg);
        for (uint32_t i = 0; i < call->n_inst; ++i) {
            XXH128_hash_t h = call->inst_hash[i];
            uint32_t k = 0;
            while (!XXH128Eq()(unique_hashes[k], h))
                k++;
            fmt(" $u %call_case_$u_r$u", call->inst_id[i], k, call_reg);
        }
        put("\n");

        for (uint32_t k = 0; k < (uint32_t) unique_hashes.size(); ++k) {
            XXH128_hash_t h = unique_hashes[k];
            fmt("%call_case_$u_r$u = OpLabel\n"
                "%call_cret_$u_r$u = OpFunctionCall ",
                k, call_reg,
                k, call_reg);
            jitc_vulkan_put_ret_type(call);
            fmt(" %func_$Q$Q", h.high64, h.low64);
            emit_call_args();
            fmt("OpBranch %call_sw_merge_r$u\n", call_reg);
        }

        fmt("%call_def_r$u = OpLabel\n"
            "OpBranch %call_sw_merge_r$u\n"
            "%call_sw_merge_r$u = OpLabel\n",
            call_reg,
            call_reg,
            call_reg);

        if (has_out) {
            fmt("%call_ret_r$u = OpPhi ", call_reg);
            jitc_vulkan_put_ret_type(call);
            put(" ");
            jitc_vulkan_put_zero_tuple(call);
            fmt(" %call_def_r$u", call_reg);
            for (uint32_t k = 0; k < (uint32_t) unique_hashes.size(); ++k)
                fmt(" %call_cret_$u_r$u %call_case_$u_r$u",
                    k, call_reg, k, call_reg);
            put("\n");
        }
        jitc_vulkan_set_label("call_sw_merge", call_reg);
    }

    if (is_masked) {
        char after_call_label[64];
        memcpy(after_call_label, current_block_label, sizeof(after_call_label));
        for (auto [v_out, field] : out_regs) {
            fmt("$v_act = OpCompositeExtract $t %call_ret_r$u $u\n",
                v_out, v_out, call_reg, field);
        }
        fmt("OpBranch %call_done_r$u\n"
            "%call_done_r$u = OpLabel\n",
            call_reg, call_reg);
        for (auto [v_out, field] : out_regs) {
            const char *zero_c = jitc_vulkan_require_zero((VarType) v_out->type);
            fmt("$v = OpPhi $t $v_act $s $s $s\n",
                v_out, v_out, v_out, after_call_label,
                zero_c, before_call_label);
        }
        jitc_vulkan_set_label("call_done", call_reg);
    } else {
        for (auto [v_out, field] : out_regs) {
            fmt("$v = OpCompositeExtract $t %call_ret_r$u $u\n",
                v_out, v_out, call_reg, field);
        }
    }
}

// ----------------------------------------------------------------------------
//  Runtime SPIR-V compilation & VkPipeline creation
// ----------------------------------------------------------------------------

/// Vulkan unit artifacts store the VkDevice in ptr[0], the VkShaderModule in
/// value, and the SPIR-V binary byte size in size.
static void jitc_vulkan_unit_release(UnitArtifact &a) {
    if (a.ptr[0] && a.value)
        vkDestroyShaderModule((VkDevice) a.ptr[0], (VkShaderModule) a.value,
                              nullptr);
}

/// Fold device identity, driver version, pipeline cache UUID, SPIRV-Tools
/// version, and cache version into a disk-cache key.
static XXH128_hash_t jitc_vulkan_disk_key(const VulkanDevice &dev,
                                          XXH128_hash_t hash) {
    XXH3_state_t xs;
    XXH3_INITSTATE(&xs);
    XXH3_128bits_reset_withSeed(&xs, DRJIT_CACHE_VERSION);
    XXH3_128bits_update(&xs, &hash, sizeof(XXH128_hash_t));
    XXH3_128bits_update(&xs, &dev.vendor_id, sizeof(uint32_t));
    XXH3_128bits_update(&xs, &dev.device_id, sizeof(uint32_t));
    XXH3_128bits_update(&xs, &dev.driver_version, sizeof(uint32_t));
    XXH3_128bits_update(&xs, dev.pipeline_cache_uuid,
                        sizeof(dev.pipeline_cache_uuid));
    if (dev.name)
        XXH3_128bits_update(&xs, dev.name, strlen(dev.name));
    if (spvSoftwareVersionString) {
        if (const char *ver = spvSoftwareVersionString())
            XXH3_128bits_update(&xs, ver, strlen(ver));
    }
    return XXH3_128bits_digest(&xs);
}

/// Format a few lines of SPIR-V source context around a 0-based line/column
/// diagnostic position.
static std::string jitc_vulkan_format_spv_diag(const char *src, size_t size,
                                               size_t err_line,
                                               size_t err_col) {
    std::string out = "\n\n--- Diagnostic context ---\n";
    size_t cur_line = 0, line_start = 0;
    for (size_t i = 0; i <= size; ++i) {
        if (i == size || src[i] == '\n') {
            if (cur_line + 2 >= err_line && cur_line <= err_line + 2) {
                char prefix[32];
                snprintf(prefix, sizeof(prefix), "%s %5zu | ",
                         cur_line == err_line ? ">" : " ", cur_line + 1);
                out += prefix;
                out.append(src + line_start, i - line_start);
                out += '\n';
                if (cur_line == err_line) {
                    out += "        | ";
                    size_t col = std::min(err_col, i - line_start);
                    out.append(col, ' ');
                    out += "^\n";
                }
            }
            if (cur_line > err_line + 2)
                break;
            cur_line++;
            line_start = i + 1;
        }
    }
    return out;
}

bool jitc_vulkan_kernel_compile(ThreadState *ts, Kernel &kernel) {
    const VulkanDevice &dev = state.vulkan_devices[ts->device];
    VkDevice device = (VkDevice) dev.device;
    uint64_t cache_salt = (uint64_t) (uintptr_t) device;

    UnitCompileJob job;
    jitc_unit_job_init(0, job);

    UnitArtifact artifact {};
    bool unit_hit = jitc_unit_cache_lookup(JitBackend::Vulkan, job.unit_hash,
                                           cache_salt, artifact);
    XXH128_hash_t disk_key = unit_hit ? XXH128_hash_t { 0, 0 }
                                      : jitc_vulkan_disk_key(dev, job.unit_hash);

    VkShaderModule module = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    bool from_disk = false;

    // Release the lock while assembling SPIR-V and creating the pipeline.
    // 'job.source' remains valid because 'state.eval_lock' is held throughout.
    {
        unlock_guard guard(state.lock);

        if (unit_hit) {
            module = (VkShaderModule) artifact.value;
        } else {
            std::vector<uint8_t> spv_data;
            from_disk = jitc_cache_blob_load(CacheKind::VulkanSpirv, disk_key,
                                             spv_data);
            if (from_disk && (spv_data.size() < 20 || (spv_data.size() & 3) != 0)) {
                from_disk = false;
                spv_data.clear();
            }

            if (!from_disk) {
                if (unlikely(!jitc_vulkan_spv_context || !spvTextToBinary))
                    jitc_fail("jitc_vulkan_kernel_compile(): SPIRV-Tools "
                              "context is not initialized!");

                spv_binary binary = nullptr;
                spv_diagnostic diag = nullptr;
                spv_result_t spv_rv = spvTextToBinary(
                    jitc_vulkan_spv_context, job.source, job.source_size,
                    &binary, &diag);

                if (unlikely(spv_rv != SPV_SUCCESS || !binary)) {
                    std::string err_msg = (diag && diag->error)
                                              ? diag->error
                                              : "<unknown error>";
                    size_t err_line = diag ? diag->position.line : 0;
                    size_t err_col  = diag ? diag->position.column : 0;
                    bool is_text    = diag ? diag->isTextSource : false;
                    if (diag)
                        spvDiagnosticDestroy(diag);
                    if (binary)
                        spvBinaryDestroy(binary);

                    std::string context_lines =
                        is_text ? jitc_vulkan_format_spv_diag(job.source,
                                                              job.source_size,
                                                              err_line, err_col)
                                : std::string();

                    jitc_fail("jitc_vulkan_kernel_compile(): SPIR-V assembly "
                              "of \"%s\" failed at line %zu, column %zu "
                              "(error %d): %s%s\n\n--- Source code ---\n%s",
                              job.symbol, err_line + 1, err_col + 1,
                              (int) spv_rv, err_msg.c_str(),
                              context_lines.c_str(), job.source);
                }

                if (diag)
                    spvDiagnosticDestroy(diag);

                size_t byte_size = binary->wordCount * sizeof(uint32_t);
                spv_data.resize(byte_size);
                memcpy(spv_data.data(), binary->code, byte_size);
                spvBinaryDestroy(binary);
            }

            VkShaderModuleCreateInfo module_info {};
            module_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
            module_info.codeSize = spv_data.size();
            module_info.pCode = (const uint32_t *) spv_data.data();

            VkResult vk_rv = vkCreateShaderModule(device, &module_info, nullptr,
                                                  &module);
            if (unlikely(vk_rv != VK_SUCCESS))
                jitc_fail("jitc_vulkan_kernel_compile(): vkCreateShaderModule() "
                          "failed for \"%s\": %s\n\n--- Source code ---\n%s",
                          job.symbol, jitc_vulkan_error_name(vk_rv), job.source);

            uint32_t code_size = (uint32_t) spv_data.size();
            if (!from_disk)
                jitc_cache_blob_store_async(CacheKind::VulkanSpirv, disk_key,
                                            std::move(spv_data));

            artifact = UnitArtifact {
                { (void *) device, nullptr },
                (uint64_t) module,
                code_size
            };
            jitc_unit_cache_insert(JitBackend::Vulkan, job.unit_hash,
                                   cache_salt, artifact,
                                   jitc_vulkan_unit_release);
            module = (VkShaderModule) artifact.value;
        }

        VkComputePipelineCreateInfo pipeline_info {};
        pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        pipeline_info.stage.sType =
            VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        pipeline_info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        pipeline_info.stage.module = module;
        pipeline_info.stage.pName = job.symbol;
        pipeline_info.layout = (VkPipelineLayout) dev.pipeline_layout;

        VkResult vk_rv = vkCreateComputePipelines(
            device, (VkPipelineCache) dev.pipeline_cache, 1, &pipeline_info,
            nullptr, &pipeline);
        if (unlikely(vk_rv != VK_SUCCESS))
            jitc_fail("jitc_vulkan_kernel_compile(): vkCreateComputePipelines() "
                      "failed for \"%s\": %s\n\n--- Source code ---\n%s",
                      job.symbol, jitc_vulkan_error_name(vk_rv), job.source);
    }

    kernel.vulkan.pipeline = (uint64_t) pipeline;
    kernel.vulkan.layout = (uint64_t) dev.pipeline_layout;
    kernel.vulkan.module = (uint64_t) module;
    kernel.vulkan.workgroup_size = active_vulkan_wg_size;
    kernel.size = artifact.size ? artifact.size : (uint32_t) job.source_size;

    return unit_hit || from_disk;
}

void jitc_vulkan_kernel_free(int device_id, const Kernel &kernel) {
    if (device_id < 0 || (size_t) device_id >= state.vulkan_devices.size())
        return;
    VkDevice device = (VkDevice) state.vulkan_devices[device_id].device;
    if (kernel.vulkan.pipeline)
        vkDestroyPipeline(device, (VkPipeline) kernel.vulkan.pipeline, nullptr);
}
