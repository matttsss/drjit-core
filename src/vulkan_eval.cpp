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

static bool emitted_types   [(size_t) VarType::Count];
static bool emitted_psb_ptrs[(size_t) VarType::Count];
static bool emitted_fn_ptrs [(size_t) VarType::Count];

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

void jitc_vulkan_assemble_reset() {
    spv_header.clear();
    spv_annotations.clear();
    spv_types_consts.clear();
    spv_func_locals.clear();
    spv_body.clear();

    memset(emitted_types, 0, sizeof(emitted_types));
    memset(emitted_psb_ptrs, 0, sizeof(emitted_psb_ptrs));
    memset(emitted_fn_ptrs, 0, sizeof(emitted_fn_ptrs));
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

static void jitc_vulkan_render(Variable *v) {
    VarType vt = (VarType) v->type;
    if (vt != VarType::Void)
        jitc_vulkan_require_type(vt);

    switch ((VarKind) v->kind) {
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

        case VarKind::Bitcast: {
            Variable *a0 = jitc_var(v->dep[0]);
            jitc_vulkan_require_type((VarType) a0->type);
            if (strcmp(type_name_vulkan[v->type],
                       type_name_vulkan[a0->type]) == 0)
                fmt("$v = OpCopyObject $t $v\n", v, v, a0);
            else
                fmt("$v = OpBitcast $t $v\n", v, v, a0);
            break;
        }

        case VarKind::Select: {
            Variable *a0 = jitc_var(v->dep[0]),
                     *a1 = jitc_var(v->dep[1]),
                     *a2 = jitc_var(v->dep[2]);
            fmt("$v = OpSelect $t $v $v $v\n", v, v, a0, a1, a2);
            break;
        }

        default:
            jitc_raise("jitc_vulkan_render(): unhandled variable kind \"%s\" "
                       "(the Vulkan backend cannot compile kernels with this "
                       "operation yet)!",
                       var_kind_name[v->kind]);
    }
}
