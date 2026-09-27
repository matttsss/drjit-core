/*
    resources/vulkan_kernels.glsl -- Precompiled GLSL compute utility kernels
    for the Vulkan backend (compiled offline to SPIR-V via glslangValidator
    and packed by resources/pack.c into resources/vulkan_kernels.lz4).

    Copyright (c) 2026 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#version 460
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_ARB_gpu_shader_int64 : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_ballot : require
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_KHR_shader_subgroup_shuffle : require

#if defined(TYPE_U8) || defined(KERNEL_COMPRESS_SCATTER)
#  extension GL_EXT_shader_explicit_arithmetic_types_int8 : require
#endif

#if defined(TYPE_F16)
#  extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#endif

#if defined(TYPE_U64) || defined(TYPE_I64)
#  extension GL_EXT_shader_subgroup_extended_types_int64 : require
#endif

layout(local_size_x = 256) in;

// ============================================================================
//  1. compress_scatter -- materialize compacted indices from a mask prefix sum
// ============================================================================

#if defined(KERNEL_COMPRESS_SCATTER)

layout(buffer_reference, scalar) readonly buffer InBuf  { uint8_t d[]; };
layout(buffer_reference, scalar) readonly buffer PreBuf { uint    d[]; };
layout(buffer_reference, scalar) writeonly buffer OutBuf { uint   d[]; };

layout(push_constant, scalar) uniform PushConstants {
    uint64_t in_addr;
    uint64_t prefix_addr;
    uint64_t out_addr;
    uint     size;
} p;

void main() {
    uint wg_id = gl_WorkGroupID.y * gl_NumWorkGroups.x + gl_WorkGroupID.x;
    uint tid   = wg_id * 256u + gl_LocalInvocationID.x;
    if (tid >= p.size)
        return;

    InBuf in_buf = InBuf(p.in_addr);
    if (in_buf.d[tid] == uint8_t(0))
        return;

    PreBuf pre_buf = PreBuf(p.prefix_addr);
    OutBuf out_buf = OutBuf(p.out_addr);
    out_buf.d[pre_buf.d[tid] - 1u] = tid;
}

#endif // KERNEL_COMPRESS_SCATTER

// ============================================================================
//  2. mkperm (global-atomic fallback: phase 1 & phase 3)
// ============================================================================

#if defined(KERNEL_MKPERM_PHASE_1) || defined(KERNEL_MKPERM_PHASE_3)

layout(buffer_reference, scalar) readonly buffer ValBuf { uint d[]; };
layout(buffer_reference, scalar) buffer          BktBuf { uint d[]; };
layout(buffer_reference, scalar) writeonly buffer PrmBuf { uint d[]; };

layout(push_constant, scalar) uniform PushConstants {
    uint64_t values_addr;
    uint64_t buckets_addr;
    uint64_t perm_addr;
    uint     size;
} p;

void main() {
    uint wg_id = gl_WorkGroupID.y * gl_NumWorkGroups.x + gl_WorkGroupID.x;
    uint tid   = wg_id * 256u + gl_LocalInvocationID.x;
    if (tid >= p.size)
        return;

    ValBuf values  = ValBuf(p.values_addr);
    BktBuf buckets = BktBuf(p.buckets_addr);
    uint val = values.d[tid];

#if defined(KERNEL_MKPERM_PHASE_1)
    atomicAdd(buckets.d[val], 1u);
#else
    PrmBuf perm = PrmBuf(p.perm_addr);
    uint pos = atomicAdd(buckets.d[val], 1u);
    perm.d[pos] = tid;
#endif
}

#endif // KERNEL_MKPERM_PHASE_1 || KERNEL_MKPERM_PHASE_3

// ============================================================================
//  3. mkperm_detect_offsets -- detect non-empty buckets (Phase 2.5)
// ============================================================================

#if defined(KERNEL_MKPERM_DETECT_OFFSETS)

layout(buffer_reference, scalar) readonly buffer  BktBuf { uint  d[]; };
layout(buffer_reference, scalar) writeonly buffer OffBuf { uvec4 d[]; };
layout(buffer_reference, scalar) buffer           CtrBuf { uint  d[]; };

layout(push_constant, scalar) uniform PushConstants {
    uint64_t buckets_addr;
    uint64_t offsets_addr;
    uint64_t counter_addr;
    uint     bucket_count;
    uint     perm_size;
    uint     row_stride;
} p;

void main() {
    uint wg_id = gl_WorkGroupID.y * gl_NumWorkGroups.x + gl_WorkGroupID.x;
    uint tid   = wg_id * 256u + gl_LocalInvocationID.x;
    if (tid >= p.bucket_count)
        return;

    BktBuf buckets = BktBuf(p.buckets_addr);
    uint a = buckets.d[tid * p.row_stride];
    uint next_tid = min(tid + 1u, p.bucket_count - 1u);
    uint b = (tid + 1u < p.bucket_count) ? buckets.d[next_tid * p.row_stride]
                                         : p.perm_size;

    if (a != b) {
        CtrBuf counter = CtrBuf(p.counter_addr);
        OffBuf offsets = OffBuf(p.offsets_addr);
        uint idx = atomicAdd(counter.d[0], 1u);
        offsets.d[idx] = uvec4(tid, a, b - a, 0u);
    }
}

#endif // KERNEL_MKPERM_DETECT_OFFSETS

// ============================================================================
//  4. mkperm (stable "tiny" variant: phase 1 & phase 4)
// ============================================================================

#if defined(KERNEL_MKPERM_PHASE_1_TINY) || defined(KERNEL_MKPERM_PHASE_4_TINY)

layout(buffer_reference, scalar) readonly buffer  ValBuf { uint d[]; };
layout(buffer_reference, scalar) buffer           BktBuf { uint d[]; };
layout(buffer_reference, scalar) writeonly buffer PrmBuf { uint d[]; };

layout(push_constant, scalar) uniform PushConstants {
    uint64_t values_addr;
    uint64_t buckets_addr;
    uint64_t perm_addr;
    uint     size;
    uint     size_per_block;
    uint     bucket_count;
    uint     block_size;
    uint     n_blocks;
    uint     gpu_blocks_per_group;
    uint     warp_count;
    uint     rows_per_group;
} p;

// 8 KiB shared histogram space: supports warp_count * bucket_count <= 2048
shared uint sh[2048];

// Bitmask of active subgroup lanes that share 'value', computed in
// subgroup-uniform control flow via O(key_bits) subgroupBallot calls.
uvec4 mkperm_match_any(uint value, uint key_bits, bool valid_lane) {
    uvec4 peers = subgroupBallot(valid_lane);
    for (uint b = 0u; b < key_bits; ++b) {
        bool bit   = ((value >> b) & 1u) != 0u;
        uvec4 ones = subgroupBallot(bit);
        peers = bit ? (peers & ones) : (peers & ~ones);
    }
    return peers;
}

void main() {
    uint wg_id = gl_WorkGroupID.y * gl_NumWorkGroups.x + gl_WorkGroupID.x;
    uint group = wg_id / p.gpu_blocks_per_group;
    uint sub_block = wg_id - group * p.gpu_blocks_per_group;
    if (group >= p.n_blocks)
        return;

    uint lid       = gl_LocalInvocationID.x;
    uint warp_id   = gl_SubgroupID;
    uint lane      = gl_SubgroupInvocationID;
    uint warp_size = gl_SubgroupSize;
    uint bc        = p.bucket_count;
    uint wc        = p.warp_count;
    uint sh_words  = bc * wc;
    uint rows      = p.rows_per_group;

    uint user_block_start = group * p.block_size;
    uint user_block_end   = min(user_block_start + p.block_size, p.size);
    uint block_start      = user_block_start + sub_block * p.size_per_block;
    uint block_end        = min(block_start + p.size_per_block, user_block_end);

    uint total = (block_end > block_start) ? (block_end - block_start) : 0u;
    uint epw   = ((total + wc - 1u) / wc + warp_size - 1u) & ~(warp_size - 1u);

    uint warp_start = block_start + warp_id * epw;
    uint warp_end   = min(warp_start + epw, block_end);
    uint key_bits   = (bc <= 1u) ? 0u : (findMSB(bc - 1u) + 1u);
    uint gbase      = group * rows * bc;
    uint row_base   = sub_block * wc;

    ValBuf values  = ValBuf(p.values_addr);
    BktBuf buckets = BktBuf(p.buckets_addr);

#if defined(KERNEL_MKPERM_PHASE_1_TINY)
    for (uint i = lid; i < sh_words; i += 256u)
        sh[i] = 0u;
    barrier();

    if (warp_id < wc) {
        uint warp_off = warp_id * bc;
        for (uint base = warp_start; base < warp_start + epw; base += warp_size) {
            uint i = base + lane;
            bool valid_lane = i < warp_end;
            uint val = valid_lane ? values.d[min(i, p.size - 1u)] : 0u;
            uvec4 peers = mkperm_match_any(val, key_bits, valid_lane);
            if (valid_lane && lane == subgroupBallotFindLSB(peers))
                sh[warp_off + val] += subgroupBallotBitCount(peers);
        }
    }
    barrier();

    for (uint i = lid; i < sh_words; i += 256u) {
        uint w = i / bc, b = i - w * bc;
        buckets.d[gbase + b * rows + row_base + w] = sh[i];
    }
#else // KERNEL_MKPERM_PHASE_4_TINY
    PrmBuf perm = PrmBuf(p.perm_addr);
    for (uint i = lid; i < sh_words; i += 256u) {
        uint w = i / bc, b = i - w * bc;
        sh[i] = buckets.d[gbase + b * rows + row_base + w];
    }
    barrier();

    if (warp_id < wc) {
        uint warp_off = warp_id * bc;
        for (uint base = warp_start; base < warp_start + epw; base += warp_size) {
            uint i = base + lane;
            bool valid_lane = i < warp_end;
            uint val = valid_lane ? values.d[min(i, p.size - 1u)] : 0u;
            uvec4 peers = mkperm_match_any(val, key_bits, valid_lane);
            uint leader = valid_lane ? subgroupBallotFindLSB(peers) : 0u;
            uint rel    = subgroupBallotExclusiveBitCount(peers);
            uint base_off = 0u;
            if (valid_lane && lane == leader) {
                base_off = sh[warp_off + val];
                sh[warp_off + val] = base_off + subgroupBallotBitCount(peers);
            }
            base_off = subgroupShuffle(base_off, leader);
            if (valid_lane)
                perm.d[user_block_start + base_off + rel] = i;
        }
    }
#endif
}

#endif // KERNEL_MKPERM_PHASE_1_TINY || KERNEL_MKPERM_PHASE_4_TINY

// ============================================================================
//  5. Type & reduction operator definitions for block_reduce / prefix_reduce / dot
// ============================================================================

#if defined(TYPE_U8)
#  define T        uint8_t
#  define ACC_T    uint
#  define LIMIT_LO 0u
#  define LIMIT_HI 0xFFFFFFFFu
#elif defined(TYPE_F16)
#  define T        float16_t
#  define ACC_T    float
#  define LIMIT_LO uintBitsToFloat(0xFF800000u)
#  define LIMIT_HI uintBitsToFloat(0x7F800000u)
#elif defined(TYPE_F32)
#  define T        float
#  define ACC_T    float
#  define LIMIT_LO uintBitsToFloat(0xFF800000u)
#  define LIMIT_HI uintBitsToFloat(0x7F800000u)
#elif defined(TYPE_F64)
#  define T        double
#  define ACC_T    double
#  define LIMIT_LO packDouble2x32(uvec2(0u, 0xFFF00000u))
#  define LIMIT_HI packDouble2x32(uvec2(0u, 0x7FF00000u))
#elif defined(TYPE_U32)
#  define T        uint
#  define ACC_T    uint
#  define LIMIT_LO 0u
#  define LIMIT_HI 0xFFFFFFFFu
#elif defined(TYPE_U64)
#  define T        uint64_t
#  define ACC_T    uint64_t
#  define LIMIT_LO 0ul
#  define LIMIT_HI 0xFFFFFFFFFFFFFFFFul
#elif defined(TYPE_I32)
#  define T        int
#  define ACC_T    int
#  define LIMIT_LO int(0x80000000u)
#  define LIMIT_HI 0x7FFFFFFF
#elif defined(TYPE_I64)
#  define T        int64_t
#  define ACC_T    int64_t
#  define LIMIT_LO int64_t(0x8000000000000000ul)
#  define LIMIT_HI 0x7FFFFFFFFFFFFFFFl
#endif

#if defined(WIDE_OUT)
#  define OUT_T ACC_T
#elif defined(T)
#  define OUT_T T
#endif

#if defined(OP_ADD)
#  define RED_INIT        ACC_T(0)
#  define RED_OP(a, b)    ((a) + (b))
#  define SG_REDUCE(v)    subgroupAdd(v)
#  define SG_EXCL_SCAN(v) subgroupExclusiveAdd(v)
#elif defined(OP_MUL)
#  define RED_INIT        ACC_T(1)
#  define RED_OP(a, b)    ((a) * (b))
#  define SG_REDUCE(v)    subgroupMul(v)
#  define SG_EXCL_SCAN(v) subgroupExclusiveMul(v)
#elif defined(OP_MIN)
#  define RED_INIT        LIMIT_HI
#  define RED_OP(a, b)    min(a, b)
#  define SG_REDUCE(v)    subgroupMin(v)
#  define SG_EXCL_SCAN(v) subgroupExclusiveMin(v)
#elif defined(OP_MAX)
#  define RED_INIT        LIMIT_LO
#  define RED_OP(a, b)    max(a, b)
#  define SG_REDUCE(v)    subgroupMax(v)
#  define SG_EXCL_SCAN(v) subgroupExclusiveMax(v)
#elif defined(OP_OR)
#  define RED_INIT        ACC_T(0)
#  define RED_OP(a, b)    ((a) | (b))
#  define SG_REDUCE(v)    subgroupOr(v)
#  define SG_EXCL_SCAN(v) subgroupExclusiveOr(v)
#elif defined(OP_AND)
#  define RED_INIT        (~ACC_T(0))
#  define RED_OP(a, b)    ((a) & (b))
#  define SG_REDUCE(v)    subgroupAnd(v)
#  define SG_EXCL_SCAN(v) subgroupExclusiveAnd(v)
#endif

// ============================================================================
//  6. block_reduce_small & block_reduce (chunked)
// ============================================================================

#if defined(KERNEL_BLOCK_REDUCE_SMALL) || defined(KERNEL_BLOCK_REDUCE_CHUNK)

layout(buffer_reference, scalar) readonly buffer  InBuf  { T     d[]; };
layout(buffer_reference, scalar) writeonly buffer OutBuf { OUT_T d[]; };

layout(push_constant, scalar) uniform PushConstants {
    uint64_t in_addr;
    uint64_t out_addr;
    uint     size;
    uint     block_size;
    uint     block_count;
    uint     chunk_size;
    uint     chunks_per_block;
} p;

#if defined(KERNEL_BLOCK_REDUCE_SMALL)

void main() {
    uint wg_id = gl_WorkGroupID.y * gl_NumWorkGroups.x + gl_WorkGroupID.x;
    uint b     = wg_id * 256u + gl_LocalInvocationID.x;
    if (b >= p.block_count)
        return;

    uint base = b * p.block_size;
    uint n    = min(p.size - base, p.block_size);

    InBuf  in_buf  = InBuf(p.in_addr);
    OutBuf out_buf = OutBuf(p.out_addr);

    ACC_T acc = RED_INIT;
    uint i = 0u;
    for (; i + 4u <= n; i += 4u) {
        ACC_T v01 = RED_OP(ACC_T(in_buf.d[base + i]),      ACC_T(in_buf.d[base + i + 1u]));
        ACC_T v23 = RED_OP(ACC_T(in_buf.d[base + i + 2u]), ACC_T(in_buf.d[base + i + 3u]));
        acc = RED_OP(acc, RED_OP(v01, v23));
    }
    for (; i < n; ++i)
        acc = RED_OP(acc, ACC_T(in_buf.d[base + i]));

    out_buf.d[b] = OUT_T(acc);
}

#else // KERNEL_BLOCK_REDUCE_CHUNK

shared ACC_T sh[64];

void main() {
    uint wg_id     = gl_WorkGroupID.y * gl_NumWorkGroups.x + gl_WorkGroupID.x;
    uint block     = wg_id / p.chunks_per_block;
    uint rel_chunk = wg_id - block * p.chunks_per_block;
    if (block >= p.block_count)
        return;

    uint tid         = gl_LocalInvocationID.x;
    uint block_start = block * p.block_size;
    uint block_len   = min(p.size - block_start, p.block_size);
    uint chunk_start = rel_chunk * p.chunk_size;

    InBuf  in_buf  = InBuf(p.in_addr);
    OutBuf out_buf = OutBuf(p.out_addr);

    ACC_T acc = RED_INIT;
    for (uint off = tid * 4u; off < p.chunk_size; off += 256u * 4u) {
        uint pib = chunk_start + off;
        uint pos = block_start + pib;
        if (pib + 3u < block_len) {
            ACC_T v01 = RED_OP(ACC_T(in_buf.d[pos]),      ACC_T(in_buf.d[pos + 1u]));
            ACC_T v23 = RED_OP(ACC_T(in_buf.d[pos + 2u]), ACC_T(in_buf.d[pos + 3u]));
            acc = RED_OP(acc, RED_OP(v01, v23));
        } else {
            for (uint j = 0u; j < 4u; ++j) {
                if (pib + j < block_len)
                    acc = RED_OP(acc, ACC_T(in_buf.d[min(pos + j, p.size - 1u)]));
            }
        }
    }

    acc = SG_REDUCE(acc);

    if (gl_NumSubgroups > 1u) {
        if (gl_SubgroupInvocationID == 0u)
            sh[gl_SubgroupID] = acc;
        barrier();
        if (gl_SubgroupID == 0u) {
            ACC_T s = RED_INIT;
            for (uint i = gl_SubgroupInvocationID; i < gl_NumSubgroups; i += gl_SubgroupSize)
                s = RED_OP(s, sh[i]);
            acc = SG_REDUCE(s);
        }
    }

    if (tid == 0u)
        out_buf.d[block * p.chunks_per_block + rel_chunk] = OUT_T(acc);
}

#endif // KERNEL_BLOCK_REDUCE_SMALL / CHUNK
#endif // KERNEL_BLOCK_REDUCE_*

// ============================================================================
//  7. block_prefix_reduce_small & block_prefix_reduce (chunked scan)
// ============================================================================

#if defined(KERNEL_BLOCK_PREFIX_REDUCE_SMALL) || defined(KERNEL_BLOCK_PREFIX_REDUCE)

layout(buffer_reference, scalar) readonly buffer InBuf  { T     d[]; };
layout(buffer_reference, scalar) buffer          OutBuf { OUT_T d[]; };
layout(buffer_reference, scalar) readonly buffer PreBuf { ACC_T d[]; };

layout(push_constant, scalar) uniform PushConstants {
    uint64_t in_addr;
    uint64_t out_addr;
    uint64_t prefixes_addr;
    uint     size;
    uint     block_size;
    uint     block_count;
    uint     chunk_size;
    uint     chunks_per_block;
    uint     exclusive;
    uint     reverse;
} p;

#if defined(KERNEL_BLOCK_PREFIX_REDUCE_SMALL)

void main() {
    uint wg_id = gl_WorkGroupID.y * gl_NumWorkGroups.x + gl_WorkGroupID.x;
    uint b     = wg_id * 256u + gl_LocalInvocationID.x;
    if (b >= p.block_count)
        return;

    uint base = b * p.block_size;
    uint n    = min(p.size - base, p.block_size);

    InBuf  in_buf  = InBuf(p.in_addr);
    OutBuf out_buf = OutBuf(p.out_addr);

    ACC_T acc = RED_INIT;
    bool excl = p.exclusive != 0u;
    bool rev  = p.reverse != 0u;

    for (uint i = 0u; i < n; ++i) {
        uint  pos  = rev ? (base + (n - 1u - i)) : (base + i);
        ACC_T val  = ACC_T(in_buf.d[pos]);
        ACC_T next = RED_OP(acc, val);
        out_buf.d[pos] = OUT_T(excl ? acc : next);
        acc = next;
    }
}

#else // KERNEL_BLOCK_PREFIX_REDUCE

shared ACC_T sh[64];

void main() {
    uint wg_id     = gl_WorkGroupID.y * gl_NumWorkGroups.x + gl_WorkGroupID.x;
    uint block     = wg_id / p.chunks_per_block;
    uint rel_chunk = wg_id - block * p.chunks_per_block;
    if (block >= p.block_count)
        return;

    uint tid         = gl_LocalInvocationID.x;
    uint lane        = gl_SubgroupInvocationID;
    uint sgid        = gl_SubgroupID;
    uint nsg         = gl_NumSubgroups;
    uint sg_size     = gl_SubgroupSize;
    uint block_start = block * p.block_size;
    uint block_len   = min(p.size - block_start, p.block_size);
    bool rev         = p.reverse != 0u;

    InBuf  in_buf  = InBuf(p.in_addr);
    OutBuf out_buf = OutBuf(p.out_addr);

    ACC_T v[4];
    bool  valid[4];
    uint  pos[4];
    for (uint j = 0u; j < 4u; ++j) {
        uint s = tid * 4u + j;
        uint pib = rev
            ? ((rel_chunk + 1u) * p.chunk_size - 1u - s)
            : (rel_chunk * p.chunk_size + s);
        valid[j] = (s < p.chunk_size) && (pib < block_len);
        pos[j]   = valid[j] ? (block_start + pib) : 0u;
        v[j]     = valid[j] ? ACC_T(in_buf.d[pos[j]]) : RED_INIT;
    }

    // Inclusive scan of the 4-element grain in registers
    v[1] = RED_OP(v[0], v[1]);
    v[2] = RED_OP(v[1], v[2]);
    v[3] = RED_OP(v[2], v[3]);

    // Exclusive scan of per-thread totals within each subgroup
    ACC_T prefix = SG_EXCL_SCAN(v[3]);

    // Exclusive scan of per-subgroup totals across the workgroup
    if (nsg > 1u) {
        if (lane == sg_size - 1u)
            sh[sgid] = RED_OP(prefix, v[3]);
        barrier();
        if (sgid == 0u) {
            ACC_T running = RED_INIT;
            for (uint base = 0u; base < nsg; base += sg_size) {
                uint  idx       = base + lane;
                ACC_T t         = (idx < nsg) ? sh[idx] : RED_INIT;
                ACC_T scan_excl = SG_EXCL_SCAN(t);
                ACC_T scan_incl = RED_OP(scan_excl, t);
                if (idx < nsg)
                    sh[idx] = RED_OP(running, scan_excl);
                running = RED_OP(running, subgroupShuffle(scan_incl, sg_size - 1u));
            }
        }
        barrier();
        prefix = RED_OP(sh[sgid], prefix);
    }

    // Add exclusive prefix of preceding chunks in the same block (if multi-chunk).
    // When chunks_per_block == 1 the host sets prefixes_addr = out_addr so any
    // speculated load by software drivers (e.g. Mesa llvmpipe) reads index 0 of
    // a valid buffer.
    if (p.chunks_per_block > 1u) {
        PreBuf pre_buf = PreBuf(p.prefixes_addr);
        uint pre_idx = (p.prefixes_addr != p.out_addr)
                           ? (block * p.chunks_per_block + rel_chunk)
                           : 0u;
        prefix = RED_OP(pre_buf.d[pre_idx], prefix);
    }

    ACC_T r[4];
    if (p.exclusive != 0u) {
        r[0] = prefix;
        r[1] = RED_OP(prefix, v[0]);
        r[2] = RED_OP(prefix, v[1]);
        r[3] = RED_OP(prefix, v[2]);
    } else {
        r[0] = RED_OP(prefix, v[0]);
        r[1] = RED_OP(prefix, v[1]);
        r[2] = RED_OP(prefix, v[2]);
        r[3] = RED_OP(prefix, v[3]);
    }

    for (uint j = 0u; j < 4u; ++j) {
        if (valid[j])
            out_buf.d[pos[j]] = OUT_T(r[j]);
    }
}

#endif // KERNEL_BLOCK_PREFIX_REDUCE_SMALL / CHUNK
#endif // KERNEL_BLOCK_PREFIX_REDUCE_*

// ============================================================================
//  8. reduce_dot -- fused dot-product chunk reduction
// ============================================================================

#if defined(KERNEL_REDUCE_DOT)

layout(buffer_reference, scalar) readonly buffer  InBuf  { T d[]; };
layout(buffer_reference, scalar) writeonly buffer OutBuf { T d[]; };

layout(push_constant, scalar) uniform PushConstants {
    uint64_t in1_addr;
    uint64_t in2_addr;
    uint64_t out_addr;
    uint     size;
    uint     chunk_size;
    uint     chunk_count;
} p;

shared ACC_T sh[64];

void main() {
    uint wg_id = gl_WorkGroupID.y * gl_NumWorkGroups.x + gl_WorkGroupID.x;
    if (wg_id >= p.chunk_count)
        return;

    uint tid = gl_LocalInvocationID.x;
    InBuf  a       = InBuf(p.in1_addr);
    InBuf  b       = InBuf(p.in2_addr);
    OutBuf out_buf = OutBuf(p.out_addr);

    uint chunk_start = wg_id * p.chunk_size;
    uint chunk_len   = min(p.size - chunk_start, p.chunk_size);

    T acc_t = T(0);
    for (uint off = tid * 4u; off < p.chunk_size; off += 256u * 4u) {
        uint pos = chunk_start + off;
        if (off + 3u < chunk_len) {
            acc_t = fma(a.d[pos],      b.d[pos],      acc_t);
            acc_t = fma(a.d[pos + 1u], b.d[pos + 1u], acc_t);
            acc_t = fma(a.d[pos + 2u], b.d[pos + 2u], acc_t);
            acc_t = fma(a.d[pos + 3u], b.d[pos + 3u], acc_t);
        } else {
            for (uint j = 0u; j < 4u; ++j) {
                if (off + j < chunk_len) {
                    uint idx = min(pos + j, p.size - 1u);
                    acc_t = fma(a.d[idx], b.d[idx], acc_t);
                }
            }
        }
    }

    ACC_T acc = subgroupAdd(ACC_T(acc_t));

    if (gl_NumSubgroups > 1u) {
        if (gl_SubgroupInvocationID == 0u)
            sh[gl_SubgroupID] = acc;
        barrier();
        if (gl_SubgroupID == 0u) {
            ACC_T s = ACC_T(0);
            for (uint i = gl_SubgroupInvocationID; i < gl_NumSubgroups; i += gl_SubgroupSize)
                s += sh[i];
            acc = subgroupAdd(s);
        }
    }

    if (tid == 0u)
        out_buf.d[wg_id] = T(acc);
}

#endif // KERNEL_REDUCE_DOT
