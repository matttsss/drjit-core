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

#if defined(TYPE_U16)
#  extension GL_EXT_shader_explicit_arithmetic_types_int16 : require
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

layout(buffer_reference, scalar) readonly buffer  InBuf    { uint8_t d[]; };
layout(buffer_reference, scalar) readonly buffer  PreBuf   { uint    d[]; };
layout(buffer_reference, scalar) writeonly buffer OutBuf   { uint    d[]; };
layout(buffer_reference, scalar) writeonly buffer CountBuf { uint    d[]; };

layout(push_constant, scalar) uniform PushConstants {
    uint64_t in_addr;
    uint64_t prefix_addr;
    uint64_t out_addr;
    uint64_t count_addr;
    uint     size;
    uint     chunk_size;
    uint     chunk_count;
    uint     chunks_scanned;
} p;

shared uint sh_sg[257];

void main() {
    uint wg_id = gl_WorkGroupID.y * gl_NumWorkGroups.x + gl_WorkGroupID.x;
    if (wg_id >= p.chunk_count)
        return;

    uint tid     = gl_LocalInvocationID.x;
    uint lane    = gl_SubgroupInvocationID;
    uint sgid    = gl_SubgroupID;
    uint nsg     = gl_NumSubgroups;
    uint sg_size = gl_SubgroupSize;

    uint chunk_running = 0u;
    if (p.chunk_count > 1u) {
        PreBuf pre_buf = PreBuf(p.prefix_addr);
        if (p.chunks_scanned != 0u) {
            chunk_running = pre_buf.d[wg_id];
        } else {
            uint pre_sum = 0u;
            for (uint i = tid; i < wg_id; i += 256u)
                pre_sum += pre_buf.d[i];
            pre_sum = subgroupAdd(pre_sum);
            if (nsg > 1u) {
                if (lane == 0u)
                    sh_sg[sgid] = pre_sum;
                barrier();
                if (sgid == 0u) {
                    uint s = 0u;
                    for (uint i = lane; i < nsg; i += sg_size)
                        s += sh_sg[i];
                    s = subgroupAdd(s);
                    if (lane == 0u)
                        sh_sg[0] = s;
                }
                barrier();
                pre_sum = sh_sg[0];
                barrier();
            }
            chunk_running = pre_sum;
        }
    }

    InBuf  in_buf  = InBuf(p.in_addr);
    OutBuf out_buf = OutBuf(p.out_addr);

    uint chunk_start = wg_id * p.chunk_size;
    uint chunk_end   = min(chunk_start + p.chunk_size, p.size);
    uint n_entries   = 4u * nsg;

    for (uint tile_base = chunk_start; tile_base < chunk_end; tile_base += 1024u) {
        uint pos0 = tile_base + tid;
        uint pos1 = pos0 + 256u;
        uint pos2 = pos0 + 512u;
        uint pos3 = pos0 + 768u;

        bool act0, act1, act2, act3;
        if (tile_base + 1024u <= chunk_end) {
            act0 = in_buf.d[pos0] != uint8_t(0);
            act1 = in_buf.d[pos1] != uint8_t(0);
            act2 = in_buf.d[pos2] != uint8_t(0);
            act3 = in_buf.d[pos3] != uint8_t(0);
        } else {
            act0 = (pos0 < chunk_end) && (in_buf.d[pos0] != uint8_t(0));
            act1 = (pos1 < chunk_end) && (in_buf.d[pos1] != uint8_t(0));
            act2 = (pos2 < chunk_end) && (in_buf.d[pos2] != uint8_t(0));
            act3 = (pos3 < chunk_end) && (in_buf.d[pos3] != uint8_t(0));
        }

        uvec4 bal0 = subgroupBallot(act0);
        uvec4 bal1 = subgroupBallot(act1);
        uvec4 bal2 = subgroupBallot(act2);
        uvec4 bal3 = subgroupBallot(act3);

        if (lane == 0u) {
            sh_sg[0u * nsg + sgid] = subgroupBallotBitCount(bal0);
            sh_sg[1u * nsg + sgid] = subgroupBallotBitCount(bal1);
            sh_sg[2u * nsg + sgid] = subgroupBallotBitCount(bal2);
            sh_sg[3u * nsg + sgid] = subgroupBallotBitCount(bal3);
        }
        barrier();

        if (sgid == 0u) {
            if (n_entries <= sg_size) {
                uint t = (lane < n_entries) ? sh_sg[lane] : 0u;
                uint scan_excl = subgroupExclusiveAdd(t);
                if (lane < n_entries)
                    sh_sg[lane] = scan_excl;
                if (lane == n_entries - 1u)
                    sh_sg[n_entries] = scan_excl + t;
            } else {
                uint running = 0u;
                for (uint base = 0u; base < n_entries; base += sg_size) {
                    uint idx       = base + lane;
                    uint t         = (idx < n_entries) ? sh_sg[idx] : 0u;
                    uint scan_excl = subgroupExclusiveAdd(t);
                    uint scan_incl = scan_excl + t;
                    if (idx < n_entries)
                        sh_sg[idx] = running + scan_excl;
                    running += subgroupShuffle(scan_incl, sg_size - 1u);
                }
                if (lane == 0u)
                    sh_sg[n_entries] = running;
            }
        }
        barrier();

        if (act0) out_buf.d[chunk_running + sh_sg[0u * nsg + sgid] + subgroupBallotExclusiveBitCount(bal0)] = pos0;
        if (act1) out_buf.d[chunk_running + sh_sg[1u * nsg + sgid] + subgroupBallotExclusiveBitCount(bal1)] = pos1;
        if (act2) out_buf.d[chunk_running + sh_sg[2u * nsg + sgid] + subgroupBallotExclusiveBitCount(bal2)] = pos2;
        if (act3) out_buf.d[chunk_running + sh_sg[3u * nsg + sgid] + subgroupBallotExclusiveBitCount(bal3)] = pos3;

        chunk_running += sh_sg[n_entries];
        if (tile_base + 1024u < chunk_end)
            barrier();
    }

    if (wg_id == p.chunk_count - 1u && tid == 0u) {
        CountBuf count_buf = CountBuf(p.count_addr);
        count_buf.d[0] = chunk_running;
    }
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
#elif defined(TYPE_U16)
#  define T        uint16_t
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
    InBuf  in_buf  = InBuf(p.in_addr);
    OutBuf out_buf = OutBuf(p.out_addr);

    if (p.chunk_size != 0u) {
        uint b = wg_id * gl_NumSubgroups + gl_SubgroupID;
        if (b >= p.block_count)
            return;

        uint base    = b * p.block_size;
        uint n       = min(p.size - base, p.block_size);
        uint lane    = gl_SubgroupInvocationID;
        uint sg_size = gl_SubgroupSize;
        uint step4   = sg_size * 4u;

        ACC_T acc = RED_INIT;
        uint i = 0u;
        for (; i + step4 <= n; i += step4) {
            uint idx = base + i + lane;
            ACC_T v0 = ACC_T(in_buf.d[idx]);
            ACC_T v1 = ACC_T(in_buf.d[idx + sg_size]);
            ACC_T v2 = ACC_T(in_buf.d[idx + sg_size * 2u]);
            ACC_T v3 = ACC_T(in_buf.d[idx + sg_size * 3u]);
            acc = RED_OP(acc, RED_OP(RED_OP(v0, v1), RED_OP(v2, v3)));
        }
        for (; i < n; i += sg_size) {
            if (i + lane < n)
                acc = RED_OP(acc, ACC_T(in_buf.d[base + i + lane]));
        }
        acc = SG_REDUCE(acc);
        if (lane == 0u)
            out_buf.d[b] = OUT_T(acc);
        return;
    }

    uint b = wg_id * 256u + gl_LocalInvocationID.x;
    if (b >= p.block_count)
        return;

    uint base = b * p.block_size;
    uint n    = min(p.size - base, p.block_size);

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
    uint chunk_end   = min(chunk_start + p.chunk_size, block_len);

    InBuf  in_buf  = InBuf(p.in_addr);
    OutBuf out_buf = OutBuf(p.out_addr);

    ACC_T acc = RED_INIT;
    for (uint base_off = chunk_start; base_off < chunk_end; base_off += 1024u) {
        uint pib = base_off + tid;
        uint pos = block_start + pib;
        if (base_off + 1024u <= chunk_end) {
            ACC_T v0 = ACC_T(in_buf.d[pos]);
            ACC_T v1 = ACC_T(in_buf.d[pos + 256u]);
            ACC_T v2 = ACC_T(in_buf.d[pos + 512u]);
            ACC_T v3 = ACC_T(in_buf.d[pos + 768u]);
            acc = RED_OP(acc, RED_OP(RED_OP(v0, v1), RED_OP(v2, v3)));
        } else {
            if (pib < chunk_end)        acc = RED_OP(acc, ACC_T(in_buf.d[pos]));
            if (pib + 256u < chunk_end) acc = RED_OP(acc, ACC_T(in_buf.d[pos + 256u]));
            if (pib + 512u < chunk_end) acc = RED_OP(acc, ACC_T(in_buf.d[pos + 512u]));
            if (pib + 768u < chunk_end) acc = RED_OP(acc, ACC_T(in_buf.d[pos + 768u]));
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
        out_buf.d[wg_id] = OUT_T(acc);
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

shared ACC_T sh[65];

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
    bool excl        = p.exclusive != 0u;

    InBuf  in_buf  = InBuf(p.in_addr);
    OutBuf out_buf = OutBuf(p.out_addr);

    ACC_T chunk_running = RED_INIT;
    if (p.chunks_per_block > 1u) {
        PreBuf pre_buf = PreBuf(p.prefixes_addr);
        uint pre_idx = (p.prefixes_addr != p.out_addr)
                           ? wg_id
                           : 0u;
        chunk_running = pre_buf.d[pre_idx];
    }

    uint chunk_start = rel_chunk * p.chunk_size;
    uint chunk_span  = rev ? p.chunk_size
                           : min(p.chunk_size,
                                 block_len > chunk_start ? (block_len - chunk_start) : 0u);

    for (uint tile_off = 0u; tile_off < chunk_span; tile_off += 1024u) {
        uint s0   = tile_off + tid * 4u;
        bool fast = !rev && (s0 + 3u < chunk_span);

        ACC_T v0, v1, v2, v3;
        bool  valid[4];
        uint  pos[4];

        if (fast) {
            uint gpos = block_start + chunk_start + s0;
            v0 = ACC_T(in_buf.d[gpos]);
            v1 = ACC_T(in_buf.d[gpos + 1u]);
            v2 = ACC_T(in_buf.d[gpos + 2u]);
            v3 = ACC_T(in_buf.d[gpos + 3u]);
        } else {
            for (uint j = 0u; j < 4u; ++j) {
                uint s = s0 + j;
                uint pib = rev
                    ? ((rel_chunk + 1u) * p.chunk_size - 1u - s)
                    : (chunk_start + s);
                valid[j] = (s < p.chunk_size) && (pib < block_len);
                pos[j]   = valid[j] ? (block_start + pib) : 0u;
            }
            v0 = valid[0] ? ACC_T(in_buf.d[pos[0]]) : RED_INIT;
            v1 = valid[1] ? ACC_T(in_buf.d[pos[1]]) : RED_INIT;
            v2 = valid[2] ? ACC_T(in_buf.d[pos[2]]) : RED_INIT;
            v3 = valid[3] ? ACC_T(in_buf.d[pos[3]]) : RED_INIT;
        }

        ACC_T s1 = RED_OP(v0, v1);
        ACC_T s2 = RED_OP(s1, v2);
        ACC_T s3 = RED_OP(s2, v3);

        ACC_T prefix = SG_EXCL_SCAN(s3);
        ACC_T sg_tot = RED_OP(prefix, s3);
        ACC_T tile_tot = sg_tot;

        if (nsg > 1u) {
            if (lane == sg_size - 1u)
                sh[sgid] = sg_tot;
            barrier();
            if (sgid == 0u) {
                if (nsg <= sg_size) {
                    ACC_T t = (lane < nsg) ? sh[lane] : RED_INIT;
                    ACC_T scan_excl = SG_EXCL_SCAN(t);
                    if (lane < nsg)
                        sh[lane] = scan_excl;
                    if (lane == nsg - 1u)
                        sh[nsg] = RED_OP(scan_excl, t);
                } else {
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
                    if (lane == 0u)
                        sh[nsg] = running;
                }
            }
            barrier();
            prefix   = RED_OP(sh[sgid], prefix);
            tile_tot = sh[nsg];
        } else {
            tile_tot = subgroupShuffle(sg_tot, sg_size - 1u);
        }

        prefix = RED_OP(chunk_running, prefix);
        chunk_running = RED_OP(chunk_running, tile_tot);

        ACC_T r0, r1, r2, r3;
        if (excl) {
            r0 = prefix;
            r1 = RED_OP(prefix, v0);
            r2 = RED_OP(prefix, s1);
            r3 = RED_OP(prefix, s2);
        } else {
            r0 = RED_OP(prefix, v0);
            r1 = RED_OP(prefix, s1);
            r2 = RED_OP(prefix, s2);
            r3 = RED_OP(prefix, s3);
        }

        if (fast) {
            uint gpos = block_start + chunk_start + s0;
            out_buf.d[gpos]      = OUT_T(r0);
            out_buf.d[gpos + 1u] = OUT_T(r1);
            out_buf.d[gpos + 2u] = OUT_T(r2);
            out_buf.d[gpos + 3u] = OUT_T(r3);
        } else {
            if (valid[0]) out_buf.d[pos[0]] = OUT_T(r0);
            if (valid[1]) out_buf.d[pos[1]] = OUT_T(r1);
            if (valid[2]) out_buf.d[pos[2]] = OUT_T(r2);
            if (valid[3]) out_buf.d[pos[3]] = OUT_T(r3);
        }

        if (tile_off + 1024u < chunk_span && nsg > 1u)
            barrier();
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

    T acc0 = T(0), acc1 = T(0);
    for (uint base_off = 0u; base_off < chunk_len; base_off += 1024u) {
        uint off = base_off + tid;
        uint pos = chunk_start + off;
        if (base_off + 1024u <= chunk_len) {
            acc0 = fma(a.d[pos],        b.d[pos],        acc0);
            acc1 = fma(a.d[pos + 256u], b.d[pos + 256u], acc1);
            acc0 = fma(a.d[pos + 512u], b.d[pos + 512u], acc0);
            acc1 = fma(a.d[pos + 768u], b.d[pos + 768u], acc1);
        } else {
            if (off < chunk_len)        acc0 = fma(a.d[pos],        b.d[pos],        acc0);
            if (off + 256u < chunk_len) acc1 = fma(a.d[pos + 256u], b.d[pos + 256u], acc1);
            if (off + 512u < chunk_len) acc0 = fma(a.d[pos + 512u], b.d[pos + 512u], acc0);
            if (off + 768u < chunk_len) acc1 = fma(a.d[pos + 768u], b.d[pos + 768u], acc1);
        }
    }

    ACC_T acc = subgroupAdd(ACC_T(acc0 + acc1));

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

// ============================================================================
//  9. deinterleave & interleave -- multi-channel texture packing/unpacking
// ============================================================================

#if defined(KERNEL_DEINTERLEAVE) || defined(KERNEL_INTERLEAVE)

layout(buffer_reference, scalar) readonly buffer  SrcBuf { T d[]; };
layout(buffer_reference, scalar) writeonly buffer DstBuf { T d[]; };

layout(push_constant, scalar) uniform PushConstants {
    uint64_t src_addr;
    uint64_t dst_addr;
    uint     size;
    uint     n_channels;
    uint     ci;
    uint     tex_base;
    uint     c_valid;
} p;

void main() {
    uint wg_id = gl_WorkGroupID.y * gl_NumWorkGroups.x + gl_WorkGroupID.x;
    uint tid   = wg_id * 256u + gl_LocalInvocationID.x;
    if (tid >= p.size)
        return;

    SrcBuf src = SrcBuf(p.src_addr);
    DstBuf dst = DstBuf(p.dst_addr);

#if defined(KERNEL_DEINTERLEAVE)
    uint pix = tid / p.ci;
    uint c   = tid - pix * p.ci;
    if (c < p.c_valid)
        dst.d[tid] = src.d[pix * p.n_channels + p.tex_base + c];
    else
        dst.d[tid] = T(0);
#else
    uint pix = tid / p.c_valid;
    uint c   = tid - pix * p.c_valid;
    dst.d[pix * p.n_channels + p.tex_base + c] = src.d[pix * p.ci + c];
#endif
}

#endif // KERNEL_DEINTERLEAVE || KERNEL_INTERLEAVE

