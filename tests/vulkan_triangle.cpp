/*
    tests/vulkan_triangle.cpp -- Vulkan ray tracing test (VK_KHR_acceleration_structure
    & VK_KHR_ray_query), analogous to tests/metal_triangle.mm and tests/optix_triangle.cpp.

    Tests:
    1. Single-triangle BLAS/TLAS construction and closest-hit 16x16 ray grid
       (with kernel caching across iterations, scene owner handle, and cleanup callback).
    2. Shadow rays, active lane mask, per-lane visibility mask (vmask), and
       triangle backface culling (geometry_types_mask = 0x8).
    3. Custom procedural AABB intersection (jit_isect_begin / jit_isect_end /
       jit_isect_bind / jit_isect_unbind) for analytic spheres, both standalone
       (geometry_types_mask = 0x2) and in a hybrid triangle + AABB scene
       (geometry_types_mask = 0x1 | 0x2).
    4. Double-precision (Float64) ray tracing and custom intersection.
*/

#include <drjit-core/array.h>
#include <drjit-core/vulkan.h>
#include "vulkan_api.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace dr = drjit;

using Float   = dr::VulkanArray<float>;
using Float64 = dr::VulkanArray<double>;
using UInt32  = dr::VulkanArray<uint32_t>;
using Int32   = dr::VulkanArray<int32_t>;
using UInt64  = dr::VulkanArray<uint64_t>;
using Mask    = dr::VulkanArray<bool>;

#define CHECK(cond)                                                            \
    do {                                                                       \
        if (!(cond)) {                                                         \
            fprintf(stderr, "Assertion failed at %s:%d: %s\n",                 \
                    __FILE__, __LINE__, #cond);                                \
            std::abort();                                                      \
        }                                                                      \
    } while (0)

// Dynamically resolved Vulkan entry points used by the test
static PFN_vkGetAccelerationStructureBuildSizesKHR
    pfn_vkGetAccelerationStructureBuildSizesKHR;
static PFN_vkCreateAccelerationStructureKHR
    pfn_vkCreateAccelerationStructureKHR;
static PFN_vkDestroyAccelerationStructureKHR
    pfn_vkDestroyAccelerationStructureKHR;
static PFN_vkGetAccelerationStructureDeviceAddressKHR
    pfn_vkGetAccelerationStructureDeviceAddressKHR;
static PFN_vkCmdBuildAccelerationStructuresKHR
    pfn_vkCmdBuildAccelerationStructuresKHR;
static PFN_vkCmdPipelineBarrier
    pfn_vkCmdPipelineBarrier;

static void resolve_vk_functions() {
    auto resolve = [](const char *name) -> void * {
        void *p = jit_vulkan_lookup(name);
        CHECK(p != nullptr);
        return p;
    };
    pfn_vkGetAccelerationStructureBuildSizesKHR =
        (PFN_vkGetAccelerationStructureBuildSizesKHR) resolve(
            "vkGetAccelerationStructureBuildSizesKHR");
    pfn_vkCreateAccelerationStructureKHR =
        (PFN_vkCreateAccelerationStructureKHR) resolve(
            "vkCreateAccelerationStructureKHR");
    pfn_vkDestroyAccelerationStructureKHR =
        (PFN_vkDestroyAccelerationStructureKHR) resolve(
            "vkDestroyAccelerationStructureKHR");
    pfn_vkGetAccelerationStructureDeviceAddressKHR =
        (PFN_vkGetAccelerationStructureDeviceAddressKHR) resolve(
            "vkGetAccelerationStructureDeviceAddressKHR");
    pfn_vkCmdBuildAccelerationStructuresKHR =
        (PFN_vkCmdBuildAccelerationStructuresKHR) resolve(
            "vkCmdBuildAccelerationStructuresKHR");
    pfn_vkCmdPipelineBarrier =
        (PFN_vkCmdPipelineBarrier) resolve("vkCmdPipelineBarrier");
}

struct AccelBuffer {
    VkAccelerationStructureKHR handle = VK_NULL_HANDLE;
    void *storage = nullptr;
    VkDeviceAddress address = 0;
};

static void destroy_accel(VkDevice device, AccelBuffer &as) {
    if (as.handle) {
        pfn_vkDestroyAccelerationStructureKHR(device, as.handle, nullptr);
        as.handle = VK_NULL_HANDLE;
    }
    if (as.storage) {
        jit_free(as.storage);
        as.storage = nullptr;
    }
    as.address = 0;
}

static AccelBuffer build_accel(VkDevice device,
                               VkAccelerationStructureTypeKHR type,
                               const VkAccelerationStructureGeometryKHR *geoms,
                               const uint32_t *prim_counts,
                               uint32_t geom_count) {
    VkAccelerationStructureBuildGeometryInfoKHR build_info {};
    build_info.sType =
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    build_info.type = type;
    build_info.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    build_info.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    build_info.geometryCount = geom_count;
    build_info.pGeometries = geoms;

    VkAccelerationStructureBuildSizesInfoKHR sizes {};
    sizes.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
    pfn_vkGetAccelerationStructureBuildSizesKHR(
        device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &build_info,
        prim_counts, &sizes);

    AccelBuffer result;
    result.storage =
        jit_malloc(JitBackend::Vulkan, sizes.accelerationStructureSize);
    CHECK(result.storage != nullptr);

    size_t storage_offset = 0;
    VkBuffer vk_buf =
        (VkBuffer) jit_vulkan_lookup_buffer(result.storage, &storage_offset);
    CHECK(vk_buf != VK_NULL_HANDLE);

    VkAccelerationStructureCreateInfoKHR create_info {};
    create_info.sType =
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
    create_info.buffer = vk_buf;
    create_info.offset = storage_offset;
    create_info.size = sizes.accelerationStructureSize;
    create_info.type = type;

    VkResult rv = pfn_vkCreateAccelerationStructureKHR(
        device, &create_info, nullptr, &result.handle);
    CHECK(rv == VK_SUCCESS);

    // Allocate aligned scratch buffer (pad by 256 bytes for minAccelerationStructureScratchOffsetAlignment)
    void *scratch =
        jit_malloc(JitBackend::Vulkan, sizes.buildScratchSize + 256);
    CHECK(scratch != nullptr);
    VkDeviceAddress scratch_addr = jit_vulkan_device_address(scratch);
    scratch_addr = (scratch_addr + 255ull) & ~255ull;

    build_info.dstAccelerationStructure = result.handle;
    build_info.scratchData.deviceAddress = scratch_addr;

    std::vector<VkAccelerationStructureBuildRangeInfoKHR> ranges(geom_count);
    std::vector<const VkAccelerationStructureBuildRangeInfoKHR *> range_ptrs(
        geom_count);
    for (uint32_t i = 0; i < geom_count; ++i) {
        memset(&ranges[i], 0, sizeof(ranges[i]));
        ranges[i].primitiveCount = prim_counts[i];
        range_ptrs[i] = &ranges[i];
    }

    VkCommandBuffer cmd = (VkCommandBuffer) jit_vulkan_command_buffer();
    CHECK(cmd != VK_NULL_HANDLE);
    pfn_vkCmdBuildAccelerationStructuresKHR(cmd, 1, &build_info,
                                            range_ptrs.data());

    VkMemoryBarrier barrier {};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
    barrier.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
    pfn_vkCmdPipelineBarrier(
        cmd, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
        VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR |
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0, 1, &barrier, 0, nullptr, 0, nullptr);

    jit_sync_thread();
    jit_free(scratch);

    VkAccelerationStructureDeviceAddressInfoKHR addr_info {};
    addr_info.sType =
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
    addr_info.accelerationStructure = result.handle;
    result.address =
        pfn_vkGetAccelerationStructureDeviceAddressKHR(device, &addr_info);
    CHECK(result.address != 0);

    return result;
}

// A single triangle, matching tests/metal_triangle.mm and tests/optix_triangle.cpp
static const float vertices[9] = {
    -0.8f, -0.8f, 0.0f,
     0.8f, -0.8f, 0.0f,
     0.0f,  0.8f, 0.0f
};
static const uint32_t indices[3] = { 0, 1, 2 };

static AccelBuffer build_triangle_blas(VkDevice device) {
    void *vb = jit_malloc(JitBackend::Vulkan, sizeof(vertices));
    void *ib = jit_malloc(JitBackend::Vulkan, sizeof(indices));
    jit_memcpy(JitBackend::Vulkan, vb, vertices, sizeof(vertices));
    jit_memcpy(JitBackend::Vulkan, ib, indices, sizeof(indices));

    VkAccelerationStructureGeometryKHR geom {};
    geom.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    geom.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
    geom.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
    geom.geometry.triangles.sType =
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
    geom.geometry.triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
    geom.geometry.triangles.vertexData.deviceAddress =
        jit_vulkan_device_address(vb);
    geom.geometry.triangles.vertexStride = 3 * sizeof(float);
    geom.geometry.triangles.maxVertex = 2;
    geom.geometry.triangles.indexType = VK_INDEX_TYPE_UINT32;
    geom.geometry.triangles.indexData.deviceAddress =
        jit_vulkan_device_address(ib);

    uint32_t prim_count = 1;
    AccelBuffer blas = build_accel(
        device, VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR,
        &geom, &prim_count, 1);

    jit_free(vb);
    jit_free(ib);
    return blas;
}

static AccelBuffer build_tlas(VkDevice device,
                              const VkAccelerationStructureInstanceKHR *insts,
                              uint32_t inst_count) {
    size_t inst_bytes = inst_count * sizeof(VkAccelerationStructureInstanceKHR);
    void *inst_buf = jit_malloc(JitBackend::Vulkan, inst_bytes);
    jit_memcpy(JitBackend::Vulkan, inst_buf, insts, inst_bytes);

    VkAccelerationStructureGeometryKHR geom {};
    geom.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    geom.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
    geom.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
    geom.geometry.instances.sType =
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
    geom.geometry.instances.arrayOfPointers = VK_FALSE;
    geom.geometry.instances.data.deviceAddress =
        jit_vulkan_device_address(inst_buf);

    AccelBuffer tlas = build_accel(
        device, VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR,
        &geom, &inst_count, 1);

    jit_free(inst_buf);
    return tlas;
}

struct SceneCleanupPayload {
    VkDevice device;
    AccelBuffer blas;
    AccelBuffer tlas;
    bool *cleaned_up;
};

static void scene_cleanup_cb(void *ptr) {
    auto *p = (SceneCleanupPayload *) ptr;
    destroy_accel(p->device, p->tlas);
    destroy_accel(p->device, p->blas);
    if (p->cleaned_up)
        *p->cleaned_up = true;
    delete p;
}

// ====================================================================
//  1. Single-triangle demo (mirrors metal_triangle.mm / optix_triangle.cpp)
// ====================================================================
static void demo() {
    VkDevice device = (VkDevice) jit_vulkan_device_handle();
    CHECK(device != VK_NULL_HANDLE);

    AccelBuffer blas = build_triangle_blas(device);

    VkAccelerationStructureInstanceKHR inst {};
    inst.transform.matrix[0][0] = 1.f;
    inst.transform.matrix[1][1] = 1.f;
    inst.transform.matrix[2][2] = 1.f;
    inst.instanceCustomIndex = 42;
    inst.mask = 0xFF;
    inst.instanceShaderBindingTableRecordOffset = 0;
    inst.flags = VK_GEOMETRY_INSTANCE_FORCE_OPAQUE_BIT_KHR;
    inst.accelerationStructureReference = blas.address;

    AccelBuffer tlas = build_tlas(device, &inst, 1);

    bool cleaned_up = false;
    {
        UInt32 scene = UInt32::steal(jit_vulkan_configure_scene(
            (void *) tlas.handle,
            /* n_isect_entries = */ 0,
            /* geometry_types_mask = */ 0x1u));

        // Verify jit_vulkan_scene_owner_handle() returns a valid UInt64 resource variable
        UInt64 owner = UInt64::steal(jit_vulkan_scene_owner_handle(scene.index()));
        CHECK(owner.valid());

        auto *cleanup_payload =
            new SceneCleanupPayload { device, blas, tlas, &cleaned_up };
        jit_vulkan_scene_set_cleanup(scene.index(), scene_cleanup_cb,
                                     cleanup_payload);

        // Twice, to verify kernel caching (like metal_triangle.mm and optix_triangle.cpp)
        for (int i = 0; i < 2; ++i) {
            int res = 16;
            UInt32 index = dr::arange<UInt32>(res * res),
                   x     = index % res,
                   y     = index / res;

            Float ox = Float(x) * (2.0f / res) - 1.0f,
                  oy = Float(y) * (2.0f / res) - 1.0f,
                  oz = -1.f;
            Float dx = 0.f, dy = 0.f, dz = 1.f;
            Float mint = 0.f, maxt = 100.f, time = 0.f;
            Mask  mask = true;

            uint32_t ray_args[9] = {
                ox.index(), oy.index(), oz.index(),
                dx.index(), dy.index(), dz.index(),
                mint.index(), maxt.index(), time.index()
            };

            uint32_t out[8];
            jit_vulkan_ray_trace(9, ray_args, mask.index(), out, 8,
                                 scene.index(), 0);

            Mask   valid     = Mask::steal(out[0]);
            Float  distance  = Float::steal(out[1]);
            Float  bary_u    = Float::steal(out[2]);
            Float  bary_v    = Float::steal(out[3]);
            UInt32 inst_id   = UInt32::steal(out[4]);
            UInt32 prim_id   = UInt32::steal(out[5]);
            UInt32 geom_id   = UInt32::steal(out[6]);
            UInt32 custom_id = UInt32::steal(out[7]);

            UInt32 hit = select(valid, UInt32(1), UInt32(0));
            hit.schedule();
            distance.schedule();
            bary_u.schedule();
            bary_v.schedule();
            inst_id.schedule();
            prim_id.schedule();
            geom_id.schedule();
            custom_id.schedule();
            jit_eval();

            UInt32 hit_host =
                UInt32::steal(jit_var_migrate(hit.index(), JitBackend::None));
            jit_sync_thread();

            const uint32_t *img = hit_host.data();
            printf("Iteration %d:\n", i);
            for (int r = 0; r < res; ++r) {
                for (int c = 0; c < res; ++c)
                    printf("%i ", img[r * res + c]);
                printf("\n");
            }
            printf("\n");
            fflush(stdout);

            // Verify every ray against analytical triangle intersection
            int hit_count = 0;
            for (int r = 0; r < res; ++r) {
                for (int c = 0; c < res; ++c) {
                    int idx = r * res + c;
                    float px = c * (2.0f / res) - 1.0f;
                    float py = r * (2.0f / res) - 1.0f;
                    // Vertices: v0=(-0.8,-0.8), v1=(0.8,-0.8), v2=(0,0.8)
                    // py = -0.8*(1-v) + 0.8*v = -0.8 + 1.6*v => v = (py + 0.8) / 1.6
                    // px = -0.8*(1-u-v) + 0.8*u = -0.8 + 1.6*u + 0.8*v => u = (px + 0.8 - 0.8*v) / 1.6
                    float v_ref = (py + 0.8f) / 1.6f;
                    float u_ref = (px + 0.8f - 0.8f * v_ref) / 1.6f;
                    bool expected_hit = (u_ref >= -1e-5f && v_ref >= -1e-5f &&
                                         (u_ref + v_ref) <= 1.0f + 1e-5f);

                    CHECK((img[idx] != 0) == expected_hit);
                    if (expected_hit) {
                        hit_count++;
                        CHECK(std::fabs(distance.read(idx) - 1.0f) < 1e-5f);
                        CHECK(std::fabs(bary_u.read(idx) - u_ref) < 1e-4f);
                        CHECK(std::fabs(bary_v.read(idx) - v_ref) < 1e-4f);
                        CHECK(inst_id.read(idx) == 0u);
                        CHECK(prim_id.read(idx) == 0u);
                        CHECK(geom_id.read(idx) == 0u);
                        CHECK(custom_id.read(idx) == 42u);
                    } else {
                        CHECK(std::isinf(distance.read(idx)));
                        CHECK(bary_u.read(idx) == 0.0f);
                        CHECK(bary_v.read(idx) == 0.0f);
                        CHECK(inst_id.read(idx) == 0u);
                        CHECK(prim_id.read(idx) == 0u);
                        CHECK(geom_id.read(idx) == 0u);
                        CHECK(custom_id.read(idx) == 0u);
                    }
                }
            }
            CHECK(hit_count == 85);
        }
    }
    CHECK(cleaned_up);
}

// ====================================================================
//  2. Shadow rays, active mask, per-lane vmask, and backface culling
// ====================================================================
static void test_shadow_mask_cull() {
    VkDevice device = (VkDevice) jit_vulkan_device_handle();
    AccelBuffer blas = build_triangle_blas(device);

    VkAccelerationStructureInstanceKHR inst {};
    inst.transform.matrix[0][0] = 1.f;
    inst.transform.matrix[1][1] = 1.f;
    inst.transform.matrix[2][2] = 1.f;
    inst.instanceCustomIndex = 77;
    inst.mask = 0x02; // Only visibility mask bit 1 matches
    inst.flags = 0;   // Allow backface culling when requested by ray flags
    inst.accelerationStructureReference = blas.address;

    AccelBuffer tlas = build_tlas(device, &inst, 1);

    UInt32 scene = UInt32::steal(jit_vulkan_configure_scene(
        (void *) tlas.handle, 0, 0x1u));
    UInt32 scene_cull = UInt32::steal(jit_vulkan_configure_scene(
        (void *) tlas.handle, 0, 0x1u | 0x8u));

    uint32_t N = 16;
    UInt32 idx = dr::arange<UInt32>(N);
    // All rays aim at the center (0, 0, 0) of the triangle
    Float ox = 0.0f, oy = 0.0f, oz = -1.0f;
    Float dx = 0.0f, dy = 0.0f, dz = 1.0f;
    Float mint = 0.0f, maxt = 10.0f, time = 0.0f;

    // 2a. Shadow ray with active mask (even lanes active) and vmask (lanes with bit 1 set match 0x02)
    Mask active_mask = eq(idx & UInt32(1u), UInt32(0u));
    UInt32 vmask = select(eq(idx & UInt32(2u), UInt32(0u)), UInt32(0x02u), UInt32(0x01u));

    uint32_t ray_args_10[10] = {
        ox.index(), oy.index(), oz.index(),
        dx.index(), dy.index(), dz.index(),
        mint.index(), maxt.index(), time.index(),
        vmask.index()
    };

    uint32_t out[8];
    jit_vulkan_ray_trace(10, ray_args_10, active_mask.index(), out, 8,
                         scene.index(), /*shadow=*/1);

    Mask   valid     = Mask::steal(out[0]);
    Float  distance  = Float::steal(out[1]);
    Float  bary_u    = Float::steal(out[2]);
    Float  bary_v    = Float::steal(out[3]);
    UInt32 inst_id   = UInt32::steal(out[4]);
    UInt32 prim_id   = UInt32::steal(out[5]);
    UInt32 geom_id   = UInt32::steal(out[6]);
    UInt32 custom_id = UInt32::steal(out[7]);

    for (uint32_t i = 0; i < N; ++i) {
        bool should_hit = ((i & 1u) == 0u) && ((i & 2u) == 0u);
        CHECK(valid.read(i) == should_hit);
        // Shadow rays only populate valid and user_instance_id
        CHECK(std::isinf(distance.read(i)));
        CHECK(bary_u.read(i) == 0.0f);
        CHECK(bary_v.read(i) == 0.0f);
        CHECK(inst_id.read(i) == 0u);
        CHECK(prim_id.read(i) == 0u);
        CHECK(geom_id.read(i) == 0u);
        CHECK(custom_id.read(i) == (should_hit ? 77u : 0u));
    }

    // 2b. Backface culling: shoot from z = +1 with dz = -1
    // In Vulkan, vertices (-0.8,-0.8,0), (0.8,-0.8,0), (0,0.8,0) viewed from z = -1
    // have clockwise screen-space winding; test both directions (+z and -z) to verify
    // that scene_cull (0x8) culls exactly one side while scene (0x1) hits from both sides!
    for (int dir_sign : { 1, -1 }) {
        Float oz_dir = (dir_sign > 0) ? -1.0f : 1.0f;
        Float dz_dir = (float) dir_sign;
        Mask all_active = dr::full<Mask>(true, 4);
        uint32_t args_dir[9] = {
            ox.index(), oy.index(), oz_dir.index(),
            dx.index(), dy.index(), dz_dir.index(),
            mint.index(), maxt.index(), time.index()
        };

        uint32_t out_nocull[8], out_cull[8];
        jit_vulkan_ray_trace(9, args_dir, all_active.index(), out_nocull, 8,
                             scene.index(), 0);
        jit_vulkan_ray_trace(9, args_dir, all_active.index(), out_cull, 8,
                             scene_cull.index(), 0);

        Mask valid_nocull = Mask::steal(out_nocull[0]);
        Mask valid_cull   = Mask::steal(out_cull[0]);
        for (int k = 1; k < 8; ++k) {
            jit_var_dec_ref(out_nocull[k]);
            jit_var_dec_ref(out_cull[k]);
        }

        // Without backface culling, both directions hit
        CHECK(valid_nocull.read(0) == true);
        // Since (v1 - v0) x (v2 - v0) = (0, 0, +2.56), the CCW front face points
        // toward +z: rays with dz = -1 hit the front face, while rays with dz = +1
        // hit the back face and are culled when 0x8 is set.
        if (dir_sign > 0)
            CHECK(valid_cull.read(0) == false);
        else
            CHECK(valid_cull.read(0) == true);
    }

    jit_sync_thread();
    destroy_accel(device, tlas);
    destroy_accel(device, blas);
}

// ====================================================================
//  3. Custom AABB procedural intersection (jit_isect_*) & hybrid scene
// ====================================================================
static uint32_t compile_sphere_isect(uint32_t sphere_buf_var) {
    uint32_t in_vars[9] = {};
    jit_isect_begin(JitBackend::Vulkan, VarType::Float32, in_vars);

    Float  ox   = Float::steal(in_vars[0]);
    Float  oy   = Float::steal(in_vars[1]);
    Float  oz   = Float::steal(in_vars[2]);
    Float  dx   = Float::steal(in_vars[3]);
    Float  dy   = Float::steal(in_vars[4]);
    Float  dz   = Float::steal(in_vars[5]);
    Float  tmax = Float::steal(in_vars[6]);
    Float  time = Float::steal(in_vars[7]);
    UInt32 prim = UInt32::steal(in_vars[8]);
    (void) time;

    // Read cx, cy, cz, r from the captured sphere_buf_var at offset prim * 4
    Float spheres = Float::borrow(sphere_buf_var);
    UInt32 base = prim * UInt32(4u);
    Float cx = dr::gather<Float>(spheres, base + UInt32(0u));
    Float cy = dr::gather<Float>(spheres, base + UInt32(1u));
    Float cz = dr::gather<Float>(spheres, base + UInt32(2u));
    Float r  = dr::gather<Float>(spheres, base + UInt32(3u));

    // Analytic ray-sphere intersection: |o + t*d - c|^2 = r^2
    Float ocx = ox - cx, ocy = oy - cy, ocz = oz - cz;
    Float a = dx * dx + dy * dy + dz * dz;
    Float half_b = ocx * dx + ocy * dy + ocz * dz;
    Float c = ocx * ocx + ocy * ocy + ocz * ocz - r * r;
    Float disc = half_b * half_b - a * c;

    Mask valid_disc = (disc >= Float(0.0f));
    Float sqrt_disc = sqrt(max(disc, Float(0.0f)));
    Float t0 = (-half_b - sqrt_disc) / a;
    Float t1 = (-half_b + sqrt_disc) / a;

    Mask t0_ok = (t0 >= Float(0.0f)) && (t0 <= tmax);
    Mask t1_ok = (t1 >= Float(0.0f)) && (t1 <= tmax);
    Mask hit = valid_disc && (t0_ok || t1_ok);
    Float t = select(t0_ok, t0, t1);

    // Compute unit normal (nx, ny) at hit point and pass as bitcast uint32 attributes
    Float hx = (ox + t * dx - cx) / r;
    Float hy = (oy + t * dy - cy) / r;
    UInt32 attr0 = UInt32::steal(jit_var_cast(hx.index(), VarType::UInt32, /*reinterpret=*/1));
    UInt32 attr1 = UInt32::steal(jit_var_cast(hy.index(), VarType::UInt32, /*reinterpret=*/1));

    uint32_t out_vars[4] = {
        hit.index(),
        t.index(),
        attr0.index(),
        attr1.index()
    };
    return jit_isect_end(JitBackend::Vulkan, "sphere_isect", out_vars);
}

static void test_custom_isect_and_hybrid() {
    VkDevice device = (VkDevice) jit_vulkan_device_handle();

    // Two spheres in one AABB geometry:
    // Sphere 0: center (-0.25, 0.0, 0.5), radius 0.45
    // Sphere 1: center ( 0.25, 0.0, 0.2), radius 0.45 (closer in z where they overlap!)
    const float h_spheres[8] = {
        -0.25f, 0.0f, 0.5f, 0.45f,
         0.25f, 0.0f, 0.2f, 0.45f
    };
    VkAabbPositionsKHR h_aabbs[2] = {
        { -0.70f, -0.45f,  0.05f,  0.20f, 0.45f, 0.95f },
        { -0.20f, -0.45f, -0.25f,  0.70f, 0.45f, 0.65f }
    };

    void *d_aabbs = jit_malloc(JitBackend::Vulkan, sizeof(h_aabbs));
    jit_memcpy(JitBackend::Vulkan, d_aabbs, h_aabbs, sizeof(h_aabbs));

    VkAccelerationStructureGeometryKHR aabb_geom {};
    aabb_geom.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    aabb_geom.geometryType = VK_GEOMETRY_TYPE_AABBS_KHR;
    aabb_geom.flags = 0; // Non-opaque so candidate intersections invoke custom callback
    aabb_geom.geometry.aabbs.sType =
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_AABBS_DATA_KHR;
    aabb_geom.geometry.aabbs.data.deviceAddress =
        jit_vulkan_device_address(d_aabbs);
    aabb_geom.geometry.aabbs.stride = sizeof(VkAabbPositionsKHR);

    uint32_t aabb_prim_count = 2;
    AccelBuffer aabb_blas = build_accel(
        device, VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR,
        &aabb_geom, &aabb_prim_count, 1);
    jit_free(d_aabbs);

    VkAccelerationStructureInstanceKHR aabb_inst {};
    aabb_inst.transform.matrix[0][0] = 1.f;
    aabb_inst.transform.matrix[1][1] = 1.f;
    aabb_inst.transform.matrix[2][2] = 1.f;
    aabb_inst.instanceCustomIndex = 123;
    aabb_inst.mask = 0xFF;
    aabb_inst.instanceShaderBindingTableRecordOffset = 0;
    aabb_inst.flags = 0;
    aabb_inst.accelerationStructureReference = aabb_blas.address;

    AccelBuffer aabb_tlas = build_tlas(device, &aabb_inst, 1);

    // Upload sphere parameters as a Dr.Jit array captured by the intersection function
    Float sphere_arr = Float::copy(h_spheres, 8);
    sphere_arr.eval();

    UInt32 isect_func = UInt32::steal(compile_sphere_isect(sphere_arr.index()));

    UInt32 aabb_scene = UInt32::steal(jit_vulkan_configure_scene(
        (void *) aabb_tlas.handle,
        /* n_isect_entries = */ 1,
        /* geometry_types_mask = */ 0x2u));

    int dummy_user = 7;
    JitIsectBinding *binding1 =
        jit_isect_bind(isect_func.index(), (uintptr_t) aabb_scene.index(), 0, &dummy_user);
    CHECK(binding1 != nullptr && binding1->user == &dummy_user && binding1->data != nullptr);

    // Trace 16x16 rays against the 2-sphere procedural scene
    int res = 16;
    UInt32 index = dr::arange<UInt32>(res * res),
           x     = index % res,
           y     = index / res;
    Float ox = Float(x) * (2.0f / res) - 1.0f,
          oy = Float(y) * (2.0f / res) - 1.0f,
          oz = -1.f;
    Float dx = 0.f, dy = 0.f, dz = 1.f;
    Float mint = 0.f, maxt = 100.f, time = 0.f;
    Mask  mask = true;

    uint32_t ray_args[9] = {
        ox.index(), oy.index(), oz.index(),
        dx.index(), dy.index(), dz.index(),
        mint.index(), maxt.index(), time.index()
    };

    uint32_t out[8];
    jit_vulkan_ray_trace(9, ray_args, mask.index(), out, 8,
                         aabb_scene.index(), /*shadow=*/0);

    Mask   valid     = Mask::steal(out[0]);
    Float  distance  = Float::steal(out[1]);
    Float  bary_u    = Float::steal(out[2]);
    Float  bary_v    = Float::steal(out[3]);
    UInt32 inst_id   = UInt32::steal(out[4]);
    UInt32 prim_id   = UInt32::steal(out[5]);
    UInt32 geom_id   = UInt32::steal(out[6]);
    UInt32 custom_id = UInt32::steal(out[7]);

    valid.schedule();
    distance.schedule();
    bary_u.schedule();
    bary_v.schedule();
    inst_id.schedule();
    prim_id.schedule();
    geom_id.schedule();
    custom_id.schedule();
    jit_eval();
    jit_sync_thread();

    int sphere0_hits = 0, sphere1_hits = 0;
    for (int r = 0; r < res; ++r) {
        for (int c = 0; c < res; ++c) {
            int idx = r * res + c;
            float px = c * (2.0f / res) - 1.0f;
            float py = r * (2.0f / res) - 1.0f;

            float best_t = INFINITY;
            uint32_t best_prim = 0;
            float best_nx = 0.f, best_ny = 0.f;

            for (uint32_t s = 0; s < 2; ++s) {
                float scx = h_spheres[s * 4 + 0];
                float scy = h_spheres[s * 4 + 1];
                float scz = h_spheres[s * 4 + 2];
                float sr  = h_spheres[s * 4 + 3];
                float d2  = (px - scx) * (px - scx) + (py - scy) * (py - scy);
                if (d2 <= sr * sr) {
                    float t_hit = (scz - (-1.0f)) - std::sqrt(sr * sr - d2);
                    if (t_hit < best_t) {
                        best_t = t_hit;
                        best_prim = s;
                        best_nx = (px - scx) / sr;
                        best_ny = (py - scy) / sr;
                    }
                }
            }

            bool expected_hit = std::isfinite(best_t);
            CHECK(valid.read(idx) == expected_hit);
            if (expected_hit) {
                if (best_prim == 0)
                    sphere0_hits++;
                else
                    sphere1_hits++;
                CHECK(std::fabs(distance.read(idx) - best_t) < 1e-4f);
                CHECK(std::fabs(bary_u.read(idx) - best_nx) < 1e-4f);
                CHECK(std::fabs(bary_v.read(idx) - best_ny) < 1e-4f);
                CHECK(inst_id.read(idx) == 0u);
                CHECK(prim_id.read(idx) == best_prim);
                CHECK(geom_id.read(idx) == 0u);
                CHECK(custom_id.read(idx) == 123u);
            } else {
                CHECK(std::isinf(distance.read(idx)));
            }
        }
    }
    CHECK(sphere0_hits > 0 && sphere1_hits > 0);

    // Also test hybrid scene: Instance 0 = Triangle (z=0), Instance 1 = Spheres (sphere 1 extends to z=-0.25 < 0, sphere 0 is at z=[0.05, 0.95] > 0)
    AccelBuffer tri_blas = build_triangle_blas(device);
    VkAccelerationStructureInstanceKHR hybrid_insts[2] = {};
    // Instance 0: Triangle at z = 0
    hybrid_insts[0].transform.matrix[0][0] = 1.f;
    hybrid_insts[0].transform.matrix[1][1] = 1.f;
    hybrid_insts[0].transform.matrix[2][2] = 1.f;
    hybrid_insts[0].instanceCustomIndex = 100;
    hybrid_insts[0].mask = 0xFF;
    hybrid_insts[0].instanceShaderBindingTableRecordOffset = 0;
    hybrid_insts[0].flags = VK_GEOMETRY_INSTANCE_FORCE_OPAQUE_BIT_KHR;
    hybrid_insts[0].accelerationStructureReference = tri_blas.address;

    // Instance 1: Spheres AABB BLAS
    hybrid_insts[1] = aabb_inst;
    hybrid_insts[1].instanceCustomIndex = 200;

    AccelBuffer hybrid_tlas = build_tlas(device, hybrid_insts, 2);
    UInt32 hybrid_scene = UInt32::steal(jit_vulkan_configure_scene(
        (void *) hybrid_tlas.handle,
        /* n_isect_entries = */ 1,
        /* geometry_types_mask = */ 0x1u | 0x2u));
    JitIsectBinding *binding2 =
        jit_isect_bind(isect_func.index(), (uintptr_t) hybrid_scene.index(), 0, nullptr);
    CHECK(binding2 != nullptr);

    uint32_t hout[8];
    jit_vulkan_ray_trace(9, ray_args, mask.index(), hout, 8,
                         hybrid_scene.index(), /*shadow=*/0);
    Mask   h_valid     = Mask::steal(hout[0]);
    Float  h_distance  = Float::steal(hout[1]);
    UInt32 h_inst_id   = UInt32::steal(hout[4]);
    UInt32 h_custom_id = UInt32::steal(hout[7]);
    for (int k : { 2, 3, 5, 6 })
        jit_var_dec_ref(hout[k]);

    h_valid.schedule();
    h_distance.schedule();
    h_inst_id.schedule();
    h_custom_id.schedule();
    jit_eval();
    jit_sync_thread();

    int tri_wins = 0, sphere_wins = 0;
    for (int i = 0; i < res * res; ++i) {
        if (h_valid.read(i)) {
            uint32_t cid = h_custom_id.read(i);
            CHECK(cid == 100u || cid == 200u);
            if (cid == 100u) {
                CHECK(h_inst_id.read(i) == 0u);
                CHECK(std::fabs(h_distance.read(i) - 1.0f) < 1e-4f);
                tri_wins++;
            } else {
                CHECK(h_inst_id.read(i) == 1u);
                sphere_wins++;
            }
        }
    }
    // Triangle (at z=0, t=1.0) occludes Sphere 0 (z >= 0.05, t >= 1.05), while
    // Sphere 1 (center z=0.2, radius 0.45 -> front at z=-0.25, t=0.75) occludes the Triangle!
    CHECK(tri_wins > 0 && sphere_wins > 0);

    jit_isect_unbind(binding2);
    jit_isect_unbind(binding1);
    destroy_accel(device, hybrid_tlas);
    destroy_accel(device, tri_blas);
    destroy_accel(device, aabb_tlas);
    destroy_accel(device, aabb_blas);
}

// ====================================================================
//  4. Double-precision (Float64) ray tracing & custom intersection
// ====================================================================
static void test_float64_rays() {
    VkDevice device = (VkDevice) jit_vulkan_device_handle();
    AccelBuffer blas = build_triangle_blas(device);

    VkAccelerationStructureInstanceKHR inst {};
    inst.transform.matrix[0][0] = 1.f;
    inst.transform.matrix[1][1] = 1.f;
    inst.transform.matrix[2][2] = 1.f;
    inst.instanceCustomIndex = 99;
    inst.mask = 0xFF;
    inst.flags = VK_GEOMETRY_INSTANCE_FORCE_OPAQUE_BIT_KHR;
    inst.accelerationStructureReference = blas.address;

    AccelBuffer tlas = build_tlas(device, &inst, 1);
    UInt32 scene = UInt32::steal(jit_vulkan_configure_scene(
        (void *) tlas.handle, 0, 0x1u));

    double h_ox[2] = { 0.0, 0.9 }; // lane 0 hits (0,0), lane 1 misses (0.9, 0.9)
    double h_oy[2] = { 0.0, 0.9 };
    Float64 ox = Float64::copy(h_ox, 2);
    Float64 oy = Float64::copy(h_oy, 2);
    Float64 oz = -2.5;
    Float64 dx = 0.0, dy = 0.0, dz = 1.0;
    Float64 mint = 0.0, maxt = 100.0, time = 0.0;
    Mask mask = true;

    uint32_t ray_args[9] = {
        ox.index(), oy.index(), oz.index(),
        dx.index(), dy.index(), dz.index(),
        mint.index(), maxt.index(), time.index()
    };

    uint32_t out[8];
    jit_vulkan_ray_trace(9, ray_args, mask.index(), out, 8, scene.index(), 0);

    Mask    valid     = Mask::steal(out[0]);
    Float64 distance  = Float64::steal(out[1]);
    Float64 bary_u    = Float64::steal(out[2]);
    Float64 bary_v    = Float64::steal(out[3]);
    UInt32  inst_id   = UInt32::steal(out[4]);
    UInt32  prim_id   = UInt32::steal(out[5]);
    UInt32  geom_id   = UInt32::steal(out[6]);
    UInt32  custom_id = UInt32::steal(out[7]);

    CHECK(valid.read(0) == true);
    CHECK(std::fabs(distance.read(0) - 2.5) < 1e-5);
    CHECK(std::fabs(bary_u.read(0) - 0.25) < 1e-4);
    CHECK(std::fabs(bary_v.read(0) - 0.50) < 1e-4);
    CHECK(custom_id.read(0) == 99u);

    CHECK(valid.read(1) == false);
    CHECK(std::isinf(distance.read(1)));
    CHECK(bary_u.read(1) == 0.0);
    CHECK(bary_v.read(1) == 0.0);
    CHECK(custom_id.read(1) == 0u);

    jit_sync_thread();
    destroy_accel(device, tlas);
    destroy_accel(device, blas);

    // 4b. Float64 custom AABB intersection (jit_isect_begin with VarType::Float64)
    VkAabbPositionsKHR h_aabb = { -0.5f, -0.5f, -0.5f, 0.5f, 0.5f, 0.5f };
    void *d_aabb = jit_malloc(JitBackend::Vulkan, sizeof(h_aabb));
    jit_memcpy(JitBackend::Vulkan, d_aabb, &h_aabb, sizeof(h_aabb));

    VkAccelerationStructureGeometryKHR aabb_geom {};
    aabb_geom.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    aabb_geom.geometryType = VK_GEOMETRY_TYPE_AABBS_KHR;
    aabb_geom.flags = 0;
    aabb_geom.geometry.aabbs.sType =
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_AABBS_DATA_KHR;
    aabb_geom.geometry.aabbs.data.deviceAddress =
        jit_vulkan_device_address(d_aabb);
    aabb_geom.geometry.aabbs.stride = sizeof(VkAabbPositionsKHR);

    uint32_t prim_count = 1;
    AccelBuffer aabb_blas = build_accel(
        device, VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR,
        &aabb_geom, &prim_count, 1);
    jit_free(d_aabb);

    inst.flags = 0;
    inst.instanceCustomIndex = 88;
    inst.accelerationStructureReference = aabb_blas.address;
    AccelBuffer aabb_tlas = build_tlas(device, &inst, 1);

    uint32_t in_f64[9] = {};
    jit_isect_begin(JitBackend::Vulkan, VarType::Float64, in_f64);
    Float64 iox   = Float64::steal(in_f64[0]);
    Float64 ioy   = Float64::steal(in_f64[1]);
    Float64 ioz   = Float64::steal(in_f64[2]);
    Float64 idx_v = Float64::steal(in_f64[3]);
    Float64 idy_v = Float64::steal(in_f64[4]);
    Float64 idz_v = Float64::steal(in_f64[5]);
    Float64 itmax = Float64::steal(in_f64[6]);
    Float64 itime = Float64::steal(in_f64[7]);
    UInt32  iprim = UInt32::steal(in_f64[8]);
    (void) idx_v; (void) idy_v; (void) idz_v; (void) itime; (void) iprim;

    // Sphere of radius 0.5 at origin
    Float64 r2 = Float64(0.25) - (iox * iox + ioy * ioy);
    Mask ihit = r2 >= Float64(0.0);
    Float64 it = -ioz - sqrt(max(r2, Float64(0.0)));
    ihit = ihit && (it >= Float64(0.0)) && (it <= itmax);
    Float u_f32 = Float(iox);
    Float v_f32 = Float(ioy);
    UInt32 ia0 = UInt32::steal(jit_var_cast(u_f32.index(), VarType::UInt32, 1));
    UInt32 ia1 = UInt32::steal(jit_var_cast(v_f32.index(), VarType::UInt32, 1));
    uint32_t out_f64[4] = { ihit.index(), it.index(), ia0.index(), ia1.index() };
    UInt32 isect_f64 = UInt32::steal(
        jit_isect_end(JitBackend::Vulkan, "sphere_isect_f64", out_f64));

    UInt32 scene_f64 = UInt32::steal(jit_vulkan_configure_scene(
        (void *) aabb_tlas.handle, 1, 0x2u));
    JitIsectBinding *bind_f64 =
        jit_isect_bind(isect_f64.index(), (uintptr_t) scene_f64.index(), 0, nullptr);

    uint32_t out2[8];
    jit_vulkan_ray_trace(9, ray_args, mask.index(), out2, 8, scene_f64.index(), 0);
    Mask    valid2    = Mask::steal(out2[0]);
    Float64 dist2     = Float64::steal(out2[1]);
    UInt32  custom2   = UInt32::steal(out2[7]);
    for (int k : { 2, 3, 4, 5, 6 })
        jit_var_dec_ref(out2[k]);

    CHECK(valid2.read(0) == true);
    CHECK(std::fabs(dist2.read(0) - 2.0) < 1e-5);
    CHECK(custom2.read(0) == 88u);
    CHECK(valid2.read(1) == false);
    CHECK(std::isinf(dist2.read(1)));

    jit_sync_thread();
    jit_isect_unbind(bind_f64);
    destroy_accel(device, aabb_tlas);
    destroy_accel(device, aabb_blas);
}

// ====================================================================
//  5. Frozen function recording & replay with scene rebinding
// ====================================================================
static void test_frozen_ray_trace() {
    VkDevice device = (VkDevice) jit_vulkan_device_handle();
    AccelBuffer blas = build_triangle_blas(device);

    // Build two TLASes at different z-offsets and custom instance IDs
    VkAccelerationStructureInstanceKHR inst1 {};
    inst1.transform.matrix[0][0] = 1.f;
    inst1.transform.matrix[1][1] = 1.f;
    inst1.transform.matrix[2][2] = 1.f;
    inst1.transform.matrix[2][3] = 0.0f; // z = 0 -> distance = 1.0 from oz = -1
    inst1.instanceCustomIndex = 11;
    inst1.mask = 0xFF;
    inst1.flags = VK_GEOMETRY_INSTANCE_FORCE_OPAQUE_BIT_KHR;
    inst1.accelerationStructureReference = blas.address;

    VkAccelerationStructureInstanceKHR inst2 = inst1;
    inst2.transform.matrix[2][3] = 2.5f; // z = 2.5 -> distance = 3.5 from oz = -1
    inst2.instanceCustomIndex = 22;

    AccelBuffer tlas1 = build_tlas(device, &inst1, 1);
    AccelBuffer tlas2 = build_tlas(device, &inst2, 1);

    UInt32 scene1 = UInt32::steal(jit_vulkan_configure_scene(
        (void *) tlas1.handle, 0, 0x1u));
    UInt32 scene2 = UInt32::steal(jit_vulkan_configure_scene(
        (void *) tlas2.handle, 0, 0x1u));

    UInt32 owner1 = UInt32::steal(jit_vulkan_scene_owner_handle(scene1.index()));
    UInt32 owner2 = UInt32::steal(jit_vulkan_scene_owner_handle(scene2.index()));

    const float h_ox[2] = { 0.0f, 0.9f };
    const float h_oy[2] = { 0.0f, 0.9f };
    Float ox = Float::copy(h_ox, 2);
    Float oy = Float::copy(h_oy, 2);
    ox.eval();
    oy.eval();

    uint32_t freeze_inputs[3] = { ox.index(), oy.index(), owner1.index() };
    jit_freeze_start(JitBackend::Vulkan, freeze_inputs, 3);

    Float oz = -1.0f, dx = 0.0f, dy = 0.0f, dz = 1.0f;
    Float mint = 0.0f, maxt = 100.0f, time = 0.0f;
    Mask mask = true;
    uint32_t ray_args[9] = {
        ox.index(), oy.index(), oz.index(),
        dx.index(), dy.index(), dz.index(),
        mint.index(), maxt.index(), time.index()
    };
    uint32_t out[8];
    jit_vulkan_ray_trace(9, ray_args, mask.index(), out, 8, scene1.index(), 0);
    Float  dist1   = Float::steal(out[1]);
    UInt32 custom1 = UInt32::steal(out[7]);
    for (int k : { 0, 2, 3, 4, 5, 6 })
        jit_var_dec_ref(out[k]);

    dist1.schedule();
    custom1.schedule();
    jit_eval();

    uint32_t freeze_outputs[2] = { dist1.index(), custom1.index() };
    Recording *rec = jit_freeze_stop(JitBackend::Vulkan, freeze_outputs, 2);
    CHECK(rec != nullptr);

    CHECK(std::fabs(dist1.read(0) - 1.0f) < 1e-5f);
    CHECK(std::isinf(dist1.read(1)));
    CHECK(custom1.read(0) == 11u);
    CHECK(custom1.read(1) == 0u);

    // Replay the frozen recording with scene2 (z = 2.5, instanceCustomIndex = 22)
    uint32_t replay_inputs[3] = { ox.index(), oy.index(), owner2.index() };
    uint32_t replay_outputs[2] = { 0, 0 };
    jit_freeze_replay(rec, replay_inputs, replay_outputs);

    Float  dist2   = Float::steal(replay_outputs[0]);
    UInt32 custom2 = UInt32::steal(replay_outputs[1]);

    CHECK(std::fabs(dist2.read(0) - 3.5f) < 1e-5f);
    CHECK(std::isinf(dist2.read(1)));
    CHECK(custom2.read(0) == 22u);
    CHECK(custom2.read(1) == 0u);

    jit_freeze_destroy(rec);
    jit_sync_thread();
    destroy_accel(device, tlas2);
    destroy_accel(device, tlas1);
    destroy_accel(device, blas);
}

int main(int, char **) {
    const char *req_env = getenv("DRJIT_TEST_REQUIRE_VULKAN");
    bool require_vulkan = req_env && strcmp(req_env, "1") == 0;

    jit_init(1u << (uint32_t) JitBackend::Vulkan);

    if (!jit_has_backend(JitBackend::Vulkan)) {
        fprintf(stderr, "Vulkan backend not available, skipping vulkan_triangle.\n");
        jit_shutdown(0);
        return require_vulkan ? 1 : 0;
    }

    if (!jit_vulkan_supports_ray_tracing()) {
        fprintf(stderr, "Vulkan ray tracing not supported on this device, skipping vulkan_triangle.\n");
        jit_shutdown(0);
        return require_vulkan ? 1 : 0;
    }

    resolve_vk_functions();

    demo();
    test_shadow_mask_cull();
    test_custom_isect_and_hybrid();
    test_float64_rays();
    test_frozen_ray_trace();

    jit_shutdown(0);
    printf("vulkan_triangle: all tests passed.\n");
    return 0;
}
