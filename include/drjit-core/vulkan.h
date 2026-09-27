/*
    drjit-core/vulkan.h -- Public API for the Vulkan backend, including ray
    tracing primitives.

    Copyright (c) 2026 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#include "jit.h"

#if defined(__cplusplus)
extern "C" {
#endif

/**
 * \brief Inform Dr.Jit about a per-scene Vulkan ray-tracing configuration.
 *
 * The application calls this once per scene to register its top-level
 * acceleration structure (TLAS) and the size of the intersection function
 * table that its bounding-box geometry indexes.
 *
 * The function returns a JIT variable index that owns the lifetime of the scene
 * state on the Dr.Jit side. This index must be passed to \ref jit_vulkan_ray_trace
 * to bind this scene's TLAS and intersection table at launch time. When the
 * variable's reference count reaches zero, Dr.Jit releases its owned per-scene
 * resources and runs any callback registered via \ref jit_vulkan_scene_set_cleanup.
 *
 * \param tlas
 *     The \c VkAccelerationStructureKHR TLAS handle (passed as <tt>void *</tt>).
 *
 * \param n_isect_entries
 *     Number of entries in the scene's custom intersection function table.
 *
 * \param geometry_types_mask
 *     Bit 0 (0x1): triangle geometry present.
 *     Bit 1 (0x2): bounding-box (custom-intersection) geometry present.
 *     Bit 2 (0x4): curve geometry present.
 *     Bit 3 (0x8): triangle backface culling required for at least one instance.
 *     Bit 4 (0x10): motion (instance motion) present.
 */
extern JIT_EXPORT uint32_t jit_vulkan_configure_scene(
    void *tlas,
    uint32_t n_isect_entries,
    uint32_t geometry_types_mask);

/**
 * \brief Perform an inline ray intersection in a Vulkan compute kernel
 *
 * Creates a \c VarKind::TraceRay IR node that, when evaluated, emits a
 * \c SPV_KHR_ray_query traversal against the scene identified by \c scene.
 * For lanes that are masked off or that miss the geometry, the distance output
 * is set to +infinity, the validity flag to \c false, and every other output
 * to zero.
 *
 * \param n_args
 *     Number of ray input arguments. Must be 9, or 10 when a per-lane
 *     visibility mask is supplied.
 *
 * \param args
 *     Array of 9 (or 10) JIT variable indices:
 *       [0] ox     (Float32 / Float64) — ray origin X
 *       [1] oy     (Float32 / Float64) — ray origin Y
 *       [2] oz     (Float32 / Float64) — ray origin Z
 *       [3] dx     (Float32 / Float64) — ray direction X
 *       [4] dy     (Float32 / Float64) — ray direction Y
 *       [5] dz     (Float32 / Float64) — ray direction Z
 *       [6] tmin   (Float32 / Float64) — minimum ray distance
 *       [7] tmax   (Float32 / Float64) — maximum ray distance
 *       [8] time   (Float32 / Float64) — ray evaluation time
 *       [9] vmask  (UInt32)            — optional visibility mask
 *
 * \param mask
 *     JIT variable index of the active lane mask (Bool).
 *
 * \param out
 *     Array of 8 JIT variable indices (output, written by this function):
 *       [0] valid            (Bool)              — true if a hit was found
 *       [1] distance         (Float32 / Float64) — distance to the closest hit
 *       [2] bary_u           (Float32 / Float64) — barycentric U coordinate, or
 *                                                  attribute 0 of an AABB hit
 *       [3] bary_v           (Float32 / Float64) — barycentric V coordinate, or
 *                                                  attribute 1 of an AABB hit
 *       [4] instance_id      (UInt32)            — instance index in the TLAS
 *       [5] primitive_id     (UInt32)            — primitive index in the BLAS
 *       [6] geometry_id      (UInt32)            — geometry index in the BLAS
 *       [7] user_instance_id (UInt32)            — per-instance custom index
 *
 * \param n_out
 *     Number of output variables. Must be 8.
 *
 * \param scene
 *     JIT variable index returned by \ref jit_vulkan_configure_scene.
 *
 * \param shadow
 *     If nonzero, performs a shadow ray test that accepts any intersection
 *     and terminates traversal early. Only outputs 0 (the hit flag) and 7
 *     (the user instance ID of whichever hit ended the traversal) are
 *     computed; outputs 1-6 keep their miss values.
 */
extern JIT_EXPORT void jit_vulkan_ray_trace(uint32_t n_args, uint32_t *args,
                                            uint32_t mask, uint32_t *out,
                                            uint32_t n_out, uint32_t scene,
                                            int shadow);

/**
 * \brief Register a cleanup callback that runs when the scene variable dies
 *
 * The application's Vulkan objects (TLAS, BLAS, buffers) and intersection
 * bindings must outlive the scene variable. \c callback runs once when the
 * variable is freed, right before the scene is destroyed, so the application
 * can release the scene by dropping its reference. \c scene_index must be a
 * variable representing a scene, as returned by \ref jit_vulkan_configure_scene.
 */
extern JIT_EXPORT void jit_vulkan_scene_set_cleanup(uint32_t scene_index,
                                                    void (*callback)(void *),
                                                    void *payload);

/**
 * \brief Create a handle exposing a scene as a frozen-function input
 *
 * Returns a \c UInt64 variable whose data pointer is drjit-core's internal
 * per-scene bookkeeping object. It is used as an identity token for
 * \c dr.freeze. The handle does not own the scene. \c scene_index must be a
 * variable representing a scene, as returned by \ref jit_vulkan_configure_scene.
 */
extern JIT_EXPORT uint32_t jit_vulkan_scene_owner_handle(uint32_t scene_index);

/**
 * \brief Return the active \c VkCommandBuffer on Dr.Jit's Vulkan command stream
 *
 * Begins a new command buffer if none is currently recording and inserts a
 * memory barrier if prior commands wrote to memory. Useful for recording
 * acceleration structure build commands directly onto Dr.Jit's stream.
 */
extern JIT_EXPORT void *jit_vulkan_command_buffer();

#if defined(__cplusplus)
}
#endif
