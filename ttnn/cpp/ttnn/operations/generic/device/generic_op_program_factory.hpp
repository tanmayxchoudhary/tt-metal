// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/program_descriptors.hpp>

#include "generic_op_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/distributed/types.hpp"

namespace ttnn::operations::generic::program {

// Custom MeshWorkloadFactoryConcept factory.
//
// The standard ProgramDescriptor adapter contracts don't fit generic_op:
//   - Contract 1 (`create_descriptor`) calls the factory once per mesh
//     coordinate and demands a `ProgramDescriptor` for every coord. The user
//     of generic_op may only bind programs to a subset of coords (e.g.
//     point-to-point binds a sender + receiver only) so coords without a
//     user program would trigger a TT_FATAL.
//   - Contract 2 (`create_workload_descriptor`) supports per-range programs
//     but has no slow-path rebuild on cache hit. generic_op users feed raw
//     `tensor.buffer_address()` integers as runtime args (not `Buffer*`
//     bindings) so on cache hits the addresses would go stale and ops like
//     fused matmul would produce wrong outputs.
//
// Implementing MeshWorkloadFactoryConcept directly lets us:
//   - Iterate only the user's `mesh_programs` on cache miss (no demand to
//     cover every mesh coord).
//   - Refresh kernel runtime args + CB sizes / page sizes / addresses from
//     the user's CURRENT descriptor on every cache hit via
//     `apply_descriptor_runtime_args` (the public Metal helper used by the
//     framework's own Contract-1 slow path).
struct GenericMeshWorkloadFactory {
    struct shared_variables_t {};
    using cached_mesh_workload_t = ttnn::device_operation::AdaptedCachedMeshWorkload<shared_variables_t>;

    static cached_mesh_workload_t create_mesh_workload(
        const operation_attributes_t& operation_attributes,
        const ttnn::MeshCoordinateRangeSet& tensor_coords,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value);

    static void override_runtime_arguments(
        cached_mesh_workload_t& cached_workload,
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value);
};

}  // namespace ttnn::operations::generic::program
