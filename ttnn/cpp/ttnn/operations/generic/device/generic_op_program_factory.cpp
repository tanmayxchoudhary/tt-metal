// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "generic_op_program_factory.hpp"

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/program.hpp>
#include <tt_stl/assert.hpp>

namespace ttnn::operations::generic::program {

using namespace tt::tt_metal;

GenericMeshWorkloadFactory::cached_mesh_workload_t GenericMeshWorkloadFactory::create_mesh_workload(
    const operation_attributes_t& operation_attributes,
    const ttnn::MeshCoordinateRangeSet& /*tensor_coords*/,
    const tensor_args_t& /*tensor_args*/,
    tensor_return_value_t& /*tensor_return_value*/) {
    const auto& mesh_programs = operation_attributes.mesh_programs;
    TT_FATAL(!mesh_programs.empty(), "generic_op: MeshProgramDescriptor.mesh_programs must not be empty");

    tt::tt_metal::distributed::MeshWorkload mesh_workload;
    std::unordered_map<ttnn::MeshCoordinateRange, shared_variables_t> shared_variables;
    shared_variables.reserve(mesh_programs.size());

    for (const auto& [range, program_descriptor] : mesh_programs) {
        mesh_workload.add_program(range, Program{program_descriptor});
        shared_variables.emplace(range, shared_variables_t{});
    }

    return cached_mesh_workload_t{std::move(mesh_workload), std::move(shared_variables)};
}

void GenericMeshWorkloadFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& /*tensor_args*/,
    tensor_return_value_t& /*tensor_return_value*/) {
    // Refresh kernel runtime args, CB sizes / page sizes, and dynamic CB
    // addresses from the user's CURRENT descriptor for every cached program.
    // The user's runtime args carry raw addresses (e.g. tensor.buffer_address())
    // that move between dispatches, so we must re-apply them here — otherwise
    // a cache hit would run with stale addresses and produce wrong outputs.
    auto& workload_programs = cached_workload.workload.get_programs();
    for (const auto& [range, program_descriptor] : operation_attributes.mesh_programs) {
        auto program_it = workload_programs.find(range);
        TT_FATAL(
            program_it != workload_programs.end(), "generic_op: cached workload missing program for range {}", range);
        apply_descriptor_runtime_args(program_it->second, program_descriptor);
    }
}

}  // namespace ttnn::operations::generic::program
