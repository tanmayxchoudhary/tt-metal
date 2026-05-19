// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "paged_fused_update_cache_device_operation_types.hpp"

#include <tt-metalium/program_descriptors.hpp>

#include <optional>

namespace ttnn::experimental::prim {

struct PagedRowMajorFusedUpdateCacheProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const PagedFusedUpdateCacheParams& operation_attributes,
        const PagedFusedUpdateCacheInputs& tensor_args,
        PagedFusedUpdateCacheResult& tensor_return_value);
};

struct PagedRowMajorFusedUpdateCacheMeshWorkloadFactory {
    // Per-coord program build.  Coordinates outside operation_attributes.mesh_coords
    // (when provided) get an empty program — the legacy mesh path skipped them
    // entirely; with the descriptor framework we still must hand back a descriptor
    // for every dispatched coord, so we return an empty one for excluded coords.
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const PagedFusedUpdateCacheParams& operation_attributes,
        const PagedFusedUpdateCacheInputs& tensor_args,
        PagedFusedUpdateCacheResult& tensor_return_value,
        const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate);
};

}  // namespace ttnn::experimental::prim
