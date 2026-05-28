// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <tt-metalium/host_api.hpp>
#include "ttnn/device_operation.hpp"
#include "ttnn/operations/data_movement/slice/device/slice_device_operation_types.hpp"

namespace ttnn::prim {

struct SliceRmShardedProgramFactory {
    struct shared_variables_t {
        tt::tt_metal::CBHandle cb_src0{};
        tt::tt_metal::CBHandle cb_output{};
    };

    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(const SliceParams& args, const SliceInputs& tensor_args, Tensor& output);

    static void override_runtime_arguments(
        cached_program_t& cached_program, const SliceParams& args, const SliceInputs& tensor_args, Tensor& output);
};

// WIDTH/TAIL-TRIM factory: HEIGHT_SHARDED ROW_MAJOR input with only the last
// dimension sliced.  Supports two output paths:
//   L1 HS output: each core trims into a locally-allocated output CB (no DRAM).
//   DRAM output:  each core writes trimmed sticks to DRAM interleaved via
//                 noc_async_write; no output L1 CB allocated — zero extra L1.
struct SliceRmShardedWidthTrimProgramFactory {
    struct shared_variables_t {
        tt::tt_metal::CBHandle cb_src0{};
        tt::tt_metal::CBHandle cb_output{};        // valid only for L1 HS output
        tt::tt_metal::KernelHandle writer_kernel{}; // valid only for DRAM output
        bool output_is_dram{false};
        bool row_major{false};
        CoreRangeSet all_cores;
        uint32_t shard_height{0};
    };

    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(const SliceParams& args, const SliceInputs& tensor_args, Tensor& output);

    static void override_runtime_arguments(
        cached_program_t& cached_program, const SliceParams& args, const SliceInputs& tensor_args, Tensor& output);
};

}  // namespace ttnn::prim
