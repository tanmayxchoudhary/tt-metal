// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "update_padded_kv_cache_device_operation.hpp"

#include <cstdint>
#include <utility>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>

#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::update_padded_kv_cache {

using namespace tt::tt_metal;
using namespace tt::constants;

namespace {

// Reused from the original kv_cache fill path; these kernels take (addr, num_tiles, start_tile_id)
// as runtime args and operate purely on tile ids — no per-device logic of their own.
constexpr auto kReaderKernelPath =
    "ttnn/cpp/ttnn/operations/kv_cache/device/kernels/dataflow/reader_fill_cache_interleaved_start_id.cpp";
constexpr auto kWriterKernelPath =
    "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp";

constexpr uint32_t kSrcCbIndex = 0;
constexpr uint32_t kNumInputTilesDoubleBuffered = 2;

// Per-chip write-start tile (within the per-chip cache slot), derived from a single global token
// count and this device's coord along the sp cluster axis. Mirrors `_update_idxt_for_chip` in
// tests/ttnn/unit_tests/operations/deepseek/test_deepseek_prefill_update_padded_kv_cache.py.
uint32_t update_idxt_for_chip(
    uint32_t kv_actual_global, uint32_t my_sp_coord, uint32_t sp_factor, uint32_t chunk_local_tokens) {
    const uint32_t kv_actual_t = kv_actual_global / TILE_HEIGHT;
    const uint32_t chunk_local_t = chunk_local_tokens / TILE_HEIGHT;
    const uint32_t chunk_global_t = sp_factor * chunk_local_t;

    const uint32_t boundary_slab_idx = kv_actual_t / chunk_global_t;
    const uint32_t boundary_chip = (kv_actual_t / chunk_local_t) % sp_factor;
    const uint32_t boundary_offset_t = kv_actual_t % chunk_local_t;

    if (my_sp_coord < boundary_chip) {
        return (boundary_slab_idx + 1) * chunk_local_t;
    }
    if (my_sp_coord == boundary_chip) {
        return boundary_slab_idx * chunk_local_t + boundary_offset_t;
    }
    return boundary_slab_idx * chunk_local_t;
}

// Count distinct values along the cluster axis among the participating devices to determine
// sp_factor without round-tripping to the mesh view.
uint32_t sp_factor_for_tensor(const Tensor& tensor, uint32_t cluster_axis) {
    const auto device_coords = tensor.device_storage().get_coords();
    TT_FATAL(!device_coords.empty(), "device_coords is empty when computing sp_factor");
    uint32_t min_v = std::numeric_limits<uint32_t>::max();
    uint32_t max_v = 0;
    for (const auto& c : device_coords) {
        TT_FATAL(c.dims() > cluster_axis, "cluster_axis {} out of range for coord rank {}", cluster_axis, c.dims());
        const uint32_t v = c[cluster_axis];
        min_v = std::min(min_v, v);
        max_v = std::max(max_v, v);
    }
    return max_v - min_v + 1;
}

}  // namespace

UpdatePaddedKvCacheDeviceOperation::program_factory_t UpdatePaddedKvCacheDeviceOperation::select_program_factory(
    const operation_attributes_t& /*args*/, const tensor_args_t& /*tensor_args*/) {
    return ProgramFactory{};
}

void UpdatePaddedKvCacheDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& cache = tensor_args.cache;
    const auto& input = tensor_args.input;

    TT_FATAL(cache.storage_type() == StorageType::DEVICE, "cache must be on device");
    TT_FATAL(input.storage_type() == StorageType::DEVICE, "input must be on device");
    TT_FATAL(cache.layout() == Layout::TILE, "cache must be TILE layout");
    TT_FATAL(input.layout() == Layout::TILE, "input must be TILE layout");
    TT_FATAL(cache.dtype() == input.dtype(), "cache and input dtype must match");

    const auto& cache_shape = cache.padded_shape();
    const auto& input_shape = input.padded_shape();
    TT_FATAL(cache_shape.rank() == 4, "cache must be 4D (got rank {})", cache_shape.rank());
    TT_FATAL(input_shape.rank() == 4, "input must be 4D (got rank {})", input_shape.rank());
    TT_FATAL(cache_shape[-1] == input_shape[-1], "cache and input head dim must match");
    TT_FATAL(cache_shape[1] == input_shape[1], "cache and input num-heads dim must match");

    const uint32_t cache_seq = cache_shape[-2];
    const uint32_t input_seq = input_shape[-2];
    TT_FATAL(input_seq % TILE_HEIGHT == 0, "input seq dim ({}) must be tile-aligned", input_seq);
    TT_FATAL(cache_seq % input_seq == 0, "cache seq ({}) must be a multiple of input seq ({})", cache_seq, input_seq);

    TT_FATAL(
        args.kv_actual_global % TILE_HEIGHT == 0, "kv_actual_global ({}) must be tile-aligned", args.kv_actual_global);

    TT_FATAL(
        args.batch_idx < cache_shape[0], "batch_idx {} out of range (cache batch {})", args.batch_idx, cache_shape[0]);

    // Verify cluster-axis sizing: the cache holds `sp_factor` copies of the per-chip slot globally.
    const uint32_t sp_factor = sp_factor_for_tensor(cache, args.cluster_axis);
    const uint32_t chunk_local_tokens = input_seq;
    const uint32_t global_cache_capacity = sp_factor * cache_seq;
    TT_FATAL(
        args.kv_actual_global + sp_factor * chunk_local_tokens <= global_cache_capacity,
        "kv_actual_global ({}) + chunk_global ({}) would overflow global cache capacity ({})",
        args.kv_actual_global,
        sp_factor * chunk_local_tokens,
        global_cache_capacity);
}

void UpdatePaddedKvCacheDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& /*args*/, const tensor_args_t& /*tensor_args*/) {}

UpdatePaddedKvCacheDeviceOperation::spec_return_value_t UpdatePaddedKvCacheDeviceOperation::compute_output_specs(
    const operation_attributes_t& /*args*/, const tensor_args_t& tensor_args) {
    // In-place: output spec = cache spec.
    return tensor_args.cache.tensor_spec();
}

UpdatePaddedKvCacheDeviceOperation::tensor_return_value_t UpdatePaddedKvCacheDeviceOperation::create_output_tensors(
    const operation_attributes_t& /*args*/, const tensor_args_t& tensor_args) {
    // In-place: return a handle to the cache.
    return tensor_args.cache;
}

ttsl::hash::hash_t UpdatePaddedKvCacheDeviceOperation::compute_program_hash(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& cache = tensor_args.cache;
    const auto& input = tensor_args.input;
    return tt::tt_metal::operation::hash_operation<UpdatePaddedKvCacheDeviceOperation>(
        args.batch_idx,
        args.kv_actual_global,
        args.cluster_axis,
        input.dtype(),
        input.memory_config(),
        input.padded_shape().volume(),
        cache.memory_config(),
        cache.padded_shape().volume());
}

tt::tt_metal::ProgramDescriptor UpdatePaddedKvCacheDeviceOperation::ProgramFactory::create_descriptor(
    const operation_attributes_t& args,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& /*output*/,
    const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate) {
    TT_FATAL(
        mesh_dispatch_coordinate.has_value(),
        "UpdatePaddedKvCache::create_descriptor requires a mesh dispatch coordinate");
    const auto& coord = mesh_dispatch_coordinate.value();

    const auto& cache = tensor_args.cache;
    const auto& input = tensor_args.input;
    auto* device = input.device();

    const auto& cache_shape = cache.padded_shape();
    const auto& input_shape = input.padded_shape();

    const tt::DataFormat data_format = datatype_to_dataformat_converter(input.dtype());
    const uint32_t single_tile_size = tt::tile_size(data_format);

    const uint32_t Wt = cache_shape[-1] / TILE_WIDTH;
    const uint32_t input_Ht = input_shape[-2] / TILE_HEIGHT;
    const uint32_t cache_HtWt = cache_shape[-2] * Wt / TILE_HEIGHT;
    const uint32_t cache_CHtWt = cache_shape[1] * cache_HtWt;

    // Per-chip update_idxt math: derive sp_factor + my_sp_coord, then compute.
    const uint32_t sp_factor = sp_factor_for_tensor(cache, args.cluster_axis);
    const uint32_t my_sp_coord = ::ttnn::ccl::get_linearized_index_from_physical_coord(cache, coord, args.cluster_axis);
    const uint32_t update_idxt = update_idxt_for_chip(args.kv_actual_global, my_sp_coord, sp_factor, input_shape[-2]);
    const uint32_t start_idx = (args.batch_idx * cache_CHtWt) + (update_idxt * Wt);

    // Work split: one tile per "block". num_blocks_of_work = input_C * input_Ht (= num_heads * seq_tiles).
    const uint32_t num_blocks_of_work = input_shape[1] * input_Ht;

    const auto compute_grid = device->compute_with_storage_grid_size();
    auto [num_cores, all_cores, core_group_1, core_group_2, num_blocks_per_core_g1, num_blocks_per_core_g2] =
        tt::tt_metal::split_work_to_cores(compute_grid, num_blocks_of_work, /*row_major=*/true);

    tt::tt_metal::ProgramDescriptor desc;

    // CB for the input tiles.
    desc.cbs.push_back(CBDescriptor{
        .total_size = kNumInputTilesDoubleBuffered * single_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = kSrcCbIndex,
            .data_format = data_format,
            .page_size = single_tile_size,
        }}},
    });

    // Reader kernel descriptor.
    KernelDescriptor::CompileTimeArgs reader_compile_args;
    TensorAccessorArgs(input.buffer()).append_to(reader_compile_args);

    KernelDescriptor reader_kernel;
    reader_kernel.kernel_source = kReaderKernelPath;
    reader_kernel.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_kernel.core_ranges = all_cores;
    reader_kernel.compile_time_args = std::move(reader_compile_args);
    reader_kernel.config = ReaderConfigDescriptor{};

    // Writer kernel descriptor.
    KernelDescriptor::CompileTimeArgs writer_compile_args = {kSrcCbIndex};
    TensorAccessorArgs(cache.buffer()).append_to(writer_compile_args);

    KernelDescriptor writer_kernel;
    writer_kernel.kernel_source = kWriterKernelPath;
    writer_kernel.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_kernel.core_ranges = all_cores;
    writer_kernel.compile_time_args = std::move(writer_compile_args);
    writer_kernel.config = WriterConfigDescriptor{};

    // Per-core runtime args (mirrors the original fill_cache work-split scheme).
    auto* src_buffer = input.buffer();
    auto* dst_buffer = cache.buffer();
    const uint32_t g1_numcores = core_group_1.num_cores();

    const auto cores = corerange_to_cores(all_cores, num_cores, /*row_major=*/true);
    reader_kernel.runtime_args.reserve(num_cores);
    writer_kernel.runtime_args.reserve(num_cores);

    uint32_t num_blocks_written = 0;
    for (uint32_t i = 0; i < num_cores; ++i) {
        const CoreCoord& core = cores.at(i);
        const uint32_t num_blocks_per_core = (i < g1_numcores) ? num_blocks_per_core_g1 : num_blocks_per_core_g2;

        // Reader: (src_addr, num_tiles, src_start_tile_id)
        reader_kernel.runtime_args.emplace_back(
            core,
            KernelDescriptor::CoreRuntimeArgs{
                src_buffer->address(),
                num_blocks_per_core * Wt,
                num_blocks_written * Wt,
            });

        // Writer: (dst_addr, num_tiles, dst_start_tile_id)
        const uint32_t cache_start_id =
            start_idx + (num_blocks_written / input_Ht * cache_HtWt) + ((num_blocks_written % input_Ht) * Wt);
        writer_kernel.runtime_args.emplace_back(
            core,
            KernelDescriptor::CoreRuntimeArgs{
                dst_buffer->address(),
                num_blocks_per_core * Wt,
                cache_start_id,
            });

        num_blocks_written += num_blocks_per_core;
    }

    desc.kernels.push_back(std::move(reader_kernel));
    desc.kernels.push_back(std::move(writer_kernel));
    return desc;
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::update_padded_kv_cache

namespace ttnn::prim {

ttnn::Tensor update_padded_kv_cache(
    const ttnn::Tensor& cache,
    const ttnn::Tensor& input,
    uint32_t batch_idx,
    uint32_t kv_actual_global,
    uint32_t cluster_axis) {
    using OperationType =
        ttnn::operations::experimental::deepseek_prefill::update_padded_kv_cache::UpdatePaddedKvCacheDeviceOperation;
    auto attrs = OperationType::operation_attributes_t{
        .batch_idx = batch_idx,
        .kv_actual_global = kv_actual_global,
        .cluster_axis = cluster_axis,
    };
    auto tensor_args = OperationType::tensor_args_t{.cache = cache, .input = input};
    return ttnn::device_operation::launch<OperationType>(attrs, tensor_args);
}

}  // namespace ttnn::prim
