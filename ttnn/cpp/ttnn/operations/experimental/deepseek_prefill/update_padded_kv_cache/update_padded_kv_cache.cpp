// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "update_padded_kv_cache.hpp"
#include "device/update_padded_kv_cache_device_operation.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::update_padded_kv_cache {

ttnn::Tensor update_padded_kv_cache(
    const ttnn::Tensor& cache,
    const ttnn::Tensor& input,
    uint32_t batch_idx,
    uint32_t kv_actual_global,
    uint32_t cluster_axis) {
    return ttnn::prim::update_padded_kv_cache(cache, input, batch_idx, kv_actual_global, cluster_axis);
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::update_padded_kv_cache
