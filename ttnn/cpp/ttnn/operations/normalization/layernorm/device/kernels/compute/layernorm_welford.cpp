// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#define BCAST_LLKOP EltwiseBinaryType::ELWMUL
#define BCAST_DIM BroadcastType::COL

#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/welford.h"
#include "api/compute/transpose_wh.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/operations/normalization/kernel_util/compute/memory.h"
#include "ttnn/operations/normalization/kernel_util/generic/blocked_range.h"
#include "api/dataflow/circular_buffer.h"
#include "experimental/kernel_args.h"

namespace kutil = norm::kernel_util;
namespace generic = kutil::generic;

void kernel_main() {
    auto NCHt = get_arg(args::NCHt);
    constexpr auto Wt = get_arg(args::Wt);
    constexpr auto blk = get_arg(args::block_size);
    constexpr auto do_gamma = get_arg(args::do_gamma);
    constexpr auto do_beta = get_arg(args::do_beta);
    constexpr bool FLOAT32_DTYPE = get_arg(args::FLOAT32_DTYPE) == 1;
    constexpr auto W = get_arg(args::W);
    constexpr auto tile_width = get_arg(args::TILE_SIZE);
    constexpr bool fuse_pre_add = static_cast<bool>(get_arg(args::fuse_pre_add));

    constexpr uint32_t onetile = 1;

    // DFB handles. Conditional bindings on the host → conditional declarations here.
    constexpr uint32_t cb_eps = dfb::cb_eps;
    constexpr uint32_t cb_in = dfb::cb_in;
    constexpr uint32_t cb_out = dfb::cb_out;
    constexpr uint32_t cb_xmm = dfb::cb_xmm;
    constexpr uint32_t cb_ex = dfb::cb_ex;
    constexpr uint32_t cb_ex2 = dfb::cb_ex2;
    constexpr uint32_t cb_ex2pe = dfb::cb_ex2pe;
    constexpr uint32_t cb_reciprocals = dfb::cb_reciprocals;
    DataflowBuffer cb_eps_obj(cb_eps);
    DataflowBuffer cb_in_obj(cb_in);
    DataflowBuffer cb_out_obj(cb_out);
    DataflowBuffer cb_xmm_obj(cb_xmm);
    DataflowBuffer cb_ex_obj(cb_ex);
    DataflowBuffer cb_ex2_obj(cb_ex2);
    DataflowBuffer cb_ex2pe_obj(cb_ex2pe);

#ifdef FUSE_PRE_ADD
    constexpr uint32_t cb_inb = dfb::cb_inb;
    DataflowBuffer cb_inb_obj(cb_inb);
#endif
#ifdef FUSE_GAMMA
    constexpr uint32_t cb_gamma = dfb::cb_gamma;
    DataflowBuffer cb_gamma_obj(cb_gamma);
#endif
#ifdef FUSE_BETA
    constexpr uint32_t cb_beta = dfb::cb_beta;
    DataflowBuffer cb_beta_obj(cb_beta);
#endif

#if defined FUSE_GAMMA || defined FUSE_BETA
    constexpr uint32_t cb_fusion = dfb::cb_fusion;
    DataflowBuffer cb_fusion_obj(cb_fusion);
    constexpr auto cb_im_or_out = (do_gamma | do_beta) ? cb_fusion : cb_out;
#else
    constexpr auto cb_im_or_out = cb_out;
#endif
    DataflowBuffer cb_im_or_out_obj(cb_im_or_out);

    //  Either in or in + b if doing fused pre-add
#ifdef FUSE_PRE_ADD
    constexpr uint32_t cb_x = dfb::cb_x;
#else
    constexpr uint32_t cb_x = cb_in;
#endif
    DataflowBuffer cb_x_obj(cb_x);

    constexpr uint32_t dst0 = 0;
    constexpr uint32_t input_dst = 0;  // Input tile for Welford's algorithm
    constexpr uint32_t mean_dst = 1;
    constexpr uint32_t var_dst = 2;

    // The number of valid columns in the last tile in width dimension.
    // Because the Welford's llk is given transposed data, skip some rows when
    // we want to skip some columns from getting processed by layer_norm.
    constexpr uint32_t last_tile_rows = (W % tile_width) == 0 ? tile_width : W % tile_width;

    cb_eps_obj.wait_front(1);  // comes from the reader

#ifdef FUSE_PRE_ADD
    binary_op_init_common(cb_in, cb_inb, cb_x);
    pack_reconfig_data_format(cb_x);
#else
    compute_kernel_hw_startup(cb_in, cb_ex);
    pack_reconfig_data_format(cb_ex);
#endif

    // Get pointer to the reciprocal LUT
    using recip_lut_t = std::array<uint32_t, W>;
    auto p_reciprocals = kutil::compute::memory::get_pointer_to_cb_data<recip_lut_t>(cb_reciprocals, 0);

    // Intermediate buffers need to be reserved/pushed/popped
    // in full blocks
    const auto total_buffer_size = generic::blocks(Wt, blk).total_with_remainder();

    for (uint32_t ncht = 0; ncht < NCHt; ncht++) {
#ifdef FUSE_PRE_ADD
        {
            // x = in + b
            add_tiles_init(cb_in, cb_inb);
            reconfig_data_format(cb_in, cb_inb);
            pack_reconfig_data_format(cb_x);
            for (auto block : generic::blocks(Wt, blk)) {
                cb_in_obj.wait_front(block.full_block_size());
                cb_inb_obj.wait_front(block.full_block_size());
                tile_regs_acquire();
                for (auto i : block.local()) {
                    add_tiles(cb_in, cb_inb, i, i, i);
                }
                tile_regs_commit();
                cb_in_obj.pop_front(block.full_block_size());
                cb_inb_obj.pop_front(block.full_block_size());

                cb_x_obj.reserve_back(block.full_block_size());
                tile_regs_wait();
                for (auto i : block.local()) {
                    pack_tile(i, cb_x);
                }
                tile_regs_release();
                cb_x_obj.push_back(block.full_block_size());
            }
            reconfig_data_format(cb_in, cb_x, cb_inb, cb_ex);
        }
#endif

        // Simultaneous calculation of E[x] and Var[x] using Welford's algorithm
        uint32_t start_N = 0;
        reconfig_data_format_srca(cb_x);
        transpose_wh_init_short(cb_x);
        tile_regs_acquire();
        welford_init();
        // Process all but the last tile
        for (uint32_t wt = 0; wt < (Wt - 1); ++wt) {
            cb_x_obj.wait_front(wt + 1);
            // Welford's needs transposed input tile
            transpose_wh_tile(cb_x, wt, input_dst);
            welford_update<W>(input_dst, start_N, *p_reciprocals);
            start_N += tile_width;
        }

        // Process the last tile
        // cb_x is synced on full blocks, so we need to wait for the
        // last tile + any remaining in the last block
        const auto num_to_wait = generic::blocks(Wt, blk).total_with_remainder();
        cb_x_obj.wait_front(num_to_wait);
        transpose_wh_tile(cb_x, Wt - 1, input_dst);
        welford_update_rows<W>(input_dst, start_N, 0, last_tile_rows, *p_reciprocals);

        // Store the mean and variance to the destination registers
        welford_finalize_to_row<W>(mean_dst, W - 1, *p_reciprocals);
        tile_regs_commit();

        // Transpose mean and var back to columns
        cb_ex_obj.reserve_back(onetile);
        cb_ex2_obj.reserve_back(onetile);
        tile_regs_wait();
        pack_reconfig_data_format(cb_ex);
        pack_tile(mean_dst, cb_ex);
        pack_reconfig_data_format(cb_ex2);
        pack_tile(var_dst, cb_ex2);
        tile_regs_release();
        cb_ex_obj.push_back(onetile);
        cb_ex2_obj.push_back(onetile);

        cb_ex_obj.wait_front(onetile);
        cb_ex2_obj.wait_front(onetile);
        reconfig_data_format_srca(cb_ex);
        transpose_wh_init_short(cb_ex);
        tile_regs_acquire();
        transpose_wh_tile(cb_ex, 0, mean_dst);
        transpose_wh_tile(cb_ex2, 0, var_dst);
        tile_regs_commit();

        cb_ex_obj.pop_front(onetile);
        cb_ex2_obj.pop_front(onetile);

        cb_ex_obj.reserve_back(onetile);
        cb_ex2_obj.reserve_back(onetile);

        pack_reconfig_data_format(cb_ex);
        tile_regs_wait();
        pack_tile(mean_dst, cb_ex);
        pack_reconfig_data_format(cb_ex2);
        pack_tile(var_dst, cb_ex2);
        tile_regs_release();

        cb_ex_obj.push_back(onetile);
        cb_ex2_obj.push_back(onetile);

        // x - E[x]
        // Reuse cb_x since we didn't pop anything from it
        if constexpr (FLOAT32_DTYPE) {
            reconfig_data_format(cb_x, cb_ex);
        }
        cb_ex_obj.wait_front(onetile);  // should have 1 tile
        cb_xmm_obj.reserve_back(total_buffer_size);
        sub_bcast_cols_init_short(cb_x, cb_ex);
        for (auto block : generic::blocks(Wt, blk)) {
            tile_regs_acquire();
            for (auto i : block.local()) {
                sub_tiles_bcast_cols(cb_x, cb_ex, i, 0, i);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (auto i : block.local()) {
                pack_tile(i, cb_xmm);
            }
            tile_regs_release();
            cb_xmm_obj.push_back(block.full_block_size());
            cb_x_obj.pop_front(block.full_block_size());
        }
        cb_ex_obj.pop_front(1);
        cb_xmm_obj.wait_front(total_buffer_size);

        if constexpr (!fuse_pre_add) {
            reconfig_data_format_srca(cb_x, cb_xmm);
        }

        // Var(x) + eps
        if constexpr (FLOAT32_DTYPE) {
            reconfig_data_format(cb_ex2, cb_eps);
        }
        cb_ex2_obj.wait_front(onetile);  // should have 1 tile
        tile_regs_acquire();
        add_tiles_init(cb_ex2, cb_eps);
        add_tiles(cb_ex2, cb_eps, 0, 0, dst0);
        rsqrt_tile_init();
        rsqrt_tile(dst0);
        tile_regs_commit();
        cb_ex2_obj.pop_front(onetile);

        cb_ex2pe_obj.reserve_back(onetile);
        tile_regs_wait();
        pack_tile(dst0, cb_ex2pe);
        tile_regs_release();
        cb_ex2pe_obj.push_back(onetile);

        // Remainder of the layernorm operation
        // norm(x) * gamma + beta,
        // where norm(x) is:
        // (x - E[x]) / sqrt(E[(x-E[x])^2] + eps)
        cb_ex2pe_obj.wait_front(onetile);
        for (auto block : generic::blocks(Wt, blk)) {
            reconfig_data_format(cb_xmm, cb_ex2pe);
#if defined FUSE_GAMMA || defined FUSE_BETA
            pack_reconfig_data_format(cb_fusion);
#else
            pack_reconfig_data_format(cb_out);
#endif

            mul_bcast_cols_init_short(cb_xmm, cb_ex2pe);
            tile_regs_acquire();
            for (auto i : block.local()) {
                mul_tiles_bcast_cols(cb_xmm, cb_ex2pe, block.to_global(i), 0, i);
            }
            tile_regs_commit();

            cb_im_or_out_obj.reserve_back(block.full_block_size());
            tile_regs_wait();
            for (auto i : block.local()) {
                pack_tile(i, cb_im_or_out);
            }
            tile_regs_release();
            cb_im_or_out_obj.push_back(block.full_block_size());

#ifdef FUSE_GAMMA
            {
#ifndef FUSE_BETA
                pack_reconfig_data_format(cb_out);
#endif
                reconfig_data_format_srcb(cb_ex2pe, cb_gamma);
#ifdef FUSE_BETA
                uint32_t cb_outg = cb_fusion;
#else
                uint32_t cb_outg = cb_out;
#endif
                DataflowBuffer cb_outg_obj(cb_outg);
                mul_bcast_rows_init_short(cb_fusion, cb_gamma);
                cb_gamma_obj.wait_front(block.start() + block.full_block_size());
                cb_fusion_obj.wait_front(block.full_block_size());
                tile_regs_acquire();
                for (auto i : block.local()) {
                    mul_tiles_bcast_rows(cb_fusion, cb_gamma, i, block.to_global(i), i);
                }
                tile_regs_commit();
                cb_fusion_obj.pop_front(block.full_block_size());

                cb_outg_obj.reserve_back(block.full_block_size());
                tile_regs_wait();
                for (auto i : block.local()) {
                    pack_tile(i, cb_outg);
                }
                tile_regs_release();
                cb_outg_obj.push_back(block.full_block_size());
            }
#endif
#ifdef FUSE_BETA
            {
                pack_reconfig_data_format(cb_out);
#ifdef FUSE_GAMMA
                reconfig_data_format_srcb(cb_gamma, cb_beta);
#else
                reconfig_data_format_srcb(cb_ex2pe, cb_beta);
#endif

                add_bcast_rows_init_short(cb_fusion, cb_beta);
                cb_beta_obj.wait_front(block.start() + block.full_block_size());
                cb_fusion_obj.wait_front(block.full_block_size());
                tile_regs_acquire();
                for (auto i : block.local()) {
                    add_tiles_bcast_rows(cb_fusion, cb_beta, i, block.to_global(i), i);
                }
                tile_regs_commit();
                cb_fusion_obj.pop_front(block.full_block_size());

                cb_out_obj.reserve_back(block.full_block_size());
                tile_regs_wait();
                for (auto i : block.local()) {
                    pack_tile(i, cb_out);
                }
                tile_regs_release();
                cb_out_obj.push_back(block.full_block_size());
            }
#endif
        }
        cb_ex2pe_obj.pop_front(onetile);
        cb_xmm_obj.pop_front(total_buffer_size);

    }  // NCHt loop
}
