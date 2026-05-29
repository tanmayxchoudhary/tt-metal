// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/noc_semaphore.h"

/**
 * ChainConfig: Runtime args for store-and-forward chain configuration.
 * Mirrors the append_to_args() layout in ring_joint_sdpa_program_factory.cpp.
 */
struct ChainConfig {
    bool participates = false;
    bool is_injector = false;
    bool is_sink = false;
    uint32_t batch = 0;
    uint32_t head = 0;
    uint32_t prev_physical_x = 0;
    uint32_t prev_physical_y = 0;
    uint32_t next_physical_x = 0;
    uint32_t next_physical_y = 0;
    uint32_t next_core_q_chunks = 0;
    uint32_t mcast_start_x = 0;
    uint32_t mcast_start_y = 0;
    uint32_t mcast_end_x = 0;
    uint32_t mcast_end_y = 0;
    uint32_t injector_physical_x = 0;
    uint32_t injector_physical_y = 0;
    uint32_t mcast_num_dests = 0;
    uint32_t mcast_sender_wait = 0;

    // Read 18 args in canonical order matching append_to_args()
    static ChainConfig read_from_args(uint32_t& argidx) {
        ChainConfig cfg;
        cfg.participates = static_cast<bool>(get_arg_val<uint32_t>(argidx++));
        cfg.is_injector = static_cast<bool>(get_arg_val<uint32_t>(argidx++));
        cfg.is_sink = static_cast<bool>(get_arg_val<uint32_t>(argidx++));
        cfg.batch = get_arg_val<uint32_t>(argidx++);
        cfg.head = get_arg_val<uint32_t>(argidx++);
        cfg.prev_physical_x = get_arg_val<uint32_t>(argidx++);
        cfg.prev_physical_y = get_arg_val<uint32_t>(argidx++);
        cfg.next_physical_x = get_arg_val<uint32_t>(argidx++);
        cfg.next_physical_y = get_arg_val<uint32_t>(argidx++);
        cfg.next_core_q_chunks = get_arg_val<uint32_t>(argidx++);
        cfg.mcast_start_x = get_arg_val<uint32_t>(argidx++);
        cfg.mcast_start_y = get_arg_val<uint32_t>(argidx++);
        cfg.mcast_end_x = get_arg_val<uint32_t>(argidx++);
        cfg.mcast_end_y = get_arg_val<uint32_t>(argidx++);
        cfg.injector_physical_x = get_arg_val<uint32_t>(argidx++);
        cfg.injector_physical_y = get_arg_val<uint32_t>(argidx++);
        cfg.mcast_num_dests = get_arg_val<uint32_t>(argidx++);
        cfg.mcast_sender_wait = get_arg_val<uint32_t>(argidx++);
        return cfg;
    }

    // Compute signal target based on mcast mode
    template <bool mcast_enabled>
    uint32_t signal_target_x() const {
        return mcast_enabled ? injector_physical_x : prev_physical_x;
    }

    template <bool mcast_enabled>
    uint32_t signal_target_y() const {
        return mcast_enabled ? injector_physical_y : prev_physical_y;
    }
};

/**
 * ChainLink: Unified abstraction for store-and-forward chaining.
 *
 * Each core in a chain is a "link" that can receive from upstream and forward downstream.
 *
 * Template parameters:
 * - mcast_enabled: selects multicast vs unicast forwarding at compile time
 * - is_head_level: true = head chain (matches batch AND head), false = batch chain (matches batch only)
 *
 * Constructor takes semaphore IDs plus a Noc reference,
 * and resolves L1 / NoC addresses once at construction so receive()/forward() can skip the
 * per-hop re-derivation in the K/V mcast hot path.
 *
 * For non-participating cores the semaphores are not used; the valid semaphore is only
 * initialized when is_participant=true.
 */
template <bool mcast_enabled, bool is_head_level>
class ChainLink {
public:
    // Participation flags
    const bool is_participant;
    const bool is_injector;
    const bool is_sink;

    // Chain scope matching
    const uint32_t chain_batch;
    const uint32_t chain_head;  // Only checked when is_head_level
    const uint32_t next_core_q_chunks;

    ChainLink(
        const Noc& noc,
        bool is_participant,
        bool is_injector,
        bool is_sink,
        uint32_t sender_sem_id,
        uint32_t receiver_sem_id,
        uint32_t valid_sem_id,
        uint32_t signal_target_x,
        uint32_t signal_target_y,
        uint32_t next_core_x,
        uint32_t next_core_y,
        uint32_t mcast_start_x,
        uint32_t mcast_start_y,
        uint32_t mcast_end_x,
        uint32_t mcast_end_y,
        uint32_t mcast_num_dests,
        uint32_t mcast_sender_wait,
        uint32_t chunk_tiles,
        uint32_t tile_bytes,
        uint32_t chain_batch,
        uint32_t chain_head,
        uint32_t next_core_q_chunks) :
        is_participant(is_participant),
        is_injector(is_injector),
        is_sink(is_sink),
        chain_batch(chain_batch),
        chain_head(chain_head),
        next_core_q_chunks(next_core_q_chunks),
        noc_id_(noc.get_noc_id()),
        sender_sem_l1_addr_(get_semaphore(sender_sem_id)),
        receiver_sem_l1_addr_(get_semaphore(receiver_sem_id)),
        valid_sem_l1_addr_(get_semaphore(valid_sem_id)),
        sender_sem_noc_addr_(get_noc_addr(signal_target_x, signal_target_y, sender_sem_l1_addr_, noc_id_)),
        receiver_sem_noc_addr_(
            mcast_enabled ? uint64_t{0} : get_noc_addr(next_core_x, next_core_y, receiver_sem_l1_addr_, noc_id_)),
        mcast_base_noc_addr_(
            (mcast_enabled && is_injector)
                ? get_noc_multicast_addr(mcast_start_x, mcast_start_y, mcast_end_x, mcast_end_y, 0u, noc_id_)
                : uint64_t{0}),
        mcast_sem_noc_addr_(
            (mcast_enabled && is_injector) ? (mcast_base_noc_addr_ | receiver_sem_l1_addr_) : uint64_t{0}),
        unicast_dst_base_x_(next_core_x),
        unicast_dst_base_y_(next_core_y),
        sender_wait_count_((mcast_enabled && is_injector) ? mcast_sender_wait : 1),
        mcast_num_dests_(mcast_num_dests),
        chunk_tiles_(chunk_tiles),
        tile_bytes_(tile_bytes) {
        // Initialize valid semaphore (only meaningful for participants; non-participants leave it alone)
        if (is_participant) {
            Semaphore<>(valid_sem_id).set(VALID);
        }
    }

    /**
     * Check if this core should receive data from upstream.
     * Head-level chains match (batch, head), batch-level chains match batch only.
     * In mcast mode, skip batch check: mcast is only enabled for B=1, and padded
     * iterations may have garbage nb values from out-of-bounds global_q_chunk.
     */
    bool should_receive(uint32_t nb, uint32_t nq) const {
        if (!is_participant || is_injector) {
            return false;
        }
        if constexpr (!mcast_enabled) {
            if (nb != chain_batch) {
                return false;
            }
        }
        if constexpr (is_head_level) {
            if (nq != chain_head) {
                return false;
            }
        }
        return true;
    }

    /**
     * Check if this core should forward data to downstream.
     * Also checks iteration count against next core's expected reads.
     * In mcast mode, skip batch check (see should_receive comment).
     */
    bool should_forward(uint32_t nb, uint32_t nq, uint32_t q_iter_local) const {
        if (!is_participant || is_sink) {
            return false;
        }
        if constexpr (!mcast_enabled) {
            if (nb != chain_batch) {
                return false;
            }
        }
        if constexpr (is_head_level) {
            if (nq != chain_head) {
                return false;
            }
        }
        if (q_iter_local >= next_core_q_chunks) {
            return false;
        }
        return true;
    }

    /**
     * Receive data from upstream link (called by non-injector participants).
     * Protocol: signal sender that we're ready, then wait for data.
     *
     * The Noc parameter is kept for API parity; the runtime path uses noc_id_ captured at
     * construction.
     */
    void receive(const Noc& /*noc*/) const {
        auto* receiver_sem_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(receiver_sem_l1_addr_);
        noc_semaphore_set(receiver_sem_ptr, INVALID);
        noc_semaphore_inc(sender_sem_noc_addr_, 1, noc_id_);
        noc_semaphore_wait(receiver_sem_ptr, VALID);
    }

    /**
     * Forward data to downstream link(s) using default chunk size.
     * In mcast mode: wait for all receivers, then broadcast.
     * In unicast mode: wait for next receiver, then point-to-point transfer.
     */
    void forward(const Noc& noc, uint32_t cb_addr) const { forward(noc, cb_addr, chunk_tiles_, tile_bytes_); }

    /**
     * Forward data to downstream link(s) with explicit size.
     * Use this when the data size differs from the default (e.g., K using head chain).
     *
     * Uses NoC/L1 addresses cached at construction. Drops to the underlying noc_async_* /
     * noc_semaphore_* primitives.
     */
    void forward(const Noc& /*noc*/, uint32_t cb_addr, uint32_t num_tiles, uint32_t tile_bytes) const {
        auto* sender_sem_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sender_sem_l1_addr_);
        if constexpr (mcast_enabled) {
            noc_semaphore_wait(sender_sem_ptr, sender_wait_count_);
            noc_semaphore_set(sender_sem_ptr, 0);
            const uint64_t mcast_addr = mcast_base_noc_addr_ | cb_addr;
            noc_async_write_multicast(
                cb_addr,
                mcast_addr,
                num_tiles * tile_bytes,
                mcast_num_dests_,
                true /* linked: companion semaphore mcast follows */,
                noc_id_);
            // Companion semaphore mcast: write local valid_sem value into remote receiver_sem
            // (different L1 offset). Must be issued back-to-back after the linked write —
            // inserting a flush between them deadlocks.
            noc_semaphore_set_multicast(valid_sem_l1_addr_, mcast_sem_noc_addr_, mcast_num_dests_, false, noc_id_);
            noc_async_writes_flushed(noc_id_);
        } else {
            noc_semaphore_wait(sender_sem_ptr, 1);
            noc_semaphore_set(sender_sem_ptr, 0);
            // Data write — cb_addr varies per call, so encode here (coords cached).
            const uint64_t unicast_addr = ::get_noc_addr(unicast_dst_base_x_, unicast_dst_base_y_, cb_addr, noc_id_);
            noc_async_write(cb_addr, unicast_addr, num_tiles * tile_bytes, noc_id_);
            noc_async_writes_flushed(noc_id_);
            noc_semaphore_set_remote(valid_sem_l1_addr_, receiver_sem_noc_addr_, noc_id_);
        }
    }

private:
    // NoC index captured once so we don't re-read it from a passed-in Noc object per call.
    uint8_t noc_id_;

    // Local L1 semaphore addresses (read directly via volatile pointer in receive/forward).
    uint32_t sender_sem_l1_addr_;
    uint32_t receiver_sem_l1_addr_;
    uint32_t valid_sem_l1_addr_;

    // Precomputed NoC addresses (avoid get_noc_addr / get_noc_multicast_addr per hop).
    uint64_t sender_sem_noc_addr_;    // Remote upstream sender semaphore (for noc_semaphore_inc).
    uint64_t receiver_sem_noc_addr_;  // Remote downstream receiver semaphore (unicast only).
    uint64_t mcast_base_noc_addr_;    // Multicast rectangle | 0 (injector only).
    uint64_t mcast_sem_noc_addr_;     // mcast_base | receiver_sem_l1_addr_ (injector only).

    // Unicast forward destination coords (the L1 addr varies per call, only coords are stable).
    uint32_t unicast_dst_base_x_;
    uint32_t unicast_dst_base_y_;

    // Configuration
    uint32_t sender_wait_count_;
    uint32_t mcast_num_dests_;
    uint32_t chunk_tiles_;
    uint32_t tile_bytes_;
};
