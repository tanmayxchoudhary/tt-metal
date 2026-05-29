// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Wire format of one DRAM-core prefetcher request page, shared by the host
// (DramCorePrefetcherManager composes the bytes) and the DRISC kernel (reads them
// out of its H2D socket page via the same structs). Keep both structs packed so the
// L1 byte layout matches on both sides.
//
// The page starts with a DramCorePrefetcherRequestHeader: a one-byte command id
// (DramCorePrefetcherBaseCmd) followed by a union of the per-command payloads,
// modeled on the dispatch CQPrefetchCmd / CQDispatchCmd encoding in
// tt_metal/impl/dispatch/kernels/cq_commands.hpp. The three commands are:
//   * STOP     — no payload; the kernel exits its request loop.
//   * PREFETCH  — followed in the page by header.prefetch.num_tensors
//                 DramCorePrefetcherTensorGeom entries (only the first num_tensors
//                 of kMaxTensorsPerRequest are valid).
//   * WAIT_CQ   — no geoms; the kernel spins until its per-CQ signal slot
//                 [wait_cq.cq_index] reaches wait_cq.cq_wait_value (wrap-safe).

#pragma once

#include <cstdint>

namespace tt::tt_metal {

// Number of per-DRAM-core CQ signal slots (one uint32 counter per command queue).
// WaitForCqOnDramCorePrefetcher writes an incrementing value into slot[cq_id] via
// the dispatcher; a WAIT_CQ request makes the kernel spin until it is reached.
constexpr uint32_t kNumCqSignalSlots = 2;

// Per-tensor geometry handed to the DRAM-core prefetcher kernel. All values are
// derived from the tensor shape + dtype + GCB ring topology + DRISC L1 stage budget;
// the host (compute_tensor_geom) picks (rows_per_sub, M) by the fit ladder documented
// in tt_metal/impl/buffers/prefetcher_matmul_design.md §6.
//
// Invariant: rows_per_sub > 1 implies M == 1 (the kernel cannot row-stride DMA).
struct DramCorePrefetcherTensorGeom {
    uint32_t bank_local_base = 0;      // GDDR offset where this tensor starts in the bank
    uint32_t num_sub = 0;              // sub-bands per ring-block
    uint32_t M = 0;                    // N-chunks per sub-band (divides num_receivers)
    uint32_t rows_per_sub = 0;         // K-rows per sub-band
    uint32_t coalesced_page_size = 0;  // bytes per K-row per receiver per coalesced page
    uint32_t coalesced_num_pages = 0;  // coalesced pages per K-row per receiver
    uint32_t sub_chunk_bytes = 0;      // bytes per DMA into one ring half
    uint32_t sub_stride_bytes = 0;     // DRAM byte stride between sub-bands within a block
    uint32_t block_stride_bytes = 0;   // DRAM byte stride between ring-blocks
    uint32_t page_bytes_per_recv = 0;  // bytes per receiver per full block (fifo_page_size)
    uint32_t block_count = 0;          // K-blocks for this tensor (per-tensor; was the shared GCB ring size)
} __attribute__((packed));

// One-byte command id at the front of every request page.
enum DramCorePrefetcherCmdId : uint8_t {
    DRAM_PREFETCHER_CMD_STOP = 0,      // exit the request loop (no payload)
    DRAM_PREFETCHER_CMD_PREFETCH = 1,  // num_tensors geoms follow the header
    DRAM_PREFETCHER_CMD_WAIT_CQ = 2,   // spin until cq slot[cq_index] >= cq_wait_value
};

struct DramCorePrefetcherBaseCmd {
    DramCorePrefetcherCmdId cmd_id;  // 1 byte
} __attribute__((packed));

// PREFETCH payload. The leading pad keeps num_layers/gcb_state_addr 4-byte aligned
// past the one-byte base (mirrors the pad fields in cq_commands.hpp commands).
struct DramCorePrefetcherPrefetchCmd {
    uint8_t pad1;
    uint16_t num_tensors;     // number of valid DramCorePrefetcherTensorGeom entries
    uint32_t num_layers;      // outer loop count: the kernel replays the tensor list this many times
    uint32_t gcb_state_addr;  // DRISC L1 base of the target GCB's sender state block
} __attribute__((packed));

// WAIT_CQ payload.
struct DramCorePrefetcherWaitCqCmd {
    uint8_t cq_index;  // which per-core CQ signal slot to wait on (0/1)
    uint16_t pad1;
    uint32_t cq_wait_value;  // wait until slot >= this value (wrap-safe int32 compare)
} __attribute__((packed));

// Header at the start of each request page: command id + per-command payload union.
struct DramCorePrefetcherRequestHeader {
    DramCorePrefetcherBaseCmd base;
    union {
        DramCorePrefetcherPrefetchCmd prefetch;
        DramCorePrefetcherWaitCqCmd wait_cq;
    } __attribute__((packed));
};

}  // namespace tt::tt_metal
