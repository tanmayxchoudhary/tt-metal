# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for ttnn.experimental.deepseek_prefill.update_padded_kv_cache.

The op writes a `chunk_local`-row input slab into a KV cache at a per-device
start offset, derived from a single global token count `kv_actual_global`.
When the global count aligns to a chunk boundary, every device writes at the
same local offset (the chunked-natural case). When it does not, devices around
the boundary write at different local offsets so that new tokens overwrite
the trailing pad cells of the prior cache before spilling into the next slab.

The torch showcase below is a pure-python spec for that math, independent of
device. It documents how the kernel must compute `update_idxt` per device and
asserts the resulting cache state matches a natural-order reference for a
range of scenarios.

A ttnn device test will be added in the same file once the op is implemented.
"""

import pytest
import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc


TILE_H = 32
TILE_W = 32


def _update_idxt_for_chip(kv_actual_global, my_sp_coord, sp_factor, chunk_local):
    """Per-chip starting tile index (in tiles) for the write.

    Mirrors the math the device kernel / host descriptor will use.

    Args:
        kv_actual_global: prior valid global KV length (tokens, tile-aligned).
        my_sp_coord: this device's index along the sp cluster axis (0..sp_factor-1).
        sp_factor: number of devices along the sp axis.
        chunk_local: per-device input seq length (tokens).

    Returns:
        Local cache tile index where this device starts writing.
    """
    assert kv_actual_global % TILE_H == 0
    assert chunk_local % TILE_H == 0
    kv_actual_t = kv_actual_global // TILE_H
    chunk_local_t = chunk_local // TILE_H
    chunk_global_t = sp_factor * chunk_local_t

    boundary_slab_idx = kv_actual_t // chunk_global_t
    boundary_chip = (kv_actual_t // chunk_local_t) % sp_factor
    boundary_offset_t = kv_actual_t % chunk_local_t

    if my_sp_coord < boundary_chip:
        return (boundary_slab_idx + 1) * chunk_local_t
    elif my_sp_coord == boundary_chip:
        return boundary_slab_idx * chunk_local_t + boundary_offset_t
    else:
        return boundary_slab_idx * chunk_local_t


def _server_rotate(new_tokens, kv_actual_global, sp_factor, chunk_local):
    """Server-side reorder of natural-order new_tokens into per-chip inputs.

    For each chip, fills the first N rows of its input with the new tokens
    destined for its write range (pad-fill rows first if applicable, then
    NEW-slab rows), in the order they will land in the cache. Trailing rows
    stay zeroed (pad — SDPA's mask drops them).

    Returns:
        per_chip_input: tensor of shape [sp_factor, chunk_local, dim].
    """
    new_actual_isl, dim = new_tokens.shape
    per_chip = torch.zeros(sp_factor, chunk_local, dim, dtype=new_tokens.dtype)

    total_pad_in_old = max(0, sp_factor * chunk_local - kv_actual_global)
    fill_count = min(total_pad_in_old, new_actual_isl)
    chip_write_count = [0] * sp_factor

    # Phase 1: fill OLD-slab pad in global-position order.
    for i in range(fill_count):
        pad_global_pos = kv_actual_global + i
        chip = (pad_global_pos // chunk_local) % sp_factor
        per_chip[chip, chip_write_count[chip]] = new_tokens[i]
        chip_write_count[chip] += 1

    # Phase 2: fill NEW slabs, starting from chip 0.
    remaining = new_actual_isl - fill_count
    token_idx = fill_count
    chip = 0
    while remaining > 0:
        per_chip[chip, chip_write_count[chip]] = new_tokens[token_idx]
        chip_write_count[chip] += 1
        token_idx += 1
        remaining -= 1
        if chip_write_count[chip] == chunk_local:
            chip += 1  # next chip's NEW slab

    return per_chip


def _apply_op(cache, per_chip_input, kv_actual_global, sp_factor, chunk_local):
    """Apply the op's per-chip write into `cache` in-place.

    `cache` shape: [sp_factor, 2 * chunk_local, dim] — 2 slabs per chip.
    Each chip's write is `chunk_local` rows starting at `_update_idxt_for_chip`'s
    tile offset, written verbatim from `per_chip_input[chip]`.
    """
    for chip in range(sp_factor):
        idxt = _update_idxt_for_chip(kv_actual_global, chip, sp_factor, chunk_local)
        idx = idxt * TILE_H
        cache[chip, idx : idx + chunk_local] = per_chip_input[chip]
    return cache


def _natural_pos_to_chip_row(p, sp_factor, chunk_local):
    """Map global token position to (chip, local cache row) in chunked-natural layout.

    Per-chip cache layout: rows [k*chunk_local, (k+1)*chunk_local) hold slab k.
    Within slab k, chip c holds global positions [k*chunk_global + c*chunk_local,
    k*chunk_global + (c+1)*chunk_local).
    """
    chunk_global = sp_factor * chunk_local
    slab = p // chunk_global
    rem = p % chunk_global
    chip = rem // chunk_local
    cell_in_slab = rem % chunk_local
    local_row = slab * chunk_local + cell_in_slab
    return chip, local_row


# Scenario parametrizations. Each: (kv_actual_global, new_actual_isl, sp_factor, chunk_local)
# IDs document the case.
_SCENARIOS = [
    # MLA 10K test scope (sp=2, chunk_local=2560 → 2*chunk_local = 5120 = cache per chip).
    (0, 5120, 2, 2560),  # iter0_full_chunk
    (0, 2560, 2, 2560),  # iter0_half_chunk (only chip 0 gets real data)
    (2560, 5120, 2, 2560),  # iter1_rotated (chip 1 fills its OLD-pad, chip 0 fills NEW slab)
    (5120, 5120, 2, 2560),  # iter1_chunk_aligned (degenerate: both chips at slab 1)
    # Smaller stress cases (sp=4, chunk_local=64) borrowed from the rotation handoff test scope.
    (0, 128, 4, 64),  # cold_start
    (64, 128, 4, 64),  # boundary at chip 1 entirely (full pad-fill on chip 1, then NEW on chip 0)
    (96, 64, 4, 64),  # boundary mid-chip-1; partial pad-fill
    (160, 64, 4, 64),  # multi-pad span (chips 2 and 3 OLD slabs are pad)
    (224, 64, 4, 64),  # single_pad_dev3 from rotation handoff
    (256, 64, 4, 64),  # no_old_pad: all new tokens go to NEW slabs from chip 0
]
_SCENARIO_IDS = [
    "mla_iter0_full",
    "mla_iter0_half",
    "mla_iter1_rotated",
    "mla_iter1_aligned",
    "small_cold_start",
    "small_full_chip1_padfill",
    "small_partial_chip1_padfill",
    "small_multi_pad",
    "small_dev3",
    "small_no_old_pad",
]


@pytest.mark.parametrize(
    "kv_actual_global, new_actual_isl, sp_factor, chunk_local",
    _SCENARIOS,
    ids=_SCENARIO_IDS,
)
def test_update_padded_kv_cache_torch_showcase(kv_actual_global, new_actual_isl, sp_factor, chunk_local):
    """Pure-torch spec for the per-chip update_idxt math.

    Builds a sentinel-filled 2-slab cache per chip, pre-seeds positions
    [0, kv_actual_global) with "prior valid" tokens at their natural cache
    cells, applies the op math with server-rotated input, and asserts:

      1. Prior valid cells are untouched.
      2. New tokens land at their natural global positions in the cache.
      3. The per-chip write range is bit-equivalent to `per_chip_input[chip]`
         (proves we write exactly chunk_local rows from the chip's input).
    """
    assert chunk_local % TILE_H == 0
    assert kv_actual_global % TILE_H == 0
    assert new_actual_isl % TILE_H == 0
    assert (
        kv_actual_global + new_actual_isl <= 2 * sp_factor * chunk_local
    ), "test scope: total valid tokens must fit in 2-slab cache"

    dim = 8
    torch.manual_seed(0)
    sentinel = -999.0

    # Init cache with sentinel; pre-seed prior valid positions in natural layout.
    cache = torch.full((sp_factor, 2 * chunk_local, dim), sentinel, dtype=torch.float32)
    prior_tokens = torch.randn(kv_actual_global, dim) if kv_actual_global > 0 else torch.empty(0, dim)
    for p in range(kv_actual_global):
        chip, row = _natural_pos_to_chip_row(p, sp_factor, chunk_local)
        cache[chip, row] = prior_tokens[p]

    # This iter's new tokens in natural order.
    new_tokens = torch.randn(new_actual_isl, dim)

    # Server-side rotation: per-chip input matching destination order.
    per_chip_input = _server_rotate(new_tokens, kv_actual_global, sp_factor, chunk_local)

    # Apply the op.
    cache_post = cache.clone()
    _apply_op(cache_post, per_chip_input, kv_actual_global, sp_factor, chunk_local)

    # Assertion 1: prior valid cells unchanged.
    for p in range(kv_actual_global):
        chip, row = _natural_pos_to_chip_row(p, sp_factor, chunk_local)
        assert torch.equal(
            cache_post[chip, row], prior_tokens[p]
        ), f"prior position {p} (chip {chip}, row {row}) was modified"

    # Assertion 2: new tokens at their natural positions in the cache.
    for p_idx in range(new_actual_isl):
        p = kv_actual_global + p_idx
        chip, row = _natural_pos_to_chip_row(p, sp_factor, chunk_local)
        assert torch.equal(
            cache_post[chip, row], new_tokens[p_idx]
        ), f"new token {p_idx} (global pos {p}, chip {chip}, row {row}) mismatch"

    # Assertion 3: per-chip write range bit-equivalent to per_chip_input.
    for chip in range(sp_factor):
        idxt = _update_idxt_for_chip(kv_actual_global, chip, sp_factor, chunk_local)
        idx = idxt * TILE_H
        assert torch.equal(
            cache_post[chip, idx : idx + chunk_local], per_chip_input[chip]
        ), f"chip {chip} write range [{idx}, {idx + chunk_local}) does not match input"


@pytest.mark.parametrize(
    "kv_actual_global, sp_factor, chunk_local, expected_per_chip_idxt",
    [
        # Chunk-aligned: degenerate case — every chip same offset (slab boundary).
        (0, 4, 64, [0, 0, 0, 0]),  # cold start: all at slab 0
        (
            256,
            4,
            64,
            [2, 2, 2, 2],
        ),  # past one chunk_global: all at slab 1 in TILES (2 tiles = slab 1 since chunk_local_t=2)
        # Boundary at chip 1: chip 0 jumps to slab 1, chips 1+ stay at slab 0.
        (64, 4, 64, [2, 0, 0, 0]),
        # Boundary mid-chip-1: chip 0 jumps to slab 1, chip 1 starts at offset_t=1, chips 2+ at slab 0.
        (96, 4, 64, [2, 1, 0, 0]),
        # Multi-pad: kv=160. Boundary chip=2, offset=32 → chip 0,1 jump to slab 1; chip 2 starts at 1; chip 3 at 0.
        (160, 4, 64, [2, 2, 1, 0]),
    ],
    ids=["aligned_cold", "aligned_slab1", "chip1_full_pad", "chip1_partial_pad", "multi_pad"],
)
def test_update_padded_kv_cache_idxt_math(kv_actual_global, sp_factor, chunk_local, expected_per_chip_idxt):
    """Direct unit test of _update_idxt_for_chip math against hand-computed values."""
    for chip, expected_idxt in enumerate(expected_per_chip_idxt):
        got = _update_idxt_for_chip(kv_actual_global, chip, sp_factor, chunk_local)
        assert got == expected_idxt, f"kv={kv_actual_global}, chip={chip}: expected idxt={expected_idxt}, got {got}"


# ===========================================================================
# TTNN device test
# ===========================================================================
#
# Exercises ttnn.experimental.deepseek_prefill.update_padded_kv_cache end-to-end:
# sets up a sharded cache pre-seeded with prior valid tokens at natural positions,
# builds a server-rotated per-chip input, calls the op, gathers the cache to host,
# and verifies that prior cells are untouched, new tokens land at their natural
# positions, and trailing pad cells get overwritten with the input's pad rows.
#
# Scenarios mirror the torch showcase, mapped onto a 2x4 mesh:
#   - sp_axis=0 (sp=2) covers the 10K MLA scope (chunk_local=2560).
#   - sp_axis=1 (sp=4) covers the smaller rotation-handoff scenarios.


def _build_per_chip_cache(prior_tokens, sp_factor, chunk_local, dim, sentinel):
    """Build per-chip cache tensor of shape [sp_factor, 2*chunk_local, dim]
    pre-seeded with prior tokens at their natural positions."""
    cache = torch.full((sp_factor, 2 * chunk_local, dim), sentinel, dtype=torch.bfloat16)
    for p in range(len(prior_tokens)):
        chip, row = _natural_pos_to_chip_row(p, sp_factor, chunk_local)
        cache[chip, row] = prior_tokens[p]
    return cache


def _to_ttnn_sp_sharded(host_tensor_per_chip, mesh_device, sp_axis):
    """Convert a [sp_factor, S, dim] host tensor into a ttnn tensor sharded
    along the mesh's sp axis. The other mesh axis gets replicated.

    Returns the ttnn tensor; the unsharded torch shape is
    [1, 1, sp_factor * S, dim].
    """
    sp_factor, S, dim = host_tensor_per_chip.shape
    full = host_tensor_per_chip.reshape(1, 1, sp_factor * S, dim).contiguous()
    shard_dims = [None, None]
    shard_dims[sp_axis] = 2
    return ttnn.from_torch(
        full,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=shard_dims),
    )


def _gather_sp_sharded(tt_tensor, mesh_device, sp_axis):
    """Inverse of `_to_ttnn_sp_sharded`. Returns a [sp_factor, S, dim] host tensor."""
    # ConcatMesh2dToTensor concatenates along the given dims; for the replicated
    # axis, take the first row's data (all rows are identical).
    concat_dims = [None, None]
    concat_dims[sp_axis] = 2
    # The non-sp mesh axis is replicated; pick the first slice along it.
    other_axis = 1 - sp_axis
    concat_dims[other_axis] = 0  # batch dim, replicated
    full = ttnn.to_torch(
        tt_tensor,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=concat_dims, mesh_shape=mesh_device.shape),
    ).to(torch.bfloat16)
    # full shape: [other_axis_size, 1, sp_factor * S, dim]; take row 0 along the replicated axis.
    full = full[:1]
    _, _, total_seq, dim = full.shape
    sp_factor = mesh_device.shape[sp_axis]
    S = total_seq // sp_factor
    return full.reshape(sp_factor, S, dim)


_DEVICE_SCENARIOS = [
    # MLA 10K scope on 2x4 (sp_axis=0 → sp=2, chunk_local=2560).
    (0, 5120, 2, 2560, 0),
    (2560, 5120, 2, 2560, 0),
    (5120, 5120, 2, 2560, 0),
    # Smaller scope on 2x4 (sp_axis=1 → sp=4, chunk_local=64).
    (0, 128, 4, 64, 1),
    (64, 128, 4, 64, 1),
    (96, 64, 4, 64, 1),
    (160, 64, 4, 64, 1),
    (224, 64, 4, 64, 1),
]
_DEVICE_IDS = [
    "mla_iter0_full",
    "mla_iter1_rotated",
    "mla_iter1_aligned",
    "small_cold",
    "small_chip1_full",
    "small_chip1_partial",
    "small_multi_pad",
    "small_dev3",
]


@pytest.mark.parametrize("mesh_device", [(2, 4)], ids=["2x4"], indirect=True)
@pytest.mark.parametrize(
    "kv_actual_global, new_actual_isl, sp_factor, chunk_local, sp_axis",
    _DEVICE_SCENARIOS,
    ids=_DEVICE_IDS,
)
@pytest.mark.timeout(0)
def test_update_padded_kv_cache_ttnn(mesh_device, kv_actual_global, new_actual_isl, sp_factor, chunk_local, sp_axis):
    """Device regression for ttnn.experimental.deepseek_prefill.update_padded_kv_cache."""
    assert (
        mesh_device.shape[sp_axis] == sp_factor
    ), f"mesh sp_axis={sp_axis} has size {mesh_device.shape[sp_axis]}; expected {sp_factor}"

    dim = TILE_W  # 32; minimal tile-aligned head dim
    sentinel = -1.5  # distinguishable bfloat16 value
    torch.manual_seed(0)

    # 1. Build prior-valid tokens and pre-seed cache.
    prior_tokens = (
        torch.randn(kv_actual_global, dim, dtype=torch.bfloat16)
        if kv_actual_global > 0
        else torch.empty(0, dim, dtype=torch.bfloat16)
    )
    cache_per_chip = _build_per_chip_cache(prior_tokens, sp_factor, chunk_local, dim, sentinel)

    # 2. Build server-rotated per-chip input.
    new_tokens = torch.randn(new_actual_isl, dim, dtype=torch.bfloat16)
    input_per_chip = _server_rotate(new_tokens, kv_actual_global, sp_factor, chunk_local)

    # 3. Push to device as 4D tensors [1, 1, total_seq, dim], sharded along sp axis.
    cache_per_chip_4d = cache_per_chip.unsqueeze(1)  # [sp_factor, 1, 2*chunk_local, dim]
    input_per_chip_4d = input_per_chip.unsqueeze(1)  # [sp_factor, 1, chunk_local, dim]

    tt_cache = _to_ttnn_sp_sharded(cache_per_chip_4d.reshape(sp_factor, 2 * chunk_local, dim), mesh_device, sp_axis)
    tt_input = _to_ttnn_sp_sharded(input_per_chip_4d.reshape(sp_factor, chunk_local, dim), mesh_device, sp_axis)

    # 4. Call the op.
    ttnn.experimental.deepseek_prefill.update_padded_kv_cache(
        tt_cache,
        tt_input,
        batch_idx=0,
        kv_actual_global=kv_actual_global,
        cluster_axis=sp_axis,
    )

    # 5. Gather cache back.
    cache_post_per_chip = _gather_sp_sharded(tt_cache, mesh_device, sp_axis)
    # cache_post_per_chip: [sp_factor, 2 * chunk_local, dim]

    # 6. Verify.
    # Prior valid cells unchanged.
    for p in range(kv_actual_global):
        chip, row = _natural_pos_to_chip_row(p, sp_factor, chunk_local)
        diff = (cache_post_per_chip[chip, row].float() - prior_tokens[p].float()).abs().max().item()
        assert diff < 1e-2, f"prior pos {p} (chip {chip}, row {row}) modified; max abs diff {diff}"

    # New tokens at natural positions match.
    for p_idx in range(new_actual_isl):
        p = kv_actual_global + p_idx
        chip, row = _natural_pos_to_chip_row(p, sp_factor, chunk_local)
        diff = (cache_post_per_chip[chip, row].float() - new_tokens[p_idx].float()).abs().max().item()
        assert diff < 1e-2, f"new pos {p} (chip {chip}, row {row}) mismatch; max abs diff {diff}"

    # Per-chip write range bit-equivalent to per_chip_input (within bfloat16 precision).
    for chip in range(sp_factor):
        idxt = _update_idxt_for_chip(kv_actual_global, chip, sp_factor, chunk_local)
        idx = idxt * TILE_H
        chip_pcc = (
            (cache_post_per_chip[chip, idx : idx + chunk_local].float() - input_per_chip[chip].float())
            .abs()
            .max()
            .item()
        )
        assert chip_pcc < 1e-2, f"chip {chip} write range mismatch; max abs diff {chip_pcc}"


# ===========================================================================
# Performance comparison vs the existing kv_cache.fill_cache_for_user_ op.
#
# Both tests use realistic MLA prefill shapes (chunk_local=2560, kvpe_dim=576,
# sp=2 on 2x4 mesh) and call their respective op exactly once. Run under tracy
# to compare DEVICE KERNEL DURATION between the new op (per-chip offset, descriptor
# pattern) and the legacy fill_cache_for_user_ (uniform offset, classic factory).
# Functional output is not asserted — both ops do equivalent work but write to
# different cache cells; only the kernel duration matters for perf comparison.
# ===========================================================================


_PERF_CHUNK_LOCAL = 2560  # 5K-token MLA chunk per chip (sp=2 → chunk_global=5120)
_PERF_KVPE_DIM = 576  # MLA kvpe head dim (kv_lora_rank=512 + qk_rope_head_dim=64)
_PERF_SP_AXIS = 0


def _build_perf_inputs(mesh_device):
    """Build a sentinel cache + random input shaped like one MLA chunk."""
    sp_factor = mesh_device.shape[_PERF_SP_AXIS]
    chunk_local = _PERF_CHUNK_LOCAL
    dim = _PERF_KVPE_DIM
    torch.manual_seed(0)
    cache_per_chip = torch.zeros(sp_factor, 2 * chunk_local, dim, dtype=torch.bfloat16)
    input_per_chip = torch.randn(sp_factor, chunk_local, dim, dtype=torch.bfloat16)
    tt_cache = _to_ttnn_sp_sharded(cache_per_chip, mesh_device, _PERF_SP_AXIS)
    tt_input = _to_ttnn_sp_sharded(input_per_chip, mesh_device, _PERF_SP_AXIS)
    return tt_cache, tt_input


@pytest.mark.parametrize("mesh_device", [(2, 4)], ids=["2x4"], indirect=True)
@pytest.mark.timeout(0)
def test_perf_update_padded_kv_cache(mesh_device):
    """Single op invocation of the new per-chip-offset op for tracy capture."""
    tt_cache, tt_input = _build_perf_inputs(mesh_device)
    # Use kv_actual_global=2560 → rotation actually kicks in (boundary on chip 1).
    ttnn.experimental.deepseek_prefill.update_padded_kv_cache(
        tt_cache,
        tt_input,
        batch_idx=0,
        kv_actual_global=2560,
        cluster_axis=_PERF_SP_AXIS,
    )
    ttnn.synchronize_device(mesh_device)


@pytest.mark.parametrize("mesh_device", [(2, 4)], ids=["2x4"], indirect=True)
@pytest.mark.timeout(0)
def test_perf_fill_cache_for_user_baseline(mesh_device):
    """Single op invocation of the legacy uniform-offset op for tracy capture."""
    tt_cache, tt_input = _build_perf_inputs(mesh_device)
    # Writes at offset 0 on every chip. Doesn't matter for perf — workload size
    # is identical to the new op's call (same chunk_local rows written per chip).
    ttnn.kv_cache.fill_cache_for_user_(tt_cache, tt_input, 0)
    ttnn.synchronize_device(mesh_device)
