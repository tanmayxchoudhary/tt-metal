# KV-Pad-Aware Rotation — MLA Integration Plan

Plan to integrate the now-validated KV-pad-aware ring-joint SDPA rotation
(`f69cc68f887`) into `models/demos/deepseek_v3_d_p/tt/mla/mla.py`'s chunked
prefill path, including a new per-chip-offset kv-cache fill op required to
write new K/V tokens into per-chip rotated cache positions.

The plan is split into two independent branches so the kv-cache op fork can
be developed and exhaustively unit-tested in isolation, then cherry-picked
into the MLA integration branch after validation.

## Status snapshot — 2026-05-29 EOD

| Phase | Status |
|---|---|
| Phase 0 — Branch setup (op branch off main) | **Done** — on `ipotkonjak/kv_cache_per_chip_offset` |
| Phase 1 — New op `update_padded_kv_cache` under `experimental/deepseek_prefill/` | **Done** — 23/23 tests passing (15 torch + 8 device on 2x4) |
| Phase 1.5 — Tracy perf comparison vs legacy `fill_cache_for_user_` | **Blocked** — Tracy issue on bh-lb-10 box; needs re-run on another machine. Test functions exist in the test file. |
| Phase 2 — MLA integration on `chunked_attn_mla_rotation` | **Not started** — waiting on perf sign-off |
| Phase 3 — MLA chunked-rotated test | **Not started** |
| Phase 4 — Regression sweep | **Not started** |

### Approach changes from the original plan
- Instead of modifying the production `ttnn.kv_cache.fill_cache_for_user_`
  op, **forked it** into a new experimental op under
  `ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/update_padded_kv_cache/`.
  This avoids touching a widely-used op and avoids cross-model approvals.
- Used the **`ProgramDescriptor` + `create_descriptor(mesh_dispatch_coordinate)`**
  pattern (same one `ring_joint_sdpa` uses). Lets per-chip `update_idxt` be
  baked into per-device writer rt-args at host time; kernels themselves stay
  the existing generic kv_cache reader + eltwise unary writer (no kernel forking).
- Op name shipped: **`ttnn.experimental.deepseek_prefill.update_padded_kv_cache`**.
  Signature: `(cache, input, batch_idx, kv_actual_global, cluster_axis)`. In-place;
  returns a handle to `cache`.

## Source-of-truth branches

| Branch | What it has |
|---|---|
| `ipotkonjak/chunked_attn_tests` | Rotation op `f69cc68f887` (Pavle-committed), Tests 1–3 fixes, handoff tests |
| `ipotkonjak/chunked_attn_mla` | Chunked MLA forward (cache-as-K, latent-V, persistent buffers at init, multi-user cache layout), `test_mla_chunked_prefill` scaffolding |
| `main` | Stable base; neither rotation op nor chunked MLA |

`chunked_attn_mla` branched before the rotation op landed, so it doesn't have
the kv-pad-aware SDPA kernel work yet.

## Branching strategy

Two branches, developed in parallel where possible:

- **Branch A — `ipotkonjak/kv_cache_per_chip_offset`** (off `main`). Op-only.
  Phase 1 (kv_cache extension) + thorough unit tests live here.
- **Branch B — `ipotkonjak/chunked_attn_mla_rotation`** (off `chunked_attn_mla`).
  MLA integration. After Branch A is validated, cherry-pick its commits here,
  then implement Phase 2 + Phase 3 on top.

```
main ─── A (op ext + unit tests) ──┐
                                   ▼ cherry-pick
chunked_attn_mla ──┐               │
                   ▼ merge         │
chunked_attn_tests ── B ──────────►├─── MLA forward wiring + rotated test
                                   │
                                   ▼
                          run Phase 4 regression
```

---

## Phase 0 — Branch setup

### A. Op branch

```bash
git checkout main
git pull
git checkout -b ipotkonjak/kv_cache_per_chip_offset
```

Touches only `ttnn/cpp/ttnn/operations/kv_cache/` files + a new unit test
under `tests/ttnn/unit_tests/operations/`.

### B. MLA integration branch

```bash
git checkout ipotkonjak/chunked_attn_mla
git checkout -b ipotkonjak/chunked_attn_mla_rotation
git merge ipotkonjak/chunked_attn_tests
```

A merge is preferred over cherry-pick — the two source branches touch disjoint
files (MLA python vs SDPA kernels). Expected conflicts:

- `models/demos/deepseek_v3_d_p/tests/test_ring_joint_sdpa_handoff.py` —
  `chunked_attn_mla` has an older copy via the `78bc4a56aba` import;
  `chunked_attn_tests` has the full 4-test + handoff iterations.
  **Resolution:** take `chunked_attn_tests` version (newer, complete).
- `models/demos/deepseek_v3_d_p/tt/mla/mla.py` — `chunked_attn_tests` does not
  touch this file; no conflict expected.

**Verify after merge** (before any new code):

```bash
pytest models/demos/deepseek_v3_d_p/tests/test_ring_joint_sdpa_handoff.py \
       -k "kv_pad_aware_rotation_torch_showcase or kv_pad_aware_rotation_ttnn"
pytest models/demos/deepseek_v3_d_p/tests/test_mla.py::test_mla_chunked_prefill \
       -k "seq10k and N2 and q32"
```

Both must pass before proceeding.

---

## Phase 1 — `fill_cache_for_user_` per-chip offset (on Branch A)

Unified single-path design: kernel always derives its `update_idxt` from a
single global token offset + its mesh coordinate. Chunk-aligned uniform writes
become the degenerate case.

### Files

- `ttnn/cpp/ttnn/operations/kv_cache/kv_cache.cpp` — top-level wrapper
- `ttnn/cpp/ttnn/operations/kv_cache/kv_cache.hpp`
- `ttnn/cpp/ttnn/operations/kv_cache/kv_cache_nanobind.cpp` — python binding
- `ttnn/cpp/ttnn/operations/kv_cache/device/update_cache_device_operation.hpp` — attribute struct
- `ttnn/cpp/ttnn/operations/kv_cache/device/update_cache_device_operation.cpp` — validation + hash
- `ttnn/cpp/ttnn/operations/kv_cache/device/fill_cache_multi_core_program_factory.cpp` — per-chip rt-arg derivation

### API

```cpp
ttnn::Tensor fill_cache_for_user_(
    const ttnn::Tensor& cache,
    const ttnn::Tensor& input,
    const uint32_t batch_index,
    const uint32_t update_idx = 0,                              // existing
    std::optional<uint32_t> kv_actual_global = std::nullopt,    // NEW
    std::optional<uint32_t> cluster_axis = std::nullopt);       // NEW; required when kv_actual_global is set
```

Python:
```python
ttnn.kv_cache.fill_cache_for_user_(
    cache, input, batch_index,
    update_idx=0,
    kv_actual_global=None,
    cluster_axis=None,
)
```

### Per-chip offset derivation (kernel-side)

Operates in tile units. Inputs to the kernel (compile-time or runtime as noted):

- `kv_actual_global_t = kv_actual_global / TILE_HEIGHT` (runtime arg).
- `chunk_local_t = input_seq / TILE_HEIGHT` (compile-time, from input shape).
- `sp_factor` (compile-time, from cluster axis size).
- `my_sp_coord` (runtime arg, from mesh placement).

```
chunk_global_t           = sp_factor * chunk_local_t
boundary_slab_idx        = kv_actual_global_t / chunk_global_t
boundary_chip            = (kv_actual_global_t / chunk_local_t) % sp_factor
boundary_offset_in_chip  = kv_actual_global_t % chunk_local_t

if my_sp_coord <  boundary_chip:
    update_idxt = (boundary_slab_idx + 1) * chunk_local_t
elif my_sp_coord == boundary_chip:
    update_idxt = boundary_slab_idx * chunk_local_t + boundary_offset_in_chip
else:
    update_idxt = boundary_slab_idx * chunk_local_t
```

Then the existing per-tile addressing math applies starting from `update_idxt`.

### Degenerate case (back-compat sanity)

When `kv_actual_global == chunk_start_global` AND it's chunk-aligned:
- `boundary_offset_in_chip == 0`
- All branches give `update_idxt = boundary_slab_idx * chunk_local_t`
- Identical to today's uniform `update_idx = chunk_start_global / sp_factor`

Bit-equivalent to the existing chunked-natural fill — unit-tested explicitly.

### Validation

In `update_cache_device_operation.cpp::validate_on_program_cache_miss`:

- `kv_actual_global` and a non-zero `update_idx` are mutually exclusive (one
  selection mechanism per call).
- When `kv_actual_global` is set: `cluster_axis` must be set too; cache seq
  dim must accommodate `kv_actual_global + sp_factor * chunk_local`.
- `kv_actual_global % TILE_HEIGHT == 0`.

### Program hash

Include `kv_actual_global` and `cluster_axis` so cached programs cannot reuse
stale rt-args across slots or callers.

### Mesh-coord availability

Confirm `update_cache` programs can read `sp_axis` coord. The AG/RS family
exposes it via the dispatcher's per-device rt-arg mechanism. If kv_cache's
program factory doesn't have direct access today, plumb it the same way AG
does (via `CreateBuffer` + `SetRuntimeArgs` per-device).

**Spike before committing to the design:** verify mesh-coord access works in
the kv_cache program factory pattern.

### Unit tests (`tests/ttnn/unit_tests/operations/test_fill_cache.py`)

New test class / function. Cover at minimum:

1. **`test_kv_actual_global_chunk_aligned_matches_uniform`** — call once with
   `update_idx=N*chunk_global/sp_factor` and once with
   `kv_actual_global=N*chunk_global`; gather cache to host; assert
   bit-identical. Proves degenerate case is back-compat.
2. **`test_kv_actual_global_pad_chip`** — pre-seed cache with sentinel
   pattern; call with non-chunk-aligned `kv_actual_global` so the pad chip
   gets a mid-slab offset; gather cache, verify per-chip writes landed at
   expected rows (using a Python helper that mirrors the kernel math).
3. **`test_kv_actual_global_multi_chip_pad`** — `kv_actual_global` such that
   pad spans ≥ 2 chips; verify each chip's write offset.
4. **`test_kv_actual_global_cold_start`** — `kv_actual_global=0`; verify
   every chip writes at `update_idxt=0` (all-pad-fill case).
5. **`test_kv_actual_global_no_old_pad`** — `kv_actual_global` exactly on a
   chunk boundary mid-cache; verify every chip writes at NEW-slab start.
6. **`test_kv_actual_global_rejects_invalid`** — combinations that should
   TT_FATAL: non-zero `update_idx` + `kv_actual_global`, missing
   `cluster_axis`, non-tile-aligned `kv_actual_global`, cache too small.
7. **`test_kv_actual_global_multi_user`** — multi-`batch_index` calls, each
   with different `kv_actual_global`; verify isolation between user slots.

Run on `(2, 4)` mesh (the validated 8-chip layout); skip `(2, 2)` per the
fabric-init quirk on the current LB box.

### Exit criteria for Branch A

- All 7 unit tests pass on 2x4 mesh.
- Degenerate-case bit-equivalence verified.
- No regression in existing `update_cache` tests.

---

## Phase 2 — MLA forward integration (on Branch B, after cherry-pick of Branch A)

### Cherry-pick

After Branch A passes its exit criteria:

```bash
git checkout ipotkonjak/chunked_attn_mla_rotation
git cherry-pick <branch-A commits, in order>
```

Run the unit tests once more to confirm the cherry-pick is clean.

### File

`models/demos/deepseek_v3_d_p/tt/mla/mla.py`

### Signature change

```python
def forward(
    self,
    hidden_states,
    rope_tensors,
    kvpe_cache,
    cache_layer_idx=0,
    on_layer_complete=None,
    actual_isl=None,
    chunk_start_global=None,
    cache_user_id=0,
    num_cache_layers=None,
    chunked_q_chunk_size=32,
    chunked_k_chunk_size=32,
    kv_actual_isl: Optional[int] = None,   # NEW
) -> ttnn.Tensor:
```

### Branch logic in the chunked path (replaces `mla.py:706+`)

```python
if kv_actual_isl is None:
    # Existing chunked-natural path; unchanged.
    local_offset = chunk_start_global // self.sp_factor
    ttnn.kv_cache.fill_cache_for_user_(
        kvpe_cache, tt_kvpe, cache_batch_idx, update_idx=local_offset,
    )
    sdpa_kwargs = dict(logical_n=chunk_end_global)
else:
    # Rotated path: per-chip offsets via kv_actual_global.
    new_actual_isl = hidden_states.shape[2] * self.sp_factor  # full chunk size
    ttnn.kv_cache.fill_cache_for_user_(
        kvpe_cache, tt_kvpe, cache_batch_idx,
        kv_actual_global=kv_actual_isl,
        cluster_axis=self.sp_axis,
    )
    sdpa_kwargs = dict(
        logical_n=kv_actual_isl + new_actual_isl,
        kv_actual_isl=kv_actual_isl,
    )

attn_out, _, _ = ttnn.transformer.ring_joint_scaled_dot_product_attention(
    tt_q, kvpe_cache, self.tt_v_latent_null,
    self.joint_q, self.joint_kv, self.joint_v_lora,
    persistent_output_buffer_k=self.chunked_persistent_k_buf,
    persistent_output_buffer_v=self.chunked_persistent_v_buf,
    joint_strategy="rear",
    program_config=ttnn.SDPAProgramConfig(...),
    compute_kernel_config=self.default_compute_kernel_config,
    dim=2,
    multi_device_global_semaphore=self.tt_ccl.ring_attention_ccl_semaphore_handles,
    num_links=self.ccl_num_links,
    cluster_axis=self.sp_axis,
    mesh_device=self.mesh_device,
    topology=self.ccl_topology,
    subdevice_id=self.tt_ccl.worker_sub_device_id,
    ccl_core_grid_offset=self.tt_ccl.ring_attention_ccl_core_grid_offset,
    use_column_major_ccl=True,
    is_causal=True,
    scale=self.scale,
    is_balanced=self.is_balanced,
    cache_batch_idx=cache_batch_idx,
    **sdpa_kwargs,
)
```

### Validation asserts (when `kv_actual_isl is not None`)

- `seq_len_local == hidden_states.shape[2]`. Hidden states are already a single
  chunk per call.
- `kvpe_cache.shape[2] == 2 * seq_len_local`. Cache is sized to exactly 2 slabs
  per chip (the rotation op's constraint).
- `kv_actual_isl + seq_len_local * sp_factor` ≤ cache capacity globally.
- `kv_actual_isl % (ttnn.TILE_SIZE * sp_factor) == 0`.
- `self.is_balanced is False`.

### Open verification (pre-coding spike)

The rotation op's input-shape check evaluates K's per-chip seq. When we pass
the full `kvpe_cache` (shape `[num_users * num_layers, 1, 2*chunk_local, kvpe_dim]`
per chip) plus `cache_batch_idx`, the op needs to interpret `N_local_kv` as
`2*chunk_local` (per-chip seq dim of the cache), not as the batch dim.

**Spike before Phase 2 coding:** re-run `test_kv_pad_aware_rotation_ttnn` with
the K input shaped as `[num_users * num_layers, 1, 2*chunk_local, kvpe]` and
`cache_batch_idx=0`, verify pass. This validates the op interaction before we
wire MLA.

---

## Phase 3 — `test_mla_chunked_prefill_rotated` (on Branch B)

### File

`models/demos/deepseek_v3_d_p/tests/test_mla.py` — new test function alongside
`test_mla_chunked_prefill`.

### Scope (10K test, 2 chunks of 5K)

```python
@pytest.mark.parametrize("mesh_device", [(2, 4)], ids=["2x4"], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}],
    ids=["line"],
    indirect=True,
)
@pytest.mark.parametrize(
    "iters",
    [
        # (kv_actual_isl_before_iter, new_actual_isl_this_iter)
        [(0, 2500), (2500, 5000)],   # iter 0 partial; iter 1 rotation
        [(0, 5000), (5000, 5000)],   # both chunk-aligned (rotation degenerates)
        [(0, 1024), (1024, 5000)],   # different partial ratio
    ],
    ids=["iter0_partial_2500", "both_aligned", "iter0_partial_1024"],
)
@pytest.mark.timeout(0)
def test_mla_chunked_prefill_rotated(mesh_device, device_params, iters):
    ...
```

### Setup

```python
seq_len = 10240            # = 2 chunks * 5120 tokens
sp_axis = 0
sp = mesh_device.shape[sp_axis]    # 2
chunk_size_global = seq_len // 2   # 5120
chunk_size_local = chunk_size_global // sp   # 2560

# Cache is exactly 2 slabs per chip — satisfies N_local_kv == 2 * N_local_q.
tt_kvpe_cache = init_kvpe_cache(
    kvpe_cache_head_dim=kvpe_dim,
    mesh_device=mesh_device,
    seq_len=seq_len,
    mesh_shape=mesh_shape,
    sp_axis=sp_axis,
    num_kvpe_cache_layers=1,
)
```

### Per-iter loop

```python
cumulative_valid_h = torch.empty(0)  # accumulating valid hidden_states for reference

for iter_idx, (kv_actual_isl, new_actual_isl) in enumerate(iters):
    # Generate this iter's hidden_states: first `new_actual_isl` rows valid,
    # rest is don't-care padding to fill chunk_size_global rows.
    chunk_h_valid = torch.randn(1, new_actual_isl, hidden_size).to(torch.bfloat16)
    chunk_h_padding = torch.zeros(1, chunk_size_global - new_actual_isl, hidden_size).to(torch.bfloat16)
    chunk_h = torch.cat([chunk_h_valid, chunk_h_padding], dim=1).unsqueeze(0)

    # Server-side rotation: reorder rows so each chip receives the new tokens
    # destined for its slab. Use _kv_pad_rotation_layout from test_ring_joint_sdpa_handoff.
    chunk_h_rotated = _server_rotate(chunk_h, kv_actual_isl, new_actual_isl, sp, chunk_size_local)

    tt_chunk_h = ttnn.from_torch(chunk_h_rotated, ...)
    rope_tensors = rope_setup.get_rope_tensors(chunk_size_global, start_pos=kv_actual_isl)

    tt_out = mla_tt.forward(
        hidden_states=tt_chunk_h,
        rope_tensors=rope_tensors,
        kvpe_cache=tt_kvpe_cache,
        kv_actual_isl=kv_actual_isl,
        chunked_q_chunk_size=q_chunk_size,
    )

    # Reference: append valid hidden_states, run full MLA up to (kv_actual_isl + new_actual_isl)
    cumulative_valid_h = torch.cat([cumulative_valid_h, chunk_h_valid.squeeze(0)], dim=0)
    ref_out_full = mla_ref(cumulative_valid_h, ...)
    ref_out_this_iter = ref_out_full[-new_actual_isl:]

    # TT output: first new_actual_isl rows in natural order (reverse the server rotation).
    tt_out_host = ttnn.to_torch(tt_out, ...)
    tt_out_natural = _server_unrotate(tt_out_host, kv_actual_isl, new_actual_isl, sp, chunk_size_local)
    tt_out_valid = tt_out_natural[:new_actual_isl]

    assert_with_pcc(ref_out_this_iter, tt_out_valid, 0.97)

# Final cache check
cache_host = kv_cache_to_host(tt_kvpe_cache, mesh_device, sp_axis)
expected_cache = _build_expected_rotated_cache(cumulative_valid_h, iters, sp, chunk_size_local)
assert_with_pcc(expected_cache, cache_host, 0.99)
```

### Helpers needed in this test file

- `_server_rotate(chunk_h, kv_actual_isl, new_actual_isl, sp, chunk_size_local)` —
  reorders the chunk-shaped hidden_states so each chip's slab matches the
  rotation layout. Uses `_kv_pad_rotation_layout` from the handoff test.
- `_server_unrotate(tt_out, ...)` — inverse.
- `_build_expected_rotated_cache(valid_h, iters, ...)` — replays the server
  rotation across iters; produces the cache as it should look after all
  writes. Used for the final cache PCC.

These can mostly be lifted/adapted from `test_kv_pad_aware_rotation_ttnn`'s
host-side rotation code.

---

## Phase 4 — Regression sweep

After Phases 2 and 3 land on Branch B:

```bash
# Existing handoff and rotation tests still pass.
pytest models/demos/deepseek_v3_d_p/tests/test_ring_joint_sdpa_handoff.py \
       -k "not 2x2"

# Existing chunked-natural MLA prefill still passes.
pytest models/demos/deepseek_v3_d_p/tests/test_mla.py::test_mla_chunked_prefill \
       -k "(seq10k or seq8k) and (N1 or N2)"

# New rotated MLA test.
pytest models/demos/deepseek_v3_d_p/tests/test_mla.py::test_mla_chunked_prefill_rotated

# fill_cache unit tests (cherry-picked from Branch A).
pytest tests/ttnn/unit_tests/operations/test_fill_cache.py -k kv_actual_global
```

Exit criteria for the whole plan: all of the above pass on the 2x4 mesh.

---

## Pre-coding spikes (before Phase 1 / Phase 2)

Two open verifications to retire risk before writing code:

1. **Mesh-coord access in `fill_cache` program factory** (gates Phase 1).
   Confirm the kv_cache device op can read its `sp_axis` coord as a per-device
   runtime arg, the same way AG does. If not, plan adjusts to plumb
   `cluster_axis` explicitly through additional rt-arg setup.

2. **Rotation op accepts full-cache K shape with `cache_batch_idx`**
   (gates Phase 2). Re-run `test_kv_pad_aware_rotation_ttnn` with K reshaped
   as `[num_users * num_layers, 1, 2*chunk_local, kvpe]` and `cache_batch_idx=0`.
   Validates that the op's `N_local_kv` resolves to the per-chip seq of the
   cache, not the batch dim, in the production MLA shape.

Both spikes are short (an afternoon each). Don't start Phase 1 / Phase 2 until
their corresponding spike passes.

---

## Files touched (summary)

### Branch A (`ipotkonjak/kv_cache_per_chip_offset`)

- `ttnn/cpp/ttnn/operations/kv_cache/kv_cache.{cpp,hpp}`
- `ttnn/cpp/ttnn/operations/kv_cache/kv_cache_nanobind.cpp`
- `ttnn/cpp/ttnn/operations/kv_cache/device/update_cache_device_operation.{cpp,hpp}`
- `ttnn/cpp/ttnn/operations/kv_cache/device/fill_cache_multi_core_program_factory.cpp`
- `tests/ttnn/unit_tests/operations/test_fill_cache.py` (new tests)

### Branch B (`ipotkonjak/chunked_attn_mla_rotation`)

- (cherry-picked from A) — all files above
- `models/demos/deepseek_v3_d_p/tt/mla/mla.py`
- `models/demos/deepseek_v3_d_p/tests/test_mla.py`

---

## Why this branching strategy

- **Op work is independently reviewable.** The kv_cache extension is a
  general-purpose feature (per-chip write offset from a global token count);
  reviewers can evaluate it without needing to understand MLA rotation.
- **MLA wiring is small and depends entirely on the op.** Keeping it on a
  separate branch lets us iterate the MLA test scenarios without churning
  the op PR.
- **Cherry-pick after detailed testing** ensures only validated op commits
  enter the MLA branch — no half-baked op behavior contaminating MLA tests.

---

## Implementation status (Phase 1) — what's actually on disk

### Branch
`ipotkonjak/kv_cache_per_chip_offset` off `main` (`8635db0bafe`). All work
listed below is uncommitted in the working tree — commit when perf sign-off
on another machine confirms no regression vs legacy `fill_cache_for_user_`.

### New op files

```
ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/update_padded_kv_cache/
├── CMakeLists.txt
├── sources.cmake
├── update_padded_kv_cache.hpp / .cpp                # top-level functor
├── update_padded_kv_cache_nanobind.hpp / .cpp       # python binding
└── device/
    └── update_padded_kv_cache_device_operation.hpp / .cpp
                                                    # device op + create_descriptor
```

Wired into:
- `ttnn/CMakeLists.txt` — link alias `TTNN::Ops::Experimental::DeepSeekPrefill::UpdatePaddedKvCache` + `add_subdirectory(...)`.
- `ttnn/sources.cmake` — adds the nanobind cpp to the `_ttnncpp` target.
- `ttnn/cpp/ttnn/operations/experimental/experimental_nanobind.cpp` — `#include` of nanobind hpp and `bind_update_padded_kv_cache(mod)` call.

### How the op works

- Per-chip `update_idxt` computed inside `create_descriptor` from
  `(kv_actual_global, my_sp_coord, sp_factor, chunk_local_tokens)`. Function
  `update_idxt_for_chip` in `device/update_padded_kv_cache_device_operation.cpp:34`.
- `my_sp_coord` derived via `ccl::get_linearized_index_from_physical_coord(cache_tensor, coord, cluster_axis)`. `sp_factor` derived by counting unique values along `cluster_axis` among `cache.device_storage().get_coords()`.
- Per-core `cache_start_id` baked into writer rt-args. No `mesh_device*` held in attributes; the descriptor pattern provides `mesh_dispatch_coordinate` directly.
- Reuses existing kernels (no fork):
  - `ttnn/cpp/ttnn/operations/kv_cache/device/kernels/dataflow/reader_fill_cache_interleaved_start_id.cpp`
  - `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp`

### Test file

`tests/ttnn/unit_tests/operations/deepseek/test_deepseek_prefill_update_padded_kv_cache.py`

Contains:
- `test_update_padded_kv_cache_idxt_math` (5 unit cases) — direct math check.
- `test_update_padded_kv_cache_torch_showcase` (10 scenarios) — pure-torch spec.
- `test_update_padded_kv_cache_ttnn` (8 device scenarios on 2x4) — end-to-end.
- `test_perf_update_padded_kv_cache` (1 op call) — for tracy capture.
- `test_perf_fill_cache_for_user_baseline` (1 op call) — for tracy capture.

Results so far on bh-lb-10 (8× p150b BH LoudBox):
- Math + torch + device: **23/23 passing**. Total runtime ~8s.

---

## Phase 1.5 — Tracy perf comparison (pending re-run)

### Goal
Confirm the new op's `DEVICE KERNEL DURATION [ns]` is not regressed vs the
legacy `ttnn.kv_cache.fill_cache_for_user_` op, at MLA-realistic shapes.

### Test shapes (both perf tests)

- 2x4 mesh, `sp_axis=0` (sp=2).
- Per-chip input: `[1, 1, chunk_local=2560, kvpe_dim=576]` bfloat16.
- Per-chip cache: `[1, 1, 2*chunk_local=5120, kvpe_dim=576]` bfloat16.
- Both tests call their op exactly once and `ttnn.synchronize_device`.

### Status on bh-lb-10
Tracy ran end-to-end but the device kernel duration columns in
`ops_perf_results_*.csv` came back with absolute-timestamp-looking values
(~3.24 × 10¹² ns ≈ ~54 min, clearly wrong) and empty `PER CORE MIN/MAX/AVG`
fields. Suspected tracy/profiler issue specific to this box — **re-run
on a different machine**.

### How to re-run on another machine

From repo root, on this branch (`ipotkonjak/kv_cache_per_chip_offset`):

```bash
# 1. Confirm everything builds.
./build_metal.sh

# 2. Smoke-test the perf tests run cleanly (no tracy yet).
source python_env/bin/activate
python -m pytest \
  tests/ttnn/unit_tests/operations/deepseek/test_deepseek_prefill_update_padded_kv_cache.py::test_perf_update_padded_kv_cache \
  tests/ttnn/unit_tests/operations/deepseek/test_deepseek_prefill_update_padded_kv_cache.py::test_perf_fill_cache_for_user_baseline

# 3. Profile the new op.
python tools/tracy/profile_this.py \
  -c "pytest tests/ttnn/unit_tests/operations/deepseek/test_deepseek_prefill_update_padded_kv_cache.py::test_perf_update_padded_kv_cache" \
  -n new_op

# 4. Profile the baseline op.
python tools/tracy/profile_this.py \
  -c "pytest tests/ttnn/unit_tests/operations/deepseek/test_deepseek_prefill_update_padded_kv_cache.py::test_perf_fill_cache_for_user_baseline" \
  -n baseline_op
```

Each run produces a CSV at
`generated/profiler/reports/<name>/<timestamp>/ops_perf_results_<name>_<timestamp>.csv`.

### How to read the results

In each CSV, filter `OP CODE` for the relevant op:
- New: `UpdatePaddedKvCacheDeviceOperation`
- Baseline: `FillCacheMultiCoreProgramFactory` (or whichever op-code label
  the legacy fill produces — the row is identifiable by being the only
  kv-cache fill in the trace).

Compare the `DEVICE KERNEL DURATION [ns]` column across the per-device rows
(one row per chip; 8 rows on 2x4). Take the mean per op.

### Decision rule

- **New ≤ 1.1 × baseline**: green-light Phase 2.
- **New > 1.1 × baseline**: investigate. Likely causes:
  - Per-chip math overhead in the host (not the kernel; should be a wash).
  - Different work-split path. Compare `num_blocks_per_core_*` between the
    two factories — they should be identical at these shapes.
  - Verify the writer kernel path is the same (`writer_unary_interleaved_start_id.cpp`)
    and the reader path matches too.

### Note on the baseline op signature

On this branch (`main`-based), `ttnn.kv_cache.fill_cache_for_user_` takes
**3 positional args** (cache, input, batch_index) and writes at offset 0 —
no `update_idx` kwarg. The user's `chunked_attn_mla` branch adds the
`update_idx` parameter; for the perf comparison we only need workload
equivalence, not destination equivalence.

---

## What to do after perf sign-off

1. Commit the Phase 1 work on `ipotkonjak/kv_cache_per_chip_offset`.
2. Switch to `ipotkonjak/chunked_attn_mla_rotation` (create from
   `chunked_attn_mla` + merge `chunked_attn_tests`; see Phase 0 section above).
3. Cherry-pick the Phase 1 commit(s).
4. Proceed with Phase 2 (`MLA.forward` wiring) and Phase 3 (rotated MLA test)
   as described above.
