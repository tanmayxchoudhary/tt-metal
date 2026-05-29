The pipeline:
```
User request (vLLM SamplingParams)
    │
    ▼
tt_model_runner._prepare_model_inputs()    ← converts to TTSamplingParams
    │
    ├── decides: device sampling or host sampling?
    │
    ▼
model.decode_forward(tokens, ..., sampling_params, prompt_tokens, output_tokens)
    │                                                     │
    │  (inside tt-metal model)                            │
    ▼                                                     ▼
SamplingGenerator.reset_sampling_params()    TTPenalties.reset_prompt_tokens()
SamplingGenerator.reset_output_state()       TTPenalties.reset_output_tokens()
    │
    ▼
logits = transformer(tokens)
    │
    ▼
SamplingGenerator.sample(logits)
    ├── TTPenalties.apply(logits)          ← presence/frequency/repetition
    ├── TTSampling(logits)                 ← top-k, top-p, temperature, multinomial
    └── TTPenalties.update_output_tokens() ← track newly sampled token
    │
    ▼
sampled token IDs returned to vLLM
    │
    ▼
tt_model_runner → ModelRunnerOutput → engine

```

---

# new_plan.md
# Plan: Split Sampling into Penalties1D + Sampling1D

## Context

TTTv1's `SamplingGenerator` bundles penalties and sampling into one orchestrator. After design review, we split into **two independent TTTv2 modules**: `Penalties1D` (logit transform) and `Sampling1D` (token selection). They have zero direct coupling — penalties produce logits, sampling consumes logits. This enables independent composition, testing, and reuse per TTTv2 principles.

## Files to Create

```
models/common/modules/
├── lazy_buffer.py        # LazyBuffer — lazy device buffer allocation (mutable sibling of LazyWeight)
models/common/modules/sampling/
├── penalties_1d.py       # Penalties1D — presence/frequency/repetition penalty transforms
├── sampling_1d.py        # Sampling1D — top-k/top-p/temperature + argmax fast path
models/common/tests/modules/sampling/
├── test_penalties_1d.py  # Penalty tests (penalty math, lifecycle, composition)
├── test_sampling_1d.py   # Sampling tests (top-k correctness, argmax, multi-device)
```

No `__init__.py` or `conftest.py` (project convention).

## Reference Files

| Purpose | Path |
|---------|------|
| TTTv2 module pattern | `models/common/modules/mlp/mlp_1d.py` |
| TTTv2 test pattern | `models/common/tests/modules/mlp/test_mlp_1d.py` |
| TTTv1 penalties source | `models/common/sampling/tt_penalties.py` |
| TTTv1 sampling source | `models/common/sampling/tt_sampling.py` |
| TTTv1 orchestrator | `models/common/sampling/generator.py` |
| Base class | `models/common/lightweightmodule.py` |
| LazyWeight (sibling pattern) | `models/common/modules/lazy_weight.py` |
| CCL utilities | `models/common/modules/tt_ccl.py` |
| LogProbsCalculator | `models/common/utils.py` |
| Test fixture | `models/common/tests/conftest.py` (`ttnn_mesh_device`) |

---

## Step 0: Create `lazy_buffer.py`

### Design rationale

TTTv1's `TTPenalties.__init__` (lines 104-115) allocates persistent device buffers via `_alloc_int_buffer()` / `_alloc_bf16_buffer()`, which call `ttnn.from_torch()` with a host source tensor (typically `torch.zeros(...)` or `torch.ones(...)`), dtype, layout, mesh_mapper, and memory_config. This is **exactly the same call pattern as `LazyWeight.get_device_weight()`**. The only thing that makes these buffers different from weights is **post-allocation mutability** (the device data is overwritten in-place via `output_tensor=` during decode) and **no disk caching** (caching a mutable buffer would corrupt state across instances).

`LazyBuffer` captures this by reusing LazyWeight's allocation contract — same fields, same `from_torch()` path — while explicitly omitting the caching/fingerprinting layer. This makes buffer allocation specs declarative, inspectable, and overridable by power users via the config dataclass, following the same "None means auto-resolve" pattern as MLP1DConfig's LazyWeight fields.

### LazyBuffer dataclass

```python
@dataclass
class LazyBuffer:
    """
    Lazy-allocated device buffer for mutable state tensors.

    Mirrors LazyWeight's allocation contract (source + from_torch() parameters) but is
    designed for buffers that are mutated in-place after allocation, NOT immutable model weights.

    Key differences from LazyWeight:
    - No disk caching: The device data is overwritten in-place via output_tensor= during
      decode (e.g., penalty masks, token counts). Caching a mutable buffer would cause
      state corruption if loaded by another instance.
    - No fingerprinting: Without caching, there is no cache to invalidate.
    - _value caching is safe: The ttnn.Tensor *handle* returned by get_device_buffer()
      never changes — only the on-device data changes via output_tensor= writes.
      So "allocate once, return same handle" is correct for mutable buffers.

    The only thing that makes these different from weights is post-allocation mutability
    and no disk caching. If a buffer becomes read-only in a future refactor, it can be
    promoted to a LazyWeight with caching enabled.

    See also: LazyWeight in models/common/modules/lazy_weight.py
    """

    # Source: initial host tensor values (e.g., torch.zeros, torch.ones).
    # Duck-typed — string annotation avoids torch import at module level.
    source: "torch.Tensor"

    # from_torch() parameters — same fields as LazyWeight (minus cache_dir_weight_name, pad_value)
    dtype: ttnn.DataType | None = ttnn.int32
    layout: ttnn.Layout | None = ttnn.TILE_LAYOUT
    device: ttnn.MeshDevice | None = None
    mesh_mapper_config: ttnn.MeshMapperConfig | None = None
    memory_config: ttnn.MemoryConfig | None = None

    # Cached device tensor handle (allocated once, device data mutated in-place)
    _value: ttnn.Tensor | None = field(default=None, repr=False)

    def _build_mesh_mapper(self):
        """Build mesh mapper from config. Shared by get_device_buffer() and update()."""
        if self.mesh_mapper_config is not None:
            return ttnn.create_mesh_mapper(self.device, self.mesh_mapper_config)
        return ttnn.replicate_tensor_to_mesh_mapper(self.device)

    def _from_torch_args(self, source, *, device):
        """
        Build the full from_torch() kwargs. Used by both get_device_buffer() and update()
        to ensure the same dtype/layout/mesh_mapper/memory_config are used consistently.
        Only `device` differs: real device for allocation, None for host-side update.
        """
        return dict(
            dtype=self.dtype,
            layout=self.layout,
            device=device,
            mesh_mapper=self._build_mesh_mapper(),
            memory_config=self.memory_config,
        )

    def get_device_buffer(self) -> ttnn.Tensor:
        """Allocate on first call, return cached handle thereafter."""
        if self._value is not None:
            return self._value

        if self.device is None:
            raise ValueError("device must be set before materializing buffer")
        if self.layout is None:
            raise ValueError("layout must be set before materializing buffer")

        self._value = ttnn.from_torch(
            self.source, **self._from_torch_args(self.source, device=self.device),
        )
        return self._value

    def update(self, new_source: "torch.Tensor") -> None:
        """
        Overwrite the device buffer contents with a new source tensor, without reallocating.

        If the buffer has not yet been materialized (get_device_buffer not called), this
        simply replaces self.source for future materialization.

        If the buffer IS already materialized, this performs an in-place device update
        using the SAME from_torch() args as the original allocation (dtype, layout,
        mesh_mapper, memory_config) but with device=None to create a host tensor:
            host_tt = ttnn.from_torch(new_source, **same_args, device=None)
            ttnn.copy_host_to_device_tensor(host_tt, self._value)

        The ttnn.Tensor handle (self._value) is preserved — no DRAM reallocation.

        This encapsulates the pattern seen in:
        - TTPenalties._copy_host_to_device (tt_penalties.py:157-159)
        - SeedManager.get_new_values (generator.py:382-383)
        """
        self.source = new_source
        if self._value is not None:
            host_tt = ttnn.from_torch(
                new_source, **self._from_torch_args(new_source, device=None),
            )
            ttnn.copy_host_to_device_tensor(host_tt, self._value)

    def is_resolved(self) -> bool:
        """Check if all required fields for materialization are set."""
        return self.device is not None and self.dtype is not None and self.layout is not None
```

### resolve_lazy_buffer() helper

Mirrors `resolve_lazy_weight()` from `lazy_weight.py:346-349`:

```python
def resolve_lazy_buffer(buf: LazyBuffer, **kwargs) -> LazyBuffer:
    """Resolve None fields of `buf` with the given kwargs; do not override non-None fields."""
    to_set = {k: v for k, v in kwargs.items() if getattr(buf, k, None) is None}
    return replace(buf, **to_set)
```

### Unit tests for LazyBuffer

Add to `models/common/tests/modules/sampling/test_penalties_1d.py` (or standalone `test_lazy_buffer.py`):

1. `test_lazy_buffer_defaults` — dtype=int32, layout=TILE, device/mesh_mapper_config/memory_config=None
2. `test_lazy_buffer_is_resolved` — True only when device+dtype+layout are set
3. `test_lazy_buffer_get_device_buffer_allocates` — first call returns ttnn.Tensor, second call returns same handle
4. `test_lazy_buffer_raises_without_device` — ValueError if device is None
5. `test_resolve_lazy_buffer` — fills None fields, preserves non-None
6. `test_lazy_buffer_update_before_materialize` — update() replaces source, get_device_buffer() uses new source
7. `test_lazy_buffer_update_after_materialize` — update() refreshes device data, handle identity unchanged (same `id()`), readback matches new source
8. `test_lazy_buffer_update_preserves_handle` — after update(), the tensor returned by get_device_buffer() is the same Python object as before

---

## Step 1: Create `penalties_1d.py`

### Penalties1DConfig dataclass

```python
@dataclass
class Penalties1DConfig:
    vocab_size: int                                    # Required. Caller pre-pads to be divisible by num_devices.
    mesh_device: ttnn.MeshDevice | None = None         # None → GetDefaultDevice()
    max_batch_size: int = 32                           # TTTv1 hardcode (tt_penalties.py:87)
    sub_core_grids: ttnn.CoreRangeSet | None = None    # From args.sub_core_grids (line 94)

    # --- Persistent buffer specs (LazyBuffer | ttnn.Tensor | None) ---
    # None = auto-filled by _resolve_penalties1d_config() with topology-aware defaults.
    # LazyBuffer = declarative spec, materialized lazily in load_device_buffers().
    # ttnn.Tensor = pre-allocated device tensor, used directly (power user bypass).
    #
    # Sharded vocab buffers: [max_batch_size, vocab_size], int32, TILE, sharded across devices
    prompt_mask: LazyBuffer | ttnn.Tensor | None = None
    output_mask: LazyBuffer | ttnn.Tensor | None = None
    output_counts: LazyBuffer | ttnn.Tensor | None = None
    # Replicated vocab buffers: [max_batch_size, vocab_size], int32, replicated
    output_counts_gathered: LazyBuffer | ttnn.Tensor | None = None
    zeros: LazyBuffer | ttnn.Tensor | None = None
    # Utility buffers
    decode_src: LazyBuffer | ttnn.Tensor | None = None  # [max_batch_size, 1], int32, ROW_MAJOR, ones
    # BF16 penalty param buffers: [max_batch_size, 1], bfloat16, TILE, replicated
    presence_penalties: LazyBuffer | ttnn.Tensor | None = None
    frequency_penalties: LazyBuffer | ttnn.Tensor | None = None
    repetition_penalties: LazyBuffer | ttnn.Tensor | None = None
    inverse_repetition_penalties: LazyBuffer | ttnn.Tensor | None = None

    @staticmethod
    def _buf_resolved(buf) -> bool:
        """A buffer field is resolved if it's a ttnn.Tensor (already on device) or a resolved LazyBuffer."""
        if buf is None:
            return False
        if isinstance(buf, ttnn.Tensor):
            return True  # already materialized — power user path
        return buf.is_resolved()  # LazyBuffer path

    def is_resolved(self) -> bool:
        return (
            self.mesh_device is not None
            and all(self._buf_resolved(getattr(self, f)) for f in (
                "prompt_mask", "output_mask", "output_counts",
                "output_counts_gathered", "zeros", "decode_src",
                "presence_penalties", "frequency_penalties",
                "repetition_penalties", "inverse_repetition_penalties",
            ))
        )
```

NOT config fields (derived internally): `cluster_shape` (from `mesh_device.shape`), `num_devices` (from `max(cluster_shape)`), `shard_dims`/`shard_dims_slice` (from cluster shape comparison at lines 98-103).

### _resolve_penalties1d_config() function

Mirrors `_resolve_mlp1d_config()` pattern: fills None fields with topology-aware defaults, returns `replace(config, **to_set)`, asserts `is_resolved()`.

```python
def _resolve_penalties1d_config(config: Penalties1DConfig) -> Penalties1DConfig:
    """
    Fill None fields in config with topology-aware defaults.
    Power users who set fields explicitly will NOT have them overwritten.
    """
    import torch  # lazy import — only needed for source tensor construction

    to_set: dict = {}

    # Phase 1: Device
    mesh_device = config.mesh_device or ttnn.GetDefaultDevice()
    to_set["mesh_device"] = mesh_device

    # Phase 2: Topology → shard_dims (port from tt_penalties.py:97-103)
    cluster_shape = mesh_device.shape
    num_devices = max(cluster_shape[-1], cluster_shape[-2])
    if cluster_shape[-1] == num_devices:
        shard_dims = (None, 1)     # shard vocab across columns
        shard_dims_slice = (None, 0)
    else:
        shard_dims = (1, None)     # shard vocab across rows
        shard_dims_slice = (0, None)

    B = config.max_batch_size
    V = config.vocab_size

    # Build mesh mapper configs
    shard_mapper = ttnn.MeshMapperConfig(shard_dims=shard_dims, mesh_shape=cluster_shape)
    replicate_mapper = None  # None → replicate_tensor_to_mesh_mapper in LazyBuffer

    # Helper: resolve a single buffer field.
    # - None → create LazyBuffer with defaults
    # - LazyBuffer → fill None fields with defaults via resolve_lazy_buffer()
    # - ttnn.Tensor → pass through unchanged (power user pre-allocated)
    def _resolve_buf(field_val, defaults, source_factory):
        if field_val is None:
            return LazyBuffer(source=source_factory(), **defaults)
        if isinstance(field_val, ttnn.Tensor):
            return field_val  # already on device — no resolution needed
        return resolve_lazy_buffer(field_val, **defaults)

    # Phase 3: Sharded vocab buffers — [B, V], int32, TILE, sharded
    sharded_vocab_defaults = dict(
        dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=mesh_device,
        mesh_mapper_config=shard_mapper, memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    zeros_BV = lambda: torch.zeros(B, V, dtype=torch.int32)
    to_set["prompt_mask"] = _resolve_buf(config.prompt_mask, sharded_vocab_defaults, zeros_BV)
    to_set["output_mask"] = _resolve_buf(config.output_mask, sharded_vocab_defaults, zeros_BV)
    to_set["output_counts"] = _resolve_buf(config.output_counts, sharded_vocab_defaults, zeros_BV)

    # Phase 4: Replicated vocab buffers — [B, V], int32, replicated
    replicated_vocab_defaults = dict(
        dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=mesh_device,
        mesh_mapper_config=replicate_mapper, memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    to_set["output_counts_gathered"] = _resolve_buf(
        config.output_counts_gathered, replicated_vocab_defaults, zeros_BV,
    )

    replicated_vocab_rm_defaults = dict(
        dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device,
        mesh_mapper_config=replicate_mapper, memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    to_set["zeros"] = _resolve_buf(config.zeros, replicated_vocab_rm_defaults, zeros_BV)

    # Phase 5: Utility buffers
    decode_src_defaults = dict(
        dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device,
        mesh_mapper_config=replicate_mapper, memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    to_set["decode_src"] = _resolve_buf(
        config.decode_src, decode_src_defaults, lambda: torch.ones(B, 1, dtype=torch.int32),
    )

    # Phase 6: BF16 penalty param buffers — [B, 1], bfloat16, TILE, replicated
    bf16_param_defaults = dict(
        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device,
        mesh_mapper_config=replicate_mapper, memory_config=None,  # bf16 params use default (no explicit DRAM)
    )
    zeros_B1 = lambda: torch.zeros(B, 1, dtype=torch.float32)
    for field_name in ("presence_penalties", "frequency_penalties",
                       "repetition_penalties", "inverse_repetition_penalties"):
        to_set[field_name] = _resolve_buf(getattr(config, field_name), bf16_param_defaults, zeros_B1)

    resolved = replace(config, **to_set)
    assert resolved.is_resolved(), f"Config not fully resolved after _resolve_penalties1d_config"
    return resolved
```

**Phase summary**:

| Phase | What fills | Derived from |
|---|---|---|
| 1 | `mesh_device` | `ttnn.GetDefaultDevice()` fallback |
| 2 | `shard_dims`, `shard_dims_slice` | `mesh_device.shape` topology (port of `tt_penalties.py:97-103`) |
| 3 | `prompt_mask`, `output_mask`, `output_counts` | Sharded vocab LazyBuffers |
| 4 | `output_counts_gathered`, `zeros` | Replicated vocab LazyBuffers |
| 5 | `decode_src` | Utility buffer (ones, ROW_MAJOR) |
| 6 | `presence_penalties`, `frequency_penalties`, `repetition_penalties`, `inverse_repetition_penalties` | BF16 param LazyBuffers |

Note: `shard_dims_slice` and `slice_start`/`slice_end` tensors are NOT in config — they are derived in `load_device_buffers()` since they depend on `num_devices` and `vocab_size` (same as TTTv1 lines 117-139).

### Penalty dataclasses: PenaltyParams + PenaltyAccumulator

Ported from `tt_penalties.py:19-29`, split into two dataclasses by update frequency. Both are caller-managed — NOT owned by the module.

**PenaltyParams** — set once per request, read-only during decode loop. Caller-constructed — no module convenience methods:

```python
@dataclass
class PenaltyParams:
    prompt_mask: ttnn.Tensor                   # [max_batch_size, vocab_per_device], int32, sharded
    presence_penalties: ttnn.Tensor             # [max_batch_size, 1], bfloat16
    frequency_penalties: ttnn.Tensor            # [max_batch_size, 1], bfloat16
    repetition_penalties: ttnn.Tensor           # [max_batch_size, 1], bfloat16
    inverse_repetition_penalties: ttnn.Tensor   # [max_batch_size, 1], bfloat16 (precomputed 1/rep)
```

Construction: caller allocates tensors directly (e.g., via LazyBuffer or their own `ttnn.from_torch()`) and passes them to `PenaltyParams(...)`. The module does NOT provide a `create_penalty_params()` convenience method — all tensors are caller-owned.

**PenaltyAccumulator** — mutated every decode step. Caller-constructed:

```python
@dataclass
class PenaltyAccumulator:
    output_mask: ttnn.Tensor                   # [max_batch_size, vocab_per_device], int32, sharded
    output_counts: ttnn.Tensor                 # [max_batch_size, vocab_per_device], int32, sharded
    output_counts_gathered: ttnn.Tensor        # [max_batch_size, vocab_size], int32, replicated
```

Construction: same pattern — caller allocates and constructs directly. Reset/update are done via `decode_forward` (which applies penalties) and `update_output_tokens` (which accumulates token history).

Key change from TTTv1: `sub_core_grids` removed from both — it's a module property from config, not per-context state. No convenience creation methods — the module is a pure compute pipeline that receives external state.

### Buffer ownership boundary

All penalty buffers are persistent device tensors allocated once and mutated in-place via `output_tensor=` to avoid per-step device memory allocation:

| Update Frequency | Tensors | TTTv2 Owner | Rationale |
|---|---|---|---|
| Every decode step | `output_mask`, `output_counts`, `output_counts_gathered` | `PenaltyAccumulator` | Token history accumulation, mutated between trace executions |
| Once per request | `prompt_mask` | `PenaltyParams` | Set from prompt tokens before decode loop |
| Once per request | `*_penalties`, `inverse_repetition_penalties` | `PenaltyParams` | Set from sampling params before decode loop (precomputed `1/rep` avoids division in hot path) |
| Once at init | `decode_src`, `zeros`, `slice_start`, `slice_end` | Module (`self._*`) | Fixed for model lifetime, depends only on `vocab_size` + `mesh_device` |
| Once at init | `sub_core_grids` | Module (`self.config`) | Model-level config |

**Invariants**:
- `PenaltyParams` holds per-request constants. Set before the decode loop, read-only during it.
- `PenaltyAccumulator` holds per-step accumulators. Mutated by `update_output_tokens()` after each sampled token.
- The module holds everything fixed for the model lifetime.
- All device buffers are allocated once and never re-allocated. Lifecycle methods mutate them in-place.

### Penalties1D class

```python
class Penalties1D(LightweightModule):

    # Construction
    def __init__(self, vocab_size: int, mesh_device: ttnn.MeshDevice | None = None):
    @classmethod
    def from_config(cls, config: Penalties1DConfig) -> "Penalties1D":
    @classmethod
    def from_model_args(cls, mesh_device, args) -> "Penalties1D":  # Rejects Galaxy

    # Device buffers (idempotent, like MLP1D.load_device_weights)
    def load_device_buffers(self):
        # Materializes module-owned buffers only (decode_src, zeros, slice_start, slice_end).
        # PenaltyParams and PenaltyAccumulator tensors are caller-owned — NOT materialized here.
        # Derives: _num_devices, _cluster_shape, _shard_dims_slice, _op_kwargs
        #
        # Example:
        #   self._decode_src = self._materialize(self.config.decode_src)
        #   self._zeros = self._materialize(self.config.zeros)
        #   self._slice_start, self._slice_end = self._build_slice_tensors()

    # Forward — params and accum are caller-constructed, passed as args
    def prefill_forward(self, logits: ttnn.Tensor, params: PenaltyParams,
                        accum: PenaltyAccumulator, prompt_tokens: "torch.Tensor") -> ttnn.Tensor:
        # Sets params.prompt_mask via scatter_add from prompt_tokens (port of init_prompt_penalties).
        # Returns logits UNCHANGED — penalties are not applied during prefill.

    def decode_forward(self, logits: ttnn.Tensor, params: PenaltyParams,
                       accum: PenaltyAccumulator) -> ttnn.Tensor:
        # Applies presence/frequency/repetition penalties to logits (port of apply_penalties).
        # Returns penalized logits.

    def update_output_tokens(self, accum: PenaltyAccumulator, new_tokens: ttnn.Tensor) -> None:
        # Updates accum.output_mask and accum.output_counts via scatter_add.
        # Called AFTER sampling, not inside decode_forward (sampling happens between).
        # Port of TTPenalties.update_output_tokens (tt_penalties.py:247-264).

    def reset_output_tokens(self, accum: PenaltyAccumulator,
                            tokens: "torch.Tensor | None" = None) -> None:
        # Zeros out accum buffers. Optionally re-initializes from provided tokens.
        # Port of TTPenalties.reset_output_tokens (tt_penalties.py:214-245).

    def forward(self, logits, params=None, accum=None) -> ttnn.Tensor:  # Dispatcher

    # Private
    # NOTE: _alloc_int_buffer / _alloc_bf16_buffer → LazyBuffer.get_device_buffer()
    # NOTE: _copy_host_to_device → LazyBuffer.update()
    # NOTE: REMOVED convenience methods (caller constructs PenaltyParams/PenaltyAccumulator directly):
    #   - create_penalty_params → caller allocates tensors via LazyBuffer or ttnn.from_torch
    #   - create_penalty_accumulator → same
    #   - update_penalty_values → caller uses LazyBuffer.update() on params fields
    #   - init_prompt_penalties → absorbed into prefill_forward
    def _materialize(self, buf):  # buf if isinstance(buf, ttnn.Tensor) else buf.get_device_buffer()
    def _build_slice_tensors(self):  # Derives slice_start/slice_end from vocab_size + num_devices (port lines 117-139)
    def _pad_batch_to_max(self, tokens_2d, pad_value) -> "torch.Tensor":
    def _token_bin_counts_and_mask(self, new_tokens, src, counts=None, mask=None, counts_sliced=None):
```

### Key porting decisions

1. **Two caller-constructed dataclasses as forward args**: TTTv1 stores all penalty state as module attributes. TTTv2 splits into `PenaltyParams` (per-request constants) and `PenaltyAccumulator` (per-step state), both passed as arguments to `prefill_forward` and `decode_forward`. No convenience creation methods — callers allocate tensors directly (via LazyBuffer or `ttnn.from_torch`) and construct the dataclasses themselves. This eliminates hidden state, enables multiple concurrent penalty contexts, and makes the module a pure compute pipeline.
2. **LazyBuffer for all persistent buffers**: TTTv1's `_alloc_int_buffer()` / `_alloc_bf16_buffer()` are replaced by `LazyBuffer | ttnn.Tensor | None` fields in `Penalties1DConfig`. Three paths: `None` → auto-resolve with topology-aware defaults; `LazyBuffer` → declarative spec with user overrides; `ttnn.Tensor` → pre-allocated device tensor used directly (power user bypass). `_resolve_penalties1d_config()` fills defaults; `load_device_buffers()` materializes via `_materialize()`.
3. **Penalty math** port from `apply_penalties()` at `tt_penalties.py:32-75`:
   - Presence: `logits -= typecast(output_mask, bf16) * presence` (lines 38-43)
   - Frequency: `logits -= typecast(output_counts, bf16) * frequency` (lines 46-51)
   - Repetition: sign-dependent scaling with `combined_mask = prompt + output` (lines 58-73)
4. **Shape handling**: Reshape to `(-1, original_shape[-1])` before penalties, reshape back after (lines 305-308).
5. **`torch` import is lazy** — only inside `_resolve_penalties1d_config()` and lifecycle methods that construct host tensors. `decode_forward()` is pure ttnn.

---

## Step 2: Create `test_penalties_1d.py`

### Unit tests (no device)
1. `test_config_defaults` — verify defaults (max_batch_size=32, mesh_device=None, sub_core_grids=None)
2. `test_penalty_params_fields` — verify 5 PenaltyParams fields; `test_penalty_accumulator_fields` — verify 3 PenaltyAccumulator fields
3. `test_resolve_param_host_*` — scalar, list, None, truncation cases
4. `test_pad_batch_to_max_*` — padding, truncation, invalid dims → ValueError

### Device tests (parametrized)

```python
@pytest.mark.parametrize("ttnn_mesh_device", [(1,1),(1,2),(1,8)], indirect=True)
@pytest.mark.parametrize("mesh_shape,vocab_size", [
    (1,1)/1024, (1,1)/32000, (1,2)/32000, (1,2)/128256, (1,8)/128256
])
```

5. `test_create_penalty_params` — all 5 tensors allocated with correct shapes; `test_create_penalty_accumulator` — all 3 tensors allocated
6. `test_presence_penalty_math` — known logits + output_mask → PCC vs torch reference
7. `test_frequency_penalty_math` — known logits + counts → PCC vs torch reference
8. `test_repetition_penalty_math` — positive/negative logits, sign-dependent scaling → PCC
9. `test_penalty_full_lifecycle` — construct PenaltyParams/PenaltyAccumulator → prefill_forward (sets prompt_mask) → decode_forward → update_output_tokens
10. `test_penalties_change_argmax` — heavy penalty changes argmax token
11. `test_decode_forward_none_context` — returns logits unchanged
12. `test_from_model_args` — backward compat
13. `test_rejects_galaxy` — ValueError

### PCC verification

Implement `reference_apply_penalties()` in pure torch:
```python
def reference_apply_penalties(logits, prompt_mask, output_mask, output_counts, presence, frequency, repetition):
    logits -= output_mask.float() * presence      # presence
    logits -= output_counts.float() * frequency    # frequency
    combined = ((prompt_mask + output_mask) > 0).float()  # repetition
    scale = torch.where(logits > 0, torch.where(combined.bool(), 1/rep, 1.0), torch.where(combined.bool(), rep, 1.0))
    logits *= scale
    return logits
```

---

## Step 3: Create `sampling_1d.py`

### Sampling1DConfig dataclass

```python
@dataclass
class Sampling1DConfig:
    vocab_size: int                                        # Required. Caller pre-pads.
    mesh_device: ttnn.MeshDevice | None = None             # None → GetDefaultDevice()
    tt_ccl: TT_CCL | None = None                           # None → get_tt_ccl(mesh_device) [if multi-device]
    max_batch_size: int = 32                               # TTTv1 hardcode (tt_sampling.py:95)
    max_top_k: int = 32                                    # TTTv1 default (tt_sampling.py:96)
    sub_core_grids: ttnn.CoreRangeSet | None = None        # From args.sub_core_grids (line 100)
    sub_core_grid_topk: ttnn.CoreRangeSet | None = None    # From args.sub_core_grid_topk (line 101)
    start_core: ttnn.CoreCoord | None = None               # None → CoreCoord(0,0) (line 102)
    num_gather_links: int = 1                              # From GALAXY_NUM_LINKS calc (lines 104-111)
    sampling_memory_config: ttnn.MemoryConfig | None = None # None → DRAM_MEMORY_CONFIG (lines 112-115)
    allow_force_argmax: bool = False                        # From SAMPLING_AG_CONFIG (lines 118-125)
    num_argmax_gather_links: int | None = None             # None → same as num_gather_links
    ag_topology: ttnn.Topology | None = None               # None → Topology.Linear

    # --- Persistent buffer specs (LazyBuffer | ttnn.Tensor | None) ---
    # Same triple-type pattern as Penalties1DConfig.
    # NOTE: k/p/temp are per-call forward args — NOT buffers. They were TTTv1 module state
    # (k_tensor, p_tensor, temp_tensor) but are eliminated in TTTv2.
    #
    # Static index buffers (computed from vocab_size + num_devices, never mutated)
    index_offsets: LazyBuffer | ttnn.Tensor | None = None    # [1,1,32,max_top_k*num_devices], int32, TILE
    local_indices: LazyBuffer | ttnn.Tensor | None = None    # [1,1,32,vocab_per_device], uint16, TILE
    # Seed/ID buffers (seeds mutable via LazyBuffer.update(), user_ids static)
    seeds: LazyBuffer | ttnn.Tensor | None = None            # [32], uint32, ROW_MAJOR
    user_ids: LazyBuffer | ttnn.Tensor | None = None         # [32], uint32, ROW_MAJOR

    def is_resolved(self) -> bool:
        if self.mesh_device is None:
            return False
        if self.mesh_device.get_num_devices() > 1 and self.tt_ccl is None:
            return False
        return True
```

NOT config fields (derived): `cluster_shape`, `multi_step_reduction` (True only for [1,1]), `sampling_all_gather_axis` (dropped — only for 2D/Galaxy, rejected for 1D).

### _resolve_sampling1d_config() function

Fills scalar config fields: `mesh_device` → GetDefaultDevice(), `tt_ccl` → `get_tt_ccl(mesh_device)` if multi-device, `start_core` → CoreCoord(0,0), `sampling_memory_config` → DRAM_MEMORY_CONFIG, `num_argmax_gather_links` → num_gather_links, `ag_topology` → Topology.Linear.

Fills LazyBuffer fields (using same `_resolve_buf` pattern as penalties):
- `index_offsets` → LazyBuffer with source computed from vocab_size/num_devices (port lines 200-216)
- `local_indices` → LazyBuffer with source `arange(0, vocab_per_device)` expanded to batch (port lines 218-229)
- `seeds` → LazyBuffer with source `arange(0, 32)`, uint32, ROW_MAJOR (port lines 168-175)
- `user_ids` → LazyBuffer with source `arange(0, 32)`, uint32, ROW_MAJOR (port lines 176-181)

Asserts `is_resolved()`.

### Sampling1D class

```python
class Sampling1D(LightweightModule):

    # Construction
    def __init__(self, vocab_size: int, mesh_device: ttnn.MeshDevice | None = None):
        # Happy path + _bind_strategy()
    @classmethod
    def from_config(cls, config: Sampling1DConfig) -> "Sampling1D":
        # Power path + _bind_strategy()
    @classmethod
    def from_model_args(cls, mesh_device, tt_ccl, args, model_config=None) -> "Sampling1D":
        # Backward compat. Rejects Galaxy. Extracts GALAXY_NUM_LINKS, SAMPLING_AG_CONFIG, etc.

    # Strategy binding (TTTv2 pattern: eliminate static branching)
    def _bind_strategy(self):
        # Binds self._topk to _topk_single_device (1x1) or _topk_multi_device (1xN)

    # Device buffers (idempotent, like MLP1D.load_device_weights)
    def load_device_buffers(self):
        # Materializes all LazyBuffer fields from resolved config:
        #   self._index_offsets = self._materialize(self.config.index_offsets)
        #   self._local_indices = self._materialize(self.config.local_indices)
        #   self._seeds = self._materialize(self.config.seeds)
        #   self._user_ids = self._materialize(self.config.user_ids)
        #   self._log_probs_calculator = LogProbsCalculator(...)  # non-buffer

    # Forward
    def decode_forward(self, logits, *, k=None, p=None, temp=None,
                       seeds=None, tt_out_tok=None, enable_log_probs=False):
        # k/p/temp all None + allow_force_argmax → _sample_argmax
        # k/p/temp all provided → _sample_topk
        # Otherwise → ValueError
        # Returns: (token_ids, log_probs_or_none)
    def forward(self, logits, **kwargs):  # Dispatcher

    # Private: argmax path — port from tt_sampling.py:310-341
    def _sample_argmax(self, logits, tt_out_tok):
        # all_gather_async (if multi-device) → untilize → argmax → log_probs

    # Private: top-k sampling — port from tt_sampling.py:343-481
    def _sample_topk(self, logits, k, p, temp, tt_out_tok):
        # typecast → self._topk() → offset → seed → ttnn.sampling → log_probs

    # Private: top-k strategies (bound at init time, no if-else in forward)
    def _topk_single_device(self, x_bf16):
        # Split vocab in half → two topk → concat (port lines 346-371)
    def _topk_multi_device(self, x_bf16):
        # Local topk → all_gather (cluster_axis=None for 1D) → gather indices (port lines 372-421)

    # Private: CCL helper
    def _perform_all_gather(self, tensor, dim, cluster_axis, memory_config, num_links, buffer_key=None, dtype=None):
        # Prefers line_all_gather if available, else ttnn.all_gather (port lines 231-259)
```

### Key porting decisions

1. **k/p/temp are per-call parameters, NOT module state**: TTTv1 stores `self.k_tensor` etc. and mutates via `reset_params()`. TTTv2 passes them as arguments to `decode_forward()`. This eliminates 3 mutable buffers entirely (no `k_tensor`, `p_tensor`, `temp_tensor`).
2. **LazyBuffer for all persistent buffers**: TTTv1's `from_torch` calls in `__init__` and `_create_indices_tensors` are replaced by `LazyBuffer | ttnn.Tensor | None` fields in `Sampling1DConfig`. Same triple-type pattern as Penalties1DConfig: `None` → auto-resolve, `LazyBuffer` → custom spec, `ttnn.Tensor` → direct bypass.
3. **Seeds via LazyBuffer.update()**: `seeds` is a `LazyBuffer` in config, materialized in `load_device_buffers()`. SeedManager (at orchestrator level) calls `self.config.seeds.update(new_seeds_tensor)` to refresh per-request. Caller can also pass `seeds=` to `decode_forward()` to override per-call.
4. **Argmax vs sampling is per-call**: TTTv1 tracks `self._force_argmax_sampling`. TTTv2 checks whether k/p/temp are None per call. `allow_force_argmax` config gates availability.
5. **Strategy binding via `_bind_strategy()`**: `self._topk` is bound to either `_topk_single_device` or `_topk_multi_device` at init. No static if-else in forward.
6. **No penalty logic**: Penalties1D runs upstream. Sampling1D never sees penalty tensors.
7. **No trace caching**: Trace management stays in caller/orchestrator (SamplingGenerator or equivalent).
8. **`torch` import is lazy** — only inside `_resolve_sampling1d_config()` and `from_model_args()`.

---

## Step 4: Create `test_sampling_1d.py`

### Unit tests (no device)
1. `test_config_defaults` — max_batch_size=32, max_top_k=32, allow_force_argmax=False, num_gather_links=1
2. `test_config_custom` — explicit config preserves values
3. `test_config_is_resolved` — checks mesh_device and tt_ccl requirements

### Device tests (parametrized)

```python
@pytest.mark.parametrize("ttnn_mesh_device", [(1,1),(1,2),(1,8)], indirect=True)
@pytest.mark.parametrize("mesh_shape,vocab_size", [
    (1,1)/1024, (1,1)/32000, (1,1)/128256,
    (1,2)/32000, (1,2)/128256,
    (1,8)/128256,
])
```

4. `test_topk1_vs_argmax` — k=1, p=0.0, temp=1.0 must match `torch.argmax` on gathered logits. Primary PCC test (exact match, PCC=1.0). Multi-device logits sharded via `ShardTensor2dMesh(dims=(None,None))`.
5. `test_force_argmax` — allow_force_argmax=True, k/p/temp=None → matches torch.argmax
6. `test_topk_distribution` — k=32, p=1.0 → sampled token is within top-32 set
7. `test_error_on_partial_params` — k provided but not p/temp → ValueError
8. `test_from_model_args` — backward compat with mock args, k=1 matches argmax
9. `test_rejects_galaxy` — ValueError

### Helper functions
```python
def make_random_logits(batch_size, vocab_size, num_devices, mesh_device) -> ttnn.Tensor
def make_sampling_params_tt(mesh_device, batch_size=32, k_val=1, p_val=0.0, temp_val=1.0) -> (k, p, temp)
```

---

## Step 5: Composition Verification

Add to `test_sampling_1d.py`:

```python
def test_penalties_sampling_composition(ttnn_mesh_device, mesh_shape, vocab_size):
    penalizer = Penalties1D(vocab_size, ttnn_mesh_device)
    sampler = Sampling1D(vocab_size, ttnn_mesh_device)

    # Caller constructs params and accum directly (no convenience methods)
    params = PenaltyParams(
        prompt_mask=make_zero_buffer(32, vocab_size, ttnn_mesh_device),
        presence_penalties=make_bf16_buffer(32, 1, ttnn_mesh_device, fill=0.0),
        frequency_penalties=make_bf16_buffer(32, 1, ttnn_mesh_device, fill=0.0),
        repetition_penalties=make_bf16_buffer(32, 1, ttnn_mesh_device, fill=1.2),
        inverse_repetition_penalties=make_bf16_buffer(32, 1, ttnn_mesh_device, fill=1/1.2),
    )
    accum = PenaltyAccumulator(
        output_mask=make_zero_buffer(32, vocab_size, ttnn_mesh_device),
        output_counts=make_zero_buffer(32, vocab_size, ttnn_mesh_device),
        output_counts_gathered=make_zero_buffer(32, vocab_size, ttnn_mesh_device),
    )

    logits = make_random_logits(32, vocab_size, max(mesh_shape), ttnn_mesh_device)

    # Prefill: sets prompt_mask, returns logits unchanged
    prompt_tokens = torch.randint(0, vocab_size, (32, 128))
    logits = penalizer.prefill_forward(logits, params, accum, prompt_tokens)

    # Decode step (repeated)
    penalized = penalizer.decode_forward(logits, params, accum)
    k, p, temp = make_sampling_params_tt(ttnn_mesh_device, k_val=1)
    tokens, log_probs = sampler.decode_forward(penalized, k=k, p=p, temp=temp)
    penalizer.update_output_tokens(accum, tokens)

    # Verify: valid token range, no NaN, penalized != original
```

---

## Step 6: Audit (CLAUDE.md Steps 10-13)

### Step 10 — TTTv2 vs TTTv1 correctness audit

| TTTv1 Location | TTTv2 Equivalent | Verify |
|---|---|---|
| `tt_penalties.py:32-75` apply_penalties | `Penalties1D.decode_forward` | Same typecast/deallocate pattern |
| `tt_penalties.py:83-139` __init__ | `Penalties1DConfig` LazyBuffers + `load_device_buffers` | Same shapes, dtypes, shard_dims — now declarative via LazyBuffer |
| `tt_penalties.py:157-159` _copy_host_to_device | `LazyBuffer.update()` | Same from_torch(host) + copy_host_to_device_tensor pattern, now encapsulated |
| `tt_penalties.py:161-170` reset_params | Caller uses `LazyBuffer.update()` directly on PenaltyParams fields | Pad-to-32 logic moves to caller; no module convenience method |
| `tt_penalties.py:194-212` reset_prompt_tokens | `Penalties1D.prefill_forward()` | Same -1 masking, scatter_add — now inside prefill_forward |
| `tt_penalties.py:214-264` reset/update_output | `Penalties1D.reset/update_output_tokens` | Same zero-out + scatter_add (writes to PenaltyAccumulator) |
| `tt_penalties.py:266-289` token_bin_counts | `Penalties1D._token_bin_counts_and_mask` | Same scatter→tilize→add→slice→gt |
| `tt_sampling.py:63-229` __init__+indices | `Sampling1DConfig` LazyBuffers + `load_device_buffers` | Same shapes/dtypes — now declarative via LazyBuffer. k/p/temp buffers eliminated (per-call args) |
| `tt_sampling.py:231-259` _perform_all_gather | `Sampling1D._perform_all_gather` | Same CCL introspection + fallback |
| `tt_sampling.py:310-341` argmax path | `Sampling1D._sample_argmax` | Same all_gather_async→untilize→argmax |
| `tt_sampling.py:343-481` sampling path | `Sampling1D._sample_topk`+`_topk_*` | Same typecast→topk→gather→offset→sample |

### Step 11 — Dependency check
- Module-level: only `ttnn`, `dataclasses`, `typing`, `LightweightModule`, `tt_ccl`, `LazyBuffer`
- NO `import torch` at module level (lazy only — inside `_resolve_penalties1d_config()` and lifecycle methods)
- NO imports from `models.tt_transformers` or `models.common.sampling` except in `from_model_args`

### Step 12 — Test check
- All PCC checks run for every mesh shape — no skips
- Coverage target: >90%
- Test cases cover `(1,1)`, `(1,2)`, `(1,8)` mesh shapes
- Test cases cover small (1024), medium (32000), large (128256) vocab sizes

### Step 13 — Code quality
- No dead code, no magic constants (32 → `cfg.max_batch_size`), no stale comments
- No static if-else in `decode_forward` (strategy binding at init)
- Proper `deallocate()` on all intermediates matching TTTv1 pattern
- No `_alloc_int_buffer` / `_alloc_bf16_buffer` — all buffer allocation via `LazyBuffer.get_device_buffer()`
- No `_copy_host_to_device` — all host→device refresh via `LazyBuffer.update()`
- `_resolve_penalties1d_config()` is idempotent (calling twice produces same result)

---

## Verification Commands

```bash
# Penalties tests with coverage
python_env/bin/python -m pytest models/common/tests/modules/sampling/test_penalties_1d.py -v \
  --cov=models.common.modules.sampling.penalties_1d \
  --cov-report=term-missing \
  --cov-config=models/common/tests/setup.cfg

# Sampling tests with coverage
python_env/bin/python -m pytest models/common/tests/modules/sampling/test_sampling_1d.py -v \
  --cov=models.common.modules.sampling.sampling_1d \
  --cov-report=term-missing \
  --cov-config=models/common/tests/setup.cfg

# Reset devices if needed
source python_env/bin/activate && tt-smi -r
```

## Implementation Order

1. `lazy_buffer.py` → 2. `penalties_1d.py` → 3. `test_penalties_1d.py` (run, includes LazyBuffer tests) → 4. `sampling_1d.py` → 5. `test_sampling_1d.py` (run) → 6. Composition test → 7. Audit

## Test Case Collection Plan
See `models/common/modules/sampling/test_case_collection.md` for the test case collection plan (Phase A-D).

---

# call_chain_report.md
# Call-Chain Report: `tt_penalties.py` & `tt_sampling.py`

> **Generated**: 2026-02-20
> **Source files**:
> - `models/common/sampling/tt_penalties.py` — penalty transforms (presence, frequency, repetition)
> - `models/common/sampling/tt_sampling.py` — token selection (top-k, top-p, temperature, argmax)
> - `models/common/sampling/generator.py` — orchestrator (`SamplingGenerator`)

---

## Architecture Overview

Both modules are **never used independently in production**. They are always
orchestrated by `SamplingGenerator` (`models/common/sampling/generator.py:26`),
which owns one `TTSampling` instance and one `TTPenalties` instance.

---

## Two Pipelines

| Pipeline | Entry Point | Penalties? | Via SamplingGenerator? |
|----------|-------------|------------|------------------------|
| **A** — tt_transformers (production) | `Transformer` → `Generator` | Yes | Yes |
| **B** — llama3_70b_galaxy (production) | `TtTransformer` → `Generator` | Yes | Yes |

---

// NOTE: shouldn't expect to use this in production until end of month
// But in general, 1D should be stable! The exception would some batched prefill in the generator.py
## Pipeline A: tt_transformers (production) -1D 1x1 1x2 1x8

### Model-level classes

- **Model**: `Transformer` in `models/tt_transformers/tt/model.py`
- **Generator**: `Generator` in `models/tt_transformers/tt/generator.py:93`

### Call chain

```
Transformer.__init__()                                          model.py:146
  └─ SamplingGenerator.__init__()                               generator.py:43    [ONCE]
       ├─ TTSampling.__init__(mesh_device, tt_ccl, args)        generator.py:58    [ONCE]
       └─ TTPenalties.__init__(mesh_device, args)               generator.py:59    [ONCE]

Generator.prefill_forward_text() — prefill sampling             tt/generator.py:296 [PER-REQUEST]
  ├─ format_sampling_params(broadcast(...), 32)                 tt/generator.py:414
  ├─ _apply_prefill_sampling_state(model, ...)                  tt/generator.py:418
  │    ├─ sampling_module.reset_sampling_params(sampling_params) tt/generator.py:68
  │    │    ├─ TTSampling.reset_params(k, p, temp, log_probs)   generator.py:110
  │    │    └─ TTPenalties.reset_params(presence, freq, rep)     generator.py:130   [conditional]
  │    ├─ sampling_module.reset_prompt_tokens(prompt_tokens)     tt/generator.py:73
  │    │    └─ TTPenalties.reset_prompt_tokens(prompt_tokens)    generator.py:98
  │    └─ sampling_module.reset_output_state()                   tt/generator.py:74
  │         └─ TTPenalties.reset_output_tokens()                 generator.py:103
  └─ self.model[model_id].sampling.sample(logits)               tt/generator.py:473
       └─ SamplingGenerator.sample()                            generator.py:227
            (see hot-path below)

Generator.decode_forward_text() — param setup                   tt/generator.py:673
  │
  │  ── runs every decode call (when sampling_params is not None) ──  [PER-TOKEN]
  ├─ format_sampling_params(sampling_params_list[i], 32)        tt/generator.py:733
  ├─ sampling_module.reset_sampling_params(formatted_params)    tt/generator.py:738
  │    ├─ TTSampling.reset_params(...)                          generator.py:110
  │    └─ TTPenalties.reset_params(...)                         generator.py:130   [conditional]
  ├─ sampling_module.seed_manager.get_new_values()              tt/generator.py:739
  │
  │  ── only when reset_batch=True ──                                [PER-REQUEST] --> make this into a separate public API that user can call to work around the reshuffling the user ids!
  ├─ sampling_module.reset_prompt_tokens(prompt_chunks[i])      tt/generator.py:741
  │    └─ TTPenalties.reset_prompt_tokens(...)                  generator.py:98
  └─ sampling_module.reset_output_state(output_chunks[i])       tt/generator.py:742
       └─ TTPenalties.reset_output_tokens(tokens)               generator.py:103

Transformer.decode_forward() — decode sampling                  model.py:532       [PER-TOKEN]
  └─ self.sampling.sample(tt_logits, tt_out_tok=x)              model.py:536
       └─ SamplingGenerator.sample()                            generator.py:227
            (see hot-path below)

Generator._decode_forward_trace_text() — trace-based decode     tt/generator.py:869 [PER-TOKEN]
  └─ sampling_module.sample(logits=..., tt_out_tok=...)         tt/generator.py:915
       └─ SamplingGenerator.sample()                            generator.py:227
            (see hot-path below)
```

Prefill_forward_text() has a loop over each user and each user in the batch is prefilled and sampled individually with its own params — the loop processes one user at a time.

---

// NOTE: wait for the testing work done on the existing code
## Pipeline B: llama3_70b_galaxy (production)

### Model-level classes

- **Model**: `TtTransformer` in `models/demos/llama3_70b_galaxy/tt/llama_model.py`
- **Generator**: `Generator` in `models/demos/llama3_70b_galaxy/tt/generator.py:45`

### Call chain

```
TtTransformer.setup_decode()                                    llama_model.py:185
  └─ SamplingGenerator.__init__()                               generator.py:43    [ONCE]
       ├─ TTSampling.__init__(mesh_device, tt_ccl, args)        generator.py:58    [ONCE]
       └─ TTPenalties.__init__(mesh_device, args)               generator.py:59    [ONCE]

Generator.prefill_forward_text()                                galaxy/generator.py:316 [PER-REQUEST]
  ├─ format_sampling_params(sampling_params, max_batch_size)    galaxy/generator.py:316
  ├─ sampling_module.reset_sampling_params(sampling_params)     galaxy/generator.py:349
  │    ├─ TTSampling.reset_params(...)                          generator.py:110
  │    └─ TTPenalties.reset_params(...)                         generator.py:130   [conditional]
  ├─ sampling_module.reset_prompt_tokens(prefill_ids)           galaxy/generator.py:351
  │    └─ TTPenalties.reset_prompt_tokens(...)                  generator.py:98
  ├─ sampling_module.reset_output_state()                       galaxy/generator.py:352
  │    └─ TTPenalties.reset_output_tokens()                     generator.py:103
  └─ sampling_module.sample(tt_logits_batch)                    galaxy/generator.py:355
       └─ SamplingGenerator.sample()                            generator.py:227
            (see hot-path below)

Generator.decode_forward_text() — batch-reset path             galaxy/generator.py:553 [PER-REQUEST]
  ├─ format_sampling_params(sampling_params, max_batch_size)    galaxy/generator.py:555
  ├─ sampling_module.reset_sampling_params(sampling_params)     galaxy/generator.py:558
  │    ├─ TTSampling.reset_params(...)                          generator.py:110
  │    └─ TTPenalties.reset_params(...)                         generator.py:130   [conditional]
  ├─ sampling_module.reset_prompt_tokens(prompt_tokens)         galaxy/generator.py:560
  │    └─ TTPenalties.reset_prompt_tokens(...)                  generator.py:98
  └─ sampling_module.reset_output_state(output_tokens)          galaxy/generator.py:561
       └─ TTPenalties.reset_output_tokens(tokens)               generator.py:103

TtTransformer.decode_forward() — decode sampling                llama_model.py:645  [PER-TOKEN]
  └─ self.sampling.sample(tt_logits[0], tt_out_tok=x)           llama_model.py:645
       └─ SamplingGenerator.sample()                            generator.py:227
            (see hot-path below)

Generator._decode_easy_trace_text() — trace split sampling      galaxy/generator.py:739 [PER-TOKEN]
  └─ self.model.sampling.sample(logits=..., tt_out_tok=...)     galaxy/generator.py:740
       └─ SamplingGenerator.sample()                            generator.py:227
            (see hot-path below)
```

---

## Hot Path: SamplingGenerator.sample() internals

This is the **per-token** path shared by both Pipeline A and Pipeline B.
It is the performance-critical section.

```
SamplingGenerator.sample(logits, tt_out_tok=...)                generator.py:227   [PER-TOKEN]
  │
  ├─ [trace path] ─────────────────────────────────────────────
  │   If trace is already captured:
  │     SamplingGenerator._execute_trace(key)                   generator.py:217
  │       └─ ttnn.execute_trace(...)                            generator.py:224
  │   If trace not yet captured:
  │     SamplingGenerator.capture_trace(logits)                 generator.py:169
  │       └─ _run_sampling() × 2 (compile + capture)           generator.py:187,194
  │
  ├─ [no-trace path] ──────────────────────────────────────────
  │   SamplingGenerator._run_sampling(logits, penalties_on)     generator.py:157
  │     ├─ TTPenalties.apply(logits)                            generator.py:164   [PER-TOKEN, if penalties_on]
  │     │    └─ apply_penalties(logits, PenaltyContext)          tt_penalties.py:32
  │     │         ├─ presence penalty                           tt_penalties.py:38-43
  │     │         ├─ frequency penalty                          tt_penalties.py:46-51
  │     │         └─ repetition penalty                         tt_penalties.py:54-73
  │     └─ TTSampling.forward(logits, tt_out_tok)               generator.py:166   [PER-TOKEN]
  │          ├─ [argmax fast path]                              tt_sampling.py:310
  │          │    ├─ all_gather_async (if multi-device)         tt_sampling.py:316
  │          │    ├─ ttnn.untilize                              tt_sampling.py:330
  │          │    └─ ttnn.argmax                                tt_sampling.py:331
  │          └─ [top-k/p/temp path]                             tt_sampling.py:344
  │               ├─ ttnn.topk (local per-device)              tt_sampling.py:374
  │               ├─ _perform_all_gather (values)               tt_sampling.py:386
  │               ├─ _perform_all_gather (indices)              tt_sampling.py:412
  │               ├─ add device offsets → global indices        tt_sampling.py:438
  │               ├─ ttnn.manual_seed                           tt_sampling.py:454
  │               └─ ttnn.sampling(values, indices, k, p, temp) tt_sampling.py:460
  │
  └─ [post-sampling penalty update] ───────────────────────────
      TTPenalties.update_output_tokens(tt_out)                  generator.py:263   [PER-TOKEN, if penalties_on]
        └─ token_bin_counts_and_mask(new_tokens, src, ...)      tt_penalties.py:266
             ├─ ttnn.scatter_add                                tt_penalties.py:267
             ├─ ttnn.tilize                                     tt_penalties.py:271
             ├─ ttnn.add (accumulate counts)                    tt_penalties.py:275
             ├─ ttnn.slice (shard to local vocab)               tt_penalties.py:278
             └─ ttnn.gt (update binary mask)                    tt_penalties.py:288
```

---

## Per-Function Frequency Summary

### tt_penalties.py

| Function | Position | Frequency | Call Sites (external) |
|----------|----------|-----------|-----------------------|
| `TTPenalties.__init__` | Pipeline start | **Once** (model init) | `generator.py:59` |
| `reset_params(presence, frequency, repetition)` | Request setup | **Per-request** | `generator.py:130` |
| `reset_prompt_tokens(prompt_tokens)` | Post-prefill setup | **Per-request** | `generator.py:98` |
| `reset_output_tokens(tokens=None)` | Post-prefill setup | **Per-request** | `generator.py:103` |
| `apply(tt_logits)` → `apply_penalties()` | Hot path (logit transform) | **Per-token** | `generator.py:164` |
| `update_output_tokens(new_tokens)` → `token_bin_counts_and_mask()` | Hot path (state update) | **Per-token** | `generator.py:263` |
| `token_bin_counts_and_mask(...)` | Internal workhorse | **Per-token** + **per-request** | 3 internal call sites |
| `_alloc_int_buffer(...)` / `_alloc_bf16_buffer()` | Buffer allocation | **Once** (init) + **per-request** | ~10 in `__init__`, 2+ in reset methods |
| `_copy_host_to_device(dst, src)` | H2D transfer | **Per-request** | 4× from `reset_params` |
| `_pad_params(values)` | Param padding | **Per-request** | 3× from `reset_params` |
| `_pad_batch_to_max(tokens_2d, pad_value)` | Batch padding | **Per-request** | from `reset_prompt_tokens`, `reset_output_tokens` |

### tt_sampling.py

| Function | Position | Frequency | Call Sites (external) |
|----------|----------|-----------|-----------------------|
| `TTSampling.__init__` | Pipeline start | **Once** (model init) | `generator.py:58` |
| `_create_indices_tensors()` | Internal (init) | **Once** | 1× from `__init__` |
| `reset_params(k, p, temp, enable_log_probs)` | Request setup | **Per-request** | `generator.py:110` |
| `forward(x, tt_out_tok=None)` | **Hot path** (token selection) | **Per-token** | `generator.py:166` |
| `_perform_all_gather(...)` | Internal (multi-device) | **Per-token** (2× in `forward`) | 2 internal: values + indices |
| `_is_force_argmax_sampling(k, p, temp)` | Fast-path check | **Per-request** | from `reset_params`, read in `sample()` |
| `clamp(value, min, max)` | Param validation | **Per-request** | `generator.py:336` |

### generator.py (SamplingGenerator)

| Function | Position | Frequency | Callers |
|----------|----------|-----------|---------
| `SamplingGenerator.__init__` | Pipeline start | **Once** | `Transformer.__init__` (model.py:146), `TtTransformer.setup_decode` (llama_model.py:185) |
| `reset_sampling_params(sampling_params)` | Param setup | **Per-request** (prefill), **Per-token** (decode†) | `_apply_prefill_sampling_state` (tt/generator.py:68), `Generator.decode_forward_text` (tt/generator.py:738, galaxy/generator.py:558), `Generator.prefill_forward_text` (galaxy/generator.py:349) |
| `reset_prompt_tokens(prompt_tokens)` | Post-prefill setup | **Per-request** | `_apply_prefill_sampling_state` (tt/generator.py:73), `Generator.decode_forward_text` (tt/generator.py:741, galaxy/generator.py:560) |
| `reset_output_state(tokens=None)` | Post-prefill setup | **Per-request** | `_apply_prefill_sampling_state` (tt/generator.py:74), `Generator.decode_forward_text` (tt/generator.py:742, galaxy/generator.py:561) |
| `sample(logits, *, enable_trace, tt_out_tok)` | Token selection | **Per-token** | `Transformer.decode_forward` (model.py:536), `TtTransformer.decode_forward` (llama_model.py:645), `Generator.prefill_forward_text` (tt/generator.py:473, galaxy/generator.py:355), `Generator._decode_forward_trace_text` (tt/generator.py:915), `Generator._decode_easy_trace_text` (galaxy/generator.py:740) |
| `_run_sampling(logits, penalties_on, tt_out_tok)` | Internal dispatch | **Per-token** | from `sample()` and `capture_trace()` |
| `capture_trace(logits, tt_out_tok)` | Trace capture | **Once** per config | from `sample()` on first call |
| `_execute_trace(key)` | Trace replay | **Per-token** | from `sample()` when trace exists |
| `reset_trace()` | Trace invalidation | **Per-config-change** | from `reset_sampling_params()` |
| `format_sampling_params(sampling_params, max_batch_size)` | Param formatting | **Per-request** (prefill), **Per-token** (decode†) | tt/generator.py:414, tt/generator.py:733, galaxy/generator.py:316, galaxy/generator.py:555 |

> **†** In Pipeline A's `decode_forward_text`, `format_sampling_params` → `reset_sampling_params` → `seed_manager.get_new_values()` run on **every decode call** when `sampling_params is not None` (lines 733-739). Only `reset_prompt_tokens` and `reset_output_state` (lines 741-742) are gated by `if reset_batch:`. This means the param-setup path is effectively **per-token** in the decode loop, while the penalty-state-reset path remains **per-request**.

---

## Import Graph

```
models/common/sampling/generator.py
  ├─ imports TTPenalties   from .tt_penalties   (only consumer in production)
  └─ imports TTSampling    from .tt_sampling    (only consumer in production)

models/tt_transformers/tt/model.py
  └─ imports SamplingGenerator from models.common.sampling.generator

models/demos/llama3_70b_galaxy/tt/llama_model.py
  └─ imports SamplingGenerator from models.common.sampling.generator

models/tt_transformers/tt/generator.py
  └─ imports format_sampling_params from models.common.sampling.generator

models/demos/llama3_70b_galaxy/tt/generator.py
  └─ imports format_sampling_params from models.common.sampling.generator

```

---

## Observations for TTTv2 Refactoring

1. **`TTPenalties` has exactly ONE external consumer**: `SamplingGenerator`.
   It is never instantiated or called from anywhere else. This makes it a
   clean extraction target — the TTTv2 `Penalties1D` module only needs to
   satisfy `SamplingGenerator`'s interface.

2. **`token_bin_counts_and_mask`** is the most-called internal function. It
   runs both **per-request** (from `reset_prompt_tokens`/`reset_output_tokens`)
   AND **per-token** (from `update_output_tokens`), making it the single
   most important function to optimize.

3. **The argmax fast path** (`_force_argmax_sampling`) bypasses the entire
   top-k/p/temp pipeline including `_perform_all_gather`. When k=1, p=1.0,
   temp=1.0, it falls through to `ttnn.argmax` directly. This path is
   checked at the `SamplingGenerator` level too (for trace key selection).

4. **Penalty activation is lazy**: `_penalties_active` is only set to `True`
   when any penalty parameter differs from the defaults (presence=0.0,
   frequency=0.0, repetition=1.0). When penalties are inactive, all
   `TTPenalties` per-request and per-token calls are short-circuited in
   `SamplingGenerator` via early returns.

---

# Core & Device Execution Model (Sampling)

> **Generated**: 2026-05-29
> **Sources**:
> - `ttnn/cpp/ttnn/operations/reduction/sampling/device/sampling_program_factory.cpp` — the `ttnn.sampling` device op
> - `models/common/modules/sampling/sampling_1d.py` — TTTv2 module
> - `models/common/sampling/tt_sampling.py` — TTTv1 source

Two orthogonal axes are commonly conflated: **single _device_** (1×1 mesh topology, one chip) vs **single _Tensix core_** (core placement within a chip). Neither the plan above nor the Python modules reason about core-level execution — they only forward `sub_core_grids` / `sub_core_grid_topk` / `start_core` to the ttnn ops. The single-core semantics live entirely in the C++ sampling op.

## Single Tensix core execution: one core per user

`ttnn.sampling` parallelizes over the **batch (user) dimension, not vocab**. The input is `[1, 1, B, K*num_devices]`; the program factory derives one core per row:

```cpp
// sampling_program_factory.cpp:55-65
uint32_t Ht = (input_shape[0] * input_shape[1] * input_shape[2]) / tile_height; // (1·1·B)/32
uint32_t Wt = input_shape[3] / tile_width;
auto num_cores = Ht * tile_height;                                              // == B

CoreRangeSet core_grid = num_cores_to_corerangeset(num_cores, compute_with_storage_grid_size, true);
if (sub_core_grids.has_value()) {
    core_grid = sub_core_grids.value();   // Python-supplied override
}
auto cores = corerange_to_cores(core_grid, num_cores, true);
```

For `B=32`: `Ht=1`, `num_cores=32` → **32 cores, one user per core**. A separate writer+compute kernel is instantiated per core, with the core index `i` (= user index) baked in as a compile-time arg (`sampling_program_factory.cpp:253-321`).

Consequences:
- Each core does the **entire** top-p / temperature / RNG / argmax over that user's `K*num_devices` candidates (e.g. 32·8 = 256 values). Vocab is **not** spread across cores in this op.
- `k`/`p`/`temp` buffers are broadcast to every core (`k_chunk_size = num_cores * 4B`, lines 196-217); each core NOC-reads its own scalar via index `i`. Hence k/p/temp are `[B]`-shaped ROW_MAJOR tensors.

## sub_core_grids / sub_core_grid_topk / start_core

Which physical cores get used is computed host-side and passed as the `sub_core_grids` override above:

```python
# sampling_1d.py:172-179 (load_device_buffers)
self._sampling_sub_core_grids = (
    ttnn.num_cores_to_corerangeset_in_subcoregrids(
        cfg.start_core, cfg.max_batch_size, cfg.sub_core_grids, row_wise=True
    )
    if cfg.sub_core_grids is not None else None
)
```

`num_cores_to_corerangeset_in_subcoregrids(start_core, max_batch_size, sub_core_grids, row_wise=True)` = "take `max_batch_size` cores from `sub_core_grids`, starting at `start_core`, row-wise." That set becomes the `core_grid` override at `sampling_program_factory.cpp:63`. So `start_core` literally decides which physical core is "user 0."

Two grids, placed independently:

| Grid | Consumed by |
|---|---|
| `sub_core_grids` | sampling op (one-core-per-user), plus `typecast`, `untilize`, `manual_seed`, log-probs |
| `sub_core_grid_topk` | only `ttnn.topk` (vocab reduction — different optimal layout) |

**Perf note**: TTTv1 recomputes this corerangeset *inside every* `forward()` (`tt_sampling.py:552-556`); `sampling_1d.py` hoists it to `load_device_buffers()` once (`self._sampling_sub_core_grids`). Same cores, out of the per-token hot path.

## Single device (1×1 mesh) = `multi_step_reduction`

Distinct from single-core — this is mesh topology (one chip), the `[1,1]` special case, bound at init:

```python
# sampling_1d.py:114-121
self._multi_step_reduction = list(cluster_shape) == [1, 1]
self._topk = self._topk_single_device if self._multi_step_reduction else self._topk_multi_device
```

Three single-device specializations:

1. **No cross-device gather.** Multi-device shards vocab across chips → local top-k → all-gather candidates. Single device has the full vocab on one chip, so it **splits the vocab in half and runs top-k twice** (`_topk_single_device`, lines 340-368), emulating a 2-virtual-device reduction. Hence `num_devices_in_mesh = 2` even on one chip (lines 530-533), giving index offsets `{0, V/2}`.
2. **Argmax skips the all-gather** — `_argmax_noop` vs `_argmax_all_gather` (lines 240-262).
3. **TTTv2 bug fix in single-device index width** (lines 559-566): TTTv1 built `tt_indices_tensor` at width `V/2` then split it *again* in `forward` → V/4-wide index halves against V/2-wide logit halves (mismatch). TTTv2 builds it at full width `V` with each half holding `arange(V/2)` twice (`_make_local_indices`), so post-split it's V/2 vs V/2. Only the `[1,1]` path was affected.

## Terminology summary

- **Single _device_** (`[1,1]` mesh, `multi_step_reduction`): split-vocab-twice top-k, no CCL, argmax noop. A Python-level concern, handled by `_bind_strategy()`.
- **Single _Tensix core_**: `ttnn.sampling` assigns one core per user (`num_cores = batch_size`); `sub_core_grids` / `start_core` only pick which physical cores. A C++ op concern — vocab work for a user is entirely on that user's one core (batch-parallel, not vocab-parallel).
