"""Will this model fit on this machine? Back-of-the-envelope arithmetic, clearly labelled as such.

The lesson says "8 GB of RAM is a realistic minimum" without saying which build of the model it means. Whether a model fits depends on three things you can calculate:
  weights   parameters x bytes per parameter (fp16 = 2, int8 = 1, int4 = 0.5)
  KV cache  2 x layers x KV heads x head size x bytes per value x tokens in the conversation. It grows with every token an agent loop appends
  overhead  runtime buffers, activations, the operating system. A GUESS (a fraction of the weights plus a fixed reserve)
These are ESTIMATES from formulas. They are not measurements, they ignore quantization formats that mix precisions (real "int4" files are somewhat larger than 0.5 bytes per parameter),
and I did not run any model to check them. The example shape below is from my recollection of one model's configuration: check the model's own config before relying on it.
"""

from dataclasses import dataclass

BYTES_PER_PARAM = {"fp32": 4.0, "fp16": 2.0, "bf16": 2.0, "int8": 1.0, "int4": 0.5}
GB = 1e9


@dataclass(frozen=True)
class ModelShape:
    name: str
    params_billion: float
    layers: int
    kv_heads: int
    head_dim: int
    max_context: int


# recalled from memory, NOT verified against the model card: check config.json before trusting these
EXAMPLE_SHAPES = {"qwen2.5-7b (recalled, verify)": ModelShape("qwen2.5-7b (recalled, verify)", 7.6, 28, 4, 128, 32768)}


def weights_gb(params_billion: float, precision: str) -> float:
    if precision not in BYTES_PER_PARAM:
        raise ValueError(f"precision must be one of {sorted(BYTES_PER_PARAM)}")
    if params_billion <= 0:
        raise ValueError("params_billion must be positive")
    return params_billion * 1e9 * BYTES_PER_PARAM[precision] / GB


def kv_cache_gb(shape: ModelShape, tokens: int, bytes_per_value: float = 2.0) -> float:
    if tokens < 0:
        raise ValueError("tokens cannot be negative")
    return 2 * shape.layers * shape.kv_heads * shape.head_dim * bytes_per_value * tokens / GB


@dataclass(frozen=True)
class Estimate:
    weights: float
    kv_cache: float
    overhead: float
    total: float
    available: float
    fits: bool
    headroom: float


def estimate(shape: ModelShape, precision: str, context_tokens: int, ram_gb: float, os_reserve_gb: float = 2.0, overhead_fraction: float = 0.10, kv_bytes: float = 2.0) -> Estimate:
    if context_tokens > shape.max_context:
        raise ValueError(f"{context_tokens} tokens is beyond this model's {shape.max_context}-token context")
    w, kv = weights_gb(shape.params_billion, precision), kv_cache_gb(shape, context_tokens, kv_bytes)
    over = w * overhead_fraction
    total, available = w + kv + over, ram_gb - os_reserve_gb
    return Estimate(round(w, 2), round(kv, 2), round(over, 2), round(total, 2), round(available, 2), total <= available, round(available - total, 2))


def max_context_tokens(shape: ModelShape, precision: str, ram_gb: float, os_reserve_gb: float = 2.0, overhead_fraction: float = 0.10, kv_bytes: float = 2.0) -> int:
    """The longest conversation that fits, capped at the model's own limit. 0 if even the weights do not fit."""
    w = weights_gb(shape.params_billion, precision)
    room = ram_gb - os_reserve_gb - w * (1 + overhead_fraction)
    if room <= 0:
        return 0
    per_token = kv_cache_gb(shape, 1, kv_bytes)
    return int(min(shape.max_context, room / per_token))
