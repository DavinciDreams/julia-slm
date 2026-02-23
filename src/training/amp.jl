"""
Automatic Mixed Precision (AMP) training utilities.

Provides:
- LossScaler: dynamic loss scaling to prevent Float16 gradient underflow
- cast_f16 / cast_f32: parameter tree type conversion via Functors.fmap
- check_overflow: scan a gradient tree for NaN / Inf values
- unscale_grads: divide gradients by the current loss scale
- update_scaler!: adaptive scale adjustment based on overflow detection

Typical training loop usage:
    scaler = LossScaler()
    ps_f16 = cast_f16(ps)
    loss_scaled = scale_loss(scaler, loss)
    grads = Zygote.gradient(ps_f16) do p ... end
    unscaled_grads, has_inf = unscale_grads(scaler, grads)
    update_scaler!(scaler, has_inf)
    if !has_inf
        Optimisers.update!(opt_state, ps, unscaled_grads)
    end

Note on Julia / Zygote AMP:
Float16 support is partial in standard Julia AD. This module provides the
infrastructure for AMP; actual speedup requires CUDA with Float16 matmul support
(e.g., Tensor Cores on Volta/Turing/Ampere GPUs). On CPU, Float16 arithmetic
typically *degrades* performance due to software emulation.
"""

# ─────────────────────────────────────────────
# LossScaler
# ─────────────────────────────────────────────

"""
    LossScaler(; init_scale=2f0^15, growth=2f0, backoff=0.5f0, interval=2000)

Dynamic loss scaler for AMP training.

Fields:
- `scale`:            current loss scale factor (Float32)
- `growth_factor`:    multiply scale by this after `growth_interval` clean steps
- `backoff_factor`:   multiply scale by this on overflow detection
- `growth_interval`:  number of consecutive clean steps before growing the scale
- `step_since_growth`: counter of consecutive non-overflow steps
"""
mutable struct LossScaler
    scale             ::Float32
    growth_factor     ::Float32
    backoff_factor    ::Float32
    growth_interval   ::Int
    step_since_growth ::Int
end

function LossScaler(;
        init_scale   ::Float32 = 2f0^15,
        growth       ::Float32 = 2f0,
        backoff      ::Float32 = 0.5f0,
        interval     ::Int     = 2000)
    return LossScaler(init_scale, growth, backoff, interval, 0)
end

# ─────────────────────────────────────────────
# Core AMP operations
# ─────────────────────────────────────────────

"""
    scale_loss(scaler::LossScaler, loss) -> scaled_loss

Multiply the scalar loss by the current loss scale.
Pass this value to the AD backend so that gradients are also scaled up,
reducing the risk of underflow when using Float16 accumulation.
"""
scale_loss(scaler::LossScaler, loss) = loss * scaler.scale

"""
    unscale_grads(scaler::LossScaler, grads) -> (unscaled_grads, has_overflow)

Divide every gradient array by the current loss scale and check for overflow.

Returns:
- `unscaled_grads`: gradient tree with each leaf array divided by `scaler.scale`
- `has_overflow`:   `true` if any NaN or Inf was found in the gradient tree

If overflow is detected, the returned gradients should be discarded and the
optimizer should NOT be stepped; call `update_scaler!(scaler, true)` instead.
"""
function unscale_grads(scaler::LossScaler, grads)
    inv_scale = 1.0f0 / scaler.scale
    unscaled  = Functors.fmap(
        g -> g isa AbstractArray ? g .* inv_scale : g,
        grads
    )
    has_overflow = check_overflow(unscaled)
    return unscaled, has_overflow
end

"""
    update_scaler!(scaler::LossScaler, has_overflow::Bool)

Adjust the loss scale based on whether the last step had gradient overflow.

- Overflow detected: scale *= backoff_factor; reset step counter.
- No overflow: increment step counter; if counter reaches growth_interval,
               scale *= growth_factor; reset counter.

The scale is clamped to [1f0, 2f0^24] to stay in a reasonable range.
"""
function update_scaler!(scaler::LossScaler, has_overflow::Bool)
    if has_overflow
        scaler.scale = max(1.0f0, scaler.scale * scaler.backoff_factor)
        scaler.step_since_growth = 0
    else
        scaler.step_since_growth += 1
        if scaler.step_since_growth >= scaler.growth_interval
            scaler.scale = min(2.0f0^24, scaler.scale * scaler.growth_factor)
            scaler.step_since_growth = 0
        end
    end
    return scaler
end

# ─────────────────────────────────────────────
# Type conversion helpers
# ─────────────────────────────────────────────

"""
    cast_f16(ps) -> ps_f16

Convert all Float32 arrays in a parameter tree to Float16.
Uses `Functors.fmap` to traverse nested NamedTuples.

Non-array values and non-Float32 arrays are passed through unchanged.
"""
cast_f16(ps) = Functors.fmap(
    x -> x isa Array{Float32} ? Float16.(x) : x,
    ps
)

"""
    cast_f32(ps) -> ps_f32

Convert all Float16 arrays in a parameter tree back to Float32.
"""
cast_f32(ps) = Functors.fmap(
    x -> x isa Array{Float16} ? Float32.(x) : x,
    ps
)

# ─────────────────────────────────────────────
# Overflow detection
# ─────────────────────────────────────────────

"""
    check_overflow(grads) -> Bool

Scan every leaf array in `grads` for NaN or Inf values.
Returns `true` as soon as the first problematic value is found.

Traversal uses `Functors.fleaves` (re-exported from Functors via the module).
"""
function check_overflow(grads)::Bool
    for leaf in Functors.fleaves(grads)
        leaf isa AbstractArray || continue
        for val in leaf
            if isnan(val) || isinf(val)
                return true
            end
        end
    end
    return false
end

# ─────────────────────────────────────────────
# Introspection / diagnostics
# ─────────────────────────────────────────────

"""
    scaler_info(scaler::LossScaler) -> NamedTuple

Return a snapshot of the current scaler state, useful for logging.
"""
function scaler_info(scaler::LossScaler)
    return (
        scale             = scaler.scale,
        growth_factor     = scaler.growth_factor,
        backoff_factor    = scaler.backoff_factor,
        growth_interval   = scaler.growth_interval,
        step_since_growth = scaler.step_since_growth,
    )
end

function Base.show(io::IO, s::LossScaler)
    print(io, "LossScaler(scale=$(s.scale), growth=$(s.growth_factor), " *
              "backoff=$(s.backoff_factor), interval=$(s.growth_interval), " *
              "steps_clean=$(s.step_since_growth))")
end
