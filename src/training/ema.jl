"""
Exponential Moving Average (EMA) of model parameters.

Maintains a shadow copy of parameters that tracks a moving average of the
training parameters.  The shadow parameters can be swapped in at inference
time to obtain smoother, better-generalising predictions.

Uses bias-corrected decay following the approach in:
    "Exponential Moving Average of Weights in Deep Learning" (Tan et al.)
    and the EMA schedule from PyTorch's implementation of EMA.
"""

# ─────────────────────────────────────────────
# EMAState
# ─────────────────────────────────────────────

"""
    EMAState(ps; decay=0.9999f0)

Holds the EMA shadow copy of model parameters `ps`.

Fields:
- `shadow`: NamedTuple that mirrors the structure of `ps`, initialised to a
            deep copy of the initial parameter values.
- `decay`:  EMA decay factor ∈ (0, 1).  Values close to 1 give a slower-
            moving (smoother) average.
- `step`:   Number of `update_ema!` calls performed so far (used for bias
            correction).
"""
mutable struct EMAState
    shadow ::Any        # NamedTuple matching parameter structure
    decay  ::Float32
    step   ::Int
end

"""
    EMAState(ps; decay=0.9999f0) -> EMAState

Create an EMA state by deep-copying the parameter tree `ps`.
"""
function EMAState(ps; decay::Float32=0.9999f0)
    # fmap over the parameter tree to produce an independent copy
    shadow = Functors.fmap(x -> x isa AbstractArray ? copy(x) : x, ps)
    return EMAState(shadow, decay, 0)
end

# ─────────────────────────────────────────────
# EMA update
# ─────────────────────────────────────────────

"""
    update_ema!(ema::EMAState, ps)

Perform one bias-corrected EMA update step.

The effective decay is clamped to prevent the EMA from moving too fast
at the start of training:
    corrected_decay = min(decay, (1 + step) / (10 + step))

Each leaf array in `ema.shadow` is updated in-place:
    shadow ← corrected_decay * shadow + (1 - corrected_decay) * ps

Uses `Functors.fmap` to traverse the parameter tree so that nested
NamedTuples (as used by Lux container layers) are handled correctly.
"""
function update_ema!(ema::EMAState, ps)
    ema.step += 1
    s = ema.step

    # Bias-corrected decay: ramps up from ~0.0909 at step 1 toward `decay`
    corrected_decay = min(ema.decay, Float32((1 + s) / (10 + s)))
    one_minus_decay = 1.0f0 - corrected_decay

    # Walk both trees in parallel using Functors.fmap on a paired structure.
    # We exploit the fact that fmap traverses leaves of a NamedTuple and
    # applies the function element-wise.  For the in-place update we need
    # to zip shadow and ps leaves together.
    ema.shadow = _ema_update_tree(ema.shadow, ps, corrected_decay, one_minus_decay)

    return ema
end

"""
    _ema_update_tree(shadow, ps, decay, one_minus_decay)

Recursively walk the shadow and ps trees, updating array leaves in-place.
Returns the updated shadow tree (same structure as input).
"""
function _ema_update_tree(shadow, ps, decay::Float32, one_minus_decay::Float32)
    # Use Functors.fmap2 (two-argument fmap) if available, otherwise recurse
    # manually over NamedTuple fields.
    return _update_node(shadow, ps, decay, one_minus_decay)
end

# Leaf: AbstractArray — update in-place and return the modified shadow array
function _update_node(shadow_leaf::AbstractArray, ps_leaf::AbstractArray,
                      decay::Float32, one_minus_decay::Float32)
    shadow_leaf .= decay .* shadow_leaf .+ one_minus_decay .* ps_leaf
    return shadow_leaf
end

# NamedTuple node — recurse into each field
function _update_node(shadow_nt::NamedTuple, ps_nt::NamedTuple,
                      decay::Float32, one_minus_decay::Float32)
    updated = map(keys(shadow_nt)) do k
        _update_node(getproperty(shadow_nt, k), getproperty(ps_nt, k),
                     decay, one_minus_decay)
    end
    return NamedTuple{keys(shadow_nt)}(updated)
end

# Tuple node — recurse positionally
function _update_node(shadow_t::Tuple, ps_t::Tuple,
                      decay::Float32, one_minus_decay::Float32)
    return map((s, p) -> _update_node(s, p, decay, one_minus_decay), shadow_t, ps_t)
end

# Scalar / non-array leaf — return shadow unchanged
function _update_node(shadow_other, ps_other, ::Float32, ::Float32)
    return shadow_other
end

# ─────────────────────────────────────────────
# Convenience
# ─────────────────────────────────────────────

"""
    ema_parameters(ema::EMAState) -> NamedTuple

Return the shadow (averaged) parameter tree.
Use these parameters at evaluation/inference time for best results.
"""
ema_parameters(ema::EMAState) = ema.shadow

"""
    copy_ema_to_model!(ps, ema::EMAState)

Overwrite the model's live parameters with the EMA shadow values (in-place).
Useful for final export or evaluation without modifying the EMAState.
"""
function copy_ema_to_model!(ps, ema::EMAState)
    _copy_tree!(ps, ema.shadow)
    return ps
end

function _copy_tree!(dst::AbstractArray, src::AbstractArray)
    copyto!(dst, src)
end

function _copy_tree!(dst::NamedTuple, src::NamedTuple)
    for k in keys(dst)
        _copy_tree!(getproperty(dst, k), getproperty(src, k))
    end
end

function _copy_tree!(dst::Tuple, src::Tuple)
    for (d, s) in zip(dst, src)
        _copy_tree!(d, s)
    end
end

_copy_tree!(::Any, ::Any) = nothing  # scalar / non-array leaves: no-op
