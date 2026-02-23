"""
Shared model layers: RMSNorm, SwiGLU, causal mask utilities.
Implemented as Lux layers with explicit parameter/state management.
"""

# ─────────────────────────────────────────────
# RMSNorm — Root Mean Square Layer Normalization
# ─────────────────────────────────────────────

struct RMSNorm <: Lux.AbstractLuxLayer
    dim::Int
    eps::Float32
end

RMSNorm(dim::Int; eps::Float32 = 1.0f-6) = RMSNorm(dim, eps)

function Lux.initialparameters(rng::AbstractRNG, l::RMSNorm)
    return (weight = ones(Float32, l.dim),)
end

Lux.initialstates(::AbstractRNG, ::RMSNorm) = NamedTuple()
Lux.parameterlength(l::RMSNorm) = l.dim
Lux.statelength(::RMSNorm) = 0

function (l::RMSNorm)(x, ps, st)
    # x: (dim, seq_len, batch) or (dim, ...)
    rms = sqrt.(mean(x .^ 2; dims=1) .+ l.eps)
    x_norm = x ./ rms
    return ps.weight .* x_norm, st
end

# ─────────────────────────────────────────────
# SwiGLU — Gated Linear Unit with Swish activation
# ─────────────────────────────────────────────

struct SwiGLU <: Lux.AbstractLuxLayer
    in_dim::Int
    hidden_dim::Int
    bias::Bool
end

"""
    SwiGLU(in_dim, hidden_dim; bias=false)

SwiGLU feed-forward block: gate = swish(W1 * x) .* (V * x), out = W2 * gate
Uses 3 projections (W1, V, W2) with hidden_dim intermediary.
"""
SwiGLU(in_dim::Int, hidden_dim::Int; bias::Bool=false) = SwiGLU(in_dim, hidden_dim, bias)

function Lux.initialparameters(rng::AbstractRNG, l::SwiGLU)
    scale = Float32(sqrt(2.0 / (l.in_dim + l.hidden_dim)))
    w1 = randn(rng, Float32, l.hidden_dim, l.in_dim) .* scale
    w2 = randn(rng, Float32, l.in_dim, l.hidden_dim) .* scale
    v  = randn(rng, Float32, l.hidden_dim, l.in_dim) .* scale
    ps = (w1=w1, w2=w2, v=v)
    if l.bias
        ps = merge(ps, (b1=zeros(Float32, l.hidden_dim),
                        b2=zeros(Float32, l.in_dim),
                        bv=zeros(Float32, l.hidden_dim)))
    end
    return ps
end

Lux.initialstates(::AbstractRNG, ::SwiGLU) = NamedTuple()

function Lux.parameterlength(l::SwiGLU)
    n = 3 * l.in_dim * l.hidden_dim
    l.bias && (n += 2 * l.hidden_dim + l.in_dim)
    return n
end

Lux.statelength(::SwiGLU) = 0

function (l::SwiGLU)(x, ps, st)
    # x: (in_dim, seq_len, batch)
    gate = ps.w1 * reshape(x, size(x, 1), :)
    val  = ps.v  * reshape(x, size(x, 1), :)
    if l.bias
        gate = gate .+ ps.b1
        val  = val .+ ps.bv
    end
    # SwiGLU: swish(gate) * val
    hidden = NNlib.swish.(gate) .* val
    out = ps.w2 * hidden
    if l.bias
        out = out .+ ps.b2
    end
    out = reshape(out, l.in_dim, size(x)[2:end]...)
    return out, st
end

# ─────────────────────────────────────────────
# Causal mask
# ─────────────────────────────────────────────

"""
    make_causal_mask(T; dtype=Float32) -> Matrix{dtype}

Create an upper-triangular causal attention mask of size (T, T).
Returns 0 for allowed positions, -Inf for masked positions.
"""
function make_causal_mask(seq_len::Int; dtype::Type{<:AbstractFloat}=Float32)
    # Non-mutating construction for Zygote compatibility
    mask = dtype[i >= j ? zero(dtype) : typemin(dtype) for i in 1:seq_len, j in 1:seq_len]
    return mask
end
