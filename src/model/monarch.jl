"""
Monarch Mixer layers — sub-quadratic sequence mixing using structured matrices.

A Monarch matrix M of size T×T (T = p²) factorizes as:
    M = Pᵀ · BlockDiag(L1) · P · BlockDiag(L2)
where L1, L2 are p block-diagonal matrices of size p×p each,
and P is a reshape-transpose permutation.

Parameters: 2p³ = 2T^{3/2}  (vs T² for dense).

References:
- Monarch Mixer (Dao et al., 2023): Sub-quadratic GEMM-based architecture
- MonarchAttention (Yaras et al., 2025): Zero-shot attention → Monarch conversion
"""

# ─────────────────────────────────────────────
# MonarchMatrix — factored T×T mixing matrix
# ─────────────────────────────────────────────

struct MonarchMatrix <: Lux.AbstractLuxLayer
    seq_len::Int   # T = p²
    p::Int         # √T — block size and block count
end

function MonarchMatrix(seq_len::Int)
    p = isqrt(seq_len)
    @assert p * p == seq_len "Monarch requires perfect-square seq_len, got $seq_len (√$seq_len ≈ $(sqrt(seq_len)))"
    return MonarchMatrix(seq_len, p)
end

function Lux.initialparameters(rng::AbstractRNG, l::MonarchMatrix)
    p = l.p
    scale = Float32(sqrt(2.0 / (p + p)))
    return (
        L1 = randn(rng, Float32, p, p, p) .* scale,
        L2 = randn(rng, Float32, p, p, p) .* scale,
    )
end

function Lux.initialstates(::AbstractRNG, l::MonarchMatrix)
    T = l.seq_len
    # Identity matrix stored in state — auto-moves to GPU with Lux.setup
    I_T = Float32[i == j ? 1.0f0 : 0.0f0 for i in 1:T, j in 1:T]
    return (identity_matrix = I_T,)
end

Lux.parameterlength(l::MonarchMatrix) = 2 * l.p^3
Lux.statelength(l::MonarchMatrix) = l.seq_len^2

"""
    realize(l::MonarchMatrix, ps, st) -> Matrix{Float32}

Materialize the full T×T Monarch matrix by pushing the identity through
the factored L2 → permute → L1 → permute pipeline.
All ops are non-mutating → Zygote-differentiable w.r.t. ps.L1, ps.L2.
"""
function realize(l::MonarchMatrix, ps, st)
    p = l.p
    T = l.seq_len

    # Identity is a constant (no gradient), but lives on the correct device
    I_T = st.identity_matrix  # (T, T)

    # Reshape columns of identity: (T, T) → (p, p, T)
    x = reshape(I_T, p, p, T)

    # ── Apply L2 block-diagonal ──
    # L2: (p, p, p)  — p blocks of (p, p)
    # x: (p, p, T)   — want: for each block k, multiply L2[:,:,k] @ x[:,k,:]
    # Rearrange for NNlib.batched_mul: need (p, T, p) so batch dim = p
    x = permutedims(x, (1, 3, 2))         # (p, T, p)
    x = NNlib.batched_mul(ps.L2, x)       # (p, p, p) × (p, T, p) → (p, T, p)
    x = permutedims(x, (1, 3, 2))         # back to (p, p, T)

    # ── Permutation P: transpose the p×p grid ──
    x = permutedims(x, (2, 1, 3))         # swap dims 1,2

    # ── Apply L1 block-diagonal ──
    x = permutedims(x, (1, 3, 2))         # (p, T, p)
    x = NNlib.batched_mul(ps.L1, x)       # (p, T, p)
    x = permutedims(x, (1, 3, 2))         # (p, p, T)

    # ── Undo permutation ──
    x = permutedims(x, (2, 1, 3))

    return reshape(x, T, T)
end


# ─────────────────────────────────────────────
# CausalDepthwiseConv1d — local causal context
# ─────────────────────────────────────────────

struct CausalDepthwiseConv1d <: Lux.AbstractLuxLayer
    channels::Int
    kernel_size::Int
end

function Lux.initialparameters(rng::AbstractRNG, l::CausalDepthwiseConv1d)
    scale = Float32(sqrt(1.0 / l.kernel_size))
    return (kernel = randn(rng, Float32, l.kernel_size, l.channels) .* scale,)
end

Lux.initialstates(::AbstractRNG, ::CausalDepthwiseConv1d) = NamedTuple()
Lux.parameterlength(l::CausalDepthwiseConv1d) = l.kernel_size * l.channels
Lux.statelength(::CausalDepthwiseConv1d) = 0

function (l::CausalDepthwiseConv1d)(x, ps, st)
    # x: (D, T, B)
    D, T, B = size(x)
    K = l.kernel_size

    # Causal pad: K-1 zeros on the left of the sequence dimension
    # Must use fill! inside @ignore to avoid NaN * 0 = NaN from uninitialized memory
    pad = Zygote.@ignore begin
        p = similar(x, D, K - 1, B)
        fill!(p, zero(eltype(x)))
        p
    end
    x_padded = cat(pad, x; dims=2)  # (D, T+K-1, B)

    # Sum over kernel taps — non-mutating for Zygote
    out = sum(1:K) do k
        reshape(ps.kernel[k:k, :], D, 1, 1) .* x_padded[:, k:k+T-1, :]
    end

    return out, st
end


# ─────────────────────────────────────────────
# LearnedGate — element-wise sigmoid gating
# ─────────────────────────────────────────────

struct LearnedGate <: Lux.AbstractLuxLayer
    dim::Int
end

function Lux.initialparameters(rng::AbstractRNG, l::LearnedGate)
    # zeros → sigmoid(0) = 0.5 — conservative initial gating
    return (weight = zeros(Float32, l.dim),)
end

Lux.initialstates(::AbstractRNG, ::LearnedGate) = NamedTuple()
Lux.parameterlength(l::LearnedGate) = l.dim
Lux.statelength(::LearnedGate) = 0

function (l::LearnedGate)(x, ps, st)
    # x: (D, T, B), gate: sigmoid(weight) broadcast over T and B
    g = NNlib.sigmoid_fast.(ps.weight)  # (D,)
    return g .* x, st
end


# ─────────────────────────────────────────────
# MonarchSequenceMixer — replaces CausalSelfAttention
# ─────────────────────────────────────────────

struct MonarchSequenceMixer <: Lux.AbstractLuxContainerLayer{(:conv, :monarchs, :gate)}
    conv::CausalDepthwiseConv1d
    monarchs::NamedTuple
    gate::LearnedGate
    embed_dim::Int
    seq_len::Int
    n_heads::Int
end

function MonarchSequenceMixer(embed_dim::Int, seq_len::Int, n_heads::Int;
                               conv_kernel::Int=4)
    @assert embed_dim % n_heads == 0 "embed_dim=$embed_dim must be divisible by n_heads=$n_heads"

    conv = CausalDepthwiseConv1d(embed_dim, conv_kernel)

    monarch_layers = [MonarchMatrix(seq_len) for _ in 1:n_heads]
    monarch_names = Tuple(Symbol("head_$i") for i in 1:n_heads)
    monarchs = NamedTuple{monarch_names}(Tuple(monarch_layers))

    gate = LearnedGate(embed_dim)

    return MonarchSequenceMixer(conv, monarchs, gate, embed_dim, seq_len, n_heads)
end

function (l::MonarchSequenceMixer)(x, ps, st; mask=nothing)
    D, T, B = size(x)
    H = l.n_heads
    HD = D ÷ H  # channels per head

    # 1. Causal depthwise conv for local context
    conv_out, st_conv = l.conv(x, ps.conv, st.conv)

    # 2. Multi-head Monarch mixing for global context
    head_names = keys(l.monarchs)

    monarch_out_slices = map(1:H) do i
        name = head_names[i]
        monarch_layer = l.monarchs[name]
        ps_m = ps.monarchs[name]
        st_m = st.monarchs[name]

        # Realize full T_max × T_max Monarch matrix (differentiable w.r.t. L1, L2)
        M = realize(monarch_layer, ps_m, st_m)  # (T_max, T_max)

        # Apply causal mask (0/1 multiplicative)
        if mask !== nothing
            M_causal = M .* mask
        else
            M_causal = M
        end

        # Slice to actual sequence length T (handles generation where T < T_max)
        M_causal = M_causal[1:T, 1:T]

        # Extract this head's channel slice: (HD, T, B)
        ch_start = (i - 1) * HD + 1
        ch_end = i * HD
        x_slice = x[ch_start:ch_end, :, :]

        # Matmul: (T, T) × (T, HD*B) → (T, HD*B)
        x_flat = reshape(permutedims(x_slice, (2, 1, 3)), T, HD * B)
        y_flat = M_causal * x_flat

        # Reshape back: (T, HD*B) → (T, HD, B) → (HD, T, B)
        permutedims(reshape(y_flat, T, HD, B), (2, 1, 3))
    end

    # Concatenate heads back along channel dimension
    monarch_out = cat(monarch_out_slices...; dims=1)

    # 3. Combine conv (local) + Monarch (global), then gate
    combined = conv_out .+ monarch_out
    gated, st_gate = l.gate(combined, ps.gate, st.gate)

    # Collect states
    monarch_state_pairs = map(head_names) do name
        name => st.monarchs[name]
    end
    st_monarchs = NamedTuple{head_names}(Tuple(last.(monarch_state_pairs)))
    new_st = (conv=st_conv, monarchs=st_monarchs, gate=st_gate)

    return gated, new_st
end


# ─────────────────────────────────────────────
# MonarchBlock — replaces TransformerBlock
# ─────────────────────────────────────────────

struct MonarchBlock <: Lux.AbstractLuxContainerLayer{(:seq_mixer, :ffn, :ln1, :ln2)}
    seq_mixer::MonarchSequenceMixer
    ffn::SwiGLU
    ln1::RMSNorm
    ln2::RMSNorm
end

function MonarchBlock(embed_dim::Int, seq_len::Int, n_heads::Int;
                      ffn_mult::Int=4, conv_kernel::Int=4)
    hidden_dim = embed_dim * ffn_mult
    # SwiGLU 2/3 adjustment (same as TransformerBlock)
    hidden_dim = Int(round(2 * hidden_dim / 3))
    hidden_dim = max(64, (hidden_dim ÷ 64) * 64)

    return MonarchBlock(
        MonarchSequenceMixer(embed_dim, seq_len, n_heads; conv_kernel),
        SwiGLU(embed_dim, hidden_dim),
        RMSNorm(embed_dim),
        RMSNorm(embed_dim),
    )
end

function (block::MonarchBlock)(x, ps, st; mask=nothing, rope_cos=nothing, rope_sin=nothing)
    # rope_cos/rope_sin accepted for interface compat with TransformerBlock, ignored

    # Pre-norm sequence mixing with residual
    normed, st_ln1 = block.ln1(x, ps.ln1, st.ln1)
    mixed, st_mixer = block.seq_mixer(normed, ps.seq_mixer, st.seq_mixer; mask)
    h = x .+ mixed

    # Pre-norm FFN with residual
    normed2, st_ln2 = block.ln2(h, ps.ln2, st.ln2)
    ffn_out, st_ffn = block.ffn(normed2, ps.ffn, st.ffn)
    out = h .+ ffn_out

    new_st = (seq_mixer=st_mixer, ffn=st_ffn, ln1=st_ln1, ln2=st_ln2)
    return out, new_st
end
