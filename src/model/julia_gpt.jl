"""
JuliaGPT — Decoder-only transformer in Lux.jl.
Follows nanoGPT/nanochat architecture with RoPE, RMSNorm, SwiGLU.
"""

# ─────────────────────────────────────────────
# Rotary Positional Embedding
# ─────────────────────────────────────────────

struct RotaryEmbedding <: Lux.AbstractLuxLayer
    dim::Int
    max_seq_len::Int
end

Lux.initialparameters(::AbstractRNG, ::RotaryEmbedding) = NamedTuple()

function Lux.initialstates(::AbstractRNG, l::RotaryEmbedding)
    half = l.dim ÷ 2
    # Frequency bands: theta_i = 1 / 10000^(2i/dim)
    freqs = Float32.(1.0 ./ (10000.0 .^ ((0:2:(l.dim-1)) ./ l.dim)))  # (half,)
    positions = Float32.(0:(l.max_seq_len-1))  # (max_seq_len,)
    # Outer product: (half, max_seq_len)
    angles = freqs * positions'
    return (cos_cache = cos.(angles), sin_cache = sin.(angles))
end

Lux.parameterlength(::RotaryEmbedding) = 0
Lux.statelength(l::RotaryEmbedding) = 2 * (l.dim ÷ 2) * l.max_seq_len

function apply_rotary_emb(x, cos_cache, sin_cache, seq_len)
    # x: (head_dim, seq_len, n_heads, batch)
    half = size(x, 1) ÷ 2
    x1 = x[1:half, :, :, :]
    x2 = x[half+1:end, :, :, :]

    # cos/sin caches: (half, max_seq_len) → slice to (half, seq_len)
    c = cos_cache[:, 1:seq_len]
    s = sin_cache[:, 1:seq_len]

    # Broadcast over heads and batch dims
    o1 = x1 .* c .- x2 .* s
    o2 = x1 .* s .+ x2 .* c
    return vcat(o1, o2)
end

# ─────────────────────────────────────────────
# Multi-Head Causal Self-Attention
# ─────────────────────────────────────────────

struct CausalSelfAttention <: Lux.AbstractLuxLayer
    embed_dim::Int
    n_heads::Int
    head_dim::Int
    dropout::Float32
end

function CausalSelfAttention(embed_dim::Int, n_heads::Int, head_dim::Int; dropout::Float32=0.0f0)
    return CausalSelfAttention(embed_dim, n_heads, head_dim, dropout)
end

function Lux.initialparameters(rng::AbstractRNG, l::CausalSelfAttention)
    total_dim = l.n_heads * l.head_dim
    scale = Float32(sqrt(2.0 / (l.embed_dim + total_dim)))
    return (
        wq = randn(rng, Float32, total_dim, l.embed_dim) .* scale,
        wk = randn(rng, Float32, total_dim, l.embed_dim) .* scale,
        wv = randn(rng, Float32, total_dim, l.embed_dim) .* scale,
        wo = randn(rng, Float32, l.embed_dim, total_dim) .* scale,
    )
end

Lux.initialstates(::AbstractRNG, ::CausalSelfAttention) = NamedTuple()
Lux.parameterlength(l::CausalSelfAttention) = 4 * l.embed_dim * l.n_heads * l.head_dim
Lux.statelength(::CausalSelfAttention) = 0

function (l::CausalSelfAttention)(x, ps, st; rope_cos=nothing, rope_sin=nothing, mask=nothing)
    D, T, B = size(x)
    H = l.n_heads
    HD = l.head_dim

    # Project to Q, K, V — flatten x to (D, T*B) for matmul
    x_flat = reshape(x, D, :)
    q = reshape(ps.wq * x_flat, HD, T, H, B)  # (head_dim, seq_len, n_heads, batch)
    k = reshape(ps.wk * x_flat, HD, T, H, B)
    v = reshape(ps.wv * x_flat, HD, T, H, B)

    # Apply RoPE if provided
    if rope_cos !== nothing && rope_sin !== nothing
        q = apply_rotary_emb(q, rope_cos, rope_sin, T)
        k = apply_rotary_emb(k, rope_cos, rope_sin, T)
    end

    # Attention scores: q^T k / sqrt(head_dim)
    # Reshape for batched matmul: treat (n_heads, batch) as batch dims
    scale = Float32(1.0 / sqrt(Float64(HD)))
    q_perm = permutedims(q, (1, 2, 3, 4))  # (HD, T, H, B)
    k_perm = permutedims(k, (1, 2, 3, 4))

    # Compute attention: (T, T, H, B)
    # scores[i,j,h,b] = sum_d q[d,i,h,b] * k[d,j,h,b] / sqrt(HD)
    q_r = reshape(q_perm, HD, T, H * B)  # (HD, T, H*B)
    k_r = reshape(k_perm, HD, T, H * B)  # (HD, T, H*B)
    attn = NNlib.batched_mul(permutedims(q_r, (2, 1, 3)), k_r)  # (T, T, H*B)
    attn = attn .* scale

    # Apply causal mask
    if mask !== nothing
        attn = attn .+ mask
    end

    # Softmax over key dimension (dim 2)
    attn = NNlib.softmax(attn; dims=2)

    # Apply attention to values: (HD, T, H*B)
    v_r = reshape(v, HD, T, H * B)
    out = NNlib.batched_mul(v_r, permutedims(attn, (2, 1, 3)))  # (HD, T, H*B)

    # Reshape back and project output
    out = reshape(out, HD * H, T, B)  # (total_dim, T, B)
    out_flat = reshape(out, HD * H, :)
    result = reshape(ps.wo * out_flat, D, T, B)

    return result, st
end

# ─────────────────────────────────────────────
# Transformer Block
# ─────────────────────────────────────────────

struct TransformerBlock <: Lux.AbstractLuxContainerLayer{(:attn, :ffn, :ln1, :ln2)}
    attn::CausalSelfAttention
    ffn::SwiGLU
    ln1::RMSNorm
    ln2::RMSNorm
end

function TransformerBlock(embed_dim::Int, n_heads::Int, head_dim::Int;
                          ffn_mult::Int=4, dropout::Float32=0.0f0)
    hidden_dim = embed_dim * ffn_mult
    # Adjust hidden_dim for SwiGLU (2/3 of 4x to match param count of standard FFN)
    hidden_dim = Int(round(2 * hidden_dim / 3))
    # Round to multiple of 64 for efficiency
    hidden_dim = max(64, (hidden_dim ÷ 64) * 64)

    return TransformerBlock(
        CausalSelfAttention(embed_dim, n_heads, head_dim; dropout),
        SwiGLU(embed_dim, hidden_dim),
        RMSNorm(embed_dim),
        RMSNorm(embed_dim),
    )
end

function (block::TransformerBlock)(x, ps, st; rope_cos=nothing, rope_sin=nothing, mask=nothing)
    # Pre-norm attention with residual
    normed, st_ln1 = block.ln1(x, ps.ln1, st.ln1)
    attn_out, st_attn = block.attn(normed, ps.attn, st.attn;
                                    rope_cos, rope_sin, mask)
    h = x .+ attn_out

    # Pre-norm FFN with residual
    normed2, st_ln2 = block.ln2(h, ps.ln2, st.ln2)
    ffn_out, st_ffn = block.ffn(normed2, ps.ffn, st.ffn)
    out = h .+ ffn_out

    new_st = (attn=st_attn, ffn=st_ffn, ln1=st_ln1, ln2=st_ln2)
    return out, new_st
end

# ─────────────────────────────────────────────
# Full Model
# ─────────────────────────────────────────────

struct JuliaGPTModel <: Lux.AbstractLuxContainerLayer{(:tok_emb, :rope, :blocks, :ln_f, :head)}
    tok_emb::Lux.Embedding
    rope::RotaryEmbedding
    blocks::NamedTuple
    ln_f::RMSNorm
    head::Lux.Dense
    config::ModelConfig
end

"""
    create_model(config::ModelConfig) -> JuliaGPTModel

Build a JuliaGPT model from a configuration struct.
"""
function create_model(config::ModelConfig)
    tok_emb = Lux.Embedding(config.vocab_size => config.embed_dim)
    rope = RotaryEmbedding(config.head_dim, config.context_length)

    block_layers = [
        TransformerBlock(config.embed_dim, config.n_heads, config.head_dim;
                         ffn_mult=config.ffn_mult, dropout=config.dropout)
        for _ in 1:config.n_layers
    ]
    # Store blocks as NamedTuple for Lux container compatibility
    block_names = Tuple(Symbol("block_$i") for i in 1:config.n_layers)
    blocks = NamedTuple{block_names}(Tuple(block_layers))

    ln_f = RMSNorm(config.embed_dim)
    head = Lux.Dense(config.embed_dim => config.vocab_size; use_bias=config.bias)

    return JuliaGPTModel(tok_emb, rope, blocks, ln_f, head, config)
end

function (model::JuliaGPTModel)(x, ps, st)
    T = size(x, 1)  # (seq_len, batch)

    # Token embedding: (seq_len, batch) → (embed_dim, seq_len, batch)
    h, st_emb = model.tok_emb(x, ps.tok_emb, st.tok_emb)

    # Get RoPE caches from state
    rope_cos = st.rope.cos_cache
    rope_sin = st.rope.sin_cache
    st_rope = st.rope

    # Causal mask for this sequence length — not differentiable (constant)
    mask = Zygote.@ignore make_causal_mask(T; dtype=eltype(h))
    mask = Zygote.@ignore _to_device(h, mask)

    # Transformer blocks
    block_states = Dict{Symbol, Any}()
    for (name, block) in pairs(model.blocks)
        h, new_st = block(h, getproperty(ps.blocks, name),
                          getproperty(st.blocks, name);
                          rope_cos, rope_sin, mask)
        block_states[name] = new_st
    end
    st_blocks = NamedTuple{keys(model.blocks)}(Tuple(block_states[k] for k in keys(model.blocks)))

    # Final norm
    h, st_ln = model.ln_f(h, ps.ln_f, st.ln_f)

    # Project to vocab logits: (embed_dim, seq_len, batch) → (vocab_size, seq_len, batch)
    logits, st_head = model.head(h, ps.head, st.head)

    new_st = (tok_emb=st_emb, rope=st_rope, blocks=st_blocks, ln_f=st_ln, head=st_head)
    return logits, new_st
end

"""
    _to_device(reference, x)

Move array `x` to the same device as `reference`.
On CPU this is a no-op; on GPU it converts to CuArray.
"""
_to_device(::AbstractArray, x::AbstractArray) = x  # CPU → CPU: no-op
# GPU dispatch will be added by CUDA extension when available

"""
    count_parameters(ps) -> Int

Count total number of trainable parameters in a parameter tree.
"""
function count_parameters(ps)
    n = 0
    for x in fleaves(ps)
        if x isa AbstractArray
            n += length(x)
        end
    end
    return n
end

function Base.show(io::IO, model::JuliaGPTModel)
    cfg = model.config
    print(io, "JuliaGPTModel($(cfg.arch), $(cfg.vocab_size)→$(cfg.embed_dim)d, $(cfg.n_layers)L, $(cfg.n_heads)H, ctx=$(cfg.context_length))")
end
