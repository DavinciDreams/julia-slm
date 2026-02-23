"""
Chunked (memory-efficient) attention computation.

Processes queries in fixed-size blocks to reduce peak memory usage from O(T^2)
to O(T * chunk_size). Compatible with the existing CausalSelfAttention convention
used in julia_gpt.jl: tensors are shaped (HD, T, H*B) where HD = head_dim,
T = sequence length, H = n_heads, B = batch size.
"""

# ─────────────────────────────────────────────
# Chunked Attention
# ─────────────────────────────────────────────

"""
    chunked_attention(q, k, v, mask; chunk_size=64) -> Array{Float32,3}

Memory-efficient attention that processes queries in blocks of `chunk_size`.

Arguments:
- `q`:    (HD, T, H*B) — queries
- `k`:    (HD, T, H*B) — keys
- `v`:    (HD, T, H*B) — values
- `mask`: (T, T) or `nothing` — additive causal mask (0 / -Inf entries)
- `chunk_size`: number of query positions to process per block

Returns:
- output of shape (HD, T, H*B), same as q

Memory behaviour:
Each chunk only materialises a (chunk_size × T × H*B) score tensor rather than
the full (T × T × H*B) tensor, giving a ~(T / chunk_size) reduction in peak
activation memory during the forward pass.

Drop-in replacement for the inner attention computation in CausalSelfAttention.
"""
function chunked_attention(q::AbstractArray{T,3},
                           k::AbstractArray{T,3},
                           v::AbstractArray{T,3},
                           mask;
                           chunk_size::Int=64) where {T<:AbstractFloat}
    HD, seq_len, HB = size(q)
    scale = T(1.0 / sqrt(Float64(HD)))

    # Pre-allocate output — same shape as q
    out = similar(q)

    n_chunks = cld(seq_len, chunk_size)  # ceiling division

    for chunk_idx in 1:n_chunks
        q_start = (chunk_idx - 1) * chunk_size + 1
        q_end   = min(chunk_idx * chunk_size, seq_len)

        # Query slice: (HD, q_chunk, HB)
        q_chunk = q[:, q_start:q_end, :]
        q_chunk_len = q_end - q_start + 1

        # Attention scores: (q_chunk, T, HB)
        # q_chunk transposed to (q_chunk, HD, HB), then batched-mul with k (HD, T, HB)
        # -> (q_chunk, T, HB)
        scores = NNlib.batched_mul(
            permutedims(q_chunk, (2, 1, 3)),  # (q_chunk, HD, HB)
            k                                  # (HD, T, HB)
        )  # -> (q_chunk, T, HB)
        scores = scores .* scale

        # Apply mask slice if provided
        # mask is (T, T); we need rows q_start:q_end, all columns
        if mask !== nothing
            mask_slice = mask[q_start:q_end, :]  # (q_chunk, T)
            # Broadcast over batch dim (HB): reshape to (q_chunk, T, 1)
            scores = scores .+ reshape(mask_slice, q_chunk_len, seq_len, 1)
        end

        # Softmax over key dimension (dim 2 of (q_chunk, T, HB))
        attn_weights = NNlib.softmax(scores; dims=2)  # (q_chunk, T, HB)

        # Weighted sum of values: (HD, q_chunk, HB)
        # v: (HD, T, HB); attn_weights transposed: (T, q_chunk, HB)
        chunk_out = NNlib.batched_mul(
            v,                                         # (HD, T, HB)
            permutedims(attn_weights, (2, 1, 3))       # (T, q_chunk, HB)
        )  # -> (HD, q_chunk, HB)

        out[:, q_start:q_end, :] = chunk_out
    end

    return out
end

# ─────────────────────────────────────────────
# Convenience wrapper: CausalSelfAttention using chunked attention
# ─────────────────────────────────────────────

"""
    chunked_causal_attention(q_r, k_r, v_r, mask; chunk_size=64) -> Array{Float32,3}

Wrapper that applies chunked attention with a causal mask.  Inputs and outputs
follow the (HD, T, H*B) convention used in CausalSelfAttention.
"""
function chunked_causal_attention(q_r::AbstractArray{T,3},
                                  k_r::AbstractArray{T,3},
                                  v_r::AbstractArray{T,3},
                                  mask;
                                  chunk_size::Int=64) where {T<:AbstractFloat}
    return chunked_attention(q_r, k_r, v_r, mask; chunk_size=chunk_size)
end
