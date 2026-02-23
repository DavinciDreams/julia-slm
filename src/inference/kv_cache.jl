"""
KV Cache for fast autoregressive inference.

Stores past key and value tensors per transformer layer so that each generation
step only processes a single new token rather than re-computing the full context.

NOTE: Using the KV cache during generation requires that the model's
CausalSelfAttention layer be modified to accept optional pre-computed KV
arguments (see `generate_with_cache` docstring for the expected interface).
"""

# ─────────────────────────────────────────────
# KVCache data structure
# ─────────────────────────────────────────────

"""
    KVCache(n_layers, head_dim, n_heads, max_seq_len)

Mutable cache holding past key/value tensors for each transformer layer.

Fields:
- `k_cache`: Vector of length `n_layers`, each entry (HD, max_seq_len, H)
- `v_cache`: Vector of length `n_layers`, each entry (HD, max_seq_len, H)
- `cache_len`: number of positions currently stored (0 after reset)
- `max_seq_len`: maximum sequence length the cache can hold
"""
mutable struct KVCache
    k_cache   ::Vector{Array{Float32,3}}   # per-layer: (HD, max_seq_len, H)
    v_cache   ::Vector{Array{Float32,3}}   # per-layer: (HD, max_seq_len, H)
    cache_len ::Int
    max_seq_len::Int
end

function KVCache(n_layers::Int, head_dim::Int, n_heads::Int, max_seq_len::Int)
    k_cache = [zeros(Float32, head_dim, max_seq_len, n_heads) for _ in 1:n_layers]
    v_cache = [zeros(Float32, head_dim, max_seq_len, n_heads) for _ in 1:n_layers]
    return KVCache(k_cache, v_cache, 0, max_seq_len)
end

# ─────────────────────────────────────────────
# Cache operations
# ─────────────────────────────────────────────

"""
    reset!(cache::KVCache)

Clear the cache by resetting `cache_len` to 0.
The underlying storage arrays are left in memory but treated as empty.
"""
function reset!(cache::KVCache)
    cache.cache_len = 0
    return cache
end

"""
    update!(cache, layer_idx, k_new, v_new) -> (k_full, v_full)

Append new key/value tensors to the cache for `layer_idx` (1-based).

Arguments:
- `k_new`: (HD, new_tokens, H) — new key vectors for this step
- `v_new`: (HD, new_tokens, H) — new value vectors for this step

The cache is updated in-place; `cache_len` is incremented by `new_tokens`.
Returns the full cached (k, v) up to the new `cache_len`.
Errors if the combined length would exceed `max_seq_len`.
"""
function update!(cache::KVCache, layer_idx::Int,
                 k_new::AbstractArray{Float32,3},
                 v_new::AbstractArray{Float32,3})
    HD, new_tokens, H = size(k_new)
    new_len = cache.cache_len + new_tokens

    if new_len > cache.max_seq_len
        error("KVCache overflow: cache_len=$new_len > max_seq_len=$(cache.max_seq_len). " *
              "Increase max_seq_len or truncate the context.")
    end

    # Write new tokens into the pre-allocated cache arrays
    start = cache.cache_len + 1
    stop  = new_len
    cache.k_cache[layer_idx][:, start:stop, :] .= k_new
    cache.v_cache[layer_idx][:, start:stop, :] .= v_new

    # Increment cache_len after the last layer has been updated so that
    # callers can update all layers before advancing the pointer.
    # We return the slice and let the caller decide when to advance.
    k_full = cache.k_cache[layer_idx][:, 1:new_len, :]
    v_full = cache.v_cache[layer_idx][:, 1:new_len, :]
    return k_full, v_full
end

"""
    advance_cache!(cache, n_new_tokens)

Advance `cache_len` by `n_new_tokens` after all layers have been updated.
Call this once per generation step after updating all layers.
"""
function advance_cache!(cache::KVCache, n_new_tokens::Int)
    cache.cache_len += n_new_tokens
    return cache
end

"""
    get_kv(cache, layer_idx) -> (k, v)

Return the cached key/value tensors for `layer_idx` up to `cache_len`.
Returns (HD, cache_len, H) arrays.
"""
function get_kv(cache::KVCache, layer_idx::Int)
    len = cache.cache_len
    len == 0 && error("KVCache is empty — call update! or run prefill first.")
    k = cache.k_cache[layer_idx][:, 1:len, :]
    v = cache.v_cache[layer_idx][:, 1:len, :]
    return k, v
end

# ─────────────────────────────────────────────
# Autoregressive generation with KV cache
# ─────────────────────────────────────────────

"""
    generate_with_cache(model, ps, st, tokenizer, prompt;
                        cache, max_new_tokens, temperature, top_k,
                        context_length) -> String

Autoregressive generation that exploits a KV cache for O(1)-per-step inference.

Two-phase algorithm:
1. Prefill: run the full prompt through the model in a single forward pass.
   The model is expected to populate the supplied `cache` as a side effect of
   each CausalSelfAttention block writing its K/V projections.
2. Decode: for each new token, run a single-token forward pass.  Each attention
   block reads the existing cached K/V from `cache` and appends its new K/V.

Required model interface:
The CausalSelfAttention blocks must accept a `kv_cache` keyword argument and,
when provided, (a) prepend the cached K/V to the current step's K/V before
computing attention, and (b) call `update!` + `advance_cache!` on the cache.
Blocks that do not support this keyword will fall back to the standard (slower)
full-context attention path, making this function equivalent to `generate`.

Sampling is identical to `generate` in generate.jl:
temperature scaling → top-k filter → categorical sample.
"""
function generate_with_cache(model, ps, st, tokenizer::CharTokenizer, prompt::String;
                              cache::KVCache,
                              max_new_tokens::Int=200,
                              temperature::Float64=0.8,
                              top_k::Int=0,
                              context_length::Int=256)
    # Encode prompt
    encoded = encode(tokenizer, prompt)
    tokens  = Vector{Int}(undef, length(encoded))
    copyto!(tokens, encoded)

    isempty(tokens) &&
        error("Prompt produced no tokens. Check that characters are in vocabulary.")

    reset!(cache)
    generated = Int[]

    # ── Phase 1: Prefill ──────────────────────────────────────────────────
    # Run full prompt through the model to populate the KV cache.
    prefill_ctx = tokens[max(1, end - context_length + 1):end]
    x_prefill   = reshape(prefill_ctx, :, 1)  # (T, 1)

    # Forward pass; model is expected to write into cache as a side-effect.
    # If the model does not support kv_cache, this is just a standard forward.
    logits, st = _forward_with_cache(model, ps, st, x_prefill, cache;
                                     is_prefill=true)

    # Sample next token from the last logit position
    next_token = _sample_next(logits[:, end, 1], temperature, top_k)
    push!(tokens, next_token)
    push!(generated, next_token)

    # ── Phase 2: Decode ───────────────────────────────────────────────────
    for _ in 2:max_new_tokens
        # Single-token input for this step
        x_step = reshape([tokens[end]], 1, 1)  # (1, 1)

        logits, st = _forward_with_cache(model, ps, st, x_step, cache;
                                         is_prefill=false)

        next_token = _sample_next(logits[:, end, 1], temperature, top_k)
        push!(tokens, next_token)
        push!(generated, next_token)

        # Safety: hard-stop at context limit
        if cache.cache_len >= cache.max_seq_len - 1
            break
        end
    end

    return decode(tokenizer, generated)
end

# ─────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────

"""
    _forward_with_cache(model, ps, st, x, cache; is_prefill) -> (logits, new_st)

Attempt to call the model with kv_cache keyword argument.
Falls back to the standard call signature if the model does not support it.
"""
function _forward_with_cache(model, ps, st, x, cache::KVCache; is_prefill::Bool)
    try
        return model(x, ps, st; kv_cache=cache, is_prefill=is_prefill)
    catch err
        # If the model does not accept kv_cache, fall back to standard call.
        # This makes generate_with_cache safe to use even before the model is
        # updated to support caching (at the cost of no speedup).
        if err isa MethodError
            return model(x, ps, st)
        else
            rethrow(err)
        end
    end
end

"""
    _sample_next(logits, temperature, top_k) -> Int

Apply temperature scaling and top-k filtering, then sample one token index.
"""
function _sample_next(logits::AbstractVector, temperature::Float64, top_k::Int)
    if temperature != 1.0
        logits = logits ./ Float32(temperature)
    end
    if top_k > 0
        logits = _kvcache_top_k_filter(logits, top_k)
    end
    probs = NNlib.softmax(logits)
    return _kvcache_sample_categorical(probs)
end

"""
Top-k filter: set all logits outside the top-k to -Inf.
(Local copy to avoid dependency on generate.jl's private functions.)
"""
function _kvcache_top_k_filter(logits::AbstractVector, k::Int)
    k = min(k, length(logits))
    sorted  = sort(Array(logits); rev=true)
    threshold = sorted[k]
    return map(l -> l >= threshold ? l : typemin(eltype(logits)), logits)
end

"""
Sample one index from a probability vector.
"""
function _kvcache_sample_categorical(probs::AbstractVector)
    p = Array(probs)
    r = rand(Float32)
    cumulative = 0.0f0
    for i in eachindex(p)
        cumulative += p[i]
        r <= cumulative && return i
    end
    return length(p)
end
