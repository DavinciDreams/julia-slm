"""
Sparse Mixture of Experts (MoE) layer with top-k routing and load-balance loss.
Implemented as a Lux layer with explicit parameter/state management.
"""

# ─────────────────────────────────────────────
# SparseMoE Layer
# ─────────────────────────────────────────────

"""
    SparseMoE(in_dim, hidden_dim; n_experts=8, top_k=2)

Sparse Mixture of Experts feed-forward layer.

Each token is routed to `top_k` of `n_experts` experts. Each expert is a
SwiGLU-style FFN (W1, V, W2). The gate network produces a distribution over
experts; the top-k experts process each token and their outputs are combined
using the gate probabilities as weights.

An auxiliary load-balance loss is tracked in state:
    load_balance_loss = n_experts * dot(f, P)
where `f` is the fraction of tokens routed to each expert (non-differentiable,
computed with Zygote.@ignore) and `P` is the mean gate softmax probability over
the batch (differentiable, encourages uniform routing).
"""
struct SparseMoE <: Lux.AbstractLuxLayer
    n_experts::Int
    top_k::Int
    in_dim::Int
    hidden_dim::Int
end

SparseMoE(in_dim::Int, hidden_dim::Int; n_experts::Int=8, top_k::Int=2) =
    SparseMoE(n_experts, top_k, in_dim, hidden_dim)

function Lux.initialparameters(rng::AbstractRNG, l::SparseMoE)
    scale_gate = Float32(sqrt(2.0 / (l.in_dim + l.n_experts)))
    scale_ffn  = Float32(sqrt(2.0 / (l.in_dim + l.hidden_dim)))

    # Gate: (n_experts, in_dim)
    gate = randn(rng, Float32, l.n_experts, l.in_dim) .* scale_gate

    # Per-expert SwiGLU weights stacked as 3D arrays: last dim = expert index
    # w1: (hidden_dim, in_dim, n_experts)  — up-projection (gating branch)
    # v:  (hidden_dim, in_dim, n_experts)  — up-projection (value branch)
    # w2: (in_dim, hidden_dim, n_experts)  — down-projection
    w1 = randn(rng, Float32, l.hidden_dim, l.in_dim, l.n_experts) .* scale_ffn
    v  = randn(rng, Float32, l.hidden_dim, l.in_dim, l.n_experts) .* scale_ffn
    w2 = randn(rng, Float32, l.in_dim, l.hidden_dim, l.n_experts) .* scale_ffn

    return (gate=gate, w1=w1, v=v, w2=w2)
end

function Lux.initialstates(::AbstractRNG, ::SparseMoE)
    return (load_balance_loss = 0.0f0,)
end

function Lux.parameterlength(l::SparseMoE)
    gate_params   = l.n_experts * l.in_dim
    expert_params = 3 * l.hidden_dim * l.in_dim * l.n_experts
    return gate_params + expert_params
end

Lux.statelength(::SparseMoE) = 1

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

"""
    _topk_indices(logits, k) -> Matrix{Int}

Return the indices of the top-k values in each column of `logits`.
`logits` is (n_experts, N); returns (k, N) integer index matrix.
Non-differentiable — must be called inside Zygote.@ignore.
"""
function _topk_indices(logits::AbstractMatrix, k::Int)
    n_experts, N = size(logits)
    indices = Matrix{Int}(undef, k, N)
    for i in 1:N
        col = @view logits[:, i]
        # partialsortperm returns the k largest in descending order
        indices[:, i] = partialsortperm(Array(col), 1:k; rev=true)
    end
    return indices
end

"""
    _gather_topk_probs(gate_probs, topk_idx) -> Matrix{Float32}

Gather gate probabilities at the selected top-k expert positions.
`gate_probs`: (n_experts, N) — softmax probabilities (differentiable).
`topk_idx`:   (k, N)         — integer indices (non-differentiable).
Returns (k, N) gathered probabilities, differentiable through gate_probs.
"""
function _gather_topk_probs(gate_probs::AbstractMatrix, topk_idx::Matrix{Int})
    k, N = size(topk_idx)
    # Build linear indices into the (n_experts, N) matrix
    # Column i contributes rows topk_idx[:, i]
    out = similar(gate_probs, k, N)
    for i in 1:N
        for j in 1:k
            out[j, i] = gate_probs[topk_idx[j, i], i]
        end
    end
    return out
end

# ─────────────────────────────────────────────
# Forward pass
# ─────────────────────────────────────────────

"""
Forward pass for SparseMoE.

Input x: (in_dim, T, B) — embed_dim × sequence × batch.
Output:  (in_dim, T, B) — same shape, tokens processed by their top-k experts.
State contains updated load_balance_loss scalar (auxiliary loss for training).

Algorithm:
1. Flatten tokens to (in_dim, N) where N = T*B.
2. Compute gate logits (n_experts, N) and softmax probabilities.
3. Select top-k expert indices per token (non-differentiable via @ignore).
4. Accumulate load-balance loss: n_experts * dot(f, P) where f = routing fractions.
5. For each expert e: gather its assigned tokens, run SwiGLU, scatter-add back.
6. Weight each token's contribution by normalized top-k gate probabilities.
"""
function (l::SparseMoE)(x, ps, st)
    in_dim, T, B = size(x)
    N = T * B  # total tokens in this call

    # Flatten: (in_dim, N)
    x_flat = reshape(x, in_dim, N)

    # Gate logits and probabilities: (n_experts, N)
    gate_logits = ps.gate * x_flat                         # (n_experts, N)
    gate_probs  = NNlib.softmax(gate_logits; dims=1)       # (n_experts, N)

    # Select top-k experts per token — non-differentiable routing decision
    topk_idx = Zygote.@ignore _topk_indices(gate_logits, l.top_k)  # (k, N)

    # Gather and normalize top-k gate probabilities (differentiable)
    topk_probs_raw = _gather_topk_probs(gate_probs, topk_idx)       # (k, N)
    topk_probs_sum = sum(topk_probs_raw; dims=1) .+ 1.0f-9          # (1, N)
    topk_probs     = topk_probs_raw ./ topk_probs_sum               # (k, N)

    # Load-balance auxiliary loss (computed inside @ignore for f, differentiable for P)
    # f: fraction of tokens routed to each expert (non-differentiable)
    # P: mean gate softmax probability per expert (differentiable)
    P = mean(gate_probs; dims=2)[:, 1]   # (n_experts,)

    lb_loss = Zygote.@ignore begin
        f = zeros(Float32, l.n_experts)
        for i in 1:N
            for j in 1:l.top_k
                e = topk_idx[j, i]
                f[e] += 1.0f0
            end
        end
        f ./= Float32(N * l.top_k)  # normalise to fractions
        f
    end

    # Combine differentiable P and non-differentiable f into the aux loss
    # dot(f, P) is differentiable through P; lb_loss holds the frozen f values
    load_balance_loss = Float32(l.n_experts) * sum(lb_loss .* P)

    # Accumulate expert outputs: (in_dim, N)
    out = zeros(Float32, in_dim, N)

    for e in 1:l.n_experts
        # Find which (token, slot) pairs use expert e — non-differentiable bookkeeping
        token_ids, slot_ids = Zygote.@ignore begin
            tids = Int[]
            sids = Int[]
            for i in 1:N
                for j in 1:l.top_k
                    if topk_idx[j, i] == e
                        push!(tids, i)
                        push!(sids, j)
                    end
                end
            end
            tids, sids
        end

        isempty(token_ids) && continue

        # Gather tokens assigned to expert e: (in_dim, n_e)
        x_e = x_flat[:, token_ids]  # (in_dim, n_e)

        # Expert SwiGLU forward: w1/v/w2 slices for expert e
        w1_e = ps.w1[:, :, e]   # (hidden_dim, in_dim)
        v_e  = ps.v[:, :, e]    # (hidden_dim, in_dim)
        w2_e = ps.w2[:, :, e]   # (in_dim, hidden_dim)

        gate_h = NNlib.swish.(w1_e * x_e) .* (v_e * x_e)  # (hidden_dim, n_e)
        y_e    = w2_e * gate_h                               # (in_dim, n_e)

        # Gather per-token weights for this expert: (n_e,)
        weights_e = [topk_probs[slot_ids[i], token_ids[i]] for i in eachindex(token_ids)]

        # Scatter-add weighted expert output back into out
        for (local_i, global_i) in enumerate(token_ids)
            out[:, global_i] .+= weights_e[local_i] .* y_e[:, local_i]
        end
    end

    # Reshape back to (in_dim, T, B)
    out = reshape(out, in_dim, T, B)
    new_st = (load_balance_loss = load_balance_loss,)
    return out, new_st
end
