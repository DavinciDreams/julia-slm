"""
Autoregressive text generation with temperature, top-k, and top-p sampling.
"""

"""
    generate(model, ps, st, tokenizer, prompt;
             max_new_tokens=200, temperature=0.8,
             top_k=0, top_p=1.0, greedy=false) -> String

Generate text autoregressively from a prompt string.
"""
function generate(model, ps, st, tokenizer::CharTokenizer, prompt::String;
                  max_new_tokens::Int=200,
                  temperature::Float64=0.8,
                  top_k::Int=0,
                  top_p::Float64=1.0,
                  greedy::Bool=false,
                  context_length::Int=256)
    # Encode prompt — allocate a fully independent mutable vector
    encoded = encode(tokenizer, prompt)
    tokens = Vector{Int}(undef, length(encoded))
    copyto!(tokens, encoded)

    if isempty(tokens)
        error("Prompt produced no tokens. Check that characters are in vocabulary.")
    end

    generated = Int[]

    for _ in 1:max_new_tokens
        # Crop to context window (copy to avoid reshape sharing memory with tokens)
        ctx = if length(tokens) > context_length
            tokens[end-context_length+1:end]
        else
            copy(tokens)
        end

        # Forward pass — input shape: (seq_len, 1) for single sequence
        x = reshape(ctx, :, 1)
        logits, _ = model(x, ps, st)

        # Get logits for the last position: (vocab_size,)
        next_logits = logits[:, end, 1]

        # Apply temperature
        if !greedy && temperature != 1.0
            next_logits = next_logits ./ Float32(temperature)
        end

        if greedy
            # Argmax
            next_token = argmax(next_logits)
        else
            # Apply top-k filtering
            if top_k > 0
                next_logits = _top_k_filter(next_logits, top_k)
            end

            # Apply top-p (nucleus) filtering
            if top_p < 1.0
                next_logits = _top_p_filter(next_logits, top_p)
            end

            # Sample from distribution
            probs = NNlib.softmax(next_logits)
            next_token = _sample_categorical(probs)
        end

        push!(tokens, next_token)
        push!(generated, next_token)
    end

    return decode(tokenizer, generated)
end

"""
Top-k filtering: set all logits outside top-k to -Inf.
"""
function _top_k_filter(logits::AbstractVector, k::Int)
    k = min(k, length(logits))
    # Find the k-th largest value
    sorted = sort(Array(logits); rev=true)
    threshold = sorted[k]
    # Mask values below threshold
    return map(l -> l >= threshold ? l : typemin(eltype(logits)), logits)
end

"""
Top-p (nucleus) filtering: keep smallest set of tokens with cumulative prob >= p.
"""
function _top_p_filter(logits::AbstractVector, p::Float64)
    sorted_indices = sortperm(Array(logits); rev=true)
    sorted_logits = logits[sorted_indices]
    probs = NNlib.softmax(sorted_logits)
    cumprobs = cumsum(Array(probs))

    # Find cutoff index
    cutoff = findfirst(>=(p), cumprobs)
    cutoff = isnothing(cutoff) ? length(probs) : cutoff

    # Mask everything after cutoff
    result = fill(typemin(eltype(logits)), length(logits))
    for i in 1:cutoff
        result[sorted_indices[i]] = logits[sorted_indices[i]]
    end
    return result
end

"""
Sample one index from a categorical distribution defined by probability vector.
"""
function _sample_categorical(probs::AbstractVector)
    p = Array(probs)  # move to CPU if on GPU
    r = rand(Float32)
    cumulative = 0.0f0
    for i in eachindex(p)
        cumulative += p[i]
        if r <= cumulative
            return i
        end
    end
    return length(p)  # fallback
end
