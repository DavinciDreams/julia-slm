"""
Evaluation metrics: perplexity, BPC, n-gram diversity, repetition rate.
"""

"""
    compute_perplexity(model, ps, st, data_loader; max_batches=0) -> (loss, perplexity, bpc)

Compute average loss, perplexity, and bits-per-character over a dataset.
"""
function compute_perplexity(model, ps, st, data_loader; max_batches::Int=0)
    reset!(data_loader)
    total_loss = 0.0
    n_batches = 0
    max_b = max_batches > 0 ? max_batches : JuliaGPT.n_batches(data_loader)

    for _ in 1:max_b
        x, y = next_batch!(data_loader)
        logits, _ = model(x, ps, st)
        loss = cross_entropy_loss(logits, y)
        total_loss += Float64(loss)
        n_batches += 1
    end

    avg_loss = total_loss / max(n_batches, 1)
    perplexity = exp(avg_loss)
    bpc = avg_loss / log(2.0)  # bits per character

    return avg_loss, perplexity, bpc
end

"""
    ngram_diversity(text, n) -> Float64

Fraction of unique n-grams in the text. Higher = more diverse.
"""
function ngram_diversity(text::String, n::Int)
    chars = collect(text)
    length(chars) < n && return 1.0
    ngrams = [chars[i:i+n-1] for i in 1:length(chars)-n+1]
    return length(Set(ngrams)) / length(ngrams)
end

"""
    repetition_rate(text; n=4) -> Float64

Fraction of n-grams that appear more than once. Lower = less repetitive.
"""
function repetition_rate(text::String; n::Int=4)
    chars = collect(text)
    length(chars) < n && return 0.0
    ngrams = [chars[i:i+n-1] for i in 1:length(chars)-n+1]
    counts = Dict{Vector{Char}, Int}()
    for ng in ngrams
        counts[ng] = get(counts, ng, 0) + 1
    end
    repeated = count(v -> v > 1, values(counts))
    return repeated / length(counts)
end

"""
    evaluate(model, ps, st, tokenizer, val_loader;
             max_batches=0, gen_samples=3, gen_length=200) -> Dict

Run full evaluation: perplexity, BPC, generation quality metrics.
"""
function evaluate(model, ps, st, tokenizer, val_loader;
                  max_batches::Int=0, gen_samples::Int=3, gen_length::Int=200,
                  context_length::Int=256)
    results = Dict{String, Any}()

    # Perplexity
    loss, ppl, bpc = compute_perplexity(model, ps, st, val_loader; max_batches)
    results["val_loss"] = loss
    results["perplexity"] = ppl
    results["bpc"] = bpc

    @printf("Evaluation Results:\n")
    @printf("  Val Loss:    %.4f\n", loss)
    @printf("  Perplexity:  %.2f\n", ppl)
    @printf("  BPC:         %.4f\n", bpc)

    # Generation quality
    prompts = ["the nature of ", "it follows that ", "we must therefore "]
    all_text = ""
    for (i, prompt) in enumerate(prompts[1:min(gen_samples, length(prompts))])
        text = generate(model, ps, st, tokenizer, prompt;
                       max_new_tokens=gen_length, temperature=0.8, top_k=40,
                       context_length=context_length)
        full = prompt * text
        all_text *= full * " "
        @printf("\n  Sample %d (prompt: \"%s\"):\n  %s\n", i, prompt, full)
    end

    # Diversity metrics
    if length(all_text) > 10
        results["distinct_1"] = ngram_diversity(all_text, 1)
        results["distinct_2"] = ngram_diversity(all_text, 2)
        results["distinct_3"] = ngram_diversity(all_text, 3)
        results["repetition_rate"] = repetition_rate(all_text; n=4)

        @printf("\n  Diversity Metrics:\n")
        @printf("    Distinct-1:      %.4f\n", results["distinct_1"])
        @printf("    Distinct-2:      %.4f\n", results["distinct_2"])
        @printf("    Distinct-3:      %.4f\n", results["distinct_3"])
        @printf("    Repetition Rate: %.4f\n", results["repetition_rate"])
    end

    return results
end
