#!/usr/bin/env julia
"""
CLI entry point for text generation with a trained JuliaGPT model.

Usage:
    julia --project scripts/generate.jl --checkpoint checkpoints/final.jld2 --prompt "the nature of"
    julia --project scripts/generate.jl --checkpoint checkpoints/final.jld2 --prompt "it follows that" --temperature 0.6 --top_k 20
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

include(joinpath(@__DIR__, "..", "src", "JuliaGPT.jl"))
using .JuliaGPT
using Random

function parse_args()
    checkpoint = "checkpoints/final.jld2"
    prompt = "the nature of "
    temperature = 0.8
    top_k = 40
    top_p = 1.0
    max_tokens = 500
    greedy = false

    i = 1
    while i <= length(ARGS)
        arg = ARGS[i]
        if arg == "--checkpoint" && i < length(ARGS)
            checkpoint = ARGS[i+1]; i += 2
        elseif arg == "--prompt" && i < length(ARGS)
            prompt = ARGS[i+1]; i += 2
        elseif arg == "--temperature" && i < length(ARGS)
            temperature = parse(Float64, ARGS[i+1]); i += 2
        elseif arg == "--top_k" && i < length(ARGS)
            top_k = parse(Int, ARGS[i+1]); i += 2
        elseif arg == "--top_p" && i < length(ARGS)
            top_p = parse(Float64, ARGS[i+1]); i += 2
        elseif arg == "--max_tokens" && i < length(ARGS)
            max_tokens = parse(Int, ARGS[i+1]); i += 2
        elseif arg == "--greedy"
            greedy = true; i += 1
        else
            @warn "Unknown argument: $arg"; i += 1
        end
    end

    return checkpoint, prompt, temperature, top_k, top_p, max_tokens, greedy
end

function main()
    checkpoint, prompt, temperature, top_k, top_p, max_tokens, greedy = parse_args()

    # Load checkpoint
    @info "Loading checkpoint from $checkpoint"
    data = JLD2.load(checkpoint)
    config = data["config"]

    # Determine device
    device = CUDA.functional() ? Lux.gpu_device() : Lux.cpu_device()

    # Rebuild tokenizer from training data
    train_text = read(config.data.train_path, String)
    tokenizer = CharTokenizer(train_text)

    # Rebuild model
    model = create_model(config.model)
    ps, st, _, _, _ = load_checkpoint(checkpoint; device)

    @info "Generating" prompt=prompt temperature=temperature top_k=top_k max_tokens=max_tokens

    text = generate(model, ps, st, tokenizer, prompt;
                   max_new_tokens=max_tokens,
                   temperature=temperature,
                   top_k=top_k,
                   top_p=top_p,
                   greedy=greedy,
                   context_length=config.model.context_length)

    println("\n", "="^60)
    println(prompt, text)
    println("="^60)
end

main()
