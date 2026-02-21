#!/usr/bin/env julia
"""
Smoke tests for JuliaGPT — verifies forward pass, backward pass,
tokenizer roundtrip, and data loading.
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

include(joinpath(@__DIR__, "..", "src", "JuliaGPT.jl"))
using .JuliaGPT
using .JuliaGPT: make_causal_mask, CausalSelfAttention, TransformerBlock,
                  cross_entropy_loss, compute_grad_norm, count_parameters,
                  RMSNorm, SwiGLU
import Lux
using Random
using Statistics
using Test
using Zygote

@testset "JuliaGPT" begin

    @testset "CharTokenizer" begin
        text = "the nature of things is such that we must consider"
        tok = CharTokenizer(text)

        @test vocab_size(tok) > 0
        @test vocab_size(tok) <= 256

        # Roundtrip
        encoded = encode(tok, text)
        decoded = decode(tok, encoded)
        @test decoded == text

        # Unknown chars skipped
        encoded2 = encode(tok, "the @#\$ nature")
        decoded2 = decode(tok, encoded2)
        @test decoded2 == "the  nature"

        @info "Tokenizer OK" vocab_size=vocab_size(tok)
    end

    @testset "DataLoader" begin
        # Create small test data
        text = "abcdefghijklmnopqrstuvwxyz " ^ 100  # ~2700 chars
        tok = CharTokenizer(text)
        tokens = encode(tok, text)
        dataset = TextDataset(tokens, length(tokens))

        loader = DataLoader(dataset, 4, 32)
        x, y = next_batch!(loader)

        @test size(x) == (32, 4)
        @test size(y) == (32, 4)
        # y should be x shifted by 1
        @test all(x[2:end, :] .== y[1:end-1, :])

        @info "DataLoader OK" x_size=size(x) y_size=size(y)
    end

    @testset "RMSNorm" begin
        rng = Random.MersenneTwister(42)
        norm = RMSNorm(64)
        ps, st = Lux.setup(rng, norm)

        x = randn(Float32, 64, 10, 2)
        y, st_new = norm(x, ps, st)

        @test size(y) == size(x)
        # RMS should be approximately 1 after normalization
        rms = sqrt.(mean(y .^ 2; dims=1))
        @test all(isapprox.(rms, 1.0; atol=0.5))

        @info "RMSNorm OK"
    end

    @testset "SwiGLU" begin
        rng = Random.MersenneTwister(42)
        swiglu = SwiGLU(64, 256)
        ps, st = Lux.setup(rng, swiglu)

        x = randn(Float32, 64, 10, 2)
        y, st_new = swiglu(x, ps, st)

        @test size(y) == size(x)
        @test !any(isnan, y)

        @info "SwiGLU OK"
    end

    @testset "CausalMask" begin
        mask = make_causal_mask(4)
        @test size(mask) == (4, 4)
        # Lower triangle should be 0
        @test mask[1, 1] == 0.0f0
        @test mask[1, 2] == 0.0f0
        @test mask[2, 2] == 0.0f0
        # Upper triangle should be -Inf
        @test mask[2, 1] == typemin(Float32)
        @test mask[3, 1] == typemin(Float32)

        @info "CausalMask OK"
    end

    @testset "CausalSelfAttention" begin
        rng = Random.MersenneTwister(42)
        attn = CausalSelfAttention(64, 2, 32)
        ps, st = Lux.setup(rng, attn)

        x = randn(Float32, 64, 8, 2)  # (embed_dim, seq_len, batch)
        mask = make_causal_mask(8)
        y, st_new = attn(x, ps, st; mask=mask)

        @test size(y) == size(x)
        @test !any(isnan, y)

        @info "CausalSelfAttention OK"
    end

    @testset "TransformerBlock" begin
        rng = Random.MersenneTwister(42)
        block = TransformerBlock(64, 2, 32; ffn_mult=4)
        ps, st = Lux.setup(rng, block)

        x = randn(Float32, 64, 8, 2)
        mask = make_causal_mask(8)
        y, st_new = block(x, ps, st; mask=mask)

        @test size(y) == size(x)
        @test !any(isnan, y)

        @info "TransformerBlock OK"
    end

    @testset "JuliaGPTModel forward" begin
        rng = Random.MersenneTwister(42)

        cfg = ModelConfig(
            vocab_size = 28,
            embed_dim = 64,
            n_layers = 2,
            n_heads = 2,
            head_dim = 32,
            ffn_mult = 4,
            context_length = 32,
            dropout = 0.0f0,
            bias = false,
            weight_tying = false,
        )

        model = create_model(cfg)
        ps, st = Lux.setup(rng, model)

        # Input: (seq_len, batch) of token indices
        x = rand(rng, 1:28, 16, 2)
        logits, st_new = model(x, ps, st)

        @test size(logits) == (28, 16, 2)  # (vocab_size, seq_len, batch)
        @test !any(isnan, logits)

        n = count_parameters(ps)
        @info "JuliaGPTModel forward OK" params=n logits_size=size(logits)
    end

    @testset "JuliaGPTModel backward" begin
        rng = Random.MersenneTwister(42)

        cfg = ModelConfig(
            vocab_size = 28,
            embed_dim = 64,
            n_layers = 2,
            n_heads = 2,
            head_dim = 32,
            ffn_mult = 4,
            context_length = 32,
            dropout = 0.0f0,
            bias = false,
            weight_tying = false,
        )

        model = create_model(cfg)
        ps, st = Lux.setup(rng, model)

        x = rand(rng, 1:28, 16, 2)
        y = rand(rng, 1:28, 16, 2)

        # Compute loss and gradients
        (loss, _), grads = Zygote.withgradient(ps) do p
            logits, st_ = model(x, p, st)
            l = cross_entropy_loss(logits, y)
            return l, st_
        end

        @test isfinite(loss)
        @test loss > 0
        @test grads !== nothing
        @test !isempty(grads)

        gnorm = compute_grad_norm(grads[1])
        @test isfinite(gnorm)
        @test gnorm > 0

        @info "JuliaGPTModel backward OK" loss=round(loss; digits=4) grad_norm=round(gnorm; digits=4)
    end

    @testset "Generate" begin
        rng = Random.MersenneTwister(42)

        text = "abcdefghijklmnopqrstuvwxyz .,"
        tok = CharTokenizer(text)

        cfg = ModelConfig(
            vocab_size = vocab_size(tok),
            embed_dim = 64,
            n_layers = 2,
            n_heads = 2,
            head_dim = 32,
            ffn_mult = 4,
            context_length = 32,
        )

        model = create_model(cfg)
        ps, st = Lux.setup(rng, model)

        output = generate(model, ps, st, tok, "the ";
                         max_new_tokens=20, temperature=1.0,
                         context_length=32)

        @test length(output) == 20
        @test all(c -> haskey(tok.char_to_idx, c), output)

        @info "Generate OK" output_length=length(output) sample=output
    end

end

@info "All tests passed!"
