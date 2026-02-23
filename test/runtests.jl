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
                  RMSNorm, SwiGLU, SparseMoE, chunked_attention,
                  KVCache, EMAState, update_ema!, ema_parameters,
                  LossScaler, scale_loss, unscale_grads, update_scaler!,
                  save_checkpoint, load_checkpoint, BPETokenizer,
                  CurriculumDataLoader, TiedEmbeddingHead
import Lux
using Functors
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
        # Diagonal and lower triangle (past+current) should be 0 (allowed)
        @test mask[1, 1] == 0.0f0
        @test mask[2, 1] == 0.0f0   # row 2, col 1 = past → allowed
        @test mask[2, 2] == 0.0f0
        @test mask[3, 1] == 0.0f0
        # Upper triangle (future) should be -Inf (blocked)
        @test mask[1, 2] == typemin(Float32)
        @test mask[1, 3] == typemin(Float32)
        @test mask[2, 3] == typemin(Float32)

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

    @testset "WeightTying" begin
        rng = Random.MersenneTwister(42)

        cfg = ModelConfig(
            vocab_size = 28,
            embed_dim = 64,
            n_layers = 2,
            n_heads = 2,
            head_dim = 32,
            context_length = 32,
            weight_tying = true,
        )

        model = create_model(cfg)
        @test model.head isa TiedEmbeddingHead

        ps, st = Lux.setup(rng, model)
        x = rand(rng, 1:28, 16, 2)
        logits, _ = model(x, ps, st)

        @test size(logits) == (28, 16, 2)
        @test !any(isnan, logits)

        # Weight-tied model should have fewer parameters (no separate head.weight)
        n_tied = count_parameters(ps)

        cfg2 = ModelConfig(
            vocab_size = 28,
            embed_dim = 64,
            n_layers = 2,
            n_heads = 2,
            head_dim = 32,
            context_length = 32,
            weight_tying = false,
        )
        model2 = create_model(cfg2)
        ps2, _ = Lux.setup(rng, model2)
        n_untied = count_parameters(ps2)
        @test n_tied < n_untied

        @info "WeightTying OK" tied_params=n_tied untied_params=n_untied
    end

    @testset "SparseMoE" begin
        rng = Random.MersenneTwister(42)
        moe = SparseMoE(64, 256; n_experts=4, top_k=2)
        ps, st = Lux.setup(rng, moe)

        x = randn(Float32, 64, 8, 2)  # (in_dim, seq_len, batch)
        y, st_new = moe(x, ps, st)

        @test size(y) == size(x)
        @test !any(isnan, y)
        @test haskey(st_new, :load_balance_loss)
        @test st_new.load_balance_loss >= 0

        @info "SparseMoE OK" lb_loss=st_new.load_balance_loss
    end

    @testset "ChunkedAttention" begin
        rng = Random.MersenneTwister(42)
        HD, T, HB = 32, 16, 4
        q = randn(rng, Float32, HD, T, HB)
        k = randn(rng, Float32, HD, T, HB)
        v = randn(rng, Float32, HD, T, HB)
        mask = make_causal_mask(T)

        out = chunked_attention(q, k, v, mask; chunk_size=4)
        @test size(out) == (HD, T, HB)
        @test !any(isnan, out)

        @info "ChunkedAttention OK"
    end

    @testset "EMA" begin
        rng = Random.MersenneTwister(42)

        cfg = ModelConfig(
            vocab_size = 28, embed_dim = 64, n_layers = 2,
            n_heads = 2, head_dim = 32, context_length = 32,
            weight_tying = false,
        )
        model = create_model(cfg)
        ps, st = Lux.setup(rng, model)

        ema = EMAState(ps; decay=0.99f0)
        @test ema.step == 0

        # Perturb parameters, then update EMA
        ps_perturbed = Functors.fmap(x -> x isa AbstractArray ? x .+ 0.1f0 : x, ps)
        update_ema!(ema, ps_perturbed)
        @test ema.step == 1

        shadow = ema_parameters(ema)
        @test shadow !== nothing

        @info "EMA OK"
    end

    @testset "LossScaler" begin
        scaler = LossScaler()
        @test scaler.scale > 0

        loss = 2.5f0
        scaled = scale_loss(scaler, loss)
        @test scaled == loss * scaler.scale

        # Test overflow detection
        update_scaler!(scaler, true)
        @test scaler.step_since_growth == 0
        @test scaler.scale < 2f0^15  # should have backed off

        @info "LossScaler OK"
    end

    @testset "KVCache" begin
        cache = KVCache(2, 32, 2, 64)
        @test cache.cache_len == 0

        # Simulate writing to layer 1
        k_new = randn(Float32, 32, 4, 2)
        v_new = randn(Float32, 32, 4, 2)
        k_full, v_full = JuliaGPT.update!(cache, 1, k_new, v_new)
        advance_cache!(cache, 4)
        @test cache.cache_len == 4
        @test size(k_full) == (32, 4, 2)

        # Reset
        JuliaGPT.reset!(cache)
        @test cache.cache_len == 0

        @info "KVCache OK"
    end

    @testset "Checkpoint roundtrip" begin
        rng = Random.MersenneTwister(42)
        cfg = ModelConfig(
            vocab_size = 28, embed_dim = 64, n_layers = 2,
            n_heads = 2, head_dim = 32, context_length = 32,
            weight_tying = false,
        )
        model = create_model(cfg)
        ps, st = Lux.setup(rng, model)

        path = tempname() * ".jld2"
        save_checkpoint(path, ps, st, nothing, 100, cfg; best_val_loss=1.5)

        ps2, st2, _, step, val_loss = load_checkpoint(path)
        @test step == 100
        @test val_loss == 1.5
        @test size(ps2.tok_emb.weight) == size(ps.tok_emb.weight)

        rm(path; force=true)
        @info "Checkpoint roundtrip OK"
    end

    @testset "LabelSmoothing" begin
        rng = Random.MersenneTwister(42)
        logits = randn(rng, Float32, 28, 8, 2)
        targets = rand(rng, 1:28, 8, 2)

        loss_plain = cross_entropy_loss(logits, targets; label_smoothing=0.0)
        loss_smooth = cross_entropy_loss(logits, targets; label_smoothing=0.1)

        @test isfinite(loss_plain)
        @test isfinite(loss_smooth)
        @test loss_smooth != loss_plain  # smoothing should change the value

        @info "LabelSmoothing OK" plain=round(Float64(loss_plain); digits=4) smooth=round(Float64(loss_smooth); digits=4)
    end

    @testset "CharTokenizer UNK" begin
        tok = CharTokenizer("abc"; add_unk=true)
        @test tok.unk_id > 0
        @test vocab_size(tok) == 4  # UNK + a, b, c

        encoded = encode(tok, "abcxyz")
        @test length(encoded) == 6  # x, y, z map to UNK instead of being dropped
        @test count(==(tok.unk_id), encoded) == 3

        @info "CharTokenizer UNK OK"
    end

    @testset "CurriculumDataLoader" begin
        text = "abcdefghij" ^ 1000
        tok = CharTokenizer(text)
        tokens = encode(tok, text)
        dataset = TextDataset(tokens, length(tokens))

        cl = CurriculumDataLoader(dataset, 4, 128; min_context=16, warmup_steps=10)

        # First batch should use min_context
        x1, y1 = next_batch!(cl)
        @test size(x1, 1) == 16  # min context rounded to multiple of 8

        # After warmup, should use max_context
        for _ in 1:15
            next_batch!(cl)
        end
        x_late, y_late = next_batch!(cl)
        @test size(x_late, 1) == 128

        @info "CurriculumDataLoader OK"
    end

end

@info "All tests passed!"
