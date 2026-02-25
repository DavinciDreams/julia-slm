#!/usr/bin/env julia
"""
CLI entry point for training JuliaGPT.

Usage:
    julia --project scripts/train.jl --config config/tiny.toml
    julia --project scripts/train.jl --config config/base.toml --resume checkpoints/step_1000.jld2
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

include(joinpath(@__DIR__, "..", "src", "JuliaGPT.jl"))
using .JuliaGPT
using .JuliaGPT: Lux, CUDA
using Random
using Printf

function parse_args()
    config_path = "config/tiny.toml"
    resume_path = nothing

    i = 1
    while i <= length(ARGS)
        if ARGS[i] == "--config" && i < length(ARGS)
            config_path = ARGS[i+1]
            i += 2
        elseif ARGS[i] == "--resume" && i < length(ARGS)
            resume_path = ARGS[i+1]
            i += 2
        else
            @warn "Unknown argument: $(ARGS[i])"
            i += 1
        end
    end

    return config_path, resume_path
end

function main()
    config_path, resume_path = parse_args()

    @info "Loading config from $config_path"
    config = load_config(config_path)

    # Set random seed
    Random.seed!(config.training.seed)

    # Determine device
    device = if CUDA.functional()
        @info "CUDA available" device=CUDA.device()
        Lux.gpu_device()
    else
        @info "Running on CPU"
        Lux.cpu_device()
    end

    # Load tokenizer
    tokenizer = if !isempty(config.data.tokenizer_dir)
        vocab_path = joinpath(config.data.tokenizer_dir, "vocab.json")
        merges_path = joinpath(config.data.tokenizer_dir, "merges.txt")
        @info "Loading BPE tokenizer from $(config.data.tokenizer_dir)"
        BPETokenizer(vocab_path, merges_path)
    else
        @info "Building character tokenizer from $(config.data.train_path)"
        train_text = read(config.data.train_path, String)
        tok = CharTokenizer(train_text)
        @info "Vocabulary" size=vocab_size(tok) chars=join(tok.idx_to_char)
        tok
    end
    @info "Tokenizer" type=typeof(tokenizer) vocab=vocab_size(tokenizer)

    # Update model config with actual vocab size
    model_config = ModelConfig(;
        arch = config.model.arch,
        vocab_size = vocab_size(tokenizer),
        embed_dim = config.model.embed_dim,
        n_layers = config.model.n_layers,
        n_heads = config.model.n_heads,
        head_dim = config.model.head_dim,
        ffn_mult = config.model.ffn_mult,
        context_length = config.model.context_length,
        dropout = config.model.dropout,
        bias = config.model.bias,
        weight_tying = config.model.weight_tying,
        n_monarch_heads = config.model.n_monarch_heads,
        conv_kernel_size = config.model.conv_kernel_size,
    )

    # Create model
    @info "Creating model" model_config
    model = create_model(model_config)

    # Initialize parameters
    rng = Random.MersenneTwister(config.training.seed)
    ps, st = Lux.setup(rng, model)

    # Count parameters
    n_params = count_parameters(ps; weight_tying=model_config.weight_tying)
    @info "Model initialized" parameters=n_params weight_tying=model_config.weight_tying params_human=if n_params >= 1_000_000
        @sprintf("%.2fM", n_params / 1_000_000)
    else
        @sprintf("%.1fK", n_params / 1_000)
    end

    # Move to device
    ps = device(ps)
    st = device(st)

    # Create data loaders — prefer pre-encoded .bin files when available
    train_bin = replace(config.data.train_path, r"\.txt$" => ".bin")
    val_bin = replace(config.data.val_path, r"\.txt$" => ".bin")

    train_dataset = if isfile(train_bin)
        @info "Loading pre-encoded tokens from $train_bin"
        TextDataset(train_bin)
    else
        @info "Encoding training data on-the-fly (slow for BPE)"
        TextDataset(config.data.train_path, tokenizer)
    end

    val_dataset = if isfile(val_bin)
        @info "Loading pre-encoded tokens from $val_bin"
        TextDataset(val_bin)
    else
        TextDataset(config.data.val_path, tokenizer)
    end

    train_loader = DataLoader(train_dataset, config.training.batch_size,
                              model_config.context_length; device)
    val_loader = DataLoader(val_dataset, config.training.batch_size,
                            model_config.context_length; device)

    @info "Data loaded" train_tokens=train_dataset.n_tokens val_tokens=val_dataset.n_tokens
    flush(stderr); flush(stdout)

    # Update config with actual vocab size
    full_config = Config(model_config, config.training, config.curriculum,
                         config.coreset, config.data, config.inference,
                         config.phase_weights)

    # Resume from checkpoint if specified
    start_step = 1
    resume_opt_state = nothing
    resume_best_val = Inf
    if resume_path !== nothing
        @info "Resuming from $resume_path"
        ps, st, resume_opt_state, start_step, resume_best_val = load_checkpoint(resume_path; device)
        start_step += 1  # resume from the next step
        @info "Resuming from step $start_step" best_val_loss=resume_best_val
    end

    # Train
    ps, st, metrics = train!(model, ps, st, train_loader, val_loader, full_config;
                              device, start_step, opt_state=resume_opt_state,
                              best_val_loss=resume_best_val)

    # Final evaluation
    @info "Running final evaluation..."
    results = evaluate(model, ps, st, tokenizer, val_loader;
                       max_batches=50, gen_samples=3, gen_length=300,
                       context_length=model_config.context_length)

    @info "Done!" final_perplexity=round(results["perplexity"]; digits=2)
end

main()
