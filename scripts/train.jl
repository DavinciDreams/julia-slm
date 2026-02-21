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

    # Load training data and build tokenizer
    @info "Loading training data from $(config.data.train_path)"
    train_text = read(config.data.train_path, String)
    tokenizer = CharTokenizer(train_text)
    @info "Vocabulary" size=vocab_size(tokenizer) chars=join(tokenizer.idx_to_char)

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
    )

    # Create model
    @info "Creating model" model_config
    model = create_model(model_config)

    # Initialize parameters
    rng = Random.MersenneTwister(config.training.seed)
    ps, st = Lux.setup(rng, model)

    # Count parameters
    n_params = count_parameters(ps)
    @info "Model initialized" parameters=n_params params_human=if n_params >= 1_000_000
        @sprintf("%.2fM", n_params / 1_000_000)
    else
        @sprintf("%.1fK", n_params / 1_000)
    end

    # Move to device
    ps = device(ps)
    st = device(st)

    # Create data loaders
    train_dataset = TextDataset(config.data.train_path, tokenizer)
    val_dataset = TextDataset(config.data.val_path, tokenizer)

    train_loader = DataLoader(train_dataset, config.training.batch_size,
                              model_config.context_length; device)
    val_loader = DataLoader(val_dataset, config.training.batch_size,
                            model_config.context_length; device)

    @info "Data loaded" train_tokens=train_dataset.n_tokens val_tokens=val_dataset.n_tokens

    # Resume from checkpoint if specified
    if resume_path !== nothing
        @info "Resuming from $resume_path"
        ps, st, _, start_step, _ = load_checkpoint(resume_path; device)
    end

    # Update config with actual vocab size
    full_config = Config(model_config, config.training, config.curriculum,
                         config.coreset, config.data, config.inference,
                         config.phase_weights)

    # Train
    ps, st, metrics = train!(model, ps, st, train_loader, val_loader, full_config; device)

    # Final evaluation
    @info "Running final evaluation..."
    results = evaluate(model, ps, st, tokenizer, val_loader;
                       max_batches=50, gen_samples=3, gen_length=300,
                       context_length=model_config.context_length)

    @info "Done!" final_perplexity=round(results["perplexity"]; digits=2)
end

main()
