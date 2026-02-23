"""
Model and training configuration, loaded from TOML files.
"""

Base.@kwdef struct ModelConfig
    # Architecture
    arch::String = "transformer"
    vocab_size::Int = 0  # set dynamically from tokenizer
    embed_dim::Int = 384
    n_layers::Int = 6
    n_heads::Int = 3
    head_dim::Int = 128
    ffn_mult::Int = 4
    context_length::Int = 256
    dropout::Float32 = 0.0f0
    bias::Bool = false
    weight_tying::Bool = true
end

Base.@kwdef struct TrainingConfig
    optimizer::String = "adamw"
    lr::Float64 = 6e-4
    min_lr::Float64 = 6e-5
    weight_decay::Float64 = 0.1
    warmup_steps::Int = 100
    max_steps::Int = 5000
    batch_size::Int = 64
    grad_clip::Float64 = 1.0
    precision::String = "f32"
    eval_interval::Int = 250
    eval_steps::Int = 50
    checkpoint_interval::Int = 1000
    seed::Int = 42
end

Base.@kwdef struct CurriculumConfig
    enabled::Bool = false
    ordering::String = "easy_first"
    warmup_epochs::Int = 1
end

Base.@kwdef struct CoresetConfig
    enabled::Bool = false
    method::String = "gradient_norm"
    keep_fraction::Float64 = 0.5
end

Base.@kwdef struct DataConfig
    train_path::String = "../text-pipeline/output/train.txt"
    val_path::String = "../text-pipeline/output/val.txt"
    enriched_path::String = "../text-pipeline/output/train_enriched.jsonl"
end

Base.@kwdef struct InferenceConfig
    precision::String = "f32"
    compile::Bool = false
    temperature::Float64 = 0.8
    top_k::Int = 40
    max_new_tokens::Int = 500
end

struct Config
    model::ModelConfig
    training::TrainingConfig
    curriculum::CurriculumConfig
    coreset::CoresetConfig
    data::DataConfig
    inference::InferenceConfig
    phase_weights::Dict{String,Float64}
end

"""
    load_config(path::String) -> Config

Load configuration from a TOML file. Returns a Config struct with all settings.
"""
function load_config(path::String)
    raw = TOML.parsefile(path)

    m = get(raw, "model", Dict())
    model = ModelConfig(;
        arch = get(m, "arch", "transformer"),
        embed_dim = get(m, "embed_dim", 384),
        n_layers = get(m, "n_layers", 6),
        n_heads = get(m, "n_heads", 3),
        head_dim = get(m, "head_dim", 128),
        ffn_mult = get(m, "ffn_mult", 4),
        context_length = get(m, "context_length", 256),
        dropout = Float32(get(m, "dropout", 0.0)),
        bias = get(m, "bias", false),
        weight_tying = get(m, "weight_tying", true),
    )

    t = get(raw, "training", Dict())
    training = TrainingConfig(;
        optimizer = get(t, "optimizer", "adamw"),
        lr = get(t, "lr", 6e-4),
        min_lr = get(t, "min_lr", 6e-5),
        weight_decay = get(t, "weight_decay", 0.1),
        warmup_steps = get(t, "warmup_steps", 100),
        max_steps = get(t, "max_steps", 5000),
        batch_size = get(t, "batch_size", 64),
        grad_clip = get(t, "grad_clip", 1.0),
        precision = get(t, "precision", "f32"),
        eval_interval = get(t, "eval_interval", 250),
        eval_steps = get(t, "eval_steps", 50),
        checkpoint_interval = get(t, "checkpoint_interval", 1000),
        seed = get(t, "seed", 42),
    )

    tc = get(t, "curriculum", Dict())
    curriculum = CurriculumConfig(;
        enabled = get(tc, "enabled", false),
        ordering = get(tc, "ordering", "easy_first"),
        warmup_epochs = get(tc, "warmup_epochs", 1),
    )

    tcs = get(t, "coreset", Dict())
    coreset = CoresetConfig(;
        enabled = get(tcs, "enabled", false),
        method = get(tcs, "method", "gradient_norm"),
        keep_fraction = get(tcs, "keep_fraction", 0.5),
    )

    d = get(raw, "data", Dict())
    data = DataConfig(;
        train_path = get(d, "train_path", "../text-pipeline/output/train.txt"),
        val_path = get(d, "val_path", "../text-pipeline/output/val.txt"),
        enriched_path = get(d, "enriched_path", "../text-pipeline/output/train_enriched.jsonl"),
    )

    i = get(raw, "inference", Dict())
    inference = InferenceConfig(;
        precision = get(i, "precision", "f32"),
        compile = get(i, "compile", false),
        temperature = get(i, "temperature", 0.8),
        top_k = get(i, "top_k", 40),
        max_new_tokens = get(i, "max_new_tokens", 500),
    )

    pw = get(get(t, "phase_weights", Dict()), "", Dict())
    if isempty(pw)
        pw = get(raw, "training", Dict())
        pw = get(pw, "phase_weights", Dict{String,Any}())
    end
    phase_weights = Dict{String,Float64}(k => Float64(v) for (k, v) in pw)

    return Config(model, training, curriculum, coreset, data, inference, phase_weights)
end

function Base.show(io::IO, cfg::ModelConfig)
    params_est = cfg.n_layers * 12 * cfg.embed_dim^2
    params_str = if params_est >= 1_000_000
        @sprintf("%.1fM", params_est / 1_000_000)
    else
        @sprintf("%.1fK", params_est / 1_000)
    end
    print(io, "ModelConfig($(cfg.arch), $(cfg.embed_dim)d, $(cfg.n_layers)L, $(cfg.n_heads)H, ~$(params_str) params)")
end
