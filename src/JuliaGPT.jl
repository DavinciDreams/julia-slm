module JuliaGPT

using Lux
using Functors: fmap, fleaves
using NNlib
using Optimisers
using Zygote
using Random
using Statistics
using CUDA
using JLD2
using JSON3
using TOML
using Printf
using Dates

# Model configuration
include("model/config.jl")
export ModelConfig, load_config

# Shared layers
include("model/layers.jl")
export RMSNorm, SwiGLU

# Tokenizer
include("data/tokenizer.jl")
export CharTokenizer, encode, decode, vocab_size

# Data loading
include("data/dataloader.jl")
export TextDataset, DataLoader, next_batch!

# Model architecture
include("model/julia_gpt.jl")
export JuliaGPTModel, create_model

# Training
include("training/metrics.jl")
export TrainMetrics, update_metrics!, log_metrics, reset_metrics!

include("training/optimizer.jl")
export create_optimizer, cosine_lr

include("training/checkpoint.jl")
export save_checkpoint, load_checkpoint

include("training/trainer.jl")
export train!

# Inference
include("inference/generate.jl")
export generate

# Evaluation
include("evaluation/evaluate.jl")
export evaluate, compute_perplexity

end # module
