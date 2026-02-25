module JuliaGPT

using Lux
using Functors
using Functors: fmap, fleaves
using NNlib
using Optimisers
using Zygote
using Random
using Statistics
using CUDA
using LuxCUDA
using OneHotArrays
using JLD2
using JSON3
using TOML
using Printf
using Dates

# Model configuration
include("model/config.jl")
export ModelConfig, TrainingConfig, Config, load_config

# Shared layers
include("model/layers.jl")
export RMSNorm, SwiGLU, make_causal_mask

# Mixture of Experts
include("model/moe.jl")
export SparseMoE

# Chunked attention
include("model/attention.jl")
export chunked_attention, chunked_causal_attention

# Monarch Mixer
include("model/monarch.jl")
export MonarchMatrix, CausalDepthwiseConv1d, MonarchSequenceMixer, MonarchBlock

# Tokenizer
include("data/tokenizer.jl")
export CharTokenizer, BPETokenizer, encode, decode, vocab_size

# Data loading
include("data/dataloader.jl")
export TextDataset, DataLoader, CurriculumDataLoader, next_batch!

# Model architecture
include("model/julia_gpt.jl")
export JuliaGPTModel, TiedEmbeddingHead, create_model, count_parameters

# Training
include("training/metrics.jl")
export TrainMetrics, update_metrics!, log_metrics, reset_metrics!

include("training/optimizer.jl")
export create_optimizer, cosine_lr

include("training/checkpoint.jl")
export save_checkpoint, load_checkpoint

include("training/ema.jl")
export EMAState, update_ema!, ema_parameters, copy_ema_to_model!

include("training/amp.jl")
export LossScaler, scale_loss, unscale_grads, update_scaler!, cast_f16, cast_f32, check_overflow

include("training/trainer.jl")
export train!, cross_entropy_loss

# Inference
include("inference/kv_cache.jl")
export KVCache, advance_cache!, generate_with_cache

include("inference/generate.jl")
export generate

# Evaluation
include("evaluation/evaluate.jl")
export evaluate, compute_perplexity

end # module
