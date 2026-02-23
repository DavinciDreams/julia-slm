"""
Optimizer setup and learning rate scheduling.
"""

"""
    cosine_lr(step, max_steps, lr, min_lr, warmup_steps) -> Float64

Cosine learning rate schedule with linear warmup.
"""
function cosine_lr(step::Int, max_steps::Int, lr::Float64, min_lr::Float64, warmup_steps::Int)
    if step < warmup_steps
        # Linear warmup
        return lr * step / max(warmup_steps, 1)
    end
    # Cosine decay
    progress = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
    progress = min(progress, 1.0)
    return min_lr + 0.5 * (lr - min_lr) * (1.0 + cos(π * progress))
end

"""
    create_optimizer(config::TrainingConfig) -> optimizer

Create an optimizer from training configuration.
"""
function create_optimizer(config::TrainingConfig)
    if config.optimizer == "adamw"
        return Optimisers.AdamW(; eta=Float32(config.lr), lambda=Float32(config.weight_decay))
    else
        error("Unknown optimizer: $(config.optimizer)")
    end
end

"""
    update_lr!(opt_state, new_lr)

Update the learning rate in an optimizer state tree.
Walks the tree and updates all AdamW leaves.
"""
function update_lr!(opt_state, new_lr::Float64)
    Optimisers.adjust!(opt_state; eta=Float32(new_lr))
    return opt_state
end
