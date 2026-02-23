"""
Training metrics tracking, logging, and experiment journaling.
"""

mutable struct TrainMetrics
    step::Int
    total_loss::Float64
    n_samples::Int
    best_val_loss::Float64
    start_time::Float64
    step_start::Float64
    tokens_processed::Int
    run_id::String
    log_path::Union{String,Nothing}
end

function TrainMetrics(; log_dir::Union{String,Nothing}=nothing)
    t = time()
    run_id = Dates.format(Dates.now(), "yyyymmdd_HHMMSSsss")
    log_path = nothing
    if log_dir !== nothing
        mkpath(log_dir)
        log_path = joinpath(log_dir, "train_$(run_id).jsonl")
    end
    return TrainMetrics(0, 0.0, 0, Inf, t, t, 0, run_id, log_path)
end

function update_metrics!(m::TrainMetrics, loss::Float64, n_tokens::Int)
    m.total_loss += loss
    m.n_samples += 1
    m.tokens_processed += n_tokens
end

function reset_metrics!(m::TrainMetrics)
    m.total_loss = 0.0
    m.n_samples = 0
    m.step_start = time()
    m.tokens_processed = 0
end

function avg_loss(m::TrainMetrics)
    m.n_samples > 0 ? m.total_loss / m.n_samples : 0.0
end

function log_metrics(m::TrainMetrics, lr::Float64, grad_norm::Float64; prefix="train")
    elapsed = time() - m.step_start
    tok_per_sec = m.tokens_processed / max(elapsed, 1e-6)
    loss = avg_loss(m)
    ppl = exp(loss)
    total_elapsed = time() - m.start_time

    @printf("[step %5d] %s loss: %.4f | ppl: %.2f | lr: %.2e | grad_norm: %.4f | tok/s: %.0f | elapsed: %.1fs\n",
            m.step, prefix, loss, ppl, lr, grad_norm, tok_per_sec, total_elapsed)

    # Append to JSONL experiment log if configured
    if m.log_path !== nothing
        entry = Dict{String,Any}(
            "run_id" => m.run_id,
            "step" => m.step,
            "prefix" => prefix,
            "loss" => round(loss; digits=6),
            "ppl" => round(ppl; digits=2),
            "lr" => lr,
            "grad_norm" => round(grad_norm; digits=6),
            "tok_per_sec" => round(tok_per_sec; digits=0),
            "elapsed" => round(total_elapsed; digits=1),
            "timestamp" => Dates.format(Dates.now(), "yyyy-mm-ddTHH:MM:SS"),
        )
        open(m.log_path, "a") do io
            println(io, JSON3.write(entry))
        end
    end
end
