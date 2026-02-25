"""
Main training loop for JuliaGPT.
"""

"""
    cross_entropy_loss(logits, targets; label_smoothing=0.0)

Compute cross-entropy loss for language modeling.
logits: (vocab_size, seq_len, batch)
targets: (seq_len, batch) — integer indices
Optional label smoothing distributes probability mass across vocabulary.
"""
function cross_entropy_loss(logits, targets; label_smoothing::Float64=0.0)
    V, T, B = size(logits)
    logits_flat = reshape(logits, V, T * B)
    targets_flat = reshape(targets, T * B)
    loss = _ce_loss(logits_flat, targets_flat, V; label_smoothing)
    return loss
end

function _ce_loss(logits, targets, vocab_size; label_smoothing::Float64=0.0)
    # GPU-compatible cross-entropy using one-hot encoding
    # (same pattern as Flux.logitcrossentropy used in the working notebooks)
    V, N = size(logits)
    oh = onehotbatch(targets, 1:V)
    log_probs = NNlib.logsoftmax(logits; dims=1)
    nll = -sum(log_probs .* oh) / N
    if label_smoothing > 0.0
        smooth_loss = -sum(log_probs) / (N * V)
        return (1 - Float32(label_smoothing)) * nll + Float32(label_smoothing) * smooth_loss
    end
    return nll
end

"""
    compute_grad_norm(grads) -> Float64

Compute the L2 norm of all gradients.
"""
function compute_grad_norm(grads)
    total = 0.0
    for x in fleaves(grads)
        if x isa AbstractArray
            total += sum(abs2, x)
        end
    end
    return sqrt(total)
end

"""
    clip_gradients(grads, max_norm) -> grads

Clip gradients to a maximum L2 norm.
"""
function clip_gradients(grads, max_norm::Float64)
    gnorm = compute_grad_norm(grads)
    if gnorm > max_norm
        scale = Float32(max_norm / gnorm)
        grads = fmap(x -> x isa AbstractArray ? x .* scale : x, grads)
    end
    return grads, gnorm
end

"""
    evaluate_loss(model, ps, st, val_loader, eval_steps) -> Float64

Compute average loss over eval_steps batches of validation data.
"""
function evaluate_loss(model, ps, st, val_loader, eval_steps::Int)
    total_loss = 0.0
    reset!(val_loader)
    for _ in 1:eval_steps
        x, y = next_batch!(val_loader)
        logits, _ = model(x, ps, st)
        loss = cross_entropy_loss(logits, y)
        total_loss += Float64(loss)
    end
    return total_loss / eval_steps
end

"""
    train!(model, ps, st, train_loader, val_loader, config;
           device=identity, start_step=1, opt_state=nothing, best_val_loss=Inf)

Main training loop. Trains the model according to the configuration.
Supports resuming from a checkpoint via start_step and opt_state.
Returns (ps, st, metrics).
"""
function train!(model, ps, st, train_loader, val_loader, config::Config;
                device=identity, start_step::Int=1, opt_state=nothing,
                best_val_loss::Float64=Inf)
    tc = config.training
    rng = Random.MersenneTwister(tc.seed)

    # Setup optimizer (use provided opt_state if resuming)
    if opt_state === nothing
        opt = create_optimizer(tc)
        opt_state = Optimisers.setup(opt, ps)
    end

    # Metrics
    metrics = TrainMetrics()
    metrics.best_val_loss = best_val_loss
    accum_steps = max(1, tc.accumulation_steps)
    ls = tc.label_smoothing

    @info "Starting training" start_step=start_step max_steps=tc.max_steps batch_size=tc.batch_size lr=tc.lr accumulation_steps=accum_steps
    @info "Model parameters" n_params=count_parameters(ps)

    for step in start_step:tc.max_steps
        metrics.step = step

        # Update learning rate
        lr = cosine_lr(step, tc.max_steps, tc.lr, tc.min_lr, tc.warmup_steps)
        opt_state = update_lr!(opt_state, lr)

        # Gradient accumulation loop
        accum_loss = 0.0
        accum_grads = nothing
        for micro in 1:accum_steps
            x, y = next_batch!(train_loader)

            (loss, st_new), grads = Zygote.withgradient(ps) do p
                logits, st_ = model(x, p, st)
                l = cross_entropy_loss(logits, y; label_smoothing=ls)
                return l, st_
            end

            st = st_new
            accum_loss += Float64(loss)
            n_tokens = length(x)
            update_metrics!(metrics, Float64(loss), n_tokens)

            # Accumulate gradients
            if accum_grads === nothing
                accum_grads = grads[1]
            else
                accum_grads = fmap((a, b) -> a isa AbstractArray ? a .+ b : a, accum_grads, grads[1])
            end
        end

        # Average accumulated gradients
        if accum_steps > 1
            inv_accum = Float32(1.0 / accum_steps)
            accum_grads = fmap(x -> x isa AbstractArray ? x .* inv_accum : x, accum_grads)
        end

        # Clip gradients
        grads_clipped, gnorm = clip_gradients(accum_grads, tc.grad_clip)

        # Update parameters
        opt_state, ps = Optimisers.update!(opt_state, ps, grads_clipped)

        # Log every eval_interval steps
        if step % tc.eval_interval == 0 || step == 1
            log_metrics(metrics, lr, gnorm; prefix="train")

            # Evaluate on validation set
            val_loss = evaluate_loss(model, ps, st, val_loader, tc.eval_steps)
            val_ppl = exp(val_loss)
            @printf("           val loss: %.4f | val ppl: %.2f\n", val_loss, val_ppl)

            if val_loss < metrics.best_val_loss
                metrics.best_val_loss = val_loss
                @info "New best validation loss" val_loss=round(val_loss; digits=4)
            end

            flush(stderr); flush(stdout)
            reset_metrics!(metrics)
        end

        # Checkpoint
        if step % tc.checkpoint_interval == 0
            ckpt_path = joinpath("checkpoints", "step_$(step).jld2")
            save_checkpoint(ckpt_path, ps, st, opt_state, step, config;
                           best_val_loss=metrics.best_val_loss)
        end
    end

    # Save final checkpoint
    save_checkpoint("checkpoints/final.jld2", ps, st, opt_state, tc.max_steps, config;
                    best_val_loss=metrics.best_val_loss)

    total_time = time() - metrics.start_time
    @info "Training complete" total_time=round(total_time; digits=1) best_val_loss=round(metrics.best_val_loss; digits=4)

    return ps, st, metrics
end
