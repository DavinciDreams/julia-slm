"""
Model checkpointing — save and restore training state.
"""

"""
    save_checkpoint(path, ps, st, opt_state, step, config; best_val_loss=Inf)

Save model parameters, optimizer state, and training progress to a JLD2 file.
Parameters are moved to CPU before saving.
"""
function save_checkpoint(path::String, ps, st, opt_state, step::Int, config;
                         best_val_loss::Float64=Inf)
    # Move parameters to CPU for portable serialization
    ps_cpu = Lux.cpu(ps)
    st_cpu = Lux.cpu(st)

    mkpath(dirname(path))
    JLD2.jldsave(path;
        parameters = ps_cpu,
        states = st_cpu,
        opt_state = opt_state,
        step = step,
        best_val_loss = best_val_loss,
        config = config,
        timestamp = Dates.now(),
    )
    @info "Checkpoint saved to $path (step $step)"
end

"""
    load_checkpoint(path; device=identity) -> (ps, st, opt_state, step, best_val_loss)

Load a checkpoint from a JLD2 file. Optionally move parameters to GPU.
"""
function load_checkpoint(path::String; device=identity)
    data = JLD2.load(path)
    ps = device(data["parameters"])
    st = device(data["states"])
    opt_state = data["opt_state"]
    step = data["step"]
    best_val_loss = get(data, "best_val_loss", Inf)
    @info "Checkpoint loaded from $path (step $step, val_loss=$(round(best_val_loss; digits=4)))"
    return ps, st, opt_state, step, best_val_loss
end
