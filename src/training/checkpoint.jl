"""
Model checkpointing — save and restore training state.
Includes schema versioning and checkpoint pruning.
"""

const CHECKPOINT_SCHEMA_VERSION = 2

"""
    save_checkpoint(path, ps, st, opt_state, step, config; best_val_loss=Inf)

Save model parameters, optimizer state, and training progress to a JLD2 file.
Parameters are moved to CPU before saving. Includes schema version and metadata.
"""
function save_checkpoint(path::String, ps, st, opt_state, step::Int, config;
                         best_val_loss::Float64=Inf)
    # Move parameters to CPU for portable serialization
    _to_cpu(x) = x isa AbstractArray ? Array(x) : x
    ps_cpu = fmap(_to_cpu, ps)
    st_cpu = fmap(_to_cpu, st)
    opt_cpu = fmap(_to_cpu, opt_state)

    mkpath(dirname(path))
    JLD2.jldsave(path;
        schema_version = CHECKPOINT_SCHEMA_VERSION,
        framework = "Lux.jl",
        julia_version = string(VERSION),
        parameters = ps_cpu,
        states = st_cpu,
        opt_state = opt_cpu,
        step = step,
        best_val_loss = best_val_loss,
        config = config,
        timestamp = Dates.now(),
    )
    @info "Checkpoint saved to $path (step $step, schema v$CHECKPOINT_SCHEMA_VERSION)"
end

"""
    load_checkpoint(path; device=identity) -> (ps, st, opt_state, step, best_val_loss)

Load a checkpoint from a JLD2 file. Optionally move parameters to GPU.
Handles older checkpoint formats without schema_version gracefully.
"""
function load_checkpoint(path::String; device=identity)
    data = JLD2.load(path)

    # Check schema version
    version = get(data, "schema_version", 1)
    if version > CHECKPOINT_SCHEMA_VERSION
        @warn "Checkpoint schema v$version is newer than supported v$CHECKPOINT_SCHEMA_VERSION"
    end

    ps = device(data["parameters"])
    st = device(data["states"])
    opt_state = get(data, "opt_state", nothing)
    if opt_state !== nothing
        opt_state = device(opt_state)
    end
    step = data["step"]
    best_val_loss = get(data, "best_val_loss", Inf)
    @info "Checkpoint loaded from $path (step $step, schema v$version, val_loss=$(round(best_val_loss; digits=4)))"
    return ps, st, opt_state, step, best_val_loss
end

"""
    prune_checkpoints(dir::String; keep_best::Int=3, keep_latest::Int=2)

Prune checkpoints in `dir`, keeping the best `keep_best` by validation loss
and the latest `keep_latest` by step number. Always keeps `final.jld2`.
"""
function prune_checkpoints(dir::String; keep_best::Int=3, keep_latest::Int=2)
    isdir(dir) || return

    files = filter(f -> endswith(f, ".jld2"), readdir(dir; join=true))
    isempty(files) && return

    # Parse checkpoint metadata
    entries = []
    for f in files
        basename(f) == "final.jld2" && continue
        try
            data = JLD2.load(f)
            push!(entries, (path=f, step=data["step"],
                           val_loss=get(data, "best_val_loss", Inf)))
        catch
            continue
        end
    end

    isempty(entries) && return

    # Select checkpoints to keep
    keep = Set{String}()

    # Keep best by validation loss
    sorted_by_loss = sort(entries; by=e -> e.val_loss)
    for e in sorted_by_loss[1:min(keep_best, length(sorted_by_loss))]
        push!(keep, e.path)
    end

    # Keep latest by step
    sorted_by_step = sort(entries; by=e -> e.step, rev=true)
    for e in sorted_by_step[1:min(keep_latest, length(sorted_by_step))]
        push!(keep, e.path)
    end

    # Remove the rest
    removed = 0
    for e in entries
        if e.path ∉ keep
            rm(e.path; force=true)
            removed += 1
        end
    end

    removed > 0 && @info "Pruned $removed checkpoints, kept $(length(keep))"
end
