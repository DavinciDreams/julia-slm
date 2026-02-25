"""
Batched sequence data loading for character-level language modeling.
Loads text, encodes it, and serves sliding-window batches.
"""

mutable struct TextDataset
    tokens::Vector{Int}
    n_tokens::Int
end

"""
    TextDataset(path::String, tokenizer) -> TextDataset

Load a text file and encode it into token indices.
Works with any tokenizer that implements `encode(tokenizer, text)`.
"""
function TextDataset(path::String, tokenizer)
    text = read(path, String)
    tokens = encode(tokenizer, text)
    return TextDataset(tokens, length(tokens))
end

"""
    TextDataset(bin_path::String) -> TextDataset

Load pre-encoded tokens from a .bin file (written by encode_corpus.py).
Format: "JTOK" magic (4B) + n_tokens (UInt64) + offset (Int32) + Int32 token data.
"""
function TextDataset(bin_path::String)
    @assert endswith(bin_path, ".bin") "Expected .bin file, got: $bin_path"
    open(bin_path, "r") do f
        magic = read(f, 4)
        @assert magic == UInt8[0x4a, 0x54, 0x4f, 0x4b] "Bad magic: expected JTOK"
        n_tokens = read(f, UInt64)
        _offset = read(f, Int32)  # already applied during encoding
        raw = Vector{Int32}(undef, n_tokens)
        read!(f, raw)
        tokens = Vector{Int}(raw)  # promote Int32 → Int (Int64)
        return TextDataset(tokens, Int(n_tokens))
    end
end

mutable struct DataLoader
    dataset::TextDataset
    batch_size::Int
    context_length::Int
    position::Int
    device::Function
    x_buf::Matrix{Int}
    y_buf::Matrix{Int}
end

"""
    DataLoader(dataset, batch_size, context_length; device=identity) -> DataLoader

Create a batched data loader that serves (input, target) pairs of shape
(context_length, batch_size). Targets are inputs shifted by one position.
"""
function DataLoader(dataset::TextDataset, batch_size::Int, context_length::Int; device=identity)
    x_buf = Matrix{Int}(undef, context_length, batch_size)
    y_buf = Matrix{Int}(undef, context_length, batch_size)
    return DataLoader(dataset, batch_size, context_length, 1, device, x_buf, y_buf)
end

"""
    next_batch!(loader) -> (x, y)

Get the next batch of (input, target) sequences.
Returns arrays of shape (context_length, batch_size).
x contains token indices, y contains the next-token targets (shifted by 1).
Wraps around to the beginning when reaching end of data.
"""
function next_batch!(loader::DataLoader)
    B = loader.batch_size
    T = loader.context_length
    tokens = loader.dataset.tokens
    n = loader.dataset.n_tokens

    # Each sequence needs T+1 tokens (T inputs + 1 target for the last position)
    seq_len = T + 1

    # Check if the entire batch would overflow — reset once before the loop
    total_needed = B * T + seq_len - T  # first item needs seq_len, rest stride by T
    if loader.position + total_needed - 1 > n
        loader.position = 1
    end

    x = loader.x_buf
    y = loader.y_buf

    for b in 1:B
        start = loader.position
        chunk = @view tokens[start:start+seq_len-1]

        x[:, b] .= chunk[1:T]
        y[:, b] .= chunk[2:T+1]

        # Stride by context_length to avoid excessive overlap
        loader.position += T
    end

    return loader.device(x), loader.device(y)
end

"""
    reset!(loader)

Reset the data loader position to the beginning.
"""
function reset!(loader::DataLoader)
    loader.position = 1
end

function n_batches(loader::DataLoader)
    usable = loader.dataset.n_tokens - (loader.context_length + 1)
    return max(1, usable ÷ (loader.context_length * loader.batch_size))
end

# ─────────────────────────────────────────────
# Curriculum DataLoader — sequence length warmup
# ─────────────────────────────────────────────

"""
    CurriculumDataLoader

A wrapper around DataLoader that progressively increases the context length
during training. Starts with `min_context` and linearly ramps to `max_context`
over `warmup_steps` optimizer steps.
"""
mutable struct CurriculumDataLoader
    dataset::TextDataset
    batch_size::Int
    min_context::Int
    max_context::Int
    warmup_steps::Int
    current_step::Int
    position::Int
    device::Function
end

"""
    CurriculumDataLoader(dataset, batch_size, max_context;
                         min_context=32, warmup_steps=1000, device=identity)

Create a curriculum data loader with progressive sequence length warmup.
"""
function CurriculumDataLoader(dataset::TextDataset, batch_size::Int, max_context::Int;
                               min_context::Int=32, warmup_steps::Int=1000, device=identity)
    return CurriculumDataLoader(dataset, batch_size, min_context, max_context,
                                 warmup_steps, 0, 1, device)
end

function current_context_length(cl::CurriculumDataLoader)
    progress = min(cl.current_step / max(cl.warmup_steps, 1), 1.0)
    len = cl.min_context + round(Int, progress * (cl.max_context - cl.min_context))
    # Round to multiple of 8 for alignment
    return max(cl.min_context, (len ÷ 8) * 8)
end

function next_batch!(cl::CurriculumDataLoader)
    T = current_context_length(cl)
    B = cl.batch_size
    tokens = cl.dataset.tokens
    n = cl.dataset.n_tokens
    seq_len = T + 1

    total_needed = B * T + seq_len - T
    if cl.position + total_needed - 1 > n
        cl.position = 1
    end

    x = Matrix{Int}(undef, T, B)
    y = Matrix{Int}(undef, T, B)

    for b in 1:B
        start = cl.position
        chunk = @view tokens[start:start+seq_len-1]
        x[:, b] .= chunk[1:T]
        y[:, b] .= chunk[2:T+1]
        cl.position += T
    end

    cl.current_step += 1
    return cl.device(x), cl.device(y)
end

function reset!(cl::CurriculumDataLoader)
    cl.position = 1
end
