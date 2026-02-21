"""
Batched sequence data loading for character-level language modeling.
Loads text, encodes it, and serves sliding-window batches.
"""

mutable struct TextDataset
    tokens::Vector{Int}
    n_tokens::Int
end

"""
    TextDataset(path::String, tokenizer::CharTokenizer) -> TextDataset

Load a text file and encode it into token indices.
"""
function TextDataset(path::String, tokenizer::CharTokenizer)
    text = read(path, String)
    tokens = encode(tokenizer, text)
    return TextDataset(tokens, length(tokens))
end

mutable struct DataLoader
    dataset::TextDataset
    batch_size::Int
    context_length::Int
    position::Int
    device::Function
end

"""
    DataLoader(dataset, batch_size, context_length; device=identity) -> DataLoader

Create a batched data loader that serves (input, target) pairs of shape
(context_length, batch_size). Targets are inputs shifted by one position.
"""
function DataLoader(dataset::TextDataset, batch_size::Int, context_length::Int; device=identity)
    return DataLoader(dataset, batch_size, context_length, 1, device)
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

    x = Matrix{Int}(undef, T, B)
    y = Matrix{Int}(undef, T, B)

    for b in 1:B
        # Wrap around if we would exceed the data
        if loader.position + seq_len - 1 > n
            loader.position = 1
        end

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
    usable = loader.dataset.n_tokens - loader.context_length
    return max(1, usable ÷ (loader.context_length * loader.batch_size))
end
