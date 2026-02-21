"""
Character-level tokenizer. Builds vocabulary dynamically from training data.
"""

struct CharTokenizer
    char_to_idx::Dict{Char,Int}
    idx_to_char::Vector{Char}
    vocab_sz::Int
end

"""
    CharTokenizer(text::String) -> CharTokenizer

Build a character-level tokenizer from a text corpus. Vocabulary is sorted
for reproducibility across runs.
"""
function CharTokenizer(text::String)
    chars = sort(collect(Set(text)))
    char_to_idx = Dict(c => i for (i, c) in enumerate(chars))
    return CharTokenizer(char_to_idx, chars, length(chars))
end

"""
    CharTokenizer(chars::Vector{Char}) -> CharTokenizer

Build a tokenizer from an explicit character list.
"""
function CharTokenizer(chars::Vector{Char})
    sorted = sort(chars)
    char_to_idx = Dict(c => i for (i, c) in enumerate(sorted))
    return CharTokenizer(char_to_idx, sorted, length(sorted))
end

vocab_size(t::CharTokenizer) = t.vocab_sz

"""
    encode(tokenizer, text) -> Vector{Int}

Encode a string into a vector of integer token indices.
Unknown characters are silently skipped.
"""
function encode(t::CharTokenizer, text::String)
    indices = Int[]
    sizehint!(indices, length(text))
    for c in text
        idx = get(t.char_to_idx, c, nothing)
        if idx !== nothing
            push!(indices, idx)
        end
    end
    return indices
end

"""
    decode(tokenizer, indices) -> String

Decode a vector of integer token indices back into a string.
"""
function decode(t::CharTokenizer, indices::AbstractVector{<:Integer})
    buf = IOBuffer()
    for idx in indices
        if 1 <= idx <= t.vocab_sz
            write(buf, t.idx_to_char[idx])
        end
    end
    return String(take!(buf))
end

function Base.show(io::IO, t::CharTokenizer)
    sample = join(t.idx_to_char[1:min(10, t.vocab_sz)])
    print(io, "CharTokenizer(vocab=$(t.vocab_sz), chars=\"$(sample)...\")")
end
