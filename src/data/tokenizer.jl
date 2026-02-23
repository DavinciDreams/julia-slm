"""
Tokenizers: character-level and BPE. Builds vocabulary dynamically from training data.
"""

# ─────────────────────────────────────────────
# Character-level tokenizer
# ─────────────────────────────────────────────

struct CharTokenizer
    char_to_idx::Dict{Char,Int}
    idx_to_char::Vector{Char}
    vocab_sz::Int
    unk_id::Int  # 0 = no UNK token (drop unknown chars)
end

"""
    CharTokenizer(text::String; add_unk=false) -> CharTokenizer

Build a character-level tokenizer from a text corpus. Vocabulary is sorted
for reproducibility across runs. If `add_unk=true`, unknown characters map to
an <UNK> token instead of being dropped.
"""
function CharTokenizer(text::String; add_unk::Bool=false)
    chars = sort(collect(Set(text)))
    if add_unk
        pushfirst!(chars, '\x00')  # placeholder for UNK
    end
    char_to_idx = Dict(c => i for (i, c) in enumerate(chars))
    unk_id = add_unk ? 1 : 0
    return CharTokenizer(char_to_idx, chars, length(chars), unk_id)
end

"""
    CharTokenizer(chars::Vector{Char}; add_unk=false) -> CharTokenizer

Build a tokenizer from an explicit character list.
"""
function CharTokenizer(chars::Vector{Char}; add_unk::Bool=false)
    sorted = sort(chars)
    if add_unk
        pushfirst!(sorted, '\x00')
    end
    char_to_idx = Dict(c => i for (i, c) in enumerate(sorted))
    unk_id = add_unk ? 1 : 0
    return CharTokenizer(char_to_idx, sorted, length(sorted), unk_id)
end

vocab_size(t::CharTokenizer) = t.vocab_sz

"""
    encode(tokenizer, text) -> Vector{Int}

Encode a string into a vector of integer token indices.
Unknown characters map to UNK token if available, otherwise are dropped with a warning.
"""
function encode(t::CharTokenizer, text::String)
    indices = Int[]
    sizehint!(indices, length(text))
    n_unk = 0
    for c in text
        idx = get(t.char_to_idx, c, nothing)
        if idx !== nothing
            push!(indices, idx)
        elseif t.unk_id > 0
            push!(indices, t.unk_id)
            n_unk += 1
        else
            n_unk += 1
        end
    end
    if n_unk > 0
        drop_rate = n_unk / max(length(text), 1)
        if drop_rate > 0.01
            @warn "High unknown char rate" n_unk=n_unk total=length(text) rate=round(drop_rate; digits=4)
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
            c = t.idx_to_char[idx]
            c != '\x00' && write(buf, c)  # skip UNK placeholder
        end
    end
    return String(take!(buf))
end

function Base.show(io::IO, t::CharTokenizer)
    sample = join(t.idx_to_char[1:min(10, t.vocab_sz)])
    unk_str = t.unk_id > 0 ? ", unk=true" : ""
    print(io, "CharTokenizer(vocab=$(t.vocab_sz)$(unk_str), chars=\"$(sample)...\")")
end

# ─────────────────────────────────────────────
# BPE Tokenizer (GPT-2 style)
# ─────────────────────────────────────────────

struct BPETokenizer
    encoder::Dict{String,Int}     # token string → id
    decoder::Dict{Int,String}     # id → token string
    merges::Vector{Tuple{String,String}}  # ordered merge rules
    merge_ranks::Dict{Tuple{String,String},Int}  # merge pair → priority rank
    byte_to_unicode::Dict{UInt8,Char}
    unicode_to_byte::Dict{Char,UInt8}
    vocab_sz::Int
    pat::Regex  # pre-tokenization pattern
end

"""
    BPETokenizer(vocab_path::String, merges_path::String) -> BPETokenizer

Load a GPT-2 style BPE tokenizer from vocab.json and merges.txt files.
"""
function BPETokenizer(vocab_path::String, merges_path::String)
    # Load vocabulary
    encoder = JSON3.read(read(vocab_path, String), Dict{String,Int})
    decoder = Dict{Int,String}(v => k for (k, v) in encoder)

    # Load merges
    merge_lines = readlines(merges_path)
    # Skip header line if present (e.g. "#version: ...")
    start = startswith(first(merge_lines), "#") ? 2 : 1
    merges = Tuple{String,String}[]
    for line in merge_lines[start:end]
        parts = split(strip(line))
        length(parts) == 2 && push!(merges, (String(parts[1]), String(parts[2])))
    end
    merge_ranks = Dict{Tuple{String,String},Int}(m => i for (i, m) in enumerate(merges))

    # Byte-to-unicode mapping (GPT-2 style)
    b2u = _build_byte_to_unicode()
    u2b = Dict{Char,UInt8}(v => k for (k, v) in b2u)

    # GPT-2 pre-tokenization pattern
    pat = r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"

    return BPETokenizer(encoder, decoder, merges, merge_ranks, b2u, u2b,
                        length(encoder), pat)
end

vocab_size(t::BPETokenizer) = t.vocab_sz

"""
    encode(t::BPETokenizer, text::String) -> Vector{Int}

Encode text using BPE. Pre-tokenizes with GPT-2 regex, then applies BPE merges.
"""
function encode(t::BPETokenizer, text::String)
    tokens = Int[]
    for m in eachmatch(t.pat, text)
        word = m.match
        # Convert to unicode representation
        encoded_chars = [string(t.byte_to_unicode[b]) for b in Vector{UInt8}(word)]
        bpe_tokens = _bpe_encode_word(encoded_chars, t.merge_ranks)
        for tok in bpe_tokens
            id = get(t.encoder, tok, nothing)
            id !== nothing && push!(tokens, id)
        end
    end
    return tokens
end

"""
    decode(t::BPETokenizer, ids::AbstractVector{<:Integer}) -> String

Decode BPE token ids back to text.
"""
function decode(t::BPETokenizer, ids::AbstractVector{<:Integer})
    token_strs = [get(t.decoder, id, "") for id in ids]
    joined = join(token_strs)
    # Convert unicode chars back to bytes
    bytes = UInt8[get(t.unicode_to_byte, c, UInt8(c)) for c in joined]
    return String(bytes)
end

function _bpe_encode_word(symbols::Vector{String}, merge_ranks::Dict{Tuple{String,String},Int})
    # Iteratively merge the highest-priority pair
    while length(symbols) > 1
        # Find the merge pair with lowest rank (highest priority)
        best_pair = nothing
        best_rank = typemax(Int)
        for i in 1:length(symbols)-1
            pair = (symbols[i], symbols[i+1])
            rank = get(merge_ranks, pair, typemax(Int))
            if rank < best_rank
                best_rank = rank
                best_pair = pair
            end
        end
        best_rank == typemax(Int) && break  # no more merges

        # Apply the merge
        new_symbols = String[]
        i = 1
        while i <= length(symbols)
            if i < length(symbols) && symbols[i] == best_pair[1] && symbols[i+1] == best_pair[2]
                push!(new_symbols, best_pair[1] * best_pair[2])
                i += 2
            else
                push!(new_symbols, symbols[i])
                i += 1
            end
        end
        symbols = new_symbols
    end
    return symbols
end

function _build_byte_to_unicode()
    # GPT-2 byte-to-unicode mapping
    bs = UInt8[]
    # Printable ASCII ranges that map to themselves
    append!(bs, UInt8('!'):UInt8('~'))
    append!(bs, UInt8('¡'):UInt8('¬'))
    append!(bs, UInt8('®'):UInt8('ÿ'))
    cs = Int[Int(b) for b in bs]
    n = 0
    for b in 0x00:0xff
        if !(b in bs)
            push!(bs, b)
            push!(cs, 256 + n)
            n += 1
        end
    end
    return Dict{UInt8,Char}(b => Char(c) for (b, c) in zip(bs, cs))
end

function Base.show(io::IO, t::BPETokenizer)
    print(io, "BPETokenizer(vocab=$(t.vocab_sz), merges=$(length(t.merges)))")
end
