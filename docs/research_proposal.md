# JuliaGPT: A Pure Julia Small Language Model Trained on Classical Texts

## Research Proposal

**Date:** 2026-02-21
**Author:** buildwithbooks
**Repository:** buildwithbooks/text-pipeline

---

## Abstract

We propose JuliaGPT, a small language model implemented entirely in Julia, trained on a curated corpus of classical philosophical and mathematical texts spanning from Euclid and Aristotle to Kant and Nietzsche. The project explores whether techniques from scientific computing — continuous-depth neural ODEs, adjoint-method backpropagation, sparse Jacobian exploitation, and sensitivity-guided data curation — can produce competitive small language models with significantly less compute than standard approaches. Training targets a single A100 GPU (Google Colab); inference targets consumer GPUs with less than 6GB VRAM. The model architecture draws from Karpathy's nanoGPT while incorporating structured matrix methods from numerical analysis and sparse attention patterns observed in particle physics transformers.

---

## 1. Motivation

### 1.1 The Case for Small

Large language models demonstrate remarkable capabilities but require enormous compute, data, and energy. A counter-thesis is emerging: small, purpose-trained models on carefully curated data can achieve domain-specific quality that rivals general-purpose models orders of magnitude larger. Microsoft's Phi-4 (14B, trained primarily on synthetic data) surpasses GPT-4 on STEM benchmarks. SmallThinker (0.6B active parameters) achieves state-of-the-art on reasoning tasks. The "Depth Delusion" paper shows most model capacity is wasted in excessive depth.

Our constraint profile — single A100 for training, consumer GPU for inference, small curated corpus — forces a disciplined approach where every parameter and every training example must earn its place.

### 1.2 The Case for Julia

Julia occupies a unique position: the expressiveness of Python with the performance of C, native GPU kernel programming, and the richest scientific computing ecosystem of any language (SciML). This means:

- Techniques from differential equations (adjoint methods, continuous-depth models) are first-class citizens, not research hacks bolted onto a deep learning framework
- Custom GPU kernels can be written in the same language as the training loop, without C++/CUDA extensions
- Multiple dispatch enables composing novel architectures from mathematical primitives naturally
- The entire pipeline — data processing, model definition, training, inference — lives in one language

No production language model has been trained in Julia. This is simultaneously the risk and the opportunity.

### 1.3 The Case for Classical Texts

The training corpus comprises ~50 foundational works of Western philosophy and mathematics, organized by the classical liberal arts:

**Trivium** (language arts):
- Logic: Aristotle's Organon (Categories, Prior/Posterior Analytics, Topics, On Interpretation), Plato's Meno and Theaetetus, Descartes' Discourse on the Method
- Rhetoric: Aristotle's Rhetoric and Poetics, Plato's dialogues (Symposium, Phaedrus, Gorgias, Protagoras), Bacon's and Emerson's Essays
- Ethics/Politics: Nicomachean Ethics, Republic, Leviathan, Meditations, On Liberty, Beyond Good and Evil

**Quadrivium** (mathematical arts):
- Geometry: Euclid's Elements
- Physics: Aristotle's Physics, On the Heavens, Lucretius' On the Nature of Things, Plato's Timaeus

Estimated total: ~2.8M words → ~10-12M characters after cleaning. This is small by modern standards (GPT-2 trained on 40GB; our corpus is ~12MB), but the texts represent 2,500 years of humanity's most refined argumentation, logical structure, and rhetorical technique. The hypothesis is that quality of training data can partially substitute for quantity when the domain is well-defined.

---

## 2. Technical Approach

### 2.1 Data Pipeline (Existing)

The text processing pipeline is operational and produces clean, chunked training data:

| Stage | Implementation | Output |
|-------|---------------|--------|
| Acquisition | gutenberg_search.py, mit_classics_search.py, ia_search.py | Raw .txt/.epub/.zip files |
| Parsing | parsers/txt_parser.py, epub_parser.py, zip_parser.py | Normalized text |
| Cleaning | cleaner.py | Lowercase, Unicode normalized, numerals converted, allowed charset enforced |
| Chunking | chunker.py | 40-256 char chunks, sentence-boundary aware |
| Splitting | pipeline.py | 90/10 train/val, shuffled, seeded |

Current allowed characters: `a-z .,;:?!'"-()`
Current output: 349KB train, 39KB val (Euclid only; full corpus not yet downloaded)

**Proposed enhancements:**
- Add difficulty scoring per chunk (compression ratio, lexical diversity, readability) for curriculum learning
- Add perplexity-based and gradient-norm-based chunk ranking for coreset selection
- Score chunks after initial model training to identify most informative examples

### 2.2 Model Architecture

#### Phase 1: Baseline Transformer (Weeks 1-3)

A nanoGPT-style decoder-only transformer in Lux.jl:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Vocabulary | ~40 chars | Current allowed charset mapped to indices |
| Embed dim | 384 | nanoGPT Shakespeare config; multiple of 128 |
| Layers | 6 | Depth Delusion sweet spot; nanochat formula |
| Heads | 3 | 384 / 128 head_dim |
| Head dim | 128 | nanochat default |
| FFN dim | 1536 | 4 × embed_dim (standard) |
| Context length | 256 | Current chunk size; expand to 512 in Phase 2 |
| Position encoding | RoPE | Modern standard; built into Lux.jl |
| Normalization | RMSNorm | nanochat default |
| Activation | SwiGLU | nanochat default |
| Dropout | 0.0 | Karpathy recommendation for small nets |
| Bias | false | Marginal efficiency gain |
| **Est. parameters** | **~10-15M** | Appropriate for corpus size |

Lux.jl implementation sketch:

```julia
using Lux, Random, LuxCUDA

function JuliaGPT(;
    vocab_size=40, dim=384, nheads=3, nlayers=6,
    max_seq_len=256, ffn_mult=4, dropout=0.0f0
)
    @compact(
        tok_emb = Embedding(vocab_size => dim),
        rope = RotaryPositionalEmbedding(dim ÷ nheads),
        blocks = [TransformerBlock(dim; nheads, ffn_mult, dropout)
                  for _ in 1:nlayers],
        ln_f = RMSNorm(dim),
        head = Dense(dim => vocab_size; use_bias=false),
    ) do x
        h = tok_emb(x)
        for block in blocks
            h = block(h)
        end
        @return head(ln_f(h))
    end
end
```

Training configuration:
- Optimizer: AdamW (via Optimisers.jl), lr=6e-4 with cosine decay
- Batch size: 64 (auto-tuned to A100 memory)
- Precision: Float16 mixed precision (Float32 master weights)
- Gradient accumulation as needed
- Training time estimate: <1 hour on A100

#### Phase 2: Structured Matrix Variant (Weeks 4-6)

Replace standard attention and MLP with structured matrix operations based on the Monarch Mixer architecture:

- **Monarch attention**: Replace softmax attention with Monarch matrix multiplication (generalized FFT). Sub-quadratic: O(N^{3/2}) vs O(N^2). Hardware-friendly GEMMs.
- **Monarch MLP**: Replace dense FFN with Monarch matrix layers.
- **Target**: Match Phase 1 quality with 27% fewer parameters and significantly higher throughput.
- **Reference**: M2-BERT matches BERT-base at this compression ratio. MonarchAttention (May 2025) enables zero-shot conversion.

This is the most promising architecture for the <6GB inference constraint. A 50M Monarch model would have the effective capacity of a ~70M standard transformer.

#### Phase 3: Continuous-Depth Neural ODE Variant (Weeks 7-10)

Replace discrete transformer layers with a continuous-depth ODE, using DiffEqFlux.jl:

```julia
using DiffEqFlux, OrdinaryDiffEq

# Single set of transformer weights, continuous "depth" variable
node_block = NeuralODE(
    TransformerBlock(dim; nheads, dropout),
    (0.0f0, 1.0f0),  # integrate from depth 0 to 1
    Tsit5();          # adaptive ODE solver
    save_everystep=false,
    reltol=1e-3, abstol=1e-3
)
```

Key advantages:
- **Parameter sharing by construction**: One set of weights replaces N layer-specific weight sets
- **Adaptive compute**: Fewer ODE solver steps for simple tokens, more for complex ones
- **O(1) memory in depth**: via SciMLSensitivity.jl adjoint methods
- **Reference**: Neural ODE Transformer (arXiv 2503.01329) — 3.5B continuous-depth with 32 iterations matches 50B traditional

Risk: No one has built a language model this way in Julia. This is research.

### 2.3 Training Strategy

#### 2.3.1 Curriculum Learning

Order training data from easy to difficult. Difficulty scoring per chunk:

1. **Compression ratio**: gzip compressed length / raw length (lower = more predictable = easier)
2. **Lexical diversity**: MTLD (Measure of Textual Lexical Diversity)
3. **Readability**: Flesch Reading Ease adapted for character-level (sentence length, word length)

Expected impact: 18-45% reduction in training steps to reach baseline performance (arXiv 2506.11300).

Implementation: Score all chunks in the text pipeline, store scores in metadata, sort training batches by difficulty with epoch-wise progression from easy to hard.

#### 2.3.2 Coreset Selection

After initial training (Phase 1 baseline), identify the most informative training examples:

1. Score each chunk by gradient norm during a single pass
2. Score by perplexity (high-perplexity chunks contain the most novel information)
3. Select top-K% for focused retraining

Expected impact: Up to 78% increase in training efficiency (PICore, 2025). Data selection via optimal control achieves 2x training acceleration.

#### 2.3.3 Sensitivity-Guided Data Curation

Use SciML's GlobalSensitivity.jl to determine:
- Which training examples most influence model behavior on the validation set
- Which regions of the corpus are under-represented (guiding future data collection)
- Which parameters are structurally identifiable (informing pruning decisions)

This is the technique that most directly leverages Julia's SciML advantage — it has no clean equivalent in PyTorch.

#### 2.3.4 Progressive Quantization for Inference

Train at full precision, then compress for deployment:

1. Train in Float16 mixed precision on A100
2. After convergence, transition to 1.58-bit quantization-aware training (BitNet strategy)
3. Fine-tune at ternary precision for 10-20% of total training steps
4. Deploy at 1.58-bit: a 50M model needs ~12MB for weights

The "16-to-1.58" progressive strategy has been validated for models as small as 100K parameters (SCITEPRESS 2025).

### 2.4 Inference Optimization

Target: responsive generation on a consumer GPU with <6GB VRAM.

| Technique | Memory Reduction | Implementation |
|-----------|-----------------|----------------|
| Float16 inference | 2x vs FP32 | Native Julia type conversion |
| 1.58-bit quantization | ~10x vs FP16 | BitNet-style ternary weights |
| Reactant.jl XLA compilation | Operation fusion, reduced overhead | `@compile` macro |
| Tensor-Train decomposition of FFN | Up to 95% for FFN layers | Julia tensor network packages |
| Sparse attention (top-k) | Reduces attention memory linearly with k | Custom implementation |
| KV cache with sliding window | Bounded memory for generation | Custom implementation |

For the Neural ODE variant: adaptive solver tolerance at inference provides a natural "quality knob" — reduce tolerance for faster generation with marginal quality loss.

---

## 3. Experimental Design

### 3.1 Baselines

| Model | Description | Purpose |
|-------|-------------|---------|
| **B1: CharRNN** | LSTM character-level model, ~5M params | Non-transformer baseline |
| **B2: nanoGPT-Julia** | Phase 1 standard transformer | Primary baseline |
| **B3: nanoGPT-PyTorch** | Same architecture in PyTorch | Julia vs Python performance comparison |

### 3.2 Experimental Models

| Model | Description | Hypothesis |
|-------|-------------|-----------|
| **E1: Monarch-Julia** | Phase 2 Monarch Mixer | Matches B2 quality with fewer params and higher throughput |
| **E2: NeuralODE-Julia** | Phase 3 continuous-depth | Adaptive compute enables better quality/memory tradeoff at inference |
| **E3: B2 + Curriculum** | Baseline + curriculum learning | 18-45% fewer training steps to match B2 |
| **E4: B2 + Coreset** | Baseline + coreset selection | Higher quality with subset of data |
| **E5: B2 + KFAC** | Baseline + second-order optimizer | Faster convergence on small dataset |
| **E6: B2 + Distillation** | Baseline initialized from larger model distillation | Better quality at same param count |

### 3.3 Evaluation Metrics

**Intrinsic:**
- Character-level perplexity on held-out validation set
- Bits per character (BPC)
- Loss curve convergence speed (steps to reach threshold perplexity)

**Generation quality:**
- Manual evaluation of coherence, argumentative structure, vocabulary usage
- N-gram diversity (distinct-1, distinct-2, distinct-3)
- Repetition rate (fraction of generated text that repeats)

**Efficiency:**
- Training time to convergence (A100 hours)
- Training FLOPs to target perplexity
- Inference throughput (characters/second) on target GPU
- Peak inference memory (MB)
- Model size on disk (MB)

**Cross-domain technique impact:**
- Perplexity improvement per technique vs baseline
- Training step reduction per technique
- Inference memory reduction per technique

### 3.4 Ablation Studies

1. **Depth vs width**: Fix param budget at 15M, vary depth (3→12) and width accordingly
2. **Context length**: 128, 256, 512, 1024 at fixed architecture
3. **Vocabulary**: Character (40) vs small BPE (1K) vs medium BPE (8K)
4. **Corpus composition**: Trivium-only vs Quadrivium-only vs combined (measure cross-domain transfer)
5. **Curriculum ordering**: Random vs easy-first vs hard-first vs mixed
6. **ODE solver tolerance**: 1e-1, 1e-2, 1e-3, 1e-4 (quality vs speed tradeoff for NeuralODE variant)

---

## 4. Implementation Plan

### 4.1 Timeline

| Phase | Weeks | Deliverable |
|-------|-------|-------------|
| **0: Corpus completion** | 1 | Download all 50 texts, run pipeline, produce full train/val splits |
| **1: Baseline transformer** | 1-3 | Lux.jl nanoGPT implementation, training on A100, generation working |
| **2: Cross-domain techniques** | 3-5 | Curriculum learning, coreset selection, difficulty scoring in pipeline |
| **3: Monarch variant** | 4-6 | Structured matrix attention/MLP, benchmarked against baseline |
| **4: NeuralODE variant** | 7-10 | Continuous-depth model via DiffEqFlux.jl, adjoint training |
| **5: Inference optimization** | 8-10 | Quantization, Reactant compilation, benchmarks on consumer GPU |
| **6: Evaluation** | 10-12 | Full ablation studies, cross-technique comparisons, writeup |

### 4.2 Julia Package Dependencies

```julia
# Project.toml (core)
[deps]
Lux = "..."           # Neural network framework
LuxCUDA = "..."       # CUDA backend
Reactant = "..."      # XLA compilation for inference
Optimisers = "..."    # AdamW, learning rate scheduling
Zygote = "..."        # Automatic differentiation (stable)
Random = "..."        # RNG management
Statistics = "..."    # Training metrics
JLD2 = "..."          # Model serialization
CUDA = "..."          # Direct GPU control

# Phase 3: Continuous-depth
DiffEqFlux = "..."    # Neural ODE layers
OrdinaryDiffEq = "..." # ODE solvers
SciMLSensitivity = "..." # Adjoint methods

# Analysis
GlobalSensitivity = "..." # Sensitivity-guided data curation
SparseDiffTools = "..."   # Sparse Jacobian exploitation

# Optional
Enzyme = "..."        # Alternative AD backend
KernelAbstractions = "..." # Custom GPU kernels
```

### 4.3 Compute Budget

| Task | Hardware | Estimated Time | Estimated Cost |
|------|----------|---------------|----------------|
| Phase 1 training (all ablations) | Colab A100 | ~8 hours | Free tier / $12 |
| Phase 2 curriculum/coreset | Colab A100 | ~4 hours | Free tier / $6 |
| Phase 3 Monarch training | Colab A100 | ~8 hours | $12 |
| Phase 4 NeuralODE training | Colab A100 | ~12 hours | $18 |
| Phase 5 inference benchmarks | Consumer GPU (<6GB) | ~2 hours | $0 |
| **Total** | | **~34 hours A100** | **~$48** |

### 4.4 Repository Structure

```
buildwithbooks/
├── text-pipeline/          # Existing data processing pipeline
│   ├── sources/            # Download scripts, source manifest
│   ├── parsed/             # Cleaned text files
│   ├── output/             # train.txt, val.txt
│   └── docs/               # This proposal
│
└── julia-gpt/              # New: model implementation
    ├── src/
    │   ├── model.jl        # JuliaGPT architecture (Lux.jl)
    │   ├── monarch.jl      # Monarch Mixer variant
    │   ├── neural_ode.jl   # Continuous-depth variant
    │   ├── data.jl         # Data loading, batching, curriculum
    │   ├── train.jl        # Training loop
    │   ├── generate.jl     # Text generation / sampling
    │   ├── quantize.jl     # BitNet 1.58-bit QAT
    │   └── evaluate.jl     # Metrics and evaluation
    ├── notebooks/
    │   ├── train_colab.ipynb
    │   └── ablations.ipynb
    ├── scripts/
    │   ├── train.jl        # CLI training entry point
    │   └── generate.jl     # CLI generation entry point
    ├── Project.toml
    └── Manifest.toml
```

---

## 5. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Lux.jl transformer layers have bugs/gaps | Medium | High | Fall back to Flux.jl; TransformersLite.jl as reference |
| BFloat16 instability on A100 | Medium | Low | Use Float16 mixed precision instead |
| No Flash Attention in Julia | Certain | Low | Context ≤512 makes quadratic attention manageable |
| Neural ODE training diverges | Medium | Medium | Phase 1 baseline is the deliverable; NeuralODE is experimental |
| Corpus too small for meaningful model | Low | High | Character-level tokenization maximizes effective training examples; curriculum learning and coreset selection stretch data |
| Julia GPU memory management (GC pauses) | Medium | Low | CUDA.jl 5.4 proactive GC; set soft memory limits |
| Colab session timeouts | Medium | Medium | Checkpoint every N steps; resume training across sessions |
| Monarch Mixer math is hard to implement from scratch | Medium | Medium | Start with standard attention; MonarchAttention paper provides conversion procedure |

---

## 6. Success Criteria

### Minimum Viable Outcome
- A working character-level transformer in pure Julia (Lux.jl) that generates coherent English text in the style of classical philosophical prose
- Validation perplexity competitive with an equivalent PyTorch implementation
- Inference running on a consumer GPU with <6GB VRAM

### Target Outcome
- At least one cross-domain technique (curriculum learning, coreset selection, or KFAC) demonstrably reduces training compute by >20% vs baseline
- Monarch Mixer variant matches baseline quality with ≥20% fewer parameters
- 1.58-bit quantized model fits in <50MB with <10% perplexity degradation

### Stretch Outcome
- Neural ODE variant demonstrates adaptive compute at inference (variable quality/speed tradeoff)
- Published results showing SciML techniques (adjoint methods, sensitivity analysis) applied to language model training for the first time in Julia
- Model generates text that captures the argumentative structure of classical philosophy (not just surface-level word patterns)

---

## 7. Related Work

- **nanoGPT** (Karpathy, 2023): 600-line GPT-2 implementation. Our starting architecture.
- **nanochat** (Karpathy, 2026): Single-dial scaling with modern architecture choices. Our hyperparameter scaling reference.
- **TransformersLite.jl** (Sinai, 2024): 44K-param Julia GPT on Shakespeare. Proof of concept for Julia transformers.
- **Phi-4** (Microsoft, 2024): 14B model trained on synthetic data surpassing GPT-4 on STEM. Validates small-model + curated-data hypothesis.
- **SmallThinker** (2025): 0.6B-active MoE achieving SOTA reasoning. Validates small-model viability.
- **Monarch Mixer** (Dao et al., 2023): Sub-quadratic architecture matching GPT quality. Our Phase 2 target.
- **Neural ODE Transformers** (2025): Continuous-depth models matching 50B traditional. Our Phase 3 inspiration.
- **BitNet b1.58** (Microsoft, 2024): Ternary-weight LLMs. Our quantization strategy.
- **"The Depth Delusion"** (2025): Width matters more than depth. Our architecture sizing guide.
