# JuliaGPT: Product Requirements Document

## Proof of Concept — Publishable Research Implementation

**Version:** 0.1.0
**Date:** 2026-02-21
**Status:** Draft
**Authors:** buildwithbooks

---

## 1. Product Overview

### 1.1 One-Line Summary

A pure Julia character-level language model trained on classical philosophical texts, demonstrating that scientific computing techniques (Neural ODEs, adjoint backpropagation, sensitivity-guided data curation) can improve training efficiency for small language models under severe compute constraints.

### 1.2 Publication Target

**Working title:** *"JuliaGPT: Cross-Domain Scientific Computing Methods for Efficient Small Language Model Training"*

**Target venues (in priority order):**
1. NeurIPS — Machine Learning and the Physical Sciences Workshop
2. ICML — Workshop on Efficient Systems for Foundation Models
3. JuliaCon Proceedings
4. arXiv preprint (immediate)

**Novel contributions the paper will claim:**
1. First language model trained end-to-end in Julia using the SciML ecosystem
2. Application of continuous-depth Neural ODE transformers to language modeling with adjoint-method backpropagation
3. Sensitivity-guided data curation for small curated corpora
4. Empirical comparison of Monarch Mixer vs. standard attention vs. continuous-depth at the <50M parameter scale
5. Curriculum learning with DoReMi-style phase weighting for domain-structured corpora

### 1.3 Non-Goals

- Competing with frontier models on general benchmarks
- Building a chat interface or instruction-following model
- Supporting distributed multi-GPU training
- Production deployment beyond proof-of-concept inference

---

## 2. Requirements

### 2.1 Functional Requirements

#### FR-1: Data Pipeline Integration

| ID | Requirement | Priority | Acceptance Criteria |
|----|------------|----------|-------------------|
| FR-1.1 | Load training data from text-pipeline output (`train.txt`, `val.txt`) | P0 | Model trains on character sequences from pipeline output |
| FR-1.2 | Load metadata-enriched JSONL (`train_enriched.jsonl`) with per-chunk author, art, phase, difficulty score | P0 | DataLoader can filter/weight chunks by metadata fields |
| FR-1.3 | Support DoReMi-style phase-weighted sampling using `phase_weights` from config | P0 | Trivium/quadrivium/philosophy sampled at 40/35/25 ratios |
| FR-1.4 | Support curriculum ordering by difficulty score (easy → hard progression) | P1 | Training can be configured to present chunks in difficulty order within each epoch |
| FR-1.5 | Support coreset selection: rank chunks by gradient norm or perplexity, train on top-K% | P1 | After initial training pass, re-rank and retrain on subset |

**Data pipeline outputs consumed:**
```
text-pipeline/output/
├── train.txt              # Plain text, one chunk per line
├── val.txt                # Validation split
└── train_enriched.jsonl   # {"text": "...", "author": "aristotle", "art": "logic",
                           #  "phase": "trivium", "difficulty": 0.42, "quality": 0.78}
```

#### FR-2: Model Architecture

| ID | Requirement | Priority | Acceptance Criteria |
|----|------------|----------|-------------------|
| FR-2.1 | Implement baseline decoder-only transformer in Lux.jl | P0 | Forward pass produces logits; backward pass computes gradients |
| FR-2.2 | Configurable: vocab_size, embed_dim, n_layers, n_heads, context_length, dropout | P0 | All hyperparameters controllable via config file |
| FR-2.3 | RoPE positional encoding (Lux.jl built-in `RotaryPositionalEmbedding`) | P0 | Position information encoded without learned position embeddings |
| FR-2.4 | RMSNorm pre-normalization | P0 | Each transformer block uses pre-norm with RMSNorm |
| FR-2.5 | SwiGLU activation in feed-forward blocks | P0 | FFN uses gated linear unit with swish activation |
| FR-2.6 | Causal masking in attention | P0 | Model cannot attend to future tokens |
| FR-2.7 | Weight tying between token embedding and output projection | P1 | Single parameter matrix shared; reduces param count |
| FR-2.8 | Implement Monarch Mixer variant (structured matrix attention + MLP) | P1 | Sub-quadratic forward pass; benchmarked against baseline |
| FR-2.9 | Implement continuous-depth Neural ODE variant via DiffEqFlux.jl | P2 | Single weight set with ODE integration replacing discrete layers |

#### FR-3: Training

| ID | Requirement | Priority | Acceptance Criteria |
|----|------------|----------|-------------------|
| FR-3.1 | Training loop with AdamW optimizer | P0 | Loss decreases; model learns to generate text |
| FR-3.2 | Cosine learning rate schedule with warmup | P0 | LR warms up linearly, then decays via cosine |
| FR-3.3 | Float16 mixed precision training on A100 | P0 | Master weights in Float32, forward/backward in Float16 |
| FR-3.4 | Gradient clipping (max norm) | P0 | Prevents gradient explosion |
| FR-3.5 | Periodic checkpointing (save model + optimizer state via JLD2) | P0 | Training resumable from any checkpoint |
| FR-3.6 | Validation loss computed every N steps | P0 | Overfitting detectable during training |
| FR-3.7 | Training metrics logged: loss, perplexity, learning rate, grad norm, tokens/sec | P0 | Metrics available for plotting and analysis |
| FR-3.8 | KFAC second-order optimizer as alternative to AdamW | P2 | Convergence speed comparison vs first-order |

#### FR-4: Text Generation

| ID | Requirement | Priority | Acceptance Criteria |
|----|------------|----------|-------------------|
| FR-4.1 | Autoregressive character-level text generation | P0 | Given a prompt, model generates continuation character by character |
| FR-4.2 | Temperature-controlled sampling | P0 | Temperature parameter scales logits before softmax |
| FR-4.3 | Top-k sampling | P0 | Only top-k most probable tokens considered |
| FR-4.4 | Top-p (nucleus) sampling | P1 | Smallest set of tokens with cumulative probability ≥ p |
| FR-4.5 | Deterministic greedy decoding | P0 | Argmax at each step for reproducible output |

#### FR-5: Evaluation

| ID | Requirement | Priority | Acceptance Criteria |
|----|------------|----------|-------------------|
| FR-5.1 | Character-level perplexity on validation set | P0 | Computed and reported after training |
| FR-5.2 | Bits per character (BPC) | P0 | BPC = log2(perplexity) |
| FR-5.3 | Training FLOPs estimation | P0 | Estimated from model size, batch size, steps |
| FR-5.4 | Inference throughput (chars/sec) on target GPU | P0 | Benchmarked with generation loop |
| FR-5.5 | Peak GPU memory during inference (MB) | P0 | Measured via CUDA.jl memory reporting |
| FR-5.6 | N-gram diversity (distinct-1, distinct-2, distinct-3) on generated text | P1 | Measured on 10K generated characters |
| FR-5.7 | Repetition rate | P1 | Fraction of 4-grams that appear more than once in generated output |
| FR-5.8 | Cross-technique comparison table | P0 | Perplexity, training steps, memory, throughput for each variant |

#### FR-6: Inference Optimization

| ID | Requirement | Priority | Acceptance Criteria |
|----|------------|----------|-------------------|
| FR-6.1 | Inference in Float16 | P0 | Model loads and generates in Float16 |
| FR-6.2 | Reactant.jl XLA compilation for inference | P1 | Compiled model runs faster than eager mode |
| FR-6.3 | 1.58-bit quantization-aware training (BitNet strategy) | P2 | Ternary-weight model generates coherent text |
| FR-6.4 | KV cache for efficient autoregressive generation | P1 | Past key/value pairs reused; no recomputation |
| FR-6.5 | Model fits in <6GB GPU VRAM during inference | P0 | Verified on target hardware |

### 2.2 Non-Functional Requirements

| ID | Requirement | Priority | Acceptance Criteria |
|----|------------|----------|-------------------|
| NF-1 | All model code in pure Julia (no Python/C++ dependencies for core model) | P0 | `src/` contains only `.jl` files; no PyCall |
| NF-2 | Reproducible: fixed RNG seeds produce identical training runs | P0 | Two runs with same seed produce same final loss |
| NF-3 | Single-file model definition readable by someone familiar with nanoGPT | P0 | `src/model/julia_gpt.jl` is self-contained and < 300 lines |
| NF-4 | Training runs on Google Colab A100 within a single session (~8 hours max) | P0 | Baseline model trains to convergence in one session |
| NF-5 | Configuration via TOML file (Julia standard) | P0 | All hyperparameters externalized |
| NF-6 | Code documented sufficiently for paper reproducibility | P0 | Reviewer can reproduce results from repo + paper |

---

## 3. Architecture

### 3.1 System Overview

```
┌──────────────────────────────────────────────────────────────┐
│                    text-pipeline (Python)                     │
│  sources → parsers → cleaner → chunker → quality scorer      │
│                                    │                         │
│         train.txt  val.txt  train_enriched.jsonl             │
└──────────────────┬───────────────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────────────┐
│                     julia-slm (Julia)                         │
│                                                              │
│  ┌─────────┐   ┌──────────┐   ┌───────────┐   ┌──────────┐ │
│  │  data/   │──▶│ training/ │──▶│  model/   │──▶│inference/│ │
│  │ loader   │   │  loop     │   │  weights  │   │ generate │ │
│  │ curriculum│   │  optim    │   │  (JLD2)   │   │ (XLA)   │ │
│  │ coreset  │   │  metrics  │   │           │   │          │ │
│  └─────────┘   └──────────┘   └───────────┘   └──────────┘ │
│                                                              │
│  ┌─────────────────────────────────────┐                     │
│  │          evaluation/                │                     │
│  │  perplexity, BPC, throughput,       │                     │
│  │  diversity, ablation runner         │                     │
│  └─────────────────────────────────────┘                     │
└──────────────────────────────────────────────────────────────┘
```

### 3.2 Module Breakdown

#### `src/model/` — Model Definitions

| File | Contents | Lines (est.) |
|------|----------|-------------|
| `julia_gpt.jl` | Baseline transformer: embeddings, MHA, FFN, causal mask, forward pass | ~250 |
| `monarch.jl` | Monarch Mixer variant: Monarch matrix layers replacing attention + MLP | ~200 |
| `neural_ode.jl` | Continuous-depth variant: NeuralODE wrapper around transformer block | ~150 |
| `layers.jl` | Shared components: RMSNorm, SwiGLU, KV cache | ~100 |
| `config.jl` | Model configuration struct, TOML loading, nanochat-style scaling | ~80 |

#### `src/data/` — Data Pipeline Interface

| File | Contents | Lines (est.) |
|------|----------|-------------|
| `tokenizer.jl` | Character-level tokenizer: char→index mapping, encode/decode | ~60 |
| `dataloader.jl` | Batched sequence loading, sliding window, GPU transfer | ~120 |
| `curriculum.jl` | Difficulty-sorted iteration, phase-weighted sampling (DoReMi) | ~100 |
| `coreset.jl` | Gradient-norm scoring, perplexity ranking, subset selection | ~80 |

#### `src/training/` — Training Loop

| File | Contents | Lines (est.) |
|------|----------|-------------|
| `trainer.jl` | Main training loop: forward, loss, backward, update, log | ~200 |
| `optimizer.jl` | AdamW setup, cosine schedule, warmup, gradient clipping | ~80 |
| `checkpoint.jl` | Save/load model params + optimizer state + training step | ~60 |
| `metrics.jl` | Loss tracking, perplexity, grad norm, tokens/sec | ~60 |

#### `src/inference/` — Generation and Deployment

| File | Contents | Lines (est.) |
|------|----------|-------------|
| `generate.jl` | Autoregressive generation: temperature, top-k, top-p, greedy | ~120 |
| `compile.jl` | Reactant.jl XLA compilation for optimized inference | ~60 |
| `quantize.jl` | BitNet 1.58-bit progressive quantization | ~150 |

#### `src/evaluation/` — Metrics and Experiments

| File | Contents | Lines (est.) |
|------|----------|-------------|
| `evaluate.jl` | Perplexity, BPC, diversity, repetition metrics | ~100 |
| `benchmark.jl` | Inference throughput, memory measurement | ~80 |
| `ablation.jl` | Ablation runner: sweep configs, collect results, output tables | ~120 |

**Total estimated:** ~2,020 lines of Julia

### 3.3 Configuration

Single TOML config file for all hyperparameters:

```toml
# config/base.toml — Baseline transformer

[model]
arch = "transformer"       # "transformer" | "monarch" | "neural_ode"
vocab_size = 40            # determined by charset
embed_dim = 384
n_layers = 6
n_heads = 3
head_dim = 128
ffn_mult = 4
context_length = 256
dropout = 0.0
bias = false
weight_tying = true

[training]
optimizer = "adamw"        # "adamw" | "kfac"
lr = 6e-4
min_lr = 6e-5
warmup_steps = 100
max_steps = 5000
batch_size = 64
grad_clip = 1.0
precision = "f16"          # "f32" | "f16" | "bf16"
eval_interval = 250
eval_steps = 50
checkpoint_interval = 1000
seed = 42

[training.curriculum]
enabled = false
ordering = "easy_first"    # "easy_first" | "hard_first" | "random"
warmup_epochs = 1          # epochs of random before curriculum kicks in

[training.coreset]
enabled = false
method = "gradient_norm"   # "gradient_norm" | "perplexity"
keep_fraction = 0.5        # train on top 50% most informative

[training.phase_weights]
trivium = 0.40
quadrivium = 0.35
philosophy = 0.25

[data]
train_path = "../text-pipeline/output/train.txt"
val_path = "../text-pipeline/output/val.txt"
enriched_path = "../text-pipeline/output/train_enriched.jsonl"

[inference]
precision = "f16"
compile = false            # Reactant.jl XLA
temperature = 0.8
top_k = 40
max_new_tokens = 500

[neural_ode]               # Phase 3 only
solver = "Tsit5"
rtol = 1e-3
atol = 1e-3
t_span = [0.0, 1.0]
```

### 3.4 Model Configurations for Experiments

| Config | Arch | Params | Layers | Dim | Heads | Context | Purpose |
|--------|------|--------|--------|-----|-------|---------|---------|
| `tiny` | transformer | ~1M | 3 | 192 | 3 | 128 | Smoke test, fast iteration |
| `base` | transformer | ~15M | 6 | 384 | 3 | 256 | Primary baseline |
| `base-512` | transformer | ~15M | 6 | 384 | 3 | 512 | Context length ablation |
| `wide` | transformer | ~15M | 3 | 512 | 4 | 256 | Depth vs width ablation |
| `deep` | transformer | ~15M | 12 | 256 | 2 | 256 | Depth vs width ablation |
| `monarch-base` | monarch | ~11M | 6 | 384 | - | 256 | Structured matrix variant |
| `ode-base` | neural_ode | ~3M | 1 (continuous) | 384 | 3 | 256 | Continuous-depth variant |
| `scaled` | transformer | ~50M | 8 | 512 | 4 | 512 | Scaling test |

---

## 4. Implementation Phases

### Phase 0: Project Scaffolding (Days 1-2)

**Deliverables:**
- [ ] Julia project initialized with `Project.toml` listing all dependencies
- [ ] TOML config system loading hyperparameters into typed structs
- [ ] Character-level tokenizer: build vocab from training data, encode/decode functions
- [ ] DataLoader: read `train.txt`, create sliding-window sequences, batch, transfer to GPU
- [ ] Smoke test: random model → forward pass → loss computation → gradient → parameter update

**Exit criteria:** A randomly initialized model can complete one training step on GPU without errors.

### Phase 1: Baseline Transformer (Days 3-10)

**Deliverables:**
- [ ] Full transformer architecture in Lux.jl (`julia_gpt.jl`)
  - Token embedding + RoPE
  - Multi-head causal self-attention
  - SwiGLU feed-forward with RMSNorm pre-normalization
  - Residual connections
  - Weight tying (embedding ↔ output projection)
- [ ] Training loop with AdamW, cosine LR schedule, gradient clipping
- [ ] Float16 mixed precision working on A100
- [ ] Checkpointing (save/resume)
- [ ] Validation loss evaluation every N steps
- [ ] Text generation (temperature, top-k, greedy)
- [ ] Baseline training run: `base` config, full corpus, report perplexity and BPC
- [ ] PyTorch comparison: same architecture trained on same data, verify Julia model is competitive

**Exit criteria:** Generated text is coherent English in the style of classical philosophical prose. Validation perplexity within 10% of equivalent PyTorch model.

### Phase 2: Data Techniques (Days 8-16)

*Overlaps with late Phase 1.*

**Deliverables:**
- [ ] Curriculum learning module
  - Load difficulty scores from `train_enriched.jsonl`
  - Sort batches by difficulty within each epoch
  - Configurable warmup epochs (random) before curriculum kicks in
- [ ] DoReMi phase-weighted sampling
  - Load phase weights from config
  - Sample training chunks proportional to trivium/quadrivium/philosophy weights
- [ ] Coreset selection module
  - Score chunks by gradient norm during a single evaluation pass
  - Score chunks by model perplexity
  - Select top-K% for focused retraining
- [ ] Ablation experiments:
  - `base` vs `base + curriculum` vs `base + coreset` vs `base + phase_weights`
  - Measure: steps to target perplexity, final perplexity, training time
- [ ] Sensitivity analysis (GlobalSensitivity.jl)
  - Identify top-50 most influential training chunks
  - Identify least-influential chunks (candidates for removal)

**Exit criteria:** At least one technique reduces training steps to baseline perplexity by >15%.

### Phase 3: Monarch Mixer (Days 14-22)

**Deliverables:**
- [ ] Monarch matrix implementation in Julia
  - Parameterized Monarch matrices (L × R factorization)
  - Batch matrix multiply replacing attention
  - Monarch MLP replacing dense FFN
- [ ] `monarch-base` model configuration
- [ ] Training run on full corpus
- [ ] Comparison vs `base`:
  - Parameter count reduction (target: ≥20%)
  - Perplexity comparison (target: within 5% of baseline)
  - Inference throughput improvement
  - Memory reduction

**Exit criteria:** Monarch variant matches baseline perplexity within 5% using ≥20% fewer parameters.

### Phase 4: Neural ODE Variant (Days 20-30)

**Deliverables:**
- [ ] NeuralODE transformer block via DiffEqFlux.jl
  - Single shared transformer block as ODE dynamics
  - Configurable solver (Tsit5, Euler, adaptive)
  - Configurable tolerance (quality/speed tradeoff)
- [ ] Adjoint backpropagation via SciMLSensitivity.jl
  - O(1) memory in effective depth
  - Symplectic adjoint for exact gradients
- [ ] `ode-base` model configuration
- [ ] Training run on full corpus
- [ ] Adaptive inference demonstration:
  - Same model at rtol=1e-1, 1e-2, 1e-3, 1e-4
  - Plot: perplexity vs inference throughput vs memory
- [ ] Comparison vs `base` at matched parameter count

**Exit criteria:** Neural ODE model trains successfully; demonstrates measurable quality/speed tradeoff via solver tolerance at inference.

### Phase 5: Inference Optimization (Days 25-32)

**Deliverables:**
- [ ] KV cache implementation for autoregressive generation
- [ ] Reactant.jl XLA compilation of inference path
- [ ] 1.58-bit quantization-aware fine-tuning (if Phase 4 complete)
- [ ] Inference benchmarks on target hardware (<6GB GPU):
  - Throughput (chars/sec) for each model variant
  - Peak memory (MB)
  - Time to generate 500 characters
- [ ] Model size on disk for each variant and precision

**Exit criteria:** All model variants run inference within 6GB VRAM budget.

### Phase 6: Evaluation and Paper (Days 28-40)

**Deliverables:**
- [ ] Full ablation tables (depth/width, context, curriculum, coreset, architecture)
- [ ] Generated text samples (curated, representative) for each model variant
- [ ] Training curves (loss, perplexity, grad norm) for all experiments
- [ ] Cross-technique comparison table: perplexity × training steps × memory × throughput
- [ ] Paper draft:
  - Introduction and motivation
  - Related work
  - Methods (architecture, training techniques, data pipeline)
  - Experiments and results
  - Analysis and discussion
  - Conclusion
- [ ] Reproducibility package: config files, scripts, random seeds for all experiments

**Exit criteria:** Paper draft complete with all figures and tables. All experiments reproducible from repo.

---

## 5. Dependency Map

```
Phase 0 ──▶ Phase 1 ──▶ Phase 2 ──▶ Phase 6
                │                      ▲
                ├──▶ Phase 3 ──────────┤
                │                      │
                └──▶ Phase 4 ──▶ Phase 5
```

- Phase 1 must complete before anything else (baseline is the comparison target)
- Phases 2, 3, 4 can proceed in parallel after Phase 1
- Phase 5 depends on Phase 4 (NeuralODE adaptive inference) but can partially start with Phase 1 output
- Phase 6 aggregates all results

---

## 6. Julia Package Dependencies

### Core (Phase 0-1)

```toml
[deps]
Lux = "b2108857-7c20-44ae-9111-449ecde12c47"
LuxCUDA = "d0bbae9a-e099-4d5b-a835-1c6931f17571"
Optimisers = "3bd65402-5787-11e9-1adc-39752487f4e2"
Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
JLD2 = "033835bb-8acc-5ee8-8aae-3f567f8a3819"
TOML = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
JSON3 = "0f8b85d8-7281-11e9-16c2-39a750bddbf1"
Printf = "de0858da-6303-5e67-8744-51eddeeeb8d7"
Dates = "ade2ca70-3891-5945-98fb-dc099432e06a"
```

### Phase 2: Data Techniques

```toml
GlobalSensitivity = "af5da776-676b-467e-8baf-acd8249e4f0f"
```

### Phase 3: Monarch Mixer

```toml
# No additional deps — implemented from mathematical primitives using Lux.jl
```

### Phase 4: Neural ODE

```toml
DiffEqFlux = "aae7a2af-3d4f-5e19-a356-7da93b79d9d0"
OrdinaryDiffEq = "1dea7af3-3e70-54e6-95c3-0bf5283fa5ed"
SciMLSensitivity = "1ed8b502-d754-442c-8d5d-10ac956f44a1"
```

### Phase 5: Inference

```toml
Reactant = "3c362404-f566-11ee-1572-e11a4b42c853"
```

---

## 7. Risk Register

| # | Risk | L | I | Mitigation | Contingency |
|---|------|---|---|-----------|-------------|
| R1 | Lux.jl MultiHeadAttention has undocumented limitations at our config | M | H | Test early in Phase 0 with smoke test | Implement custom MHA from Dense + batched_mul |
| R2 | Corpus too small for model to learn meaningful patterns beyond memorization | L | H | Character-level maximizes effective examples; curriculum and coreset stretch data | Augment with synthetic data from larger model (Phi-4 strategy) |
| R3 | DiffEqFlux.jl NeuralODE + transformer block fails to converge | M | M | Phase 4 is experimental; baseline (Phase 1) is the primary deliverable | Report negative result — this is still publishable |
| R4 | CUDA.jl Float16 mixed precision has numerical instability | M | M | Test in Phase 0 smoke test; fall back to Float32 | Train in Float32 (fits easily on A100 at 15M params) |
| R5 | Monarch matrix implementation is mathematically complex | M | M | Start from the M2-BERT paper pseudocode; simplify to block-diagonal + FFT | Use MonarchAttention zero-shot conversion approach instead |
| R6 | Colab session timeouts interrupt long training runs | M | L | Checkpoint every 500 steps; design for resume | Split training across sessions |
| R7 | Reactant.jl XLA compilation fails on custom layers | M | L | Phase 5 is P1; inference works without XLA | Report eager-mode inference benchmarks |
| R8 | Paper reviewers question novelty of "small model on small data" | M | M | Frame contribution as cross-domain method transfer, not model quality | Emphasize SciML techniques (adjoint, sensitivity) as primary contribution |

---

## 8. Metrics and Success Criteria

### 8.1 Proof of Concept (Minimum Viable)

| Metric | Target | Measurement |
|--------|--------|-------------|
| Baseline perplexity | <8.0 char-level on val set | Computed at end of training |
| Julia vs PyTorch parity | Within 10% perplexity | Same architecture, same data, both trained to convergence |
| Training time | <4 hours on Colab A100 | Wall clock for `base` config |
| Inference memory | <500 MB for `base` config at FP16 | CUDA.jl memory reporting |
| Generated text quality | Coherent English, philosophical vocabulary | Manual evaluation of 10 samples |
| Reproducibility | Bit-identical across runs with same seed | Two runs compared |

### 8.2 Research Contribution (Target)

| Metric | Target | Measurement |
|--------|--------|-------------|
| Curriculum learning speedup | ≥15% fewer steps to baseline perplexity | Steps to reach baseline's final perplexity |
| Coreset selection efficiency | Match baseline with ≤60% of training data | Perplexity at matched data fraction |
| Monarch param reduction | ≥20% fewer params at ≤5% perplexity increase | Direct comparison |
| Cross-technique stacking | ≥25% combined improvement (curriculum + coreset + phase weights) | Compared to baseline with uniform random sampling |

### 8.3 Stretch Goals

| Metric | Target | Measurement |
|--------|--------|-------------|
| Neural ODE adaptive inference | Measurable quality/speed curve across solver tolerances | Perplexity vs chars/sec at 4+ tolerance settings |
| 1.58-bit model quality | <10% perplexity increase vs FP16 | Direct comparison |
| Model size on disk | <20 MB for deployable model | File size |
| Sensitivity-guided curation | Identify ≥10 "high-influence" training examples with interpretable content | GlobalSensitivity.jl output + manual inspection |

---

## 9. Paper Outline

### Title
*JuliaGPT: Cross-Domain Scientific Computing Methods for Efficient Small Language Model Training*

### Structure

**1. Introduction** (1 page)
- The efficiency gap: most LM research targets frontier scale; techniques for small, constrained models are understudied
- Julia's SciML ecosystem offers techniques (Neural ODEs, adjoint methods, sensitivity analysis) with no equivalent in the PyTorch LM training stack
- Contribution summary: first Julia LM, empirical evaluation of 4 cross-domain techniques at <50M scale

**2. Related Work** (1.5 pages)
- Small language models: Phi-4, SmolLM2, TinyLlama, SmallThinker
- Efficient architectures: Monarch Mixer, Mamba, linear attention
- Cross-domain methods: Neural ODEs in NLP, tensor networks for compression, physics-informed training
- Julia ML ecosystem: Lux.jl, DiffEqFlux.jl, SciML

**3. Data** (0.5 pages)
- Corpus description: 50 classical texts, trivium + quadrivium organization
- Text pipeline: cleaning, chunking, quality scoring, DoReMi phase weights, MinHash dedup
- Character-level tokenization rationale

**4. Methods** (3 pages)
- 4.1 Baseline architecture (nanoGPT-style in Lux.jl)
- 4.2 Curriculum learning with difficulty scoring
- 4.3 Coreset selection via gradient norms
- 4.4 Monarch Mixer variant
- 4.5 Continuous-depth Neural ODE variant with adjoint backpropagation
- 4.6 Sensitivity-guided data curation

**5. Experiments** (3 pages)
- 5.1 Experimental setup (hardware, hyperparameters, baselines)
- 5.2 Baseline results and Julia vs PyTorch comparison
- 5.3 Curriculum learning ablation
- 5.4 Coreset selection ablation
- 5.5 Monarch Mixer comparison
- 5.6 Neural ODE results and adaptive inference demonstration
- 5.7 Combined technique stacking
- 5.8 Inference optimization results

**6. Analysis** (1.5 pages)
- Which techniques compose well? Which are redundant?
- Depth vs width at small scale: do our results match "The Depth Delusion"?
- Qualitative analysis of generated text: does the model capture argumentative structure?
- Sensitivity analysis: what does the model learn from each part of the corpus?

**7. Limitations and Future Work** (0.5 pages)
- Character-level only; BPE comparison needed
- Single domain (classical texts); generalization unknown
- Julia ecosystem maturity gaps (Flash Attention, BFloat16)

**8. Conclusion** (0.5 pages)

**Appendix:**
- A: Generated text samples
- B: Full ablation tables
- C: Training curves for all experiments
- D: Reproducibility details (seeds, hardware, package versions)

---

## 10. Repository Structure

```
julia-slm/
├── docs/
│   ├── research_findings.md       # Background research
│   ├── research_proposal.md       # Research proposal
│   └── PRD.md                     # This document
│
├── config/
│   ├── base.toml                  # Primary baseline config
│   ├── tiny.toml                  # Smoke test config
│   ├── monarch.toml               # Monarch Mixer config
│   ├── neural_ode.toml            # Neural ODE config
│   └── ablations/                 # Per-ablation config overrides
│       ├── depth_3.toml
│       ├── depth_12.toml
│       ├── context_512.toml
│       └── curriculum.toml
│
├── src/
│   ├── JuliaGPT.jl               # Package module definition
│   ├── model/
│   │   ├── julia_gpt.jl          # Baseline transformer
│   │   ├── monarch.jl            # Monarch Mixer variant
│   │   ├── neural_ode.jl         # Continuous-depth variant
│   │   ├── layers.jl             # RMSNorm, SwiGLU, KV cache
│   │   └── config.jl             # Model config structs
│   ├── data/
│   │   ├── tokenizer.jl          # Character-level tokenizer
│   │   ├── dataloader.jl         # Batching, GPU transfer
│   │   ├── curriculum.jl         # Difficulty ordering, phase weights
│   │   └── coreset.jl            # Gradient-norm / perplexity ranking
│   ├── training/
│   │   ├── trainer.jl            # Training loop
│   │   ├── optimizer.jl          # AdamW, cosine schedule, warmup
│   │   ├── checkpoint.jl         # Save/load state
│   │   └── metrics.jl            # Loss, perplexity, grad norm
│   ├── inference/
│   │   ├── generate.jl           # Autoregressive generation
│   │   ├── compile.jl            # Reactant.jl XLA compilation
│   │   └── quantize.jl           # BitNet 1.58-bit QAT
│   └── evaluation/
│       ├── evaluate.jl           # Perplexity, BPC, diversity
│       ├── benchmark.jl          # Throughput, memory measurement
│       └── ablation.jl           # Experiment runner
│
├── notebooks/
│   ├── train_colab.ipynb         # Colab training notebook (A100)
│   └── analysis.ipynb            # Results visualization
│
├── scripts/
│   ├── train.jl                  # CLI: julia scripts/train.jl --config config/base.toml
│   ├── generate.jl               # CLI: julia scripts/generate.jl --prompt "the nature of"
│   ├── evaluate.jl               # CLI: julia scripts/evaluate.jl --checkpoint latest
│   └── ablation.jl               # CLI: julia scripts/ablation.jl --suite depth_width
│
├── test/
│   ├── runtests.jl               # Test runner
│   ├── test_model.jl             # Model forward/backward tests
│   ├── test_tokenizer.jl         # Tokenizer encode/decode roundtrip
│   └── test_dataloader.jl        # DataLoader shape/type tests
│
├── Project.toml                  # Julia dependencies
├── .gitignore
└── LICENSE                       # MIT
```
