# JuliaGPT: Research Findings for a Pure Julia Small Language Model

**Date:** 2026-02-21
**Project:** buildwithbooks/text-pipeline → JuliaGPT
**Scope:** Framework selection, architecture design, and cross-domain training techniques for a character-level SLM trained on classical philosophical and mathematical texts

---

## 1. Julia ML Ecosystem Assessment

### 1.1 Framework Comparison

| Framework | Stars | Last Release | Status | Verdict |
|-----------|-------|-------------|--------|---------|
| **Lux.jl** | 680 | v1.4.2 (active) | Rapid growth, MIT CSAIL backing | **Recommended** |
| **Flux.jl** | 4,700 | v0.16.9 (Feb 2026) | Mature, largest community | Good reference, not recommended for new projects |
| **Knet.jl** | 1,400 | v1.4.10 (Feb 2022) | Dormant, incompatible with modern stack | **Do not use** |
| **Transformers.jl** | 567 | v0.3.1 (Jun 2024) | Flux-based, inference-focused | Reference only |
| **TransformersLite.jl** | ~50 | Feb 2025 | Educational, has GPT-style decoder | Architecture reference |

### 1.2 Lux.jl: Why It Wins

Lux.jl separates model architecture from parameters and state — the model is an immutable struct, parameters are NamedTuples passed as function arguments:

```julia
model = Chain(Dense(784 => 256, relu), Dense(256 => 10))
ps, st = Lux.setup(rng, model)
y, st_new = model(x, ps, st)  # pure function
```

This design provides concrete advantages for a from-scratch GPT:

- **Weight tying** (sharing embedding/output weights) is trivial — just pass the same parameter slice
- **Gradient checkpointing** works naturally because all layers are pure functions with no hidden state
- **Serialization** is straightforward — parameters are already separate data structures
- **Multi-AD backend**: Zygote (stable), Enzyme (faster, LLVM-level), Reactant/XLA (compiled inference)
- **SciML integration**: DiffEqFlux.jl, SciMLSensitivity.jl, and Optimization.jl all target Lux

Built-in transformer primitives:
- `MultiHeadAttention` with configurable Q/K/V dimensions and causal masking
- `RotaryPositionalEmbedding` (RoPE) — critical for modern GPT architectures
- `SinusoidalPositionalEmbedding` — classic Vaswani-style
- `SkipConnection` — built-in residual connections
- `@compact` macro for concise layer definitions

### 1.3 Flux.jl: Value as Reference

Flux has the only proven Julia GPT implementations:
- **TransformersLite.jl** — 44K-param GPT trained on Shakespeare, character-level, 3 transformer blocks, 4 attention heads, 32-dim embeddings, context length 64. Perplexity: 72 → 6.4.
- **Lior Sinai's blog series** — from-first-principles transformer in Flux.jl, updated Feb 2025 for Flux 0.16
- **Transformers.jl** — HuggingFace model loading, tokenization via TextEncoders

Key finding from TransformersLite: attention heads mostly focus on 4 or fewer preceding tokens, suggesting small context windows can work for structured text.

### 1.4 Supporting Packages

| Package | Purpose | Status |
|---------|---------|--------|
| **CUDA.jl** 5.4+ | GPU backend for A100 | Production-ready, stream-ordered allocations, proactive GC |
| **Reactant.jl** | XLA compilation for optimized inference | Very active (2,565 commits, Feb 2026) |
| **Enzyme.jl** | LLVM-level AD, up to 10x over Zygote for mutating code | Active (548 stars, 240 releases) |
| **Optimisers.jl** | AdamW and friends, shared by Flux/Lux | v0.4.7 (Dec 2025) |
| **KernelAbstractions.jl** | Vendor-neutral GPU kernels in Julia | <7% overhead vs native CUDA C |
| **DiffEqFlux.jl** | Neural ODEs with O(1) backprop | v4.2.0 (Feb 2025) |
| **SciMLSensitivity.jl** | Adjoint methods, sensitivity analysis | Active |
| **SparseDiffTools.jl** | Sparse Jacobian computation (up to 1000x speedup) | Active |
| **BytePairEncoding.jl** | BPE tokenizer (loads tiktoken formats) | v0.5.2 (Jun 2024) |

### 1.5 Julia-Specific Advantages and Gaps

**Advantages:**
- Single-language stack from data pipeline to training to inference
- Custom GPU kernels in Julia without C++/CUDA extension modules
- JIT compilation specializes kernels to exact model dimensions
- Multiple dispatch enables composing novel architectures naturally
- Julia 1.11 native BFloat16 codegen support

**Gaps:**
- No Flash Attention in Julia (fine for context ≤512, problematic at 2048+)
- BFloat16 has rough edges on A100 (CUDA.jl Issue #2306) — use Float16 initially
- No equivalent to DeepSpeed/FSDP (irrelevant for single-GPU)
- ~1/100th community support vs PyTorch for NLP
- No production Julia GPT has been published

---

## 2. Architecture Analysis

### 2.1 Karpathy's nanoGPT/nanochat Reference

nanoGPT's Shakespeare character-level config: 6 layers, 6 heads, 384 embed dim, 256 context.

nanochat (Jan 2026) introduces a single-dial scaling system:
```
width = depth × aspect_ratio       # aspect_ratio = 64
heads = width / head_dim            # head_dim = 128
target_tokens = 10.5 × num_params  # data/param ratio
```

Architecture modernizations in nanochat: Flash Attention 3 + GQA, RoPE, RMSNorm, SwiGLU, MuonAdamW optimizer.

Key scaling insight: parameters grow **linearly with depth** but **quadratically with width** (each layer = 12 × n_embd² params). The "Depth Delusion" paper (Jan 2025) confirms: optimal width scaling W* ~ C^0.34 grows 2.8x faster than depth D* ~ C^0.12.

### 2.2 Feasible Model Sizes

**Training on A100 80GB (mixed precision):**
| Model Size | Memory (train) | Train Time (est.) | Notes |
|-----------|---------------|-------------------|-------|
| 10-15M | ~2 GB | Minutes | Character-level, appropriate for current corpus |
| 50M | ~5 GB | ~1 hour | Sweet spot for expanded corpus |
| 124M | ~12 GB | ~4-8 hours | GPT-2 small equivalent |
| 350M | ~30 GB | ~1-2 days | Maximum practical for single A100 |

**Inference on <6GB GPU:**
| Precision | Params in 6GB | Notes |
|-----------|--------------|-------|
| FP16 | ~3B (model only) | No KV cache budget |
| INT8 | ~6B | Requires quantization |
| INT4 | ~12B | Aggressive quantization |
| 1.58-bit (BitNet) | ~24B | Ternary weights {-1, 0, +1} |

For a 28-char vocabulary model at 50M params: inference uses <100MB at FP16. Trivially deployable.

### 2.3 Character-Level vs BPE Tokenization

Current config uses expanded charset: `a-z .,;:?!'"-()`

**Character-level advantages for this corpus:**
- Embedding table is essentially free (40 chars × 384 dim = 15,360 params vs 50K × 384 = 19.2M for BPE)
- All model capacity goes to transformer blocks learning linguistic patterns
- No tokenizer training needed, no risk of poor BPE splits on small specialized corpus
- Al-Rfou et al. (2019) showed character-level transformers achieve SOTA with appropriate depth
- Character-level models learn spelling and morphology from scratch — appropriate for classical text

**Tradeoff:** Sequences are ~4x longer than BPE, requiring more context length or accepting shorter effective context. At 256-char context, the model sees ~1-2 sentences. At 1024, roughly a paragraph.

**Recommendation:** Stay character-level. If scaling to >5M words of training data, consider 8K-16K BPE vocabulary. NeurIPS 2024 finding: vocabularies >60K on small datasets create undertrained embeddings with 2.8% higher loss.

### 2.4 Corpus Analysis

From `source_manifest.json`: 50 texts, ~2.8M estimated words (~14-17M characters raw). After cleaning/lowercasing/stripping: estimated ~10-12M characters of training data.

| Category | Texts | Est. Words | Examples |
|----------|-------|-----------|---------|
| Trivium (Logic) | 7 | ~214K | Aristotle's Categories, Prior/Posterior Analytics, Topics, Plato's Meno, Theaetetus, Descartes' Method |
| Trivium (Rhetoric) | 8 | ~285K | Aristotle's Rhetoric/Poetics, Plato's Symposium/Phaedrus/Gorgias/Protagoras, Bacon, Emerson |
| Trivium (Ethics) | 10 | ~560K | Nicomachean Ethics, Meditations, Enchiridion, Seneca, Nietzsche, Mill, Thoreau |
| Trivium (Politics) | 6 | ~560K | Republic, Laws, Leviathan, Locke, Rousseau, Machiavelli |
| Trivium (Metaphysics) | 8 | ~345K | Metaphysics, On the Soul, Phaedo, Spinoza, Kant, Hume, Boethius, Schopenhauer |
| Quadrivium | 5 | ~285K | Elements of Euclid, Physics, On the Heavens, Generation & Corruption, Lucretius, Timaeus |

---

## 3. Cross-Domain Techniques for Efficient Training

### 3.1 From Physics

**Sparse Attention (Particle Physics)**
"Why Is Attention Sparse in Particle Transformer?" (Dec 2025) shows that attention naturally becomes sparse and nearly binary on structured data. The sparsity encodes physically meaningful correlations. For curated classical texts with consistent argumentative structure, enforcing top-k attention sparsity can reduce compute with minimal quality loss.

**Tensor-Train Decomposition (Quantum Physics)**
Tensor networks from quantum many-body physics can compress feed-forward layers by up to 95%. "Variational tensor neural networks" (Nature 2024) demonstrates scalable architectures with Tensor-Train decomposition. For short-context models (256-2048 tokens), TT-decomposition of weight matrices is highly practical for inference compression.

**Neural ODE Transformers (Dynamical Systems)**
"Neural ODE Transformers" (arXiv 2503.01329) parameterizes all attention/FFN weights as functions of a continuous layer index. Results: a 3.5B continuous-depth model with 32 iterations matches a 50B traditional model. Adaptive compute at inference — fewer ODE solver steps for simple tokens — directly addresses the <6GB constraint. Implementable via DiffEqFlux.jl + Lux.jl.

**Renormalization Group (Statistical Physics)**
RG theory (arXiv 2510.25553, Jan 2026) shows self-similarity and its breakdown in learning curves for neural networks trained on power-law data. Suggests progressive training (coarse → fine features) has principled physics basis. Universality results suggest scaling laws from larger models partially transfer to smaller ones.

**Spin Glass Theory (Statistical Mechanics)**
Multiple 2024-2025 papers connect loss landscape structure to spin glass physics. "Neural Networks as Spin Models" (Aug 2024) shows the spin glass phase is quickly destroyed during training and replaced by "hidden order." For small models on curated data, understanding this glass-to-order transition can inform learning rate scheduling and early stopping.

### 3.2 From Mathematics / Numerical Analysis

**Monarch Mixer (Structured Matrices)**
Replaces both attention AND MLP with Monarch matrices (generalized FFT). M2-BERT matches BERT-base with 27% fewer parameters and 9.1x higher throughput at sequence length 4K. At 360M params, matches GPT-quality on The Pile. Sub-quadratic, hardware-friendly GEMMs. "MonarchAttention" (May 2025) enables zero-shot conversion of existing attention to Monarch form.

**BitNet 1.58-bit Quantization-Aware Training**
Every weight restricted to {-1, 0, +1}. Microsoft's BitNet b1.58-2B-4T (2025) demonstrates competitive performance. "When Are 1.58 Bits Enough?" (2025) shows the "16-to-1.58" progressive strategy works for models as small as 100K-48M params. A 2B model at 1.58-bit needs ~500MB for weights.

**KFAC Second-Order Optimization**
Kronecker-Factored Approximate Curvature extended to transformers (NeurIPS 2023). Faster convergence on small datasets. KFAC for PINNs (NeurIPS 2024) consistently outperforms Adam and LBFGS. Available via SciML's Optimization.jl.

**Randomized Numerical Linear Algebra**
Sketching approximates large matrix operations (gradient covariance, Fisher information) with sublinear cost. Enables second-order optimization at first-order cost. RandNLA survey (KDD 2024) covers direct application to curvature matrices.

**Structured Matrices for Attention**
Toeplitz Neural Network (ICLR 2023): O(n log n) with only 2n-1 parameters for n×n. Vision Transformers shown to approximate Block Circulant matrices (Dec 2025), enabling FFT-based O(N log N) multiplication.

### 3.3 From Astronomy

**Coreset Selection (Survey Optimization)**
PICore (2025) identifies the most informative training samples based on physics-informed loss, achieving 78% increase in training efficiency. Data selection via optimal control achieves 2x training acceleration. Directly applicable to scoring text chunks by perplexity or gradient norm.

**Simulation-to-Real Transfer (Pre-train on Synthetic)**
Astronomy's paradigm: pre-train on simulated data, fine-tune on real observations. Directly maps to Microsoft's Phi-4 approach — trained primarily on synthetic data (~400B tokens), fine-tuning produces a 14B model that surpasses GPT-4 on STEM. For small models: generate synthetic training data from a larger model, fine-tune on curated real corpus.

**Active Learning / Data Curation**
Astronomical surveys generating 100+ PB/year use active learning to select what to observe next. For text: after initial training, identify what the model is most uncertain about, curate additional data in those domains.

**Dimensionality Reduction (Spectral Data)**
VAEs with 2 latent parameters reconstruct astronomical spectra as well as PCA with 6 components. Suggests embedding layers can benefit from non-linear compression (VAE-style bottleneck) for significant model size reduction.

### 3.4 Efficient Architecture Alternatives

**State Space Models (Mamba)**
Mamba-3B outperforms transformers of the same size. 5x throughput, linear time in sequence length. Mamba-2 adds chunkwise parallelism for hardware efficiency. No Julia implementation exists yet.

**Small Mixture of Experts**
SmallThinker: 4B total / 0.6B active parameters, ~1GB at Q4 quantization, >20 tokens/s on consumer CPUs. Demonstrates MoE is viable at small scale.

**Knowledge Distillation**
MiniPLM (ICLR 2025): offline teacher inference reduces data demand by 2.4x using 1.8B teacher for 200M-1.2B students. YODA progressive learning: +17% on GSM8K for LLaMA2.

**Curriculum Learning**
"Curriculum learning reduces training steps by 18-45%" (arXiv 2506.11300, 2025). Best difficulty signals: compression ratio, lexical diversity, readability. Curriculum-guided layer scaling progressively increases model depth alongside sample difficulty.

---

## 4. SciML-Specific Techniques

### 4.1 Adjoint Methods for O(1) Memory Backprop

SciMLSensitivity.jl provides multiple adjoint schemes:
- **Standard adjoint**: O(1) memory in depth, some numerical error accumulation
- **Symplectic adjoint**: Exact gradients, memory proportional to (uses + network_size)
- **Interpolating adjoint**: Trades memory for reduced recomputation

Combined with Neural ODEs, this enables arbitrary effective depth with constant memory overhead during training.

### 4.2 Sparse Jacobian Exploitation

SparseDiffTools.jl + SparseConnectivityTracer.jl automatically detect sparsity patterns in Jacobians/Hessians and exploit them for up to 1000x speedup. Applications:
- Sparse attention patterns → automatic sparse gradient computation during backprop
- Pruning-aware training → avoid wasted FLOPs on zeroed weights
- Tractable second-order optimization via sparse Hessians

### 4.3 Sensitivity Analysis for Data Curation

GlobalSensitivity.jl can identify which training examples most influence model behavior — directly supporting the "highly curated small datasets" strategy. Structural identifiability analysis determines whether the architecture has redundant parameters that can be removed.

### 4.4 Physics-Informed Regularization

NeuralPDE.jl's constraint-based approach can be adapted: linguistic priors (Zipf's law, information-theoretic entropy bounds, syntactic dependency length) formulated as soft constraints in the loss function. NeuralPDE's curriculum regularization (constraints grow complex during training) reduces error by 1-2 orders of magnitude.

---

## 5. Key References

### Julia Ecosystem
- Lux.jl Documentation: https://lux.csail.mit.edu/
- Flux.jl: https://github.com/FluxML/Flux.jl
- CUDA.jl 5.4 Memory Management: https://juliagpu.org/post/2024-05-28-cuda_5.4/
- Reactant.jl (XLA compilation): https://github.com/EnzymeAD/Reactant.jl
- DiffEqFlux.jl: https://github.com/SciML/DiffEqFlux.jl
- TransformersLite.jl: https://github.com/LiorSinai/TransformersLite.jl
- Lior Sinai, "Generative Transformer from First Principles in Julia": https://liorsinai.github.io/machine-learning/2024/03/23/transformers-gpt.html

### Architecture and Training
- Karpathy, nanoGPT: https://github.com/karpathy/nanoGPT
- Karpathy, nanochat: https://github.com/karpathy/nanochat
- "The Depth Delusion" (Jan 2025): https://arxiv.org/abs/2601.20994
- "Architectural Trade-offs in Small Language Models" (Dec 2025): https://arxiv.org/html/2512.20877
- Neural ODE Transformers (Mar 2025): https://arxiv.org/abs/2503.01329
- Monarch Mixer (Oct 2023): https://arxiv.org/abs/2310.12109
- MonarchAttention (May 2025): https://arxiv.org/html/2505.18698v1
- BitNet b1.58 (Feb 2024): https://arxiv.org/abs/2402.17764
- "When Are 1.58 Bits Enough?" (2025): https://www.scitepress.org/Papers/2025/133824/133824.pdf
- Mamba (Dec 2023): https://arxiv.org/abs/2312.00752

### Cross-Domain Methods
- "Why Is Attention Sparse in Particle Transformer?" (Dec 2025): https://arxiv.org/abs/2512.00210
- Variational Tensor Neural Networks (Nature 2024): https://www.nature.com/articles/s41598-024-69366-8
- Renormalization Group for DNNs (Oct 2025): https://arxiv.org/abs/2510.25553
- PICore Coreset Selection (2025): https://arxiv.org/html/2507.17151
- Phi-4 Technical Report (Dec 2024): https://arxiv.org/abs/2412.08905
- KFAC for Transformers (NeurIPS 2023): https://arxiv.org/abs/2311.00636
- Curriculum Learning for LM Pretraining (2025): https://arxiv.org/abs/2506.11300
- MiniPLM Knowledge Distillation (ICLR 2025): https://arxiv.org/abs/2410.17215
- Simulation-Based Pretraining for Astronomy (Oct 2025): https://arxiv.org/abs/2510.12958
- "The End of Transformers?" (Oct 2025): https://arxiv.org/html/2510.05364v1
- Toeplitz Neural Network (ICLR 2023): https://arxiv.org/pdf/2305.04749
