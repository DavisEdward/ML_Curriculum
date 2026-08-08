# Tiny Transformer Research Lab

## Goal

Build and train a small language model from first principles to develop a deep understanding of the architecture and training process underlying modern frontier language models.

This should eventually become a portfolio-quality research-engineering project.

The objective is NOT to create a useful competitor to existing LLMs. The objective is to understand every important component well enough to implement, modify, test and explain it.

## Stage 0 — Tensor fluency

Before building the language model, become comfortable reasoning about tensor shapes without relying on trial and error.

Practice transformations such as:

- `[B, T, d_model]`
- `[B, h, T, d_k]`
- projections and reshaping
- broadcasting
- matrix multiplication dimensions

Predict dimensions before running code.

Also practice:

- parameter counting
- initialization
- train/eval behavior
- gradient norms
- optimizer state
- basic memory accounting

## Stage 1 — Understand language modelling

Develop an understanding of:

- tokenization
- embeddings
- next-token prediction
- cross-entropy
- autoregressive modelling

Start with extremely small experiments.

## Stage 2 — Attention from scratch

Implement scaled dot-product attention myself.

Before implementing it, require me to derive and explain:

- Q, K and V
- matrix dimensions
- why QKᵀ represents similarity
- why scaling is used
- softmax
- causal masking
- how the output is produced

Do not initially allow me to use PyTorch's built-in attention module.

## Stage 3 — Transformer

Build:

- embeddings
- positional representations
- multi-head attention
- residual connections
- normalization
- feed-forward layers
- Transformer blocks
- output projection
- weight tying where appropriate

Gradually assemble these into a small decoder-only Transformer.

## Stage 4 — Training

Create a complete training pipeline including:

- datasets
- batches
- optimizer
- validation
- checkpoints
- learning curves
- generation
- reproducibility

Keep individual experiments small enough that most training runs can finish in roughly 30–60 minutes when practical.

## Stage 5 — Turn it into research

Once the baseline works, stop merely building and begin experimenting.

Choose questions such as:

- How does model depth versus width affect learning at fixed parameter count?
- How do positional encoding strategies compare?
- How does normalization placement affect training?
- What happens with different learning-rate schedules?
- How does context length affect results?
- How does dataset size affect validation loss?
- What attention patterns emerge?

Form a hypothesis before each experiment.

## Stage 6 — Research report

Produce a technical report containing:

- question
- mathematical background
- architecture
- experimental setup
- baselines
- results
- plots
- ablations
- failures
- conclusions
- potential next experiments

The GitHub repository should be reproducible by another engineer.

## AI rules

AI is my research mentor, not my code generator.

Do not provide full implementations of the Transformer components I'm trying to learn.

Help with:

- conceptual explanations
- mathematical derivations
- code review after I write code
- debugging strategies
- experimental design
- interpreting results

If my code is broken, point me toward the underlying problem before rewriting it.

As I improve, give me progressively less scaffolding.
