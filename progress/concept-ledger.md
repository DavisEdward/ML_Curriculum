# Concept Ledger

Use this file to track demonstrated understanding across every lab and curriculum phase.

| Concept | Status | Evidence | Last tested | Revisit |
|---|---|---|---|---|
| OLS | Comfortable | Derived normal equations, MSE gradient, projection interpretation, conditioning behavior | 2026-09-03 | Retrieval later |
| Ridge regression | Comfortable | Derived L2 gradient/closed form; connected regularization to conditioning and coefficient shrinkage | 2026-09-03 | Later |
| Sigmoid derivative | Comfortable | Reconstructed $\sigma'(z)=\sigma(z)(1-\sigma(z))$ | 2026-09-03 | During NN activations |
| Binary cross-entropy | Comfortable | Reconstructed BCE from desired behavior; derived BCE + sigmoid gradient | 2026-09-03 | During classification NN |
| Logistic regression | Comfortable | Derived/vectorized gradients, implemented NumPy training, tested noise and regularization effects | 2026-09-03 | Later |
| Logistic convexity / Hessian | Comfortable | Reasoned from $H=X^TRX/n$, PSD structure, rank deficiency and conditioning | 2026-09-03 | Optimization revisit |
| Covariance / PCA geometry | Comfortable | Derived maximum-variance objective and eigenvalue interpretation | 2026-09-03 | Later |
| PCA eigendecomposition | Comfortable | Implemented PCA, explained variance, projection and reconstruction | 2026-09-03 | Later |
| SVD | Comfortable | Connected PCA to SVD; derived $\lambda_i=\sigma_i^2/n$ and $Z=U_k\Sigma_k$ | 2026-09-03 | Low-rank / attention later |
| Low-rank approximation | Comfortable | Connected discarded singular values to reconstruction error; understands truncated SVD role | 2026-09-03 | Later |
| Numerical gradient checking | Comfortable | Derived central-difference accuracy/cancellation tradeoff and implemented checker | 2026-09-03 | Use to debug Stage 4 |
| Jacobian / VJP intuition | Comfortable | Correct chain-rule relation; understands why backprop avoids materializing full Jacobians | 2026-09-03 | Stage 4 immediately |
| Recall | Comfortable | Correctly explained TP / (TP + FN) |  | Later |
| Attention | Not learned | Not yet demonstrated in this curriculum |  | Phase 3 |

## Status definitions

### Not learned
Not yet demonstrated in this curriculum. This does not necessarily mean never encountered before.

### Rusty
Previously learned, but reconstruction or application is unreliable without prompting.

### Comfortable
Can derive or explain the core idea and use it correctly in normal problems.

### Strong enough to teach
Can reconstruct the idea, explain why it works, identify common misconceptions and failure modes, and connect it to adjacent concepts.

## Evidence standard

Do not update a concept based only on “that makes sense.”

Evidence should come from at least one of:

- derivation
- explanation from memory
- implementation
- debugging
- retrieval question
- experimental design decision

## Pacing rule

The learner has an engineering/mathematics background. For clearly familiar material, use one or two diagnostic derivations/questions and move on once understanding is demonstrated. Avoid repetitive exercises whose only novelty is bookkeeping. Spend more time on genuinely new ML/deep-learning concepts and on ideas that transfer to research engineering.
