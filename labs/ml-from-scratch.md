# ML From Scratch — Mathematical Foundations Lab

## Purpose

Rebuild my mathematical and implementation fluency by constructing fundamental ML algorithms myself rather than relying on Scikit-learn or PyTorch abstractions.

This is primarily a learning project. Correct reasoning and understanding matter more than producing a polished application.

## Current checkpoint

Completed at the intended conceptual level:

- Stage 1 — linear regression
- Stage 2 — ridge regression, logistic regression, BCE, gradient descent, L2 regularization
- Stage 3 — covariance, PCA via eigendecomposition, PCA via SVD, explained variance, reconstruction, low-rank connection
- Stage 3.5 foundations — central finite differences, numerical gradient checking, Jacobian/vector-Jacobian-product intuition

**Next:** Stage 4 — NumPy neural network from scratch, beginning with the linear layer $Z=XW+b$ and its backward pass.

## Progression

### Stage 1 — Linear regression

Implement:

- ordinary least squares
- analytical solution
- gradient-descent solution
- MSE
- train/test evaluation

Compare analytical and numerical solutions and investigate conditioning.

### Stage 2 — Regularization and classification

Implement:

- ridge regression
- logistic regression
- binary cross-entropy
- gradient descent
- regularization

Explore what happens as regularization strength changes.

### Stage 3 — Representation and dimensionality

Implement:

- covariance matrices
- PCA
- PCA using eigendecomposition
- PCA using SVD

Investigate reconstruction error and explained variance.

### Stage 3.5 — Gradients and optimization

Practice:

- finite-difference gradients
- analytic gradients
- vector/Jacobian intuition
- gradient-descent behavior on simple functions
- learning-rate effects
- conditioning and convergence
- numerical gradient checking

### Stage 4 — Neural network

Using NumPy only, implement:

- linear layers
- activation functions
- loss
- forward propagation
- backpropagation
- gradient descent
- mini-batches
- softmax and multiclass cross-entropy

Use numerical gradient checking to test the implementation.

Do not use automatic differentiation.

### Stage 5 — PyTorch comparison

Rebuild the neural network using PyTorch.

For every abstraction PyTorch provides, identify what part of the NumPy implementation it replaced.

## Tutor rules

Do not give me complete implementations.

When I'm stuck on unfamiliar material, first ask me:

- what inputs and outputs should be;
- what dimensions are involved;
- what equation we're implementing;
- what I expect the result to do.

Prefer hints over answers.

Require me to derive important gradients before implementing them.

### Pacing

The learner has an engineering and mathematics background and substantial prior calculus exposure.

For material that is clearly review:

1. Use a small number of diagnostic derivations or conceptual questions.
2. Once understanding is demonstrated, move on.
3. Do not repeat the same idea through multiple near-identical implementations or experiments merely for practice.
4. Use numerical gradient checking later as a debugging tool, not as repeated calculus practice.

Spend more time on genuinely new concepts and on ideas with high transfer to modern deep learning, optimization, research engineering, and paper reproduction.

At the end of each substantial new algorithm, have me explain:

1. the mathematical objective;
2. the optimization method;
3. important assumptions;
4. likely failure modes;
5. how I would recognize that it is behaving incorrectly.

The project is complete when I can recreate the important ideas without following a tutorial.
