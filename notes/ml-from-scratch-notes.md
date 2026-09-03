# ML From Scratch — Living Notes

These notes collect concepts that have been reconstructed, implemented, or experimentally verified in the ML From Scratch lab. They are intended as a compact reference for ideas that will be reused in later stages.

## Notation and shapes

For a linear model with `n` samples, `d` input features, and one output:

$$
X \in \mathbb{R}^{n\times d}, \qquad w \in \mathbb{R}^{d\times 1}, \qquad y \in \mathbb{R}^{n\times 1}
$$

Prediction:

$$
\hat y = Xw
$$

With a bias/intercept:

$$
\hat y = Xw+b
$$

where the same bias is shared across samples. A common alternative is to append a column of ones to $X$ and absorb the bias into the weight vector.

Important viewpoint:

- A **row** of $X$ contains the features for one sample.
- A **column** of $X$ describes how one weight changes predictions across all samples.

---

# Stage 1 — Linear Regression

## Ordinary least squares objective

The residual is

$$
r = \hat y-y = Xw-y.
$$

OLS chooses $w$ to minimize the squared residual magnitude:

$$
J(w)=\|Xw-y\|_2^2.
$$

Using mean squared error instead gives

$$
J(w)=\frac{1}{n}\|Xw-y\|_2^2
=\frac{1}{n}\sum_{i=1}^n(\hat y_i-y_i)^2.
$$

Multiplying the objective by a positive constant such as $1/n$ does not change the minimizing $w$.

## Expanding the OLS loss

Using

$$
\|a\|_2^2=a^Ta,
$$

we get

$$
J(w)=(Xw-y)^T(Xw-y).
$$

Expanding:

$$
J(w)=w^TX^TXw-y^TXw-w^TX^Ty+y^Ty.
$$

Since $y^TXw$ is a scalar,

$$
y^TXw=(y^TXw)^T=w^TX^Ty,
$$

so

$$
\boxed{J(w)=w^TX^TXw-2w^TX^Ty+y^Ty}.
$$

## Important matrix derivatives

For symmetric $A$,

$$
\nabla_w(w^TAw)=2Aw.
$$

Since $X^TX$ is symmetric,

$$
\nabla_w(w^TX^TXw)=2X^TXw.
$$

Also,

$$
\nabla_w(w^Tc)=c,
$$

and a quantity independent of $w$ has derivative zero.

Therefore,

$$
\boxed{\nabla_w\|Xw-y\|_2^2=2X^T(Xw-y)}.
$$

For MSE,

$$
\boxed{\nabla_w J(w)=\frac{2}{n}X^T(Xw-y)}.
$$

Interpretation: the residual says how wrong each prediction is, while multiplication by $X^T$ converts those sample-level errors into one gradient component for each weight.

## Normal equations and analytical OLS

At the optimum,

$$
\nabla_wJ(w)=0.
$$

Therefore,

$$
X^T(Xw-y)=0,
$$

which gives the normal equation

$$
\boxed{X^TXw=X^Ty}.
$$

If $X^TX$ is invertible,

$$
\boxed{w=(X^TX)^{-1}X^Ty}.
$$

In numerical code, prefer solving the linear system

$$
(X^TX)w=X^Ty
$$

directly rather than explicitly computing $(X^TX)^{-1}$.

## Geometric interpretation

Because

$$
\hat y=Xw,
$$

all possible predictions lie in the column space of $X$:

$$
\hat y\in\mathrm{Col}(X).
$$

At the OLS optimum,

$$
X^Tr=0.
$$

This means

$$
r\perp\mathrm{Col}(X).
$$

Therefore,

$$
\boxed{\hat y=\mathrm{proj}_{\mathrm{Col}(X)}(y)}.
$$

A useful optimization interpretation:

> If any component of the residual still lies along a direction the model can move, changing the corresponding weight can reduce the error. At the optimum, the remaining residual contains nothing the model can fix.

## Gradient descent

Gradient descent updates weights by moving opposite the gradient:

$$
\boxed{w_{t+1}=w_t-\eta\nabla J(w_t)}.
$$

For MSE linear regression:

$$
\boxed{w_{t+1}=w_t-\eta\frac{2}{n}X^T(Xw_t-y)}.
$$

A negative gradient component means increasing that weight locally decreases the loss. A positive gradient component means decreasing that weight locally decreases the loss.

Analytical OLS and gradient descent aim for the same optimum. Analytical OLS solves for it directly; gradient descent approaches it iteratively.

## Hessian and learning-rate stability

For

$$
J(w)=\frac{1}{n}\|Xw-y\|_2^2,
$$

the Hessian is

$$
\boxed{H=\frac{2}{n}X^TX}.
$$

Along an eigen-direction with curvature $\lambda$, gradient-descent error behaves like

$$
e_{t+1}=(1-\eta\lambda)e_t.
$$

Therefore:

- $0<1-\eta\lambda<1$: error shrinks without changing sign.
- $-1<1-\eta\lambda<0$: error oscillates while shrinking.
- $1-\eta\lambda<-1$: oscillations grow and training diverges.

For this quadratic objective, stability requires approximately

$$
0<\eta<\frac{2}{\lambda_{\max}(H)}.
$$

Since

$$
H=\frac{2}{n}X^TX,
$$

this is equivalent to

$$
\boxed{\eta<\frac{n}{\lambda_{\max}(X^TX)}}.
$$

In the Stage 1 experiment, the empirical stability limit was approximately

$$
\eta_{\max}\approx0.012593.
$$

A learning rate of $0.01$ converged, $0.1$ diverged, and very small values such as $0.0001$ converged extremely slowly.

## Conditioning and collinearity

For positive-definite $X^TX$, the condition number can be written as

$$
\boxed{\kappa(X^TX)=\frac{\lambda_{\max}}{\lambda_{\min}}}.
$$

A large condition number means the loss surface has very different curvature in different directions. Gradient descent must use one learning rate for all of them, so steep directions constrain the learning rate while shallow directions converge slowly.

If two columns of $X$ become nearly identical, then $X$ becomes nearly rank-deficient and

$$
\lambda_{\min}(X^TX)\to0,
$$

so

$$
\kappa(X^TX)\to\infty.
$$

This creates coefficient instability: very different weight vectors can produce nearly identical predictions.

Example when $x_2\approx x_3$:

$$
5x_2+1x_3\approx3x_2+3x_3\approx100x_2-94x_3.
$$

Important distinction:

$$
\boxed{\text{stable predictions do not imply stable coefficients}}.
$$

For identical columns, the model may only identify the sum $w_2+w_3$, not the individual weights.

With nearly identical columns, analytical OLS may recover the unique optimum directly, while gradient descent can quickly learn the important sum direction and then move extremely slowly along the nearly-flat difference direction.

The singular-value relationship is

$$
\boxed{\kappa(X^TX)=\kappa(X)^2}.
$$

## Train/test evaluation

Training and evaluating on the same samples does not measure generalization. A test set should represent genuinely unseen data for the real task.

Random shuffling is appropriate for independent, identically distributed samples when the original row order may be arbitrary or sorted.

Random row shuffling may be inappropriate when:

- data are time-dependent and future observations must not leak into training;
- multiple rows come from the same person, device, subject, or group and related samples must remain together;
- a deliberately structured split has already been constructed.

Always ask:

> What should “unseen data” mean for this real problem?

## Noise and parameter recovery

Synthetic experiments used

$$
y=Xw_{\text{true}}+\epsilon.
$$

If

$$
\mathbb{E}[\epsilon\mid X]=0,
$$

then OLS can recover the underlying relationship well with enough informative data.

Substituting the noisy target into the analytical solution gives

$$
w_{\text{OLS}}
=w_{\text{true}}+(X^TX)^{-1}X^T\epsilon.
$$

This shows directly how noise perturbs the fitted coefficients.

With many independent zero-mean noise terms, positive and negative effects tend to cancel. Larger noise still increases prediction MSE because the random component itself is not predictable.

Parameter recovery depends jointly on:

- noise magnitude;
- sample size;
- conditioning/information content of $X$.

## Useful implementation checks

For a from-scratch linear-regression implementation:

1. Compare analytical and gradient-descent solutions:

$$
w_{\text{GD}}\approx w_{\text{OLS}}.
$$

2. Check the optimality condition:

$$
X^T(Xw-y)\approx0.
$$

3. Compare train and test MSE.
4. On synthetic data, compare fitted weights against known $w_{\text{true}}$.
5. Inspect whether MSE decreases during gradient descent.
6. If coefficients are unstable but predictions are good, inspect conditioning and collinearity.
7. If optimization behaves badly, inspect the learning rate and Hessian/eigenvalue scale.
8. Remember that low training loss does not prove the model form is correct; the true relationship may not be linear.

## NumPy shape reminders

A NumPy array with shape `(n,)` is not the same as one with shape `(n, 1)`.

For example, adding shapes

```text
(n, 1) + (n,)
```

can broadcast to `(n, n)` rather than performing elementwise column-vector addition.

Also, a Python list comprehension such as

```python
[X[i] for i in indices]
```

creates a Python `list`, even if each element came from a NumPy array. In contrast,

```python
X[indices]
```

uses NumPy advanced indexing and returns a NumPy array.

---

# Stage 2 — Ridge Regression (in progress)

## Ridge objective

Ridge regression adds an L2 penalty to OLS:

$$
\boxed{J(w)=\|Xw-y\|_2^2+\lambda\|w\|_2^2}
$$

where

$$
\|w\|_2^2=w^Tw=\sum_jw_j^2.
$$

The regularization term penalizes large coefficients. This is especially useful when multiple nearly-collinear coefficient combinations make similar predictions.

For example, if $(100,-94)$ and $(3,3)$ produce similar prediction error, ridge strongly prefers $(3,3)$ because

$$
100^2+(-94)^2 \gg 3^2+3^2.
$$

Ridge generally **shrinks** coefficients toward zero rather than forcing them to exactly zero.

As

$$
\lambda\to0,
$$

ridge approaches ordinary OLS.

As

$$
\lambda\to\infty,
$$

the penalty dominates and coefficients are pushed toward zero, even at the cost of prediction accuracy.

## Ridge gradient

Since

$$
\nabla_w(w^Tw)=2w,
$$

we get

$$
\boxed{\nabla_wJ(w)=2X^T(Xw-y)+2\lambda w}.
$$

Setting this equal to zero:

$$
X^TXw-X^Ty+\lambda w=0.
$$

Because $\lambda w=(\lambda I)w$, where $I$ is the $d\times d$ identity matrix,

$$
\boxed{(X^TX+\lambda I)w=X^Ty}.
$$

If the matrix is invertible, the analytical ridge solution is

$$
\boxed{w=(X^TX+\lambda I)^{-1}X^Ty}.
$$

The identity matrix matters because $X^TX$ is a $d\times d$ matrix. Ridge adds $\lambda$ to each diagonal direction of $X^TX$, increasing small eigenvalues and improving conditioning.

This is the key connection between ridge and the Stage 1 collinearity experiment.


---

# Stage 2 — Logistic Regression, BCE, and L2 Regularization

## Sigmoid and binary cross-entropy

Logistic regression maps a linear logit to a probability:

$$
z=Xw+b, \qquad \hat y=\sigma(z)=\frac{1}{1+e^{-z}}.
$$

The sigmoid derivative was reconstructed as

$$
\boxed{\sigma'(z)=\sigma(z)(1-\sigma(z))}.
$$

For binary targets $y\in\{0,1\}$, BCE is

$$
\boxed{
L(y,\hat y)=
-y\log \hat y-(1-y)\log(1-\hat y)
}.
$$

For a batch, mean BCE is

$$
\boxed{
L_{\mathrm{BCE}}
=
-\frac1n\sum_i
\left[
y_i\log \hat y_i+(1-y_i)\log(1-\hat y_i)
\right].
}
$$

The two terms act as a selector: when $y=1$, only $-\log\hat y$ remains; when $y=0$, only $-\log(1-\hat y)$ remains.

Important numerical detail: clip probabilities away from exactly $0$ and $1$ before taking logs.

## BCE + sigmoid gradient simplification

Starting from

$$
\frac{\partial L}{\partial \hat y}
=
-\frac{y}{\hat y}
+
\frac{1-y}{1-\hat y}
$$

and

$$
\frac{\partial \hat y}{\partial z}
=
\hat y(1-\hat y),
$$

the terms cancel to give

$$
\boxed{
\frac{\partial L}{\partial z}
=
\hat y-y.
}
$$

This is an important reusable result: sigmoid + BCE passes back a prediction-error signal in logit space.

For mean BCE,

$$
\boxed{
\nabla_w L
=
\frac1n X^T(\hat y-y)
}
$$

and

$$
\boxed{
\frac{\partial L}{\partial b}
=
\frac1n\sum_i(\hat y_i-y_i).
}
$$

Per sample,

$$
\nabla_wL_i=x_i^T(\hat y_i-y_i).
$$

The matrix form is the sum of those per-sample feature-weighted residuals.

## L2-regularized logistic regression

Using

$$
L_{\mathrm{total}}
=
L_{\mathrm{BCE}}+\lambda\|w\|_2^2,
$$

the gradients are

$$
\boxed{
\nabla_wL_{\mathrm{total}}
=
\frac1nX^T(\hat y-y)+2\lambda w
}
$$

and

$$
\boxed{
\frac{\partial L_{\mathrm{total}}}{\partial b}
=
\frac1n\sum_i(\hat y_i-y_i).
}
$$

Bias is generally not regularized. Penalizing $b$ would impose an artificial preference for decision boundaries near the coordinate origin. The intercept should be free to shift the boundary to where the data live.

Regularization experiments showed the expected behavior: increasing $\lambda$ shrinks $\|w\|$, while BCE can rise even if the classification boundary changes little. BCE measures confidence/calibration as well as 0/1 correctness.

## Convexity and conditioning

The logistic-regression Hessian has the form

$$
\boxed{
H=\frac1nX^TRX,
}
$$

where

$$
R=\operatorname{diag}(\hat y_i(1-\hat y_i)).
$$

Since $R\succeq0$,

$$
v^THv
=
\frac1n(Xv)^TR(Xv)
\ge0,
$$

so the objective is convex. Any local minimum is global, although the optimum need not be unique if the Hessian is singular.

Rank deficiency or near-collinearity in $X$ can make the Hessian singular or poorly conditioned. Curvature can also become small when predictions saturate near $0$ or $1$, because $\hat y_i(1-\hat y_i)$ becomes small.

---

# Stage 3 — PCA, Eigendecomposition, and SVD

## Covariance matrix and PCA objective

For centered data

$$
X_c=X-\mu,
$$

the covariance matrix is

$$
\boxed{
C=\frac1nX_c^TX_c.
}
$$

Each entry is a covariance between feature columns; diagonal entries are feature variances.

For any unit direction $v$, projected coordinates are

$$
z=X_cv,
$$

and the variance along that direction is

$$
\boxed{
\mathrm{Var}(z)=v^TCv.
}
$$

If $Cv=\lambda v$ and $\|v\|=1$, then

$$
v^TCv=\lambda.
$$

Therefore the eigenvalue is exactly the variance captured along its eigenvector. The first principal component is the eigenvector with largest eigenvalue.

## Why the principal components are orthogonal

A covariance matrix is symmetric. If

$$
Cv_1=\lambda_1v_1,\qquad
Cv_2=\lambda_2v_2
$$

with $\lambda_1\ne\lambda_2$, symmetry gives

$$
\lambda_1v_1^Tv_2
=
v_1^TCv_2
=
\lambda_2v_1^Tv_2,
$$

so

$$
(\lambda_1-\lambda_2)v_1^Tv_2=0
$$

and therefore

$$
\boxed{v_1^Tv_2=0}.
$$

Geometrically, orthogonality ensures later components capture new variance rather than reusing directions already represented.

Using the orthonormal eigenbasis $v_1,\dots,v_d$, any unit vector can be written

$$
v=\sum_j a_jv_j,\qquad \sum_j a_j^2=1.
$$

Then

$$
\boxed{
v^TCv=\sum_j\lambda_ja_j^2.
}
$$

The largest eigenvalue wins the unconstrained optimization. Requiring $v\perp v_1$ forces $a_1=0$, so the next-largest eigenvalue wins, and so on.

## Projection, reconstruction, and explained variance

Keeping the top $k$ eigenvectors as columns of $V_k$,

$$
\boxed{
Z=X_cV_k
}
$$

and reconstruction is

$$
\boxed{
\hat X=ZV_k^T+\mu.
}
$$

The matrix $V_kV_k^T$ projects centered data onto the retained principal subspace.

Explained variance ratio is

$$
\boxed{
\mathrm{EVR}_j=
\frac{\lambda_j}{\sum_i\lambda_i}.
}
$$

The denominator must use all eigenvalues, not only the retained $k$.

Discarding components removes the variance in those orthogonal directions. Reconstruction error therefore decreases monotonically as $k$ increases.

## PCA via SVD

For centered data,

$$
X_c=U\Sigma V^T.
$$

Then

$$
X_c^TX_c
=
V\Sigma^2V^T
$$

and hence

$$
C
=
\frac1nV\Sigma^2V^T.
$$

Therefore:

- PCA eigenvectors are the right singular vectors, the columns of $V$.
- Covariance eigenvalues and singular values satisfy

$$
\boxed{
\lambda_i=\frac{\sigma_i^2}{n}
}
$$

for the $1/n$ covariance convention.

PCA coordinates can be computed either as

$$
\boxed{Z=X_cV_k}
$$

or directly as

$$
\boxed{Z=U_k\Sigma_k}.
$$

NumPy's `np.linalg.svd` returns `Vt = V^T`, so the top-$k$ PCA component matrix can be obtained with

```python
components = Vt[:k, :].T
```

Use `np.linalg.eigh` rather than general `eig` for covariance matrices because they are symmetric.

## Low-rank approximation

Truncated SVD

$$
X_k=U_k\Sigma_kV_k^T
$$

is the best rank-$k$ approximation under the Frobenius norm (Eckart-Young theorem).

The squared reconstruction error is

$$
\boxed{
\|X-X_k\|_F^2
=
\sum_{j=k+1}^r\sigma_j^2.
}
$$

This links PCA's discarded variance directly to low-rank approximation error.

---

# Stage 3.5 — Numerical Gradients and Backprop Scaffolding

## Central finite differences

For a scalar function $f:\mathbb R^d\to\mathbb R$,

$$
\frac{\partial f}{\partial x_i}
\approx
\frac{
f(x+\epsilon e_i)-f(x-\epsilon e_i)
}{
2\epsilon
}.
$$

Forward difference has truncation error $O(\epsilon)$, while central difference cancels the leading even-order error term and has truncation error $O(\epsilon^2)$.

Making $\epsilon$ arbitrarily small is not optimal. When the two function evaluations become nearly equal, floating-point subtraction suffers catastrophic cancellation. Gradient checking therefore balances truncation error against floating-point error.

A numerical gradient checker was implemented and verified against

$$
f(w)=\sum_iw_i^2,\qquad \nabla f=2w.
$$

This concept is understood; do not repeat equivalent bookkeeping exercises unless needed to debug later backprop.

## Jacobians and vector-Jacobian products

For

$$
y=f(x),\qquad
x\in\mathbb R^d,\quad
y\in\mathbb R^m,
$$

the Jacobian is

$$
J=\frac{\partial y}{\partial x}\in\mathbb R^{m\times d}.
$$

For scalar loss $L(y)$, the chain rule gives

$$
\boxed{
\nabla_xL
=
J^T\nabla_yL.
}
$$

Backprop/reverse-mode autodiff generally does not materialize the full Jacobian. It propagates vector-Jacobian products through the computation graph. This is crucial because a full Jacobian can require $O(md)$ storage even though the gradient needed downstream has only $d$ entries.

---

# Current checkpoint and pacing

Stages 1–3 are complete at the intended conceptual level. Stage 3.5 has covered the finite-difference and Jacobian/VJP foundations needed to begin neural-network backprop.

Next: **Stage 4 — NumPy neural network from scratch**, beginning with a linear layer

$$
Z=XW+b
$$

and deriving/implementing backward passes for $X$, $W$, and $b$, then activations, loss, backpropagation, mini-batches, softmax, and multiclass cross-entropy.

## Teaching/pacing note

The learner has an engineering and mathematics background and substantial prior calculus exposure. Much of classical regression/calculus is review.

For review material:
- use fast diagnostic questions to verify reconstruction;
- avoid repeating the same concept in multiple near-identical exercises;
- once understanding is demonstrated, move on.

Spend deeper time on concepts that are new or directly important for modern ML/deep learning, research engineering, optimization, representation learning, and later paper reproduction.
