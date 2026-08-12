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
\hat y\in\operatorname{Col}(X).
$$

At the OLS optimum,

$$
X^Tr=0.
$$

This means

$$
r\perp\operatorname{Col}(X).
$$

Therefore,

$$
\boxed{\hat y=\operatorname{proj}_{\operatorname{Col}(X)}(y)}.
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
