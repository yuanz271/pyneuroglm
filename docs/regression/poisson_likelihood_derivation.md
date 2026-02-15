# Poisson GLM: gradient and Hessian for a general inverse link

This note derives the gradient and Hessian of the Poisson log-likelihood for a GLM
with rate \(\lambda = f(\eta)\) and linear predictor \(\eta = Xw\).

## Model

- \(y_i \sim \mathrm{Poisson}(\lambda_i)\)
- \(\eta_i = x_i^\top w\)
- \(\lambda_i = f(\eta_i)\) (inverse link)

## Log-likelihood (up to constants)

\[
L(w) = \sum_{i=1}^n \left[y_i\log(\lambda_i) - \lambda_i\right]
\]

## Gradient

Define \(\dot{\lambda}_i = \frac{d\lambda_i}{d\eta_i}\). Then

\[
\nabla_w L(w) = X^\top \left(\left(\frac{y}{\lambda} - 1\right) \odot \dot{\lambda}\right)
\]

where division and \(\odot\) are elementwise.

## Hessian

Define \(\ddot{\lambda}_i = \frac{d^2\lambda_i}{d\eta_i^2}\). The Hessian can be written as

\[
\nabla_w^2 L(w) = X^\top \mathrm{diag}(h)\,X
\]

with per-observation diagonal terms

\[
h_i = \left(\frac{y_i}{\lambda_i} - 1\right)\ddot{\lambda}_i - y_i\frac{\dot{\lambda}_i^2}{\lambda_i^2}.
\]

This matches the implementation in `pyneuroglm.regression.likelihood`.

## Canonical exponential link

For \(\lambda_i = \exp(\eta_i)\), we have \(\dot{\lambda}_i = \ddot{\lambda}_i = \lambda_i\), so

\[
h_i = -\lambda_i
\]

and therefore

\[
\nabla_w^2 L(w) = -X^\top \mathrm{diag}(\lambda)\,X,
\]

which is negative semidefinite (concave log-likelihood).

## Gaussian prior

A zero-mean Gaussian prior with inverse covariance (precision) matrix \(C^{-1}\) gives

\[
P(w) = -\tfrac{1}{2}\,w^\top C^{-1} w,
\qquad
\nabla_w P = -C^{-1} w,
\qquad
\nabla_w^2 P = -C^{-1}.
\]

For a ridge prior, \(C^{-1} = \alpha I\). The intercept is typically excluded from
penalisation by setting the corresponding diagonal entry to zero.

This matches the implementation in `pyneuroglm.regression.prior`.

## MAP objective

The log-posterior is the sum of log-likelihood and log-prior:

\[
\log p(w \mid y) \propto L(w) + P(w).
\]

The MAP estimate \(\hat{w}\) maximises this, or equivalently minimises the negative
log-posterior \(-L(w) - P(w)\). At the optimum the gradient vanishes:

\[
\nabla_w L(\hat{w}) + \nabla_w P(\hat{w}) = 0.
\]

## Laplace approximation

The posterior covariance is approximated by the inverse of the negative Hessian of the
log-posterior evaluated at the MAP:

\[
\Sigma_{\mathrm{post}} \approx \left[-\nabla_w^2 L(\hat{w}) - \nabla_w^2 P(\hat{w})\right]^{-1}.
\]

Marginal standard deviations are \(\sqrt{\mathrm{diag}(\Sigma_{\mathrm{post}})}\).

The Laplace approximation to the log marginal likelihood (log evidence) is

\[
\log p(y) \approx L(\hat{w}) + P(\hat{w}) - \tfrac{1}{2}\log\left|\Sigma_{\mathrm{post}}^{-1}\right|.
\]

The normalising constant \(\tfrac{d}{2}\log(2\pi)\) is omitted since it is independent
of the hyperparameters and cancels during model comparison.

This is used for hyperparameter selection (e.g., choosing \(\alpha\)).

These match the implementations in `pyneuroglm.regression.posterior` and
`pyneuroglm.regression.empirical_bayes`.
