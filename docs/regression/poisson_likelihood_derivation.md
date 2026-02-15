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
