"""scikit-learn-compatible wrappers around pyneuroglm regressors."""

# https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/linear_model/_linear_loss.py#L207
# L2 penalty term l2_reg_strength/2 *||w||_2^2.

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from .empirical_bayes import log_evidence
from .likelihood import poisson as poisson_loglik
from .posterior import get_posterior_weights, initialize_zero, initialize_lstsq
from .prior import gaussian_zero_mean_inv, ridge_Cinv


def _poisson_link_fun(eta):
    """
    Exponential (Poisson inverse link) with derivatives.

    Computes the exponential function and its first and second derivatives
    for use as the inverse link function in Poisson regression.

    Parameters
    ----------
    eta : array-like
        Linear predictor values.

    Returns
    -------
    lam : array-like
        Exponential of eta (lambda parameter).
    dlam : array-like
        First derivative of lambda with respect to eta.
    ddlam : array-like
        Second derivative of lambda with respect to eta.
    """
    lam = np.exp(eta)
    dlam = lam
    ddlam = lam
    return lam, dlam, ddlam


def make_loglikelihood(X, y):
    """
    Create a log-likelihood function for Poisson regression.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Design matrix.
    y : array-like of shape (n_samples,)
        Response variable.

    Returns
    -------
    callable
        Log-likelihood function that takes weights as input.
    """

    def loglik(w, *args):
        """
        Compute the Poisson log-likelihood for weight vector `w`.

        Parameters
        ----------
        w : array-like of shape (n_features,)
            Weight vector at which to evaluate the log-likelihood.
        *args
            Ignored additional arguments (for sklearn compatibility).

        Returns
        -------
        float
            Log-likelihood value.
        """
        return poisson_loglik(w, X, y, _poisson_link_fun)

    return loglik


def log_evidence_scorer(estimator, X, y):
    """
    Scikit-learn scorer using empirical Bayes log_evidence.

    Parameters
    ----------
    estimator : BayesianGLMRegressor
        Fitted estimator with coef_ and optionally intercept_ attributes.
    X : array-like of shape (n_samples, n_features)
        Test samples.
    y : array-like of shape (n_samples,)
        True target values.

    Returns
    -------
    float
        Log evidence score (higher is better).

    Notes
    -----
    Only supports BayesianGLMRegressor and similar estimators with
    coef_, alpha, and optionally intercept_ attributes.
    """
    # Validate estimator is fitted and extract MAP weights, prior, intercept
    check_is_fitted(estimator, attributes=["coef_"])

    w = np.asarray(estimator.coef_).ravel()
    X_ = X
    has_intercept = getattr(estimator, "fit_intercept", False)
    if has_intercept:
        w = np.concatenate([[estimator.intercept_], w])
        X_ = np.column_stack((np.ones(X.shape[0]), X))

    m = X_.shape[1]
    alpha = getattr(estimator, "alpha", 1.0)
    # Prior: ridge with (usually) shape (m, m)
    Cinv = ridge_Cinv(alpha, m, has_intercept)

    loglik = make_loglikelihood(X_, y)
    # Assemble log_evidence params
    log_ev = log_evidence(
        param=w,
        hyperparam=Cinv,
        loglik=loglik,
        llargs=(),
        logprior=gaussian_zero_mean_inv,
        lpargs=(),
    )
    return log_ev


class BayesianGLMRegressor(RegressorMixin, BaseEstimator):
    """
    Bayesian Generalized Linear Model Regressor using MAP estimation.

    This class wraps the `get_posterior_weights` function to provide a
    scikit-learn compatible interface for Bayesian GLM fitting with
    uncertainty quantification.

    Parameters
    ----------
    alpha : float, default=1.0
        Regularization strength for the ridge (L2) prior. Higher values
        specify stronger regularization.
    dist : {'poisson'}, default='poisson'
        Distribution family for the GLM. Currently only 'poisson' is supported.
    fit_intercept : bool, default=True
        Whether to fit an intercept term.
    initialize : {'zero', 'lstsq'}, default='lstsq'
        Initialization strategy for the optimization.
        - 'zero' : Initialize with zeros
        - 'lstsq' : Initialize with least squares solution
    init_kwargs : dict, default=None
        Additional keyword arguments for the initialization function.

    Attributes
    ----------
    coef_ : ndarray of shape (n_features,)
        Fitted coefficients (excluding intercept if fit_intercept=True).
    intercept_ : float
        Fitted intercept term (only if fit_intercept=True).
    coef_std_ : ndarray of shape (n_features,)
        Posterior standard deviations of the fitted coefficients.
    intercept_std_ : float
        Posterior standard deviation of the intercept (only if fit_intercept=True).
    hessian_inv_ : ndarray of shape (n_features, n_features) or (n_features + 1, n_features + 1)
        Full inverse Hessian at the MAP estimate (includes intercept dimension
        when fit_intercept=True).
    n_features_in_ : int
        Number of features seen during fit.

    Examples
    --------
    >>> import numpy as np
    >>> from pyneuroglm.regression.sklearn import BayesianGLMRegressor
    >>> X = np.random.randn(100, 5)
    >>> y = np.random.poisson(np.exp(X @ np.random.randn(5)))
    >>> model = BayesianGLMRegressor(alpha=1.0)
    >>> model.fit(X, y)
    >>> y_pred = model.predict(X)

    Notes
    -----
    This implementation uses Maximum A Posteriori (MAP) estimation with
    a Gaussian prior on the coefficients. The inverse Hessian provides
    uncertainty estimates for predictions.
    """

    def __init__(
        self, alpha=1.0, dist="poisson", fit_intercept=True, initialize="lstsq", init_kwargs=None
    ):
        """
        Initialize the regressor with prior strength and optimization choices.

        Parameters
        ----------
        alpha : float, default=1.0
            Regularization strength for the ridge prior applied to the weights.
        dist : {'poisson'}, default='poisson'
            Response distribution to fit. Currently only Poisson is supported.
        fit_intercept : bool, default=True
            If True, prepend a bias column to the design matrix and fit an intercept.
        initialize : {'zero', 'lstsq'}, default='lstsq'
            Strategy used to initialize the weight vector before optimization.
        init_kwargs : dict or None, optional
            Extra keyword arguments forwarded to the initialization routine.

        Raises
        ------
        ValueError
            If an unsupported distribution or initialization method is supplied.
        """
        if dist not in ["poisson"]:
            raise ValueError(
                f"Unsupported distribution: {dist}. Supported distributions: ['poisson']"
            )
        if initialize not in ["zero", "lstsq"]:
            raise ValueError(
                f"Unsupported initialization: {initialize}. Supported methods: ['zero', 'lstsq']"
            )

        self.alpha = alpha
        self.dist = dist
        self.fit_intercept = fit_intercept
        self.initialize = initialize
        self.init_kwargs = init_kwargs

    def fit(self, X, y):
        """
        Fit the Bayesian GLM model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data feature matrix.
        y : array-like of shape (n_samples,)
            Target values. For Poisson distribution, should be non-negative
            integer counts.

        Returns
        -------
        self : BayesianGLMRegressor
            Fitted estimator instance.

        Raises
        ------
        ValueError
            If the initialization method is unknown or if input validation fails.
        """
        # Validate input
        X, y = check_X_y(X, y, accept_sparse=False, y_numeric=True)
        self.n_features_in_ = X.shape[1]

        # Prepare design matrix
        if self.fit_intercept:
            X_ = np.column_stack((np.ones(X.shape[0]), X))
        else:
            X_ = X

        # Create inverse covariance matrix for prior (must match X_ dimensions)
        Cinv = ridge_Cinv(self.alpha, X_.shape[1], self.fit_intercept)

        # Map initialization string to function
        if self.initialize == "zero":
            init_func = initialize_zero
        elif self.initialize == "lstsq":
            init_func = initialize_lstsq
        else:
            raise ValueError(f"Unknown initialization method: {self.initialize}")

        # Prepare initialization kwargs
        init_kwargs = self.init_kwargs if self.init_kwargs is not None else {}

        # For zero initialization with Poisson, the intercept should be
        # log(mean(y)) so that exp(w0) = mean(y) at the starting point.
        if self.initialize == "zero" and self.dist == "poisson":
            init_kwargs.setdefault("nlin", np.log)

        # Fit the model using get_posterior_weights
        w, sd, invH = get_posterior_weights(
            X_, y, Cinv, dist=self.dist, cvfolds=None, initialize=init_func, init_kwargs=init_kwargs
        )

        # Store results
        if self.fit_intercept:
            self.intercept_ = w[0]
            self.coef_ = w[1:]
            self.intercept_std_ = sd[0]
            self.coef_std_ = sd[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = w
            self.intercept_std_ = 0.0
            self.coef_std_ = sd

        self.hessian_inv_ = invH

        return self

    def predict(self, X):
        """
        Predict using the fitted model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples for prediction.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted mean values. For Poisson distribution, these are
            the expected counts (lambda parameter).

        Raises
        ------
        NotFittedError
            If the model has not been fitted yet.
        ValueError
            If X has a different number of features than during training.
        """
        check_is_fitted(self)
        X = check_array(X, accept_sparse=False)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but this regressor "
                f"was fitted with {self.n_features_in_} features."
            )

        # Compute linear predictor
        eta = X @ self.coef_
        if self.fit_intercept:
            eta += self.intercept_

        # Apply inverse link function based on distribution
        if self.dist == "poisson":
            y_pred = np.exp(eta)
        else:
            raise NotImplementedError(f"Prediction for distribution '{self.dist}' not implemented")

        return y_pred

    def score(self, X, y, sample_weight=None):
        """
        Return the log evidence score of the prediction.

        The log evidence provides a principled way to evaluate model
        quality that balances fit and complexity.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,)
            True target values for X.
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. Not currently implemented.

        Returns
        -------
        score : float
            Log evidence score. Higher values indicate better model quality.
            Falls back to log-likelihood if evidence computation fails.

        Raises
        ------
        NotImplementedError
            If sample_weight is provided (not currently supported).

        Notes
        -----
        The log evidence integrates over parameter uncertainty and provides
        automatic complexity penalization. If computation fails due to
        numerical issues, falls back to log-likelihood scoring.
        """
        if sample_weight is not None:
            raise NotImplementedError("Sample weights not supported")

        try:
            return log_evidence_scorer(self, X, y)
        except ValueError as e:
            if "positive definite" in str(e):
                # Return a fallback score based on likelihood only
                import warnings

                warnings.warn(
                    f"Log evidence computation failed ({e}). Using likelihood-based fallback score."
                )

                # Compute log-likelihood as fallback
                y_pred = self.predict(X)
                # For Poisson: log P(y|lambda) = y*log(lambda) - lambda - log(y!)
                # We'll ignore the factorial term since it's constant for scoring
                log_lik = np.sum(y * np.log(y_pred + 1e-10) - y_pred)
                return log_lik
            else:
                raise
