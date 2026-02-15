# Introduction

*pyneuroglm* is porting [neuroGLM](https://github.com/pillowlab/neuroGLM) to Python. Its main functionality is to allow you to **expand and transform your experimental variables to a feature space as a [design matrix](http://en.wikipedia.org/wiki/Design_matrix) such that a simple linear analysis could yield desired results**.

This tutorial follows its [counterpart](https://github.com/pillowlab/neuroGLM/blob/master/docs/tutorial.md) in *neuroGLM* that explains how to import your experimental data, and build appropriate features spaces to do fancy regression analysis easily. The implementation is slightly different from *neuroGLM*. For the concepts, please see *neuroGLM*.

# Concepts

We assume the experimental variables are observations over time, and organized into **trials**.
If your data don't have a trial structure, you can put all your data in a single trial.
There are 4 types of variables that constitute data: *spike train*, *timing*, *continuous*, and *value*.
This framework uses string labels to address each variable later.

## Types of experimental variables
### Spike train

Each spike train is a sequence of spike timings from a single neuron.
The spike timings are relative to the beginning of the trial.

### Timing (events)

A typical trial based experiment may have a cue that indicates the beginning of a trial, cues that indicate waiting period, or presentation of a target at random times.
These covariates are best represented as events.
However, if two or more timings are perfectly correlated, it is often sufficient to include just one.
Note that many behaviors are also recorded as timing: reaction time, button press time, etc.

### Continuous

Continuous data are measured continuously over time.
For instance, eye position of the animal may be correlated with the activities of neurons of the study, so the experimentalist could carefully measure it throughout the experiment.
Note that the sampling rate should match the bin size of the analysis, otherwise up-sampling, or down-sampling (with appropriate filtering) is necessary.

### Value

Each trial can have a single value associated with it.
In many cases these are trial specific parameters such as strength of the stimulus, type of cue, or the behavioral category.
These values can be used to build a feature space, or  to include specific feature in trials only when certain conditions are met.

## Registering variables to the experiment

Each experimental variable must be registered before the data are loaded.
First, create an experiment object using `pyneuroglm.Experiment`:
```python
from pyneuroglm import Experiment

expt = Experiment(time_unit="ms", binsize=10, eid="example", meta={})
```
where `time_unit` is a string for the time unit that's going to be used consistently throughout (e.g., 's' or 'ms'), `binsize` is the duration of the time bin to discretize the timings.
`eid` is a string to uniquely identify the experiment among other experiments (mostly for the organizational purpose).
`meta` can be anything that you want to associate with the experiment structure for easy access later, since it will be carried around throughout the code.

Then, each experimental variable is registered by indicating the type, label, and user friendly name of the variable.
```python
expt.register_continuous("LFP", "Local Field Potential", ndim=1)  # 1D observation over time
expt.register_continuous("eyepos", "Eye Position", ndim=2)  # 2D observation
expt.register_timing("fpon", "Fixation On")  # events that happen 0 or more times per trial
expt.register_timing("fpoff", "Fixation Off")
expt.register_timing("saccade", "Monkey's Saccade Timing")
expt.register_spike_train("sptrain", "Our Neuron")  # spike train
expt.register_value("coh", "Dots Coherence")  # per-trial scalar value
```

## Loading the data for each trial

For each trial, we load each of the possible covariate into the experiment structure.

For each trial, we make a temporary object `trial` to load the data:
```python
from pyneuroglm import Trial

trial = Trial(tid=1, duration=1000)
```
where `tid` is the unique identifier, and `duration` is the length of the current trial in `time_unit`.

`trial` is an object where you add each experimental variable you registered for the experiment as a key-value pair. Below is a minimal example.

```python
import numpy as np

trial["fpon"] = 100.0  # timing variable (in ms)
trial["fpoff"] = 700.0
trial["sptrain"] = np.array([120.0, 180.0, 350.0])  # spike times (in ms)
trial["coh"] = 0.5
```

Finally, we add the trial object to the experiment object:
```python
expt.add_trial(trial)
```

Repeat this for all your trials, and you are done loading your data. The experiment object will validate the trial when you add it.

# Forming your feature space
Once you have your data loaded as an experiment object, you are now ready to specify how your experimental variables will be represented, and hence how your design matrix will be formed.

## Design specification
We start by creating a **design matrix object**.
```python
from pyneuroglm.design import DesignMatrix

dm = DesignMatrix(expt)
```
You can have multiple such objects per experiment to analyze your experiments in different ways and compare models.
The design matrix object contains the covariate specifications (basis functions, temporal offsets, etc.) and can compile them into a NumPy design matrix.

For a timing variable, the following syntax adds a **delta function** at the time of the event:
```python
dm.add_covariate_timing(label="fpon", description="Fixation On")
```
However, this is seldom what you want. You probably want to have a temporal basis to represent delayed effects of the covariate on the response variable.
Let's make a set of 8 boxcar basis functions to cover 300 ms evenly:
```python
from pyneuroglm.basis import make_smooth_temporal_basis

fix_basis = make_smooth_temporal_basis("boxcar", duration=300, n_bases=8, binfun=expt.binfun)
dm.add_covariate_boxcar(
    label="fixation",
    on_label="fpon",
    off_label="fpoff",
    description="Fixation",
    basis=fix_basis,
)
```
and use this to represent the effect of timing event instead.

If you want to use autoregressive point process modeling (often known as GLM in neuroscience) by adding the spike history filter, you can do the following:
```python
dm.add_covariate_spike(label="hist", stim_label="sptrain", description="History filter")
```
This adds spike history filters with default history basis functions.

## Building the design matrix
The ultimate output is the design matrix:
```python
X = dm.compile_design_matrix(trial_indices=None)
```
where `trial_indices` are the trials to include in making the design matrix (or None for all trials). This function can be memory intensive for large datasets.

# Regression analysis
Once you have designed your features, and obtained the design matrix, it's finally time to do some analysis!

## Get the dependent variable
You need to obtain the response variable of the same length as the number of rows in the design matrix to do regression. For **point process** regression, where we want to predict the observed spike train from covariates, this would be a finely binned spike train concatenated over the trials of interest:
```python
# Get the spike trains back to regress against
y = dm.get_binned_spike(label="sptrain", trial_indices=None)
```

For predicting some continuous observation, such as predicting the LFP, you can do:
```python
y = dm.get_response("LFP")
```
Make sure your `y` is a column vector; `get_response` returns a matrix if the experimental variable is more than 1 dimension.

## Doing the actual regression
You can use any regression library with the design matrix and response variable. [scikit-learn](https://scikit-learn.org/) and [statsmodels](https://www.statsmodels.org/) are two popular Python packages that provide a variety of regression models.

## MAP estimation with BayesianGLMRegressor

pyneuroglm includes a scikit-learn-compatible Poisson GLM regressor that performs Maximum A Posteriori estimation with a ridge (Gaussian) prior:

```python
from pyneuroglm.regression.sklearn import BayesianGLMRegressor

model = BayesianGLMRegressor(alpha=1.0)
model.fit(X, y)
y_pred = model.predict(X)
```

The `alpha` parameter controls prior strength (higher values shrink weights toward zero). The optimizer uses analytical gradient and Hessian via `trust-ncg`, matching the MATLAB `fminunc` trust-region approach.

After fitting, the model provides:
- `model.coef_` and `model.intercept_` -- MAP weight estimates
- `model.coef_std_` and `model.intercept_std_` -- marginal posterior standard deviations (Laplace approximation)
- `model.hessian_inv_` -- full approximate posterior covariance matrix (includes intercept dimension)

## Reconstructing weights per covariate

To inspect weights grouped by covariate (e.g., for plotting temporal filters):

```python
ws = dm.combine_weights(model.coef_)
```

This returns a dictionary keyed by covariate label, each containing the weight vector and time axis for that covariate.
