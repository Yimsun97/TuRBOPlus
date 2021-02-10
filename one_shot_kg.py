#!/usr/bin/env python3
# coding: utf-8

# ## The one-shot Knowledge Gradient acquisition function
# 
# The *Knowledge Gradient* (KG) (see [2, 3]) is a look-ahead acquisition function that quantifies the expected
# increase in the maximum of the modeled black-box function $f$ from obtaining additional (random) observations
# collected at the candidate set $\mathbf{x}$. KG often shows improved Bayesian Optimization performance relative to
# simpler acquisition functions such as Expected Improvement, but in its traditional form it is computationally
# expensive and hard to implement.
# 
# BoTorch implements a generalized variant of parallel KG [3] given by $$ \alpha_{\text{KG}}(\mathbf{x}) = \mathbb{
# E}_{\mathcal{D}_{\mathbf{x}}} \Bigl[\, \max_{x' \in \mathbb{X}} \mathbb{E} \left[ g(\xi)\right] \Bigr] - \mu,
# $$ where $\xi \sim \mathcal{P}(f(x') \mid \mathcal{D} \cup \mathcal{D}_{\mathbf{x}})$ is the posterior at $x'$
# conditioned on $\mathcal{D}_{\mathbf{x}}$, the (random) dataset observed at $\mathbf{x}$, and $\mu := \max_{
# x}\mathbb{E}[g(f(x)) \mid \mathcal{D}]$.
# 
# 
# #### Optimizing KG
# 
# The conventional approach for optimizing parallel KG (where $g(\xi) = \xi$) is to apply stochastic gradient ascent,
# with each gradient observation potentially being an average over multiple samples. For each sample $i$,
# the inner optimization problem $\max_{x_i \in \mathbb{X}} \mathbb{E} \left[ \xi^i \mid \mathcal{D}_{\mathbf{x}}^i
# \right]$ for the posterior mean is solved numerically. An unbiased stochastic gradient of KG can then be computed
# by leveraging the envelope theorem and the optimal points $\{x_i^*\}$. In this approach, every iteration requires
# solving numerous inner optimization problems, one for each outer sample, in order to estimate just one stochastic
# gradient.
# 
# The "one-shot" formulation of KG in BoTorch treats optimizing $\alpha_{\text{KG}}(\mathbf{x})$ as an entirely
# deterministic optimization problem. It involves drawing $N_{\!f} = $ `num_fantasies` fixed base samples $\mathbf{
# Z}_f:= \{ \mathbf{Z}^i_f \}_{1\leq i \leq N_{\!f}}$ for the outer expectation, sampling fantasy data $\{\mathcal{
# D}_{\mathbf{x}}^i(\mathbf{Z}_f^i)\}_{1\leq i \leq N_{\!f}}$, and constructing associated fantasy models $\{
# \mathcal{M}^i(\mathbf{Z}_f^i)\}_{1 \leq i \leq N_{\!f}}$. The inner maximization can then be moved outside of the
# sample average, resulting in the following optimization problem: $$ \max_{\mathbf{x} \in \mathbb{X}}\alpha_{\text{
# KG}}(\mathbf{x}) \approx \max_{\mathbf{x}\in \mathbb{X}, \mathbf{X}' \in \mathbb{X}^{N_{\!f}} } %=1}^{\!N_{\!f}}}
# \sum_{i=1}^{N_{\!f}} \mathbb{E}\left[g(\xi^i)\right], $$ where $\xi^i \sim \mathcal{P}(f(x'^i) \mid \mathcal{D}
# \cup \mathcal{D}_{\mathbf{x}}^i(\mathbf{Z}_f^i))$ and $\mathbf{X}' := \{x'^i\}_{1 \leq i \leq N_{\!f}}$.
# 
# If the inner expectation does not have an analytic expression, one can also draw fixed base samples $\mathbf{Z}_I:=
# \{ \mathbf{Z}^i_I \}_{1\leq i\leq N_{\!I}}$ and use an MC approximation as with the standard MC acquisition
# functions of type `MCAcquisitionFunction`. In either case one is left with a deterministic optimization problem.
# 
# The key difference from the envelope theorem approach is that we do not solve the inner optimization problem to
# completion for every fantasy point for every gradient step with respect to $\mathbf{x}$. Instead, we solve the
# nested optimization problem jointly over $\mathbf{x}$ and the fantasy points $\mathbf{X}'$. The resulting
# optimization problem is of higher dimension, namely $(q + N_{\!f})d$ instead of $qd$, but unlike the envelope
# theorem formulation it can be solved as a single optimization problem, which can be solved using standard methods
# for deterministic optimization.
# 
# 
# [1] M. Balandat, B. Karrer, D. R. Jiang, S. Daulton, B. Letham, A. G. Wilson, and E. Bakshy. BoTorch: A Framework
# for Efficient Monte-Carlo Bayesian Optimization. Advances in Neural Information Processing Systems 33, 2020.
# 
# [2] P. Frazier, W. Powell, and S. Dayanik. A Knowledge-Gradient policy for sequential information collection. SIAM
# Journal on Control and Optimization, 2008.
# 
# [3] J. Wu and P. Frazier. The parallel knowledge gradient method for batch bayesian optimization. NIPS 2016.

# ### Setting up a toy model
# 
# We'll fit a standard `SingleTaskGP` model on noisy observations of the synthetic function $f(x) = \sin(2 \pi x_1) *
# \cos(2 \pi x_2)$ in `d=2` dimensions on the hypercube $[0, 1]^2$.

# In[1]:


import math
import torch

from botorch.fit import fit_gpytorch_model
from botorch.models import SingleTaskGP
from botorch.utils import standardize
from gpytorch.mlls import ExactMarginalLogLikelihood


# In[2]:


bounds = torch.stack([torch.zeros(2), torch.ones(2)])

train_X = bounds[0] + (bounds[1] - bounds[0]) * torch.rand(20, 2)
train_Y = torch.sin(2 * math.pi * train_X[:, [0]]) * torch.cos(2 * math.pi * train_X[:, [1]])

train_Y = standardize(train_Y + 0.05 * torch.randn_like(train_Y))

model = SingleTaskGP(train_X, train_Y)
mll = ExactMarginalLogLikelihood(model.likelihood, model)
fit_gpytorch_model(mll);


# ### Defining the qKnowledgeGradient acquisition function
# 
# The `qKnowledgeGradient` complies with the standard `MCAcquisitionFunction` API. The only mandatory argument in
# addition to the model is `num_fantasies` the number of fantasy samples. More samples result in a better
# approximation of KG, at the expense of both memory and wall time.
# 
# `qKnowledgeGradient` also supports the other parameters of `MCAcquisitionFunction`, such as a generic objective
# `objective` and pending points `X_pending`. It also accepts a `current_value` argument that is the maximum
# posterior mean of the current model (which can be obtained by maximizing `PosteriorMean` acquisition function).
# This does not change the optimizer so it is not required, but it means that the acquisition value is some constant
# shift of the actual "Knowledge Gradient" value.

# In[3]:


from botorch.acquisition import qKnowledgeGradient

qKG = qKnowledgeGradient(model, num_fantasies=128)


# ### Optimizing qKG
# 
# `qKnowledgeGradient` subclasses `OneShotAcquisitionFunction`, which makes sure that the fantasy parameterization
# $\mathbf{X}'$ is automatically generated and optimized when calling `optimize_acqf` on the acquisition function.
# This means that optimizing one-shot KG in BoTorch is just a easy as optimizing any other acquisition function (from
# an API perspective, at least). It turns out that a careful initialization of the fantasy points can significantly
# help with the optimization (see the logic in `botorch.optim.initializers.gen_one_shot_kg_initial_conditions` for
# more information).
# 
# 
# Here we use `num_restarts=10` random initial `q`-batches with `q=2` in parallel, with the intialization heuristic
# starting from `raw_samples = 512` raw points (note that since `qKnowledgeGradient` is significantly more expensive
# to evaluate than other acquisition functions, large values of `num_restarts` and `raw_samples`, which are typically
# feasible in other settings, can result in long wall times and potential memory issues).
# 
# Finally, since we do not pass a `current_value` argument, this value is not actually the KG value, but offset by
# the constant (w.r.t. the candidates) $\mu := \max_{x}\mathbb{E}[g(f(x)) \mid \mathcal{D}]$.

# In[4]:


from botorch.optim import optimize_acqf
from botorch.utils.sampling import manual_seed

with manual_seed(1234):
    candidates, acq_value = optimize_acqf(
        acq_function=qKG, 
        bounds=bounds,
        q=2,
        num_restarts=10,
        raw_samples=512,
    )


# In[5]:


candidates


# In[6]:


acq_value


# ### Computing the actual KG value
# 
# We first need to find the maximum posterior mean - we can use a large number of random restarts and raw_samples to
# increase the likelihood that we do indeed find it (this is a non-convex optimization problem, after all).

# In[7]:


from botorch.acquisition import PosteriorMean

argmax_pmean, max_pmean = optimize_acqf(
    acq_function=PosteriorMean(model), 
    bounds=bounds,
    q=1,
    num_restarts=20,
    raw_samples=2048,
)


# Now we can optimize KG after passing the current value. We also pass in the `sampler` from the original `qKG`
# above, which containst the fixed base samples $\mathbf{Z}_f$. This is to ensure that we optimize the same
# approximation and so our values are an apples-to-apples comparison (as `num_fantasies` increases, the effect of
# this randomness will get less and less important).

# In[8]:


qKG_proper = qKnowledgeGradient(
    model,
    num_fantasies=128,
    sampler=qKG.sampler,
    current_value=max_pmean,
)

with manual_seed(1234):
    candidates_proper, acq_value_proper = optimize_acqf(
        acq_function=qKG_proper, 
        bounds=bounds,
        q=2,
        num_restarts=10,
        raw_samples=512,
    )


# In[9]:


candidates_proper


# In[10]:


acq_value_proper

