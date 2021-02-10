#!/usr/bin/env python3
# coding: utf-8

# The one-shot Knowledge Gradient acquisition function

import torch
import numpy as np
from botorch.fit import fit_gpytorch_model
from botorch.models import SingleTaskGP
from botorch.utils import standardize
from gpytorch.mlls import ExactMarginalLogLikelihood

from botorch.acquisition import qKnowledgeGradient, PosteriorMean
from botorch.test_functions import Ackley, Hartmann
from botorch.optim import optimize_acqf
from botorch.utils.sampling import manual_seed
from torch.quasirandom import SobolEngine
from botorch.utils.transforms import unnormalize

from matplotlib import pyplot as plt

SEED = 1024
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double

dim = 6
fun = Hartmann(dim=dim, negate=True).to(dtype=dtype, device=device)

fun.bounds[0, :].fill_(0)
fun.bounds[1, :].fill_(1)
lb, ub = fun.bounds

Initial_Samples = 10
N_TRIALS = 20


def get_initial_points(dim, n_pts):
    sobol = SobolEngine(dimension=dim, scramble=True)
    X_init = sobol.draw(n=n_pts).to(dtype=dtype, device=device)
    return X_init


def eval_objective(x):
    """This is a helper function we use to unnormalize and evalaute a point"""
    return fun(unnormalize(x, fun.bounds))


with manual_seed(1234):
    X_okg = get_initial_points(dim, Initial_Samples)
    Y_okg = torch.tensor(
        [eval_objective(x) for x in X_okg], dtype=dtype, device=device
    ).unsqueeze(-1)

    for i in range(1, N_TRIALS + 1):
        model = SingleTaskGP(X_okg, Y_okg)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_model(mll)

        qKG = qKnowledgeGradient(model, num_fantasies=64)

        argmax_pmean, max_pmean = optimize_acqf(
            acq_function=PosteriorMean(model),
            bounds=fun.bounds,
            q=1,
            num_restarts=4,
            raw_samples=256,
        )

        qKG_proper = qKnowledgeGradient(
            model,
            num_fantasies=64,
            sampler=qKG.sampler,
            current_value=max_pmean,
        )
        candidates_proper, acq_value_proper = optimize_acqf(
            acq_function=qKG_proper,
            bounds=fun.bounds,
            q=1,
            num_restarts=4,
            raw_samples=256,
        )

        candidates_proper_y = torch.tensor(
            [eval_objective(x) for x in candidates_proper], dtype=dtype, device=device
        ).unsqueeze(-1)

        X_okg = torch.cat((X_okg, candidates_proper), dim=0)
        Y_okg = torch.cat((Y_okg, candidates_proper_y), dim=0)
        print(np.maximum.accumulate(Y_okg.cpu())[-1])

        if i < N_TRIALS:
            X_new = get_initial_points(dim, 2)
            Y_new = torch.tensor(
                [eval_objective(x) for x in X_new], dtype=dtype, device=device
            ).unsqueeze(-1)
            X_okg = torch.cat((X_okg, X_new), dim=0)
            Y_okg = torch.cat((Y_okg, Y_new), dim=0)

# ## Sobol

X_Sobol = (SobolEngine(dim, scramble=True).draw(len(X_okg)).to(dtype=dtype, device=device))
Y_Sobol = torch.tensor([eval_objective(x) for x in X_Sobol],
                       dtype=dtype, device=device).unsqueeze(-1)
names = ["OKG", "Sobol"]
runs = [Y_okg, Y_Sobol]

# %% Plot the results
fig, ax = plt.subplots(figsize=(8, 6))
fx = np.maximum.accumulate(Y_okg.cpu())

for name, run in zip(names, runs):
    fx = np.maximum.accumulate(run.cpu())
    plt.plot(fx, marker="", lw=3)

plt.plot([0, len(Y_okg)], [fun.optimal_value, fun.optimal_value], "k--", lw=3)
plt.xlabel("Function value", fontsize=12)
plt.xlabel("Number of evaluations", fontsize=12)
plt.title(f"{dim}D {fun._get_name()}", fontsize=14)
plt.xlim([0, len(Y_okg)])
# plt.ylim([4, 5])

plt.grid(True)
plt.tight_layout()
plt.legend(
    names + ["Global optimal value"],
    # loc="lower center",
    # bbox_to_anchor=(0, -0.01, 1, 1),
    # bbox_transform=plt.gcf().transFigure,
    # ncol=4,
    fontsize=12,
)
plt.show()

max_idx = Y_okg.argmax()
X_max = X_okg[max_idx]
X_max_unnorm = unnormalize(X_max, fun.bounds).cpu().detach().numpy()
print(f"The distance to the optimum is "
      f"{np.sqrt(((X_max_unnorm-np.array(fun._optimizers))**2.0).sum())}")
