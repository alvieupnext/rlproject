import torch
import copy
#Population Methods
#Population Methods
# Zeroth-order optimization still depends on a learning rate: once the perturbations are evaluated,
# they are used to compute a direction in which to change θ, and the direction is followed with a small
# learning rate. This still has problems with local minima.
# Population methods follow a simpler approach: many perturbations of θ are produced, and the best
# one is copied into θ:
# 1. The policy π has parameters θ.
# 2. Produce N perturbations of θ, now called θi with i going from 1 to N.
# 3. Evaluate each θi in the environment, by performing one or more episodes (as with Zeroth-
# order optimization). This produces one “score” per θi.
# 4. Select the θi that has the largest score, and put it in θ. The policy has now been updated.
# 5. Go back to Step 1.
# Population methods can be made more complex (selecting the top-K perturbations and doing
# something on them, for instance), but the simple method described above is already enough for this
# project.
# Population methods are not sensitive to local optima, and are generally better at eventually learning
# the optimal policy. With an infinite amount of time, they will try every possible θ, and find the one
# that leads to the best returns. However, population methods are not sample-efficient. Each
# application of the algorithm above requires at least N full episodes in the environment, to compute
# only a single updated θ.

#Write a function that given a policy, will produce N perturbations of θ, with a perturbation strength of σ.
# This function should receive a parametric policy and return a list of perturbed policies.
def perturb_policy_zeroth_order(original_policy, sigma=0.1):

  # Get the parameters of the original policy
  original_parameters = [p.data for p in original_policy.parameters()]

  # Copy the original policy to create a new instance for perturbation
  perturbed_policy_plus = copy.deepcopy(original_policy)
  perturbed_policy_minus = copy.deepcopy(original_policy)

  # Perturb each parameter
  for param, param_plus, param_minus in zip(original_parameters, perturbed_policy_plus.parameters(),
                                            perturbed_policy_minus.parameters()):
    # Create the perturbation vector
    perturbation = torch.randn_like(param) * sigma

    # Apply the perturbation
    param_plus.data += perturbation
    param_minus.data -= perturbation

  return perturbed_policy_plus, perturbed_policy_minus

#Write a function that given a policy, will produce N perturbations of θ, with a perturbation strength of σ.
def generate_perturbed_policies(original_policy, N, sigma):
  perturbed_policies = []

  # Get the parameters of the original policy
  original_parameters = [p.data for p in original_policy.parameters()]

  for _ in range(N):
    # Copy the original policy to create a new instance for perturbation
    perturbed_policy = copy.deepcopy(original_policy)

    # Perturb each parameter
    for original_param, perturbed_param in zip(original_parameters, perturbed_policy.parameters()):
      # Create the perturbation
      perturbation = torch.randn_like(original_param) * sigma

      # Apply the perturbation
      perturbed_param.data = original_param + perturbation

    # Add the perturbed policy to the list
    perturbed_policies.append(perturbed_policy)

  return perturbed_policies