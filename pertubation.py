import torch
import copy

#Write a function that given a policy, will produce N perturbations of θ, with a perturbation strength of σ.
# This function should receive a parametric policy and return a list of perturbed policies.
def perturb_policy_zeroth_order(original_policy, sigma):

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