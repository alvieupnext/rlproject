import gym
from elegantrl.train.run import *

from plot import plot_rewards, plot_reward_curves
from policy import ParametricPolicy, AffineThrottlePolicy
from pertubation import *
from utils import evaluate_policy, read_project, generate_summary

gym.logger.set_level(40)  # Block warning

# Initialize the environment using the provided utility functions and arguments
env = gym.make('LunarLanderContinuous-v2')

seed = 42

# Seed the environment for reproducibility
env.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

state_dim = 8
action_dim = 2
gamma = 0.99


def average_top_k_policies(policies, rewards, k=1):
  """
  Given a list of policies and their rewards, averages the parameters of the top k policies.

  Args:
      policies (list): The list of policy objects.
      rewards (list): The list of rewards corresponding to each policy.
      k (int): The number of top policies to average.

  Returns:
      average_policy: The policy obtained by averaging the top k policies.
      best_reward: The highest reward achieved among the policies.
  """
  # Identify the best policy and its reward
  #From the array of arrays, get the mean of each array
  mean_rewards = np.mean(rewards, axis=1)
  best_index = np.argmax(mean_rewards)
  best_policy = policies[best_index]
  best_rewards = rewards[best_index]
  #From the best rewards, average to get the best average reward
  best_reward = np.mean(best_rewards)


  # Find the indices of the top k policies
  top_indices = np.argsort(mean_rewards)[-k:][::-1]

  # top_rewards = [rewards[i] for i in top_indices]

  # Print the top k policies and their rewards
  for i, index in enumerate(top_indices):
    print(f'Top {i + 1} Policy: Index {index}, Reward: {rewards[index]}')

  # Start averaging the parameters of the top k policies
  average_policy = copy.deepcopy(best_policy)
  weight = 1  # Initial weight for the best policy
  for i, policy_index in enumerate(top_indices[1:], start=1):  # Skip the best policy itself
    for (param, avg_param) in zip(policies[policy_index].parameters(), average_policy.parameters()):
      # Adjust parameters based on rank and add to average
      adjustment_weight = 1 / (2 * (i + 1))
      avg_param.data += param.data * adjustment_weight
      weight += adjustment_weight

  # Normalize the averaged parameters by the total weight
  for param in average_policy.parameters():
    param.data /= weight

  return average_policy, best_rewards


def run_population_experiment(project_name, num_runs, num_generations, num_episodes, N, sigma, k, max_steps, keep_previous_best=True):
  print(f'Running population method experiment for project {project_name}')
  print(f'Number of runs: {num_runs}')
  print(f'Number of generations: {num_generations}')
  print(f'Number of episodes: {num_episodes}')
  print(f'N: {N}')
  print(f'Sigma: {sigma}')
  print(f'Top k: {k}')
  print(f'Max steps: {max_steps}')
  print(f'Keep previous best: {keep_previous_best}')
  # Ensure results directory exists
  results_dir = os.path.join('results/population', project_name)
  if not os.path.exists(results_dir):
    os.makedirs(results_dir)
  # Make a file called config.txt that stores num_runs, num_generations, N, and sigma
  with open(os.path.join(results_dir, 'config.txt'), 'w') as f:
    f.write(f'num_runs: {num_runs}\n')
    f.write(f'num_generations: {num_generations}\n')
    f.write(f'N: {N}\n')
    f.write(f'sigma: {sigma}\n')
    f.write(f'num_episodes: {num_episodes}\n')
    f.write(f'k: {k}\n')
    f.write(f'max_steps: {max_steps}\n')
    f.write(f'keep_previous_best: {keep_previous_best}\n')

  total_rewards = []
  for run in range(7, num_runs):
    policy = AffineThrottlePolicy(input_size=state_dim, hidden_size=128, output_size=action_dim)
    run_rewards = []
    # Open the file in 'w' mode to clear it, then immediately close it.
    with open(os.path.join(results_dir, f'run{run}.txt'), 'w') as f:
      pass  # This will clear the file contents at the beginning of the run

    policy_reward = evaluate_policy(env, policy, num_episodes, max_steps)
    print(f'Generation 0 Reward: {policy_reward}')
    run_rewards.append(policy_reward)
    # Now open the file in append mode to start adding data
    # with open(os.path.join(results_dir, f'run{run}.txt'), 'a') as f:
    #   # Convert all elements in best_rewards to string and join them with a comma
    #   reward_string = ','.join(map(str, policy_reward))
    #   f.write(reward_string)

    for gen in range(num_generations):
      print(f'Generation {gen + 1}')
      policies = generate_perturbed_policies(policy, N, sigma=sigma)
      if keep_previous_best:
        policies.append(policy)
      rewards = [evaluate_policy(env, policy, num_episodes, max_steps) for policy in policies]

      #Rewards contains N arrays, make them a single array
      all_rewards = [num for sublist in rewards for num in sublist]


      average_policy, best_rewards = average_top_k_policies(policies, rewards, k=k)

      policy = average_policy
      # Continue using append mode for subsequent writes within the same run
      with open(os.path.join(results_dir, f'run{run}.txt'), 'a') as f:
        #Write every reward of best_rewards to the file
        for episode_reward in all_rewards:
          f.write(f',{episode_reward}')

    total_rewards.append(run_rewards)
  return total_rewards

if __name__ == '__main__':
  # Train the policy
  # num_episodes = 20
  # num_generations = 2000
  # num_runs = 10
  # max_steps = 500
  # N = 10
  # sigma = 0.01
  # k = 1
  # alpha = 0.001
  num_episodes = 20
  num_generations = 1500
  num_runs = 10
  max_steps = 500
  N = 9
  sigma = 0.01
  k = 1
  experiment = 'lunar_lander_population_all_ep'
  run_population_experiment(experiment, num_runs, num_generations, num_episodes, N,
                            sigma, k, max_steps)
  generate_summary(experiment, type='population')
  # population_rewards, population_std_rewards, population_config = read_project(experiment,
  #                                                                              type='population', single_run=False)
  # plot_rewards(population_rewards, population_config, population_std_rewards)

  # run_population_experiment('lunar_lander_optimal_1eval_20_runs_test', num_runs, num_generations, num_episodes, N, sigma, k, max_steps)
  # total_rewards, config = read_project(
  #   'lunar_lander_optimal_1eval_20_runs')
