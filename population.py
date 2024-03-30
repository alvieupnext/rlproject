import elegantrl
# from elegantrl.run import *
import gym
import gym.envs.box2d.lunar_lander as lunar_lander
from elegantrl.agents import AgentModSAC
from elegantrl.train.config import get_gym_env_args, Config
from elegantrl.train.run import *

from plot import plot_rewards
from policy import ParametricPolicy, AffineThrottlePolicy
from pertubation import *

gym.logger.set_level(40)  # Block warning

# env = gym.make('LunarLanderContinuous-v2')

# env_func = gym.make
# env_args = {
#     # "env_num": 1,
#     "env_name": "LunarLanderContinuous-v2",
#     "max_step": 1000,
#     "state_dim": 8,
#     "action_dim": 2,
#     "if_discrete": False,
#     "target_return": 200,
#     "id": "LunarLanderContinuous-v2",
# }

# # Define the number of episodes for training
# num_episodes = 1000  # Or any other number of episodes you wish to train for

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



def evaluate_policy(policy, num_episodes, max_steps):
  policy_reward = 0
  for episode in range(num_episodes):
    state = env.reset()
    episode_reward = 0

    for t in range(max_steps):
      # Convert state into a tensor
      state_tensor = torch.tensor(state, dtype=torch.float32)

      # Generate an action from the policy
      action = policy(state_tensor)

      # Perform the action in the environment
      next_state, reward, done, _ = env.step(action.detach().numpy())

      # Discount the reward
      # episode_reward += reward * (gamma ** t)
      episode_reward += reward

      # Prepare for the next iteration
      state = next_state

      # If the episode is done, exit the loop
      if done:
        break

    # Add the episode reward to the policy reward
    policy_reward += episode_reward
    # # Log the episode's results
    # print(f'Episode {episode}: Total Reward: {episode_reward}')
  # Average the policy reward over the number of episodes
  policy_reward /= num_episodes
  return policy_reward


def average_top_k_policies(policies, rewards, k):
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
  best_index = np.argmax(rewards)
  best_policy = policies[best_index]
  best_reward = rewards[best_index]

  # Find the indices of the top k policies
  top_indices = np.argsort(rewards)[-k:][::-1]
  top_rewards = [rewards[i] for i in top_indices]

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

  return average_policy, best_reward


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
  for run in range(num_runs):
    policy = AffineThrottlePolicy(input_size=state_dim, hidden_size=128, output_size=action_dim)
    run_rewards = []
    # Open the file in 'w' mode to clear it, then immediately close it.
    with open(os.path.join(results_dir, f'run{run}.txt'), 'w') as f:
      pass  # This will clear the file contents at the beginning of the run

    policy_reward = evaluate_policy(policy, num_episodes, max_steps)
    print(f'Generation 0: Reward: {policy_reward}')
    run_rewards.append(policy_reward)
    # Now open the file in append mode to start adding data
    with open(os.path.join(results_dir, f'run{run}.txt'), 'a') as f:
      f.write(f'{policy_reward}')  # Write the initial reward

    for gen in range(num_generations):
      print(f'Generation {gen + 1}')
      policies = generate_perturbed_policies(policy, N, sigma=sigma)
      if keep_previous_best:
        policies.append(policy)
      rewards = [evaluate_policy(policy, num_episodes, max_steps) for policy in policies]

      average_policy, best_reward = average_top_k_policies(policies, rewards, k=k)

      policy = average_policy
      # Continue using append mode for subsequent writes within the same run
      with open(os.path.join(results_dir, f'run{run}.txt'), 'a') as f:
        f.write(f',{best_reward}')  # Append the best_reward for this generation

    total_rewards.append(run_rewards)

  # Convert to numpy array for analysis
  total_rewards = np.array(total_rewards)
  average_total_rewards = np.mean(total_rewards, axis=0)
  std_total_rewards = np.std(total_rewards, axis=0)

  # Write summary statistics to files
  with open(os.path.join(results_dir, 'summary_average.txt'), 'w') as f:
    f.write(','.join(map(str, average_total_rewards)))
  with open(os.path.join(results_dir, 'summary_std.txt'), 'w') as f:
    f.write(','.join(map(str, std_total_rewards)))

  return average_total_rewards

def run_zeroth_order_experiment(project_name, num_runs, num_generations, num_episodes, sigma, alpha, max_steps):
  print(f'Running zeroth-order experiment for project {project_name}')
  print(f'Number of runs: {num_runs}')
  print(f'Number of generations: {num_generations}')
  print(f'Number of episodes: {num_episodes}')
  print(f'Sigma: {sigma}')
  print(f'Alpha: {alpha}')
  print(f'Max steps: {max_steps}')
  # Ensure results directory exists
  results_dir = os.path.join('results/zeroth', project_name)
  if not os.path.exists(results_dir):
    os.makedirs(results_dir)
  # Make a file called config.txt that stores num_runs, num_generations, N, and sigma
  with open(os.path.join(results_dir, 'config.txt'), 'w') as f:
    f.write(f'num_runs: {num_runs}\n')
    f.write(f'num_generations: {num_generations}\n')
    f.write(f'sigma: {sigma}\n')
    f.write(f'alpha: {alpha}\n')
    f.write(f'num_episodes: {num_episodes}\n')
    f.write(f'max_steps: {max_steps}\n')
  total_rewards = []
  for run in range(num_runs):
    policy = AffineThrottlePolicy(input_size=state_dim, hidden_size=128, output_size=action_dim)
    run_rewards = []
    # Open the file in 'w' mode to clear it, then immediately close it.
    with open(os.path.join(results_dir, f'run{run}.txt'), 'w') as f:
      pass  # This will clear the file contents at the beginning of the run
    policy_reward = evaluate_policy(policy, num_episodes, max_steps)
    print(f'Generation 0: Reward: {policy_reward}')
    run_rewards.append(policy_reward)
    # Now open the file in append mode to start adding data
    with open(os.path.join(results_dir, f'run{run}.txt'), 'a') as f:
      f.write(f'{policy_reward}')  # Write the initial reward

    for gen in range(num_generations):
      pertubations = perturb_policy_zeroth_order(policy, sigma)
      p_plus, p_min = pertubations
      rewards = [evaluate_policy(policy, num_episodes, max_steps) for policy in pertubations]
      p_plus_score, p_min_score = rewards
      # A sort of gradient of θ, that we did not have to compute, is now given by “0.5 × (score of θ+ -
      # score of θ-) × θ+”.
      # Calculate this gradient, we already apply alpha here
      gradient = alpha * (0.5 * (p_plus_score - p_min_score))
      #Move the positive perturbation in the direction of the gradient
      for param in p_plus.parameters():
        param.data *= gradient
      #Move θ, the parameters of the policy, in the direction of the gradient.
      #The step size is α, and the direction is the gradient.
      policy.add_policy(p_plus)
      # Evaluate the new policy
      policy_reward = evaluate_policy(policy, num_episodes, max_steps)
      print(f'Generation {gen + 1}: Reward: {policy_reward}')
      run_rewards.append(policy_reward)
      # Continue using append mode for subsequent writes within the same run
      with open(os.path.join(results_dir, f'run{run}.txt'), 'a') as f:
        f.write(f',{policy_reward}')  # Append the best_reward for this generation

    total_rewards.append(run_rewards)


def read_project(project_name, single_run=True, type='population'):
  """
  Reads and returns the average and standard deviation of rewards
  for a given project.

  Args:
      project_name (str): The name of the project.

  Returns:
      tuple: A tuple containing two numpy arrays, the first being the average
      rewards and the second being the standard deviation of rewards.
  """
  results_dir = os.path.join(f'results/{type}', project_name)
  config_file_path = os.path.join(results_dir, 'config.txt')
  # Placeholder, read instead from run0.txt
  if single_run:
    average_file_path = os.path.join(results_dir, 'run0.txt')
  else:
    average_file_path = os.path.join(results_dir, 'summary_average.txt')
  std_file_path = os.path.join(results_dir, 'summary_std.txt')

  # Read configuration
  with open(config_file_path, 'r') as file:
    for line in file:
      if 'num_runs' in line:
        num_runs = int(line.split(':')[-1])
      elif 'num_generations' in line:
        num_generations = int(line.split(':')[-1])
      elif 'N' in line:
        N = int(line.split(':')[-1])
      elif 'sigma' in line:
        sigma = float(line.split(':')[-1])
      elif 'num_episodes' in line:
        num_episodes = int(line.split(':')[-1])
      elif 'keep_previous_best' in line:
        keep_previous_best = bool(line.split(':')[-1])
      elif 'k' in line:
        k = int(line.split(':')[-1])
      elif 'max_steps' in line:
        max_steps = int(line.split(':')[-1])
      elif 'alpha' in line:
        alpha = float(line.split(':')[-1])



  # Read average rewards
  with open(average_file_path, 'r') as file:
    average_rewards = np.array([float(value) for value in file.read().split(',')])
  config = None
  if type == 'population':
    config = ('population', num_runs, num_generations, num_episodes, N, sigma, k, max_steps, keep_previous_best)
  elif type == 'zeroth':
    config = ('zeroth', num_runs, num_generations, num_episodes, sigma, alpha, max_steps)

  # # Read standard deviation of rewards
  if not single_run:
    with open(std_file_path, 'r') as file:
      std_rewards = np.array([float(value) for value in file.read().split(',')])
      return average_rewards, std_rewards, config
  # with open(std_file_path, 'r') as file:
  #   std_rewards = np.array([float(value) for value in file.read().split(',')])




  return average_rewards, config


# run_experiment('lunar_lander_tanh', num_runs, num_generations, num_episodes, N, sigma, k)
#
def generate_project_name(num_runs, num_generations, num_episodes, N, sigma, k, max_steps, keep_previous_best):
  title = f'll_{num_runs}_{num_generations}_{num_episodes}_{N}_{sigma}_{k}_{max_steps}_{keep_previous_best}'
  if keep_previous_best:
    title += '_kept'
  return title


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
  num_generations = 2000
  num_runs = 1
  max_steps = 500
  N = 10
  sigma = 0.5
  k = 1
  alpha = 0.001
  # run_population_experiment('lunar_lander_optimal', num_runs, num_generations, num_episodes, N,
  #                           sigma, k, max_steps)
  run_zeroth_order_experiment('lunar_lander_zeroth_order', num_runs, num_generations, num_episodes, sigma, alpha, max_steps)
  # total_rewards, config = read_project('lunar_lander_zeroth_order_test4', type='population')
  # print(total_rewards)
  # plot_rewards(total_rewards, config)

  # run_population_experiment('lunar_lander_optimal_1eval_20_runs_test', num_runs, num_generations, num_episodes, N, sigma, k, max_steps)
  # total_rewards, config = read_project(
  #   'lunar_lander_optimal_1eval_20_runs')
