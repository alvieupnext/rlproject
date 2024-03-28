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



def evaluate_policy(policy, num_episodes):
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
      episode_reward += reward * (gamma ** t)

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


def run_experiment(project_name, num_runs, num_generations, num_episodes, N, sigma, k, max_steps, keep_previous_best=True):
  print(f'Running experiment for project {project_name}')
  print(f'Number of runs: {num_runs}')
  print(f'Number of generations: {num_generations}')
  print(f'Number of episodes: {num_episodes}')
  print(f'N: {N}')
  print(f'Sigma: {sigma}')
  print(f'Top k: {k}')
  print(f'Max steps: {max_steps}')
  print(f'Keep previous best: {keep_previous_best}')
  # Ensure results directory exists
  results_dir = os.path.join('results', project_name)
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

    policy_reward = evaluate_policy(policy, num_episodes)
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
      rewards = []

      for perturbed_policy in policies:
        policy_reward = evaluate_policy(perturbed_policy, num_episodes)
        rewards.append(policy_reward)

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


def read_project(project_name):
  """
  Reads and returns the average and standard deviation of rewards
  for a given project.

  Args:
      project_name (str): The name of the project.

  Returns:
      tuple: A tuple containing two numpy arrays, the first being the average
      rewards and the second being the standard deviation of rewards.
  """
  results_dir = os.path.join('results', project_name)
  config_file_path = os.path.join(results_dir, 'config.txt')
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
      elif 'k' in line:
        k = int(line.split(':')[-1])
      elif 'max_steps' in line:
        max_steps = int(line.split(':')[-1])
      elif 'keep_previous_best' in line:
        keep_previous_best = bool(line.split(':')[-1])


  # Read average rewards
  with open(average_file_path, 'r') as file:
    average_rewards = np.array([float(value) for value in file.read().split(',')])

  # Read standard deviation of rewards
  with open(std_file_path, 'r') as file:
    std_rewards = np.array([float(value) for value in file.read().split(',')])

  return average_rewards, std_rewards, (num_runs, num_generations, num_episodes, N, sigma, k, max_steps, keep_previous_best)


# Train the policy
num_episodes = 10
num_generations = 300
num_runs = 1
max_steps = 100_000
N = 20
sigma = 0.01
k = 1

# run_experiment('lunar_lander_tanh', num_runs, num_generations, num_episodes, N, sigma, k)
#
def generate_project_name(num_runs, num_generations, num_episodes, N, sigma, k, max_steps, keep_previous_best):
  title = f'll_{num_runs}_{num_generations}_{num_episodes}_{N}_{sigma}_{k}_{max_steps}_{keep_previous_best}'
  if keep_previous_best:
    title += '_kept'
  return title


if __name__ == '__main__':
  run_experiment('lunar_lander_tanh', num_runs, num_generations, num_episodes, N, sigma, k, max_steps)
  total_rewards, _, (num_runs, num_generations, num_episodes, N, sigma, k, max_steps, keep_previous_best) = read_project(
    'lunar_lander_tanh')
  plot_rewards(total_rewards, sigma, N, num_generations, num_episodes)
# total_rewards = []
# # Start with a policy
# # Initialize policy
# for run in range(num_runs):
#   policy = ParametricPolicy(input_size=state_dim,
#                             hidden_size=128,
#                             output_size=action_dim)
#   run_rewards =[]
#   #Evaluate the first policy
#   policy_reward = evaluate_policy(policy)
#   print(f'Generation {0}: Reward: {policy_reward}')
#   #Add the reward to the list of rewards
#   run_rewards.append(policy_reward)
#   for gen in range(num_generations):
#     # Get N amount of pertubations (remember that each N produces two pertubations)
#     policies = generate_perturbed_policies(policy, N, sigma=sigma)
#     #Add the original policy to the list of policies
#     policies.append(policy)
#     rewards = []
#
#     #We evaluate each policy
#     for perturbed_policy in policies:
#       policy_reward = evaluate_policy(perturbed_policy)
#       #Add the policy reward to the list of rewards
#       rewards.append(policy_reward)
#     #Select the best policy
#     best_policy = policies[np.argmax(rewards)]
#     #Select the max reward
#     best_reward = np.max(rewards)
#     #Log per generation the best policy (index) and best reward
#     print(f'Generation {gen +1}: Best Reward: {best_reward}, Best Policy: {np.argmax(rewards)}')
#     #Get the top 5 policies
#     top_policies = np.argsort(rewards)[-3:]
#     #Reverse the order to get the best policies first
#     top_policies = top_policies[::-1]
#     #Get the top 10 rewards
#     top_rewards = [rewards[i] for i in top_policies]
#     #Print the top 10 policies and rewards
#     print(f'Top Policies: {top_policies}')
#     print(f'Top Rewards: {top_rewards}')
#     #Form the average of the the 10 best rewards (by averaging all the weights/parameters)
#     average_policy = copy.deepcopy(best_policy)
#     weight = 1
#     for i in range(1, 3):
#       for param, avg_param in zip(policies[top_policies[i]].parameters(), average_policy.parameters()):
#         avg_param.data += param.data * (1/(2*(i+1)))
#         weight += (1/(2**i))
#     for param in average_policy.parameters():
#       param.data /= weight
#     #Update the policy
#     policy = average_policy
#     #Add the best reward to the list of rewards
#     run_rewards.append(best_reward)
#   total_rewards.append(run_rewards)
#
# # Convert the rewards to a numpy array
# total_rewards = np.array(total_rewards)
# # Average the rewards over the number of runs
# average_total_rewards = np.mean(total_rewards, axis=0)

# Plot the rewards

# args = Config(AgentModSAC, env_class=env_func, env_args=env_args)
#
# args.target_step = args.max_step
# args.gamma = 0.99
# args.eval_times = 2**5
# args.random_seed = 2022
#
# train_agent(args)
