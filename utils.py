#Functions that are shared between the different scripts
from elegantrl.train.run import *

hidden_size = 128
def evaluate_policy(env, policy, num_episodes, max_steps):
  policy_reward = []
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
    policy_reward.append(episode_reward)
    # # Log the episode's results
    # print(f'Episode {episode}: Total Reward: {episode_reward}')
  # Return the array of policy rewards
  return policy_reward

def read_project(project_name, single_run=1, type='population', amount_of_runs=10):
  """
  Reads and returns the average and standard deviation of rewards
  for a given project.

  Args:
      project_name (str): The name of the project.

  Returns:
      tuple: A tuple containing two numpy arrays, the first being the average
      rewards and the second being the standard deviation of rewards.
  """
  results_dir = os.path.join('results', type, project_name)
  print(results_dir)
  config_file_path = os.path.join(results_dir, 'config.txt')
  run_paths = []
  # Placeholder, read instead from run0.txt
  if single_run:
    average_file_path = os.path.join(results_dir, f'run{single_run-1}.txt')
  else:
    average_file_path = os.path.join(results_dir, 'summary_average.txt')
    for run in range(amount_of_runs):
      run_file_path = os.path.join(results_dir, f'run{run}.txt')
      run_paths.append(run_file_path)
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
    average_rewards = np.array([float(value) for value in file.read().split(',') if value])
  config = None
  if type == 'population':
    config = ('population', num_runs, num_generations, num_episodes, N, sigma, k, max_steps, keep_previous_best)
  elif type == 'zeroth':
    config = ('zeroth', num_runs, num_generations, num_episodes, sigma, alpha, max_steps)

  # # Read standard deviation of rewards
  if not single_run:
    with open(std_file_path, 'r') as file:
      std_rewards = np.array([float(value) for value in file.read().split(',')])
    #For every run_path, read the rewards and append them to the rewards array
    rewards = []
    for run_path in run_paths:
      with open(run_path, 'r') as file:
        rewards.append(np.array([float(value) for value in file.read().split(',') if value]))
    return average_rewards, std_rewards, rewards, config
  # with open(std_file_path, 'r') as file:
  #   std_rewards = np.array([float(value) for value in file.read().split(',')])




  return average_rewards, config

#Generate the average reward and std reward files from a project
def generate_summary(project_name, type='zeroth'):
    # Locate the project
    results_dir = os.path.join(f'results/{type}', project_name)
    # Get all files that start with run and are a txt file
    run_files = [file for file in os.listdir(results_dir) if file.startswith('run') and file.endswith('.txt')]
    # Filter all files that have the word index in the name of the file
    run_files = [file for file in run_files if 'index' not in file]
    # From all run_files, get their rewards (delimited by commas)
    rewards = []
    for run_file in run_files:
        with open(os.path.join(results_dir, run_file), 'r') as f:
            content = f.read()
            # Split by comma and filter out empty strings
            reward_values = [reward for reward in content.split(',') if reward.strip()]
            rewards.append(np.array([float(reward) for reward in reward_values]))
    # Get the average and std of the rewards
    average_rewards = np.mean(rewards, axis=0)
    std_rewards = np.std(rewards, axis=0)
    # Write the average and std to the summary files
    with open(os.path.join(results_dir, 'summary_average.txt'), 'w') as f:
        f.write(','.join(map(str, average_rewards)))
    with open(os.path.join(results_dir, 'summary_std.txt'), 'w') as f:
        f.write(','.join(map(str, std_rewards)))









# run_experiment('lunar_lander_tanh', num_runs, num_generations, num_episodes, N, sigma, k)
#
def generate_project_name(config):
  type = config[0]
  if type == 'population':
    _, num_runs, num_generations, num_episodes, N, sigma, k, max_steps, keep_previous_best = config
    title = f'll_{type}_{num_runs}_{num_generations}_{num_episodes}_{N}_{sigma}_{k}_{max_steps}'
    if keep_previous_best:
      title += '_kept'
    return title
  else:
    _, num_runs, num_generations, num_episodes, sigma, alpha, max_steps = config
    title = f'll_{type}_{num_runs}_{num_generations}_{num_episodes}_{sigma}_{alpha}_{max_steps}'
    return title