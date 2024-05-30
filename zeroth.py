import gym
from elegantrl.train.run import *

from plot import plot_rewards, plot_reward_curves
from policy import ParametricPolicy, AffineThrottlePolicy
from pertubation import *
from utils import evaluate_policy, generate_summary, read_project
import ray

gym.logger.set_level(40)  # Block warning

# seed = 42
#
# # Seed the environment for reproducibility
# env.seed(seed)
# torch.manual_seed(seed)
# np.random.seed(seed)

state_dim = 8
action_dim = 2
gamma = 0.99

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
  remotes = []
  for run in range(num_runs):
    remotes.append(
      zeroth_experiment_run.remote(results_dir, run, num_generations, num_episodes, sigma, alpha, max_steps))

  ray.get(remotes)

@ray.remote
def zeroth_experiment_run(results_dir, run, num_generations, num_episodes, sigma, alpha, max_steps):
  # Initialize the environment using the provided utility functions and arguments
  env = gym.make('LunarLanderContinuous-v2')
  policy = AffineThrottlePolicy(input_size=state_dim, hidden_size=128, output_size=action_dim)
  run_rewards = []
  # Open the file in 'w' mode to clear it, then immediately close it.
  with open(os.path.join(results_dir, f'run{run}.txt'), 'w') as f:
    pass  # This will clear the file contents at the beginning of the run
  policy_reward = evaluate_policy(env, policy, num_episodes, max_steps)
  print(f'Generation 0: Reward: {policy_reward}')
  run_rewards.append(policy_reward)

  for gen in range(num_generations):
    pertubations = perturb_policy_zeroth_order(policy, sigma)
    p_plus, p_min = pertubations
    rewards = [evaluate_policy(env, policy, num_episodes, max_steps) for policy in pertubations]
    # Rewards contains two arrays, one for the positive perturbation and one for the negative perturbation
    pos_rewards, neg_rewards = rewards
    # The rewards are arrays of the evaluation episode rewards
    mean_rewards = np.mean(rewards, axis=1)
    p_plus_score, p_min_score = mean_rewards
    # A sort of gradient of θ, that we did not have to compute, is now given by “0.5 × (score of θ+ -
    # score of θ-) × θ+”.
    # Calculate this gradient, we already apply alpha here
    gradient = alpha * (0.5 * (p_plus_score - p_min_score))
    # Move the positive perturbation in the direction of the gradient
    for param in p_plus.parameters():
      param.data *= gradient
    # Move θ, the parameters of the policy, in the direction of the gradient.
    # The step size is α, and the direction is the gradient.
    policy.add_policy(p_plus)
    # Evaluate the new policy (OPTIONAL)
    policy_reward = evaluate_policy(env, policy, num_episodes, max_steps)
    print(f'Generation {gen + 1}: Reward: {policy_reward}')
    run_rewards.append(policy_reward)
    # # Continue using append mode for subsequent writes within the same run
    # with open(os.path.join(results_dir, f'run{run}.txt'), 'a') as f:
    #   for episode_reward in policy_reward:
    #     f.write(f',{episode_reward}')
    # Write every positive and negative reward to the file
    with open(os.path.join(results_dir, f'run{run}.txt'), 'a') as f:
      for episode_reward in pos_rewards + neg_rewards:
        f.write(f',{episode_reward}')

if __name__ == '__main__':
  # Train the policy
  num_episodes = 40
  num_generations = 1875
  # num_generations = 10
  num_runs = 10
  max_steps = 500
  sigma = 0.5
  alpha = 0.001
  experiment = 'lunar_lander_zeroth_order_ray'
  #Run the zeroth order experiment
  run_zeroth_order_experiment(experiment, num_runs, num_generations, num_episodes, sigma, alpha, max_steps)
  # generate_summary(experiment)
  # total_rewards, std_rewards, config = read_project(experiment, type='zeroth', single_run=False)
  # plot_rewards(total_rewards, config)
