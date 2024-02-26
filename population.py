import elegantrl
# from elegantrl.run import *
import gym
import gym.envs.box2d.lunar_lander as lunar_lander
from elegantrl.agents import AgentModSAC
from elegantrl.train.config import get_gym_env_args, Config
from elegantrl.train.run import *

from policy import ParametricPolicy
from pertubation import *

gym.logger.set_level(40) # Block warning

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

# Train the policy
num_episodes = 2
num_generations = 20
num_runs = 12
target_step = 1000000000000
gamma = 0.99
N = 50
state_dim = 8
action_dim = 2

def evaluate_policy(policy):
  policy_reward = 0
  for episode in range(num_episodes):
    state = env.reset()
    episode_reward = 0

    for t in range(target_step):
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



total_rewards = []
# Start with a policy
# Initialize policy
for run in range(num_runs):
  policy = ParametricPolicy(input_size=state_dim,
                            hidden_size=128,
                            output_size=action_dim)
  run_rewards =[]
  #Evaluate the first policy
  policy_reward = evaluate_policy(policy)
  print(f'Generation {0}: Reward: {policy_reward}')
  #Add the reward to the list of rewards
  run_rewards.append(policy_reward)
  for gen in range(num_generations):
    # Get N amount of pertubations (remember that each N produces two pertubations)
    policies = generate_perturbed_policies(policy, N, sigma=0.01)
    #Add the original policy to the list of policies
    policies.append(policy)
    rewards = []

    #We evaluate each policy
    for perturbed_policy in policies:
      policy_reward = evaluate_policy(perturbed_policy)
      #Add the policy reward to the list of rewards
      rewards.append(policy_reward)
    #Select the best policy
    best_policy = policies[np.argmax(rewards)]
    #Select the max reward
    best_reward = np.max(rewards)
    #Log per generation the best policy (index) and best reward
    print(f'Generation {gen +1}: Best Reward: {best_reward}, Best Policy: {np.argmax(rewards)}')
    #Get the top 5 policies
    top_policies = np.argsort(rewards)[-5:]
    #Reverse the order to get the best policies first
    top_policies = top_policies[::-1]
    #Get the top 10 rewards
    top_rewards = [rewards[i] for i in top_policies]
    #Print the top 10 policies and rewards
    # print(f'Top 5 Policies: {top_policies}')
    # print(f'Top 5 Rewards: {top_rewards}')
    #Form the average of the the 10 best rewards (by averaging all the weights/parameters)
    average_policy = copy.deepcopy(best_policy)
    weight = 1
    for i in range(1, 3):
      for param, avg_param in zip(policies[top_policies[i]].parameters(), average_policy.parameters()):
        avg_param.data += param.data * (1/(2*(i+1)))
        weight += (1/(2**i))
    for param in average_policy.parameters():
      param.data /= weight
    #Update the policy
    policy = average_policy
    #Add the best reward to the list of rewards
    run_rewards.append(best_reward)
  total_rewards.append(run_rewards)

# Convert the rewards to a numpy array
total_rewards = np.array(total_rewards)
# Average the rewards over the number of runs
average_total_rewards = np.mean(total_rewards, axis=0)

# Plot the rewards
import matplotlib.pyplot as plt
import seaborn as sns
# plt.style.use('seaborn-darkgrid')  # Set a style to make the plot look nicer
sns.set(style='whitegrid')  # Set a style to make the plot look nicer

plt.figure(figsize=(10, 6))  # Set the figure size for better readability

plt.plot(average_total_rewards, color='steelblue', linewidth=2, linestyle='-', marker='o', markersize=8, markerfacecolor='gold', markeredgewidth=2, markeredgecolor='navy')  # Customize line and marker

plt.xlabel('Generation', fontsize=14, fontweight='bold', color='navy')  # Customize the x-label
plt.ylabel('Reward', fontsize=14, fontweight='bold', color='navy')  # Customize the y-label
plt.title('Reward over Generations', fontsize=16, fontweight='bold', color='darkred')  # Customize the title

plt.xticks(fontsize=12)  # Customize the x-ticks
plt.yticks(fontsize=12)  # Customize the y-ticks

plt.grid(True, which='both', linestyle='--', linewidth=0.5)  # Add gridlines for better readability

plt.tight_layout()  # Adjust layout to not cut off labels

plt.show()


# args = Config(AgentModSAC, env_class=env_func, env_args=env_args)
#
# args.target_step = args.max_step
# args.gamma = 0.99
# args.eval_times = 2**5
# args.random_seed = 2022
#
# train_agent(args)

