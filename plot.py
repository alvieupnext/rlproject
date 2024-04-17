import matplotlib.pyplot as plt
import seaborn as sns

def plot_rewards(average_total_rewards, config, std_rewards=None):
  #Get the first element of the config file
  type = config[0]
  if type == 'population':
    num_runs, num_generations, num_episodes, N, sigma, k, max_steps, keep_previous_best = config[1:]
  else:
    num_runs, num_generations, num_episodes, sigma, alpha, max_steps = config[1:]
  sns.set(style='whitegrid')  # Set a style to make the plot look nicer

  plt.figure(figsize=(10, 6))  # Set the figure size for better readability

  plt.plot(average_total_rewards, color='steelblue', linewidth=2, linestyle='-', markersize=8,
           markerfacecolor='gold', markeredgewidth=2, markeredgecolor='navy')  # Customize line and marker
  if std_rewards is not None:
    #Plot the 95% confidence interval around the average
    upper_bound = average_total_rewards + 1.96 * std_rewards / (num_runs ** 0.5)
    lower_bound = average_total_rewards - 1.96 * std_rewards / (num_runs ** 0.5)
    plt.fill_between(range(len(average_total_rewards)), lower_bound, upper_bound, color='lightsteelblue', alpha=0.5)

  # plt.fill_between(
    #   range(len(average_total_rewards)),
    #   average_total_rewards - std_rewards,
    #   average_total_rewards + std_rewards,
    #   color='lightsteelblue', alpha=0.5)  # Add a shaded area to show the standard deviation


  plt.xlabel('Generation', fontsize=14, fontweight='bold', color='navy')  # Customize the x-label
  plt.ylabel('Reward', fontsize=14, fontweight='bold', color='navy')  # Customize the y-label
  run_text = f'{num_runs} runs' if num_runs > 1 else f'{num_runs} run'
  episode_text = f'{num_episodes} episodes' if num_episodes > 1 else f'{num_episodes} episode'
  if type == 'population':
    title = f'Reward Curve ({run_text}, {episode_text}, {max_steps} max steps, {N} perturbations, σ: {sigma}, CI: 95%)'
  else:
    title = f'Reward Curve ({run_text}, {episode_text}, {max_steps} max steps, σ: {sigma}, α: {alpha}, CI: 95%)'
  plt.title(
    title,
    fontsize=16, fontweight='bold', color='darkred')


  plt.xticks(fontsize=12)  # Customize the x-ticks
  plt.yticks(fontsize=12)  # Customize the y-ticks

  plt.grid(True, which='both', linestyle='--', linewidth=0.5)  # Add gridlines for better readability

  plt.tight_layout()  # Adjust layout to not cut off labels

  # Save the plot
  plt.savefig(
    f'{title}.pdf',
    format='pdf', dpi=300)  # Save the plot to a file

  plt.show()

def plot_reward_curves(rewards, configs, std_rewards=None):
  # Make a figure and a line per configuration
  sns.set(style='whitegrid')  # Set a style to make the plot look nicer

  plt.figure(figsize=(10, 6))  # Set the figure size for better readability

  #From the first config, get the number of runs, generation count and the number of episodes
  config = configs[0]
  num_runs, num_generations, num_episodes = config[1:4]
  run_text = f'{num_runs} runs' if num_runs > 1 else f'{num_runs} run'
  episode_text = f'{num_episodes} evaluation episodes' if num_episodes > 1 else f'{num_episodes} evaluation episode'

  plt.title(f'Reward Curves ({run_text}, {episode_text})', fontsize=16, fontweight='bold', color='darkred')

  for i, (average_total_rewards, config) in enumerate(zip(rewards, configs)):
    #Get the first element of the config file
    type = config[0]
    if type == 'population':
      num_runs, num_generations, num_episodes, N, sigma, k, max_steps, keep_previous_best = config[1:]
      configuration_name = f'Population (N: {N}, σ: {sigma}, k: {k}, max steps: {max_steps})'
    else:
      num_runs, num_generations, num_episodes, sigma, alpha, max_steps = config[1:]
      configuration_name = f'Zeroth Order (σ: {sigma}, α: {alpha}, max steps: {max_steps})'
    # Plot the average rewards
    plt.plot(average_total_rewards, label=configuration_name)
    if std_rewards is not None:
      #Plot the 95% confidence interval around the average
      upper_bound = average_total_rewards + 1.96 * std_rewards[i] / (num_runs ** 0.5)
      lower_bound = average_total_rewards - 1.96 * std_rewards[i] / (num_runs ** 0.5)
      plt.fill_between(range(len(average_total_rewards)), lower_bound, upper_bound, alpha=0.5)

  plt.legend(fontsize=12)  # Add a legend to the plot

  plt.xlabel('Generation', fontsize=14, fontweight='bold', color='navy')  # Customize the x-label
  plt.ylabel('Reward', fontsize=14, fontweight='bold', color='navy')  # Customize the y-label
  plt.xticks(fontsize=12)  # Customize the x-ticks
  plt.yticks(fontsize=12)  # Customize the y-ticks

  plt.grid(True, which='both', linestyle='--', linewidth=0.5)  # Add gridlines for better readability

  plt.tight_layout()  # Adjust layout to not cut off labels
  plt.show()
