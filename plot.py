import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def plot_rewards(average_total_rewards, config, std_rewards=None, rolling_window=1, individual_runs = None, std=True):
  if individual_runs is None:
    individual_runs = []
  #Get the first element of the config file
  type = config[0]
  if type == 'population':
    num_runs, num_generations, num_episodes, N, sigma, k, max_steps, keep_previous_best = config[1:]
  else:
    num_runs, num_generations, num_episodes, sigma, alpha, max_steps = config[1:]
  sns.set(style='whitegrid')  # Set a style to make the plot look nicer

  plt.figure(figsize=(10, 6))  # Set the figure size for better readability

  # Plot the average reward
  average_rewards = average_total_rewards
  if rolling_window > 1:
    average_rewards = np.convolve(average_total_rewards, np.ones(rolling_window)/rolling_window, mode='same')

  plt.plot(average_rewards, color='steelblue', linewidth=2, linestyle='-', markersize=8,
           markerfacecolor='gold', markeredgewidth=2, markeredgecolor='navy')  # Customize line and marker
  if std_rewards is not None and std:
    #Plot the 95% confidence interval around the average
    upper_bound = average_total_rewards + 1.96 * std_rewards / (num_runs ** 0.5)
    lower_bound = average_total_rewards - 1.96 * std_rewards / (num_runs ** 0.5)
    upper_bound = np.convolve(upper_bound, np.ones(rolling_window)/rolling_window, mode='same')
    lower_bound = np.convolve(lower_bound, np.ones(rolling_window)/rolling_window, mode='same')
    plt.fill_between(range(len(average_total_rewards)), lower_bound, upper_bound, color='lightsteelblue', alpha=0.5)

  # Plot individual runs
  for run in individual_runs:
    run = np.convolve(run, np.ones(rolling_window)/rolling_window, mode='same')
    plt.plot(run, color='lightgray', linewidth=1, linestyle='-', alpha=0.2)

  # plt.fill_between(
    #   range(len(average_total_rewards)),
    #   average_total_rewards - std_rewards,
    #   average_total_rewards + std_rewards,
    #   color='lightsteelblue', alpha=0.5)  # Add a shaded area to show the standard deviation


  plt.xlabel('Evaluation Episode', fontsize=14, fontweight='bold', color='navy')  # Customize the x-label
  plt.ylabel('Reward', fontsize=14, fontweight='bold', color='navy')  # Customize the y-label
  run_text = f'{num_runs} runs' if num_runs > 1 else f'{num_runs} run'
  episode_text = f'{num_episodes} episodes' if num_episodes > 1 else f'{num_episodes} episode'
  generation_text = f'{num_generations} generations' if num_generations > 1 else f'{num_generations} generation'
  if type == 'population':
    title = f'Population Curve ({run_text}, {generation_text}, {episode_text}, {N} perturbations, CI: 95%)'
  else:
    title = f'Zeroth Order Curve ({run_text}, {generation_text}, {episode_text}, α: {alpha}, CI: 95%)'
  plt.title(
    title,
    fontsize=16, fontweight='bold', color='darkred')


  plt.xticks(fontsize=12)  # Customize the x-ticks
  plt.yticks(fontsize=12)  # Customize the y-ticks

  plt.grid(True, which='both', linestyle='--', linewidth=0.5)  # Add gridlines for better readability

  plt.tight_layout()  # Adjust layout to not cut off labels

  # Change the title to be a filename
  title = title.replace(' ', '_').replace('(', '').replace(')', '').replace(',', '').replace(':', '')

  # Save the plot
  plt.savefig(
    f'{title}.pdf',
    format='pdf', dpi=300)  # Save the plot to a file

  plt.show()

def plot_reward_curves(rewards, configs, std_rewards=None, std=True, ep_saved=True, rolling_window=1):
  # Make a figure and a line per configuration
  sns.set(style='whitegrid')  # Set a style to make the plot look nicer

  plt.figure(figsize=(10, 6))  # Set the figure size for better readability

  #From the first config, get the number of runs, generation count and the number of episodes
  config = configs[0]
  num_runs = config[1]
  run_text = f'{num_runs} runs' if num_runs > 1 else f'{num_runs} run'
  title = f'Reward Curves over ' + run_text
  # If std happens to be false, then add (No CI) to the title
  if not std:
    title += ' (No CI)'
  else:
    title += ' (CI: 95%)'

  plt.title(title, fontsize=16, fontweight='bold', color='darkred')

  for i, (average_total_rewards, config) in enumerate(zip(rewards, configs)):
    #Get the first element of the config file
    type = config[0]
    if type == 'population':
      num_runs, num_generations, num_episodes, N, sigma, k, max_steps, keep_previous_best = config[1:]
      episode_text = f'{num_episodes} evaluation episodes' if num_episodes > 1 else f'{num_episodes} evaluation episode'
      generation_text = f'{num_generations} generations' if num_generations > 1 else f'{num_generations} generation'
      configuration_name = f'Population ({generation_text}, {episode_text}, N: {N}, σ: {sigma}, top-k: {k}, max steps: {max_steps})'
    else:
      num_runs, num_generations, num_episodes, sigma, alpha, max_steps = config[1:]
      episode_text = f'{num_episodes} evaluation episodes' if num_episodes > 1 else f'{num_episodes} evaluation episode'
      generation_text = f'{num_generations} generations' if num_generations > 1 else f'{num_generations} generation'
      configuration_name = f'Zeroth Order ({generation_text}, {episode_text}, σ: {sigma}, α: {alpha}, max steps: {max_steps})'
    # Plot the average rewards
    average_rewards = np.convolve(average_total_rewards, np.ones(rolling_window)/rolling_window, mode='same')
    plt.plot(average_rewards, label=configuration_name)
    if std_rewards is not None and std:
      #Plot the 95% confidence interval around the average
      upper_bound = average_total_rewards + 1.96 * std_rewards[i] / (num_runs ** 0.5)
      lower_bound = average_total_rewards - 1.96 * std_rewards[i] / (num_runs ** 0.5)
      upper_bound = np.convolve(upper_bound, np.ones(rolling_window)/rolling_window, mode='same')
      lower_bound = np.convolve(lower_bound, np.ones(rolling_window)/rolling_window, mode='same')
      plt.fill_between(range(len(average_total_rewards)), lower_bound, upper_bound, alpha=0.5)

  plt.legend(fontsize=12)  # Add a legend to the plot

  xlabel = 'Evaluation Episode' if ep_saved else 'Generation'

  plt.xlabel(xlabel, fontsize=14, fontweight='bold', color='navy')  # Customize the x-label
  plt.ylabel('Reward', fontsize=14, fontweight='bold', color='navy')  # Customize the y-label
  plt.xticks(fontsize=12)  # Customize the x-ticks
  plt.yticks(fontsize=12)  # Customize the y-ticks

  plt.grid(True, which='both', linestyle='--', linewidth=0.5)  # Add gridlines for better readability

  plt.tight_layout()  # Adjust layout to not cut off labels

  title = title.replace(' ', '_').replace('(', '').replace(')', '').replace(',', '').replace(':', '')

  plt.savefig(
    f'{title}.pdf',
    format='pdf', dpi=300)  # Save the plot to a file
  plt.show()

#Plot for every reward run, a boxplot of the rewards
def plot_boxplot(rewards, config):
  # Get the first element of the config file
  type = config[0]
  if type == 'population':
    num_runs, num_generations, num_episodes, N, sigma, k, max_steps, keep_previous_best = config[1:]
  else:
    num_runs, num_generations, num_episodes, sigma, alpha, max_steps = config[1:]
  sns.set(style='whitegrid')  # Set a style to make the plot look nicer

  plt.figure(figsize=(10, 6))  # Set the figure size for better readability

  #Per rewards, make it a boxplot
  sns.boxplot(data=rewards, showfliers=False, palette='Set3')
  #Change the xlabel of each boxplot to say Run i
  plt.xlabel('Runs', fontsize=14, fontweight='bold', color='navy')  # Customize the x-label
  plt.ylabel('Reward', fontsize=14, fontweight='bold', color='navy')  # Customize the y-label
  episode_text = f'{num_episodes} evaluation episodes' if num_episodes > 1 else f'{num_episodes} evaluation episode'
  generation_text = f'{num_generations} generations' if num_generations > 1 else f'{num_generations} generation'
  if type == 'population':
    title = f'Population Method Per Run ({generation_text}, {episode_text}, {N} perturbations)'
  else:
    title = f'Zeroth Order Per Run ({generation_text}, {episode_text}, α: {alpha})'
  plt.title(
    title,
    fontsize=16, fontweight='bold', color='darkred')
  plt.xticks(fontsize=12)  # Customize the x-ticks
  plt.yticks(fontsize=12)  # Customize the y-ticks

  plt.grid(True, which='both', linestyle='--', linewidth=0.5)  # Add gridlines for better readability

  plt.tight_layout()  # Adjust layout to not cut off labels

  # Change the title to be a filename
  title = title.replace(' ', '_').replace('(', '').replace(')', '').replace(',', '').replace(':', '')

  # Save the plot
  plt.savefig(
    f'{title}.pdf',
    format='pdf', dpi=300)  # Save the plot to a file

  plt.show()

