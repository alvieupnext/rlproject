import matplotlib.pyplot as plt
import seaborn as sns

def plot_rewards(average_total_rewards, config):
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

  plt.xlabel('Generation', fontsize=14, fontweight='bold', color='navy')  # Customize the x-label
  plt.ylabel('Reward', fontsize=14, fontweight='bold', color='navy')  # Customize the y-label
  episode_text = f'{num_episodes} episodes' if num_episodes > 1 else f'{num_episodes} episode'
  if type == 'population':
    title = f'Average Reward ({episode_text}, {max_steps} max steps, {N} perturbations, σ: {sigma}, top-{k})'
  else:
    title = f'Average Reward ({episode_text}, {max_steps} max steps, σ: {sigma}, α: {alpha})'
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