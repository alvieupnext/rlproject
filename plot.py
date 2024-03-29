import matplotlib.pyplot as plt
import seaborn as sns

def plot_rewards(project_name, average_total_rewards, sigma, N, num_generations, num_episodes):
  sns.set(style='whitegrid')  # Set a style to make the plot look nicer

  plt.figure(figsize=(10, 6))  # Set the figure size for better readability

  plt.plot(average_total_rewards, color='steelblue', linewidth=2, linestyle='-', markersize=8,
           markerfacecolor='gold', markeredgewidth=2, markeredgecolor='navy')  # Customize line and marker

  plt.xlabel('Generation', fontsize=14, fontweight='bold', color='navy')  # Customize the x-label
  plt.ylabel('Reward', fontsize=14, fontweight='bold', color='navy')  # Customize the y-label
  plt.title(
    f'Reward over Generations (sigma: {sigma}, {N} perturbations, {num_generations} generations, {num_episodes} episodes)',
    fontsize=16, fontweight='bold', color='darkred')
  plt.suptitle(project_name, fontsize=16, fontweight='bold', color='darkred')


  plt.xticks(fontsize=12)  # Customize the x-ticks
  plt.yticks(fontsize=12)  # Customize the y-ticks

  plt.grid(True, which='both', linestyle='--', linewidth=0.5)  # Add gridlines for better readability

  plt.tight_layout()  # Adjust layout to not cut off labels

  # Save the plot
  plt.savefig(
    f'Reward over Generations (sigma: {sigma}, {N} perturbations, {num_generations} generations, {num_episodes} episodes).pdf',
    format='pdf', dpi=300)  # Save the plot to a file

  plt.show()