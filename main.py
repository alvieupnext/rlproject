import os

import numpy as np

from plot import plot_rewards
from population import read_project

if __name__ == 'main':
  # Get all folders in the results directory, that start with ll
  project_names = [folder for folder in os.listdir('results/population') if folder.startswith('ll')]
  # Get the average rewards and standard deviations for each project
  for project_name in project_names:
    average_rewards, config = read_project(project_name)
    print(f'Project: {project_name}')
    type, num_runs, num_generations, num_episodes, N, sigma, k, max_steps, keep_previous_best = config
    # Take the average of the rewards
    average = np.mean(average_rewards)
    print(f'Project: {project_name}, Average Reward: {average}')
    # #Get the average of the second half of the rewards
    # average = np.mean(average_rewards[len(average_rewards)//2:])
    # print(f'Project: {project_name}, Second Half Average Reward: {average}')
    # #Get the average of the last 100 rewards
    # average = np.mean(average_rewards[-100:])
    # print(f'Project: {project_name}, Last 100 Average Reward: {average}')
    #If the overall average is lower than 40
    if max_steps > 5001:
      continue
    # print(f'Project: {project_name}')
    # print(f'Average Rewards: {average_rewards}')
    # print(f'Configuration: {config}')
    plot_rewards(average_rewards, config)