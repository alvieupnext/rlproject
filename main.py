import os

import numpy as np

from plot import plot_rewards, plot_reward_curves
from population import read_project

if __name__ == '__main__':
  population_rewards, population_std, _, population_config = read_project('lunar_lander_population_ray_2nd',
                                                                               type='population', single_run=False)
  print(len(population_rewards))
  zeroth_rewards, zeroth_std, _, zeroth_config = read_project('lunar_lander_zeroth_order_ray_2nd', type='zeroth',
                                                                   single_run=False)
  print(len(zeroth_rewards))
  # print(total_rewards)
  plot_reward_curves([population_rewards, zeroth_rewards],
                     [population_config, zeroth_config],
                     std_rewards=[population_std, zeroth_std],
                     rolling_window=10,)

  # # Get all folders in the results directory, that start with ll
  # population_project_names = [folder for folder in os.listdir('results/population') if folder.startswith('ll')]
  # zeroth_project_names = [folder for folder in os.listdir('results/zeroth') if folder.startswith('l')]
  # best_population_project = None
  # best_population_average = -1
  # best_population_rewards = None
  # best_zeroth_project = None
  # best_zeroth_average = -1
  # best_zeroth_rewards = None
  # # Get the average rewards and standard deviations for each project
  # for project_name in population_project_names:
  #   average_rewards, config = read_project(project_name, type='population')
  #   # print(f'Project: {project_name}')
  #   # type, num_runs, num_generations, num_episodes, N, sigma, k, max_steps, keep_previous_best = config
  #   # Take the average of the rewards
  #   average = np.mean(average_rewards)
  #
  #   # #Get the average of the second half of the rewards
  #   # average = np.mean(average_rewards[len(average_rewards)//2:])
  #   # print(f'Project: {project_name}, Second Half Average Reward: {average}')
  #   # #Get the average of the last 100 rewards
  #   # average = np.mean(average_rewards[-100:])
  #   # print(f'Project: {project_name}, Last 100 Average Reward: {average}')
  #   #If the overall average is lower than 40
  #   if average > best_population_average:
  #     best_population_average = average
  #     best_population_rewards = average_rewards
  #     best_population_project = config
  # for project_name in zeroth_project_names:
  #   average_rewards, config = read_project(project_name, type='zeroth')
  #   # print(f'Project: {project_name}')
  #   average = np.mean(average_rewards)
  #   if average > best_zeroth_average:
  #     best_zeroth_average = average
  #     best_zeroth_rewards = average_rewards
  #     best_zeroth_project = config
  # print(f'Best population project: {best_population_project}')
  # print(f'Best zeroth project: {best_zeroth_project}')
  # rewards = [best_zeroth_rewards, best_population_rewards]
  # configs = [best_zeroth_project, best_population_project]
  # plot_reward_curves(rewards, configs)

    # print(f'Project: {project_name}, Average Reward: {average}')
    # print(f'Project: {project_name}')
    # print(f'Average Rewards: {average_rewards}')
    # print(f'Configuration: {config}')
    # plot_rewards(average_rewards, config)