import os

import numpy as np

from plot import plot_rewards, plot_reward_curves, plot_boxplot
from population import read_project

if __name__ == '__main__':
  population_average_rewards, population_std, population_rewards, population_config = read_project('lunar_lander_population_method',
                                                                                  type='population', single_run=False)
  zeroth_average_rewards, zeroth_std, zeroth_rewards, zeroth_config = read_project('lunar_lander_zeroth_order', type='zeroth',
                                                                      single_run=False)
  plot_rewards(population_average_rewards, population_config, population_std,
               rolling_window=10, std=True)
  plot_boxplot(population_rewards, population_config)
  plot_rewards(zeroth_average_rewards, zeroth_config, zeroth_std,
               rolling_window=10, std=True)
  plot_boxplot(zeroth_rewards, zeroth_config)
  plot_reward_curves([population_average_rewards, zeroth_average_rewards],
                     [population_config, zeroth_config],
                     std_rewards=[population_std, zeroth_std],
                     rolling_window=60, ylim=-200, std=True)