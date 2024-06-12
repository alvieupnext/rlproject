import os

import numpy as np

from plot import plot_rewards, plot_reward_curves
from population import read_project

if __name__ == '__main__':
  # population_rewards, population_std, _, population_config = read_project('lunar_lander_population_ray_2nd',
  #                                                                                type='population', single_run=False)
  #Import the second population rewards using lunar_lander_population_method_top2_weight_adjusted
  population_rewards, population_std, _, population_config = read_project('lunar_lander_population_method_top2_full_weight',
                                                                                 type='population', single_run=False)
  # population_rewards2, population_std2, _, population_config2 = read_project('lunar_lander_population_method_top2_weight_adjusted',
  #                                                                                type='population', single_run=False)
  # population_rewards3, population_std3, _, population_config3 = read_project('lunar_lander_population_method_top3_full_weight',
  #                                                                               type='population', single_run=False)
  population_rewards4, population_std4, _, population_config4 = read_project('lunar_lander_population_method_top3_weight_adjusted',
                                                                                type='population', single_run=False)
  # population_rewards5, population_std5, _, population_config5 = read_project('lunar_lander_population_ray_2nd',
  #                                                                                type='population', single_run=False)
  # population_rewards6, population_std6, _, population_config6 = read_project('lunar_lander_population_method_top10_weight_adjusted',
  #                                                                                type='population', single_run=False)
  zeroth_rewards, zeroth_std, _, zeroth_config = read_project('lunar_lander_zeroth_order_0.001_2500_2', type='zeroth',
                                                                   single_run=False)
  plot_reward_curves([population_rewards, population_rewards4, zeroth_rewards],
                [population_config, population_config4, zeroth_config],
                std_rewards=[population_std, population_std4, zeroth_std],
                rolling_window=60, ylim=-200, std=True)