from population import run_experiment, generate_project_name
import ray

# The various hyperparameters
max_step_configs = [10000, 50000, 100000]
N_configs = [5, 10, 20]
k_s = [1, 2, 3]
generation_counts = [500]
sigmas = [0.01]
episode_counts = [5, 10, 20]
kept_previous = True
num_runs = 1

#Make a function that returns a ray remote, it should just call run experiment with a config
@ray.remote
def run_experiment_ray(config):
    return run_experiment(*config)

#Make a function that will generate all the configs
def generate_configs():
    configs = []
    for max_steps in max_step_configs:
        for N in N_configs:
            for k in k_s:
                for generation_count in generation_counts:
                    for sigma in sigmas:
                        for episode_count in episode_counts:
                            #Generate a project name
                            project_name = generate_project_name(num_runs, generation_count, episode_count,
                                                                 N, sigma, k, max_steps, kept_previous)
                             #Append the config to the list
                            configs.append((project_name, num_runs, generation_count, episode_count, N,
                                            sigma, k, max_steps, kept_previous))
    return configs

#Initialize ray and add the remotes
if __name__ == '__main__':
    ray.init()
    configs = generate_configs()
    futures = [run_experiment_ray.remote(config) for config in configs]
    ray.get(futures)
    ray.shutdown()