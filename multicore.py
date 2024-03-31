from population import run_population_experiment, generate_project_name, run_zeroth_order_experiment
import ray

# The various hyperparameters
max_step_configs = [100, 500, 1000]
N_configs = [10]
k_s = [1]
generation_counts = [2000]
generation_count = 2000
sigmas = [0.5, 0.1, 0.8]
episode_counts = [20]
episode_count = 20
kept_previous = True
alphas = [0.1, 0.01, 0.001]
num_runs = 1

#Make a function that returns a ray remote, it should just call run experiment with a config
@ray.remote
def run_experiment_ray(config):
    type = config[0]
    if type == 'zeroth':
        return run_zeroth_order_experiment(*config[1:])
    else:
        return run_population_experiment(*config[1:])

#Make a function that will generate all the configs
def generate_population_configs():
    configs = []
    for max_steps in max_step_configs:
        for N in N_configs:
            for k in k_s:
                for generation_count in generation_counts:
                    for sigma in sigmas:
                        for episode_count in episode_counts:
                            config = ('population', num_runs, generation_count, episode_count, N, sigma, k, max_steps, kept_previous)
                            #Generate a project name
                            project_name = generate_project_name(config)
                             #Append the config to the list
                            configs.append(('population', project_name, num_runs, generation_count, episode_count, N,
                                            sigma, k, max_steps, kept_previous))
    return configs

def generate_zeroth_configs():
    configs = []
    for sigma in sigmas:
        for max_step in max_step_configs:
            for alpha in alphas:
                config = ('zeroth', num_runs, generation_count, episode_count, sigma, alpha, max_step)
                project_name = generate_project_name(config)
                configs.append(('zeroth', project_name, num_runs, generation_count, episode_count, sigma, alpha, max_step))
    return configs


#Initialize ray and add the remotes
if __name__ == '__main__':
    # ray.init()
    ray.init(num_cpus=20)
    configs = generate_zeroth_configs()
    futures = [run_experiment_ray.remote(config) for config in configs]
    ray.get(futures)
    ray.shutdown()