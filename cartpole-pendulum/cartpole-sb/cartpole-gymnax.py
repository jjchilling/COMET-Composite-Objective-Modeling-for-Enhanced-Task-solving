import os
import gymnax as gym
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import jax

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnvWrapper
from stable_baselines3.common.utils import set_random_seed

class MyGymnaxVecEnvWrapper(VecEnvWrapper):
    def __init__(self, venv: DummyVecEnv):
        super(MyGymnaxVecEnvWrapper, self).__init__(venv)
    def reset(self):
        return np.array([env.reset() for env in self.venv.envs])
    def step_async(self, actions):
        for env, action in zip(self.venv.envs, actions):
            env.step_async(action)
    def step_wait(self):
        results = [env.step_wait() for env in self.venv.envs]
        obs, rewards, dones, infos = map(np.array, zip(*results))
        return obs, rewards, dones, infos

# Function to plot training progress
def plot_training_progress(log_dir):
    results_path = os.path.join(log_dir, 'progress.csv')
    if os.path.exists(results_path):
        results = pd.read_csv(results_path)
        plt.plot(results['timesteps'], results['r'])
        plt.xlabel('Timesteps')
        plt.ylabel('Reward')
        plt.title('Training Progress')
        plt.show()
    else:
        print(f"No training results found at: {results_path}")

def make_env(env_id: str, rank: int, seed: int = 0):
    """
    Utility function for multiprocessed env.

    :param env_id: the environment ID
    :param rank: index of the subprocess
    :param seed: the initial seed for RNG
    """
    def _init():
        rng = jax.random.PRNGKey(0)
        rng, key_reset, key_act, key_step = jax.random.split(rng, 4)        
        env, env_params = gym.make("CartPole-v1")
        obs, state = env.reset(key_reset, env_params)
        # # env.seed(seed + rank)
        # obs, info = env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init

if __name__ == "__main__":
    env_id = "CartPole-v1"
    num_cpu = 4  # Number of processes to use

    # Create the Gymnax environments
    envs = [make_env(env_id, i) for i in range(num_cpu)]
    # Convert Gymnax environments to VecEnvWrapper
    vec_env = MyGymnaxVecEnvWrapper(DummyVecEnv(envs))

    print("post, before PPO")

    model = PPO("MlpPolicy", vec_env, verbose=1)

    log_dir = "ppo_cartpole_logs"
    os.makedirs(log_dir, exist_ok=True)  # Create log directory if it doesn't exist

    model.learn(total_timesteps=25000, callback=lambda locals, globals: locals['self'].logger.dump(log_dir))  # Increased total_timesteps and manually log progress

    print("post PPO")

    # Plot training progress
    plot_training_progress(log_dir)

    # Close the environments
    for env in envs:
        env.close()
