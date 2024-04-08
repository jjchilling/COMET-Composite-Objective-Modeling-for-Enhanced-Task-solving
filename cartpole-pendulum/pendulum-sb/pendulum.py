import os
import gym
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.results_plotter import plot_results

# Create a vectorized environment
vec_env = make_vec_env("Pendulum-v1", n_envs=4, seed=0)

# We collect 4 transitions per call to `env.step()` and perform 2 gradient steps per call to `env.step()`
# If gradient_steps=-1, then we would do 4 gradient steps per call to `env.step()`
model = SAC("MlpPolicy", vec_env, train_freq=1, gradient_steps=2, verbose=1)

# Train the model
model.learn(total_timesteps=10_000)

# Manually load the training results
log_dir = "sac_pendulum_logs"
results_path = os.path.join(log_dir, 'progress.csv')
if os.path.exists(results_path):
    results = pd.read_csv(results_path)
else:
    results = pd.DataFrame(columns=['timesteps', 'r'])  # Create an empty DataFrame

# Visualize the trained agent
obs = vec_env.reset()
for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render()

# Plot rewards per timestep
plt.plot(results['timesteps'], results['r'])
plt.xlabel('Timesteps')
plt.ylabel('Reward')
plt.title('Training Progress')
plt.show()

# Close the environment
vec_env.close()
