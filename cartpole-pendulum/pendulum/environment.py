import os
import jax
import jax.numpy as jnp
import gymnax

num_devices = 4
os.environ['XLA_FLAGS'] = f"--xla_force_host_platform_device_count={num_devices}"

def create_environment():
    env, env_params = gymnax.make("Pendulum-v1")
    return env, env_params

def create_cartpole_environment():
    env, env_params = gymnax.make("CartPole-v1")
    return env, env_params 

def reset_environment(env, env_params, key_reset):
    obs, state = env.reset(key_reset, env_params)
    return obs, state

def step_environment(env, env_params, key_step, state, action):
    n_obs, n_state, reward, done, _ = env.step(key_step, state, action, env_params)
    return n_obs, n_state, reward, done

