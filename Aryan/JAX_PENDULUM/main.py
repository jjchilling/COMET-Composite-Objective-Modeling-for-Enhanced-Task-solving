import jax
import jax.numpy as jnp
import gymnax as gym
import optax
import matplotlib.pyplot as plt
from gymnax.visualize import Visualizer
from pointrobot import PointRobot
from matplotlib.animation import PillowWriter

plt.rcParams['animation.writer'] = 'ffmpeg'

import jax_ppo

if __name__ == "__main__":
    k = jax.random.PRNGKey(101)
    
    print("Reached here 1")
    # env, env_params = gym.make("PointRobot-misc")
    env = PointRobot()

    # Get the default environment parameters
    env_params = env.default_params
    # env, env_params = gym.make("Pendulum-v1")

    print("Reached here 2")

     # Number of policy updates
    N_TRAIN = 2500
    # Number of training environments
    N_TRAIN_ENV = 32
    # Number of test environments
    N_TEST_ENV = 5
    # Number of enviroment steps
    N_ENV_STEPS = env_params.max_steps_in_episode
    # Number of training loops per poliy update
    N_EPOCHS = 2
    # Mini-batch sized used for actual training
    MINI_BATCH_SIZE = 256

    N_STEPS = N_TRAIN * N_TRAIN_ENV * N_ENV_STEPS * N_EPOCHS // MINI_BATCH_SIZE


    print("Reached here 3")
    params = jax_ppo.default_params._replace(
        gamma=0.95, gae_lambda=0.9, entropy_coeff=0.0001, adam_eps=1e-8, clip_coeff=0.2
    )
    
    
    print("Reached here 4")
    
    train_schedule = optax.linear_schedule(2e-3, 2e-5, N_STEPS)


    print("Reached here 5")
    k, agent = jax_ppo.init_agent(
        k, 
        params,
        env.action_space().shape,
        env.observation_space(env_params).shape,
        train_schedule,
        layer_width=16,
    )
    
    print("Reached here 6")
    _k, trained_agent, losses, ts, rewards, _, state_seq = jax_ppo.train(
        k, env, env_params, agent,
        N_TRAIN, 
        N_TRAIN_ENV, 
        N_EPOCHS, 
        MINI_BATCH_SIZE, 
        N_TEST_ENV, 
        params, 
        env_params.max_steps_in_episode,
        greedy_test_policy=True
    )
    
    print("Reached here 7")
    plt.plot(jnp.mean(jnp.sum(rewards[:, :, :], axis=2), axis=1));
    plt.xlabel("Training Step")
    plt.ylabel("Avg Total Rewards");
    plt.show()
    
    cumulative_rewards = jnp.cumsum(rewards, axis=2)  # Compute cumulative rewards
    plt.plot(jnp.mean(jnp.sum(cumulative_rewards[:, :, :], axis=2), axis=1))
    plt.xlabel("Training Step")
    plt.ylabel("Cumulative Total Rewards")
    plt.show()
    

    # Visualize the recorded states and rewards
    vis = Visualizer(env, env_params, state_seq, cumulative_rewards)
    vis.animate("docs/anim.gif")