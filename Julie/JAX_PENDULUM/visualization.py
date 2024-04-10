from gymnax.visualize import Visualizer
import jax
import jax.numpy as jnp

def visualize(env, env_params, rng, doc_path, cumulative_rewards, state_seq_2):
    state_seq, reward_seq = [], []
    rng, rng_reset = jax.random.split(rng)
    obs, env_state = env.reset(rng_reset, env_params)
    t_counter = 0
    while True:
        state_seq.append(env_state)
        rng, rng_act, rng_step = jax.random.split(rng, 3)
        action = env.action_space(env_params).sample(rng_act)
        next_obs, next_env_state, reward, done, info = env.step(rng_step, env_state, action, env_params)
        reward_seq.append(reward)
        t_counter += 1
        if done or t_counter >= 50:
            break
        else:
            obs = next_obs
            env_state = next_env_state

    cum_rewards = jnp.cumsum(jnp.array(reward_seq))
    # print(cum_rewards.shape)
    # print(cumulative_rewards.shape)
    print(state_seq.shape)
    print(state_seq_2.shape)
    vis = Visualizer(env, env_params, state_seq, cumulative_rewards)
    vis.animate(doc_path)
