import jax
import jax.numpy as jnp

def rollout(env, model, policy_params, env_params, rng_input, steps_in_episode):
    rng_reset, rng_episode = jax.random.split(rng_input)
    obs, state = env.reset(rng_reset, env_params)

    def policy_step(state_input, tmp):
        obs, state, policy_params, rng = state_input
        rng, rng_step, rng_net = jax.random.split(rng, 3)
        action = model.apply(policy_params, obs, rng_net)
        next_obs, next_state, reward, done, _ = env.step(rng_step, state, action, env_params)
        carry = [next_obs, next_state, policy_params, rng]
        return carry, [obs, action, reward, next_obs, done]

    _, scan_out = jax.lax.scan(policy_step, [obs, state, policy_params, rng_episode], (), steps_in_episode)
    obs, action, reward, next_obs, done = scan_out
    return obs, action, reward, next_obs, done
