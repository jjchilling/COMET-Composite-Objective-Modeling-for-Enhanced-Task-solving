import jax
import jax.numpy as jnp
import flax.linen as nn
from gymnax.visualize import Visualizer
import numpy as np
import optax
import tqdm
import gymnax
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Any
from flax.training.train_state import TrainState
import distrax

from wrappers import (
    LogWrapper,
    GymnaxWrapper,
    VecEnv,
    NormalizeVecObservation,
    NormalizeVecReward,
    ClipAction,
)


class ActorCritic(nn.Module):
    action_dim: Sequence[int]
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
        actor_mean = nn.Dense(
            256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(actor_mean)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        actor_logtstd = self.param("log_std", nn.initializers.zeros, (self.action_dim,))
        pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(actor_logtstd))

        critic = nn.Dense(
            256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        critic = activation(critic)
        critic = nn.Dense(
            256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(critic)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return pi, jnp.squeeze(critic, axis=-1)


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray


def make_train(config):
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )
    # env, env_params = BraxGymnaxWrapper(config["ENV_NAME"]), None
    # env = GymnaxWrapper(config["ENV_NAME"])
    env, env_params = gymnax.make(config["ENV_NAME"])
    env = LogWrapper(env)
    env = ClipAction(env)
    env = VecEnv(env)
    if config["NORMALIZE_ENV"]:
        env = NormalizeVecObservation(env)
        env = NormalizeVecReward(env, config["GAMMA"])

    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

    def train(rng):
        # INIT NETWORK
        network = ActorCritic(
            env.action_space(env_params).shape[0], activation=config["ACTIVATION"]
        )
        rng, _rng = jax.random.split(rng)
        init_x = jnp.zeros(env.observation_space(env_params).shape)
        network_params = network.init(_rng, init_x)
        if config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = env.reset(reset_rng, env_params)
        
        
        # state_seq, reward_seq = [], []
        obsv, env_state = env.reset(reset_rng, env_params)
        # state_seq.append(env_state)
        
        pbar = tqdm.tqdm(total=config["NUM_UPDATES"], desc="Training")

        # TRAIN LOOP
        for _ in range(int(config["NUM_UPDATES"])):
            def _update_step(runner_state, unused):
                # COLLECT TRAJECTORIES
                @jax.jit
                def _env_step(runner_state, unused):
                    train_state, env_state, last_obs, rng = runner_state

                    # SELECT ACTION
                    rng, _rng = jax.random.split(rng)
                    pi, value = network.apply(train_state.params, last_obs) #this line and the next line for actions
                    action = pi.sample(seed=_rng)
                    log_prob = pi.log_prob(action)

                    # STEP ENV
                    rng, _rng = jax.random.split(rng)
                    rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                    obsv, env_state, reward, done, info = env.step(
                        rng_step, env_state, action, env_params
                    )
                    
                    # env.render(env_state, env_params)
                    # Print relevant variables
                    print("Reward: ", jax.device_get(reward))
                    # print("Done:", done)
                    # print("Info:", info)
                    # reward_seq.append(reward)
                    transition = Transition(
                        done, action, value, reward, log_prob, last_obs, info
                    )
                    runner_state = (train_state, env_state, obsv, rng)
                    return runner_state, transition

                runner_state, traj_batch = jax.lax.scan(
                    _env_step, runner_state, None, config["NUM_STEPS"]
                )

                # CALCULATE ADVANTAGE
                train_state, env_state, last_obs, rng = runner_state
                _, last_val = network.apply(train_state.params, last_obs)

                def _calculate_gae(traj_batch, last_val):
                    def _get_advantages(gae_and_next_value, transition):
                        gae, next_value = gae_and_next_value
                        done, value, reward = (
                            transition.done,
                            transition.value,
                            transition.reward,
                        )
                        delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                        gae = (
                            delta
                            + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                        )
                        return (gae, value), gae

                    _, advantages = jax.lax.scan(
                        _get_advantages,
                        (jnp.zeros_like(last_val), last_val),
                        traj_batch,
                        reverse=True,
                        unroll=16,
                    )
                    return advantages, advantages + traj_batch.value

                advantages, targets = _calculate_gae(traj_batch, last_val)

                # UPDATE NETWORK
                def _update_epoch(update_state, unused):
                    def _update_minbatch(train_state, batch_info):
                        traj_batch, advantages, targets = batch_info

                        def _loss_fn(params, traj_batch, gae, targets):
                            # RERUN NETWORK
                            pi, value = network.apply(params, traj_batch.obs)
                            log_prob = pi.log_prob(traj_batch.action)

                            # CALCULATE VALUE LOSS
                            value_pred_clipped = traj_batch.value + (
                                value - traj_batch.value
                            ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                            value_losses = jnp.square(value - targets)
                            value_losses_clipped = jnp.square(value_pred_clipped - targets)
                            value_loss = (
                                0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                            )

                            # CALCULATE ACTOR LOSS
                            ratio = jnp.exp(log_prob - traj_batch.log_prob)
                            gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                            loss_actor1 = ratio * gae
                            loss_actor2 = (
                                jnp.clip(
                                    ratio,
                                    1.0 - config["CLIP_EPS"],
                                    1.0 + config["CLIP_EPS"],
                                )
                                * gae
                            )
                            loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                            loss_actor = loss_actor.mean()
                            entropy = pi.entropy().mean()

                            total_loss = (
                                loss_actor
                                + config["VF_COEF"] * value_loss
                                - config["ENT_COEF"] * entropy
                            )
                            return total_loss, (value_loss, loss_actor, entropy)

                        grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                        total_loss, grads = grad_fn(
                            train_state.params, traj_batch, advantages, targets
                        )
                        train_state = train_state.apply_gradients(grads=grads)
                        return train_state, total_loss

                    train_state, traj_batch, advantages, targets, rng = update_state
                    rng, _rng = jax.random.split(rng)
                    batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
                    assert (
                        batch_size == config["NUM_STEPS"] * config["NUM_ENVS"]
                    ), "batch size must be equal to number of steps * number of envs"
                    permutation = jax.random.permutation(_rng, batch_size)
                    batch = (traj_batch, advantages, targets)
                    batch = jax.tree_util.tree_map(
                        lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
                    )
                    shuffled_batch = jax.tree_util.tree_map(
                        lambda x: jnp.take(x, permutation, axis=0), batch
                    )
                    minibatches = jax.tree_util.tree_map(
                        lambda x: jnp.reshape(
                            x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
                        ),
                        shuffled_batch,
                    )
                    train_state, total_loss = jax.lax.scan(
                        _update_minbatch, train_state, minibatches
                    )
                    update_state = (train_state, traj_batch, advantages, targets, rng)
                    return update_state, total_loss

                update_state = (train_state, traj_batch, advantages, targets, rng)
                update_state, loss_info = jax.lax.scan(
                    _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
                )
                train_state = update_state[0]
                metric = traj_batch.info
                rng = update_state[-1]
                if config.get("DEBUG"):

                    def callback(info):
                        return_values = info["returned_episode_returns"][
                            info["returned_episode"]
                        ]
                        timesteps = (
                            info["timestep"][info["returned_episode"]] * config["NUM_ENVS"]
                        )
                        for t in range(len(timesteps)):
                            print(
                                f"global step={timesteps[t]}, episodic return={return_values[t]}"
                            )

                    jax.debug.callback(callback, metric)

                runner_state = (train_state, env_state, last_obs, rng)
                return runner_state, metric
            # state_seq.append(env_state)
            # reward_seq.append(reward)
            pbar.update(1)

        pbar.close()
        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, env_state, obsv, _rng)
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        # cum_rewards = jnp.cumsum(jnp.array(reward_seq))
        # vis = Visualizer(env, env_params, state_seq, cum_rewards)
        # vis.animate("docs/anim.gif")  # Save animation
        return {"runner_state": runner_state, "metrics": metric}

    return train


if __name__ == "__main__":
    config = {
        "LR": 3e-4,
        "NUM_ENVS": 2048,
        "NUM_STEPS": 10,
        "TOTAL_TIMESTEPS": 5e7 ,
        "UPDATE_EPOCHS": 4,
        "NUM_MINIBATCHES": 32,
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
        "CLIP_EPS": 0.2,
        "ENT_COEF": 0.0,
        "VF_COEF": 0.5,
        "MAX_GRAD_NORM": 0.5,
        "ACTIVATION": "tanh",
        # "ENV_NAME": "hopper",
        "ENV_NAME": "PointRobot-misc",
        "ANNEAL_LR": False,
        "NORMALIZE_ENV": True,
        "DEBUG": True,
    }
    print("Reached here 1")
    rng = jax.random.PRNGKey(30)
    print("Reached here 2")
    train_jit = jax.jit(make_train(config))
    print("Reached here 3")
    out = train_jit(rng)
    print("Reached here 5")
    import matplotlib.pyplot as plt
    # print(out['metrics']['returned_episode'])
    # print(out["metrics"]["returned_episode_returns"])
    plt.plot(out["metrics"]["returned_episode_returns"].mean(-1).reshape(-1))
    plt.xlabel("Updates")
    plt.ylabel("Return")
    plt.show()
    
    # env, env_params = gymnax.make(config["ENV_NAME"]);
    # network = ActorCritic(env.action_space(env_params).shape[0], activation=config["ACTIVATION"])
    # network_params = out['runner_state'][0].params
    # for _ in range(5):
    #     obs = env.reset(env_params)
    #     done = False
    #     episode_return = 0
    #     while not done:
    #         pi, _ = network.apply(network_params, obs)
    #         action = pi.sample(seed=np.random.randint(0, 1000))  # Sample action
    #         obs, reward, done, _ = env.step(action, env_params)
    #         episode_return += reward
    #         env.render()
    #     print("Episode return:", episode_return)
    # env.close()  # Close the environment after visualization
    
    # Visualize the environment after training
    # env, _ = gymnax.make(config["ENV_NAME"])
    # env_params = env.default_params()

    # # Initialize the agent with the trained parameters
    # network = ActorCritic(env.action_space(env_params).shape[0], activation=config["ACTIVATION"])
    # network_params = out['runner_state'][0].params

    # # Run for 5 episodes
    # for _ in range(5):
    #     obs, env_state = env.reset(env_params)
    #     done = False
    #     episode_return = 0
    #     while not done:
    #         pi, _ = network.apply(network_params, obs)
    #         action = pi.sample(seed=np.random.randint(0, 1000))  # Sample action
    #         obs, reward, done, _ = env.step(action, env_params, env_state)
    #         episode_return += reward
    #         env.render()
    #     print("Episode return:", episode_return)
    # env.close()  # Close the environment after visualization
    
    # rng = jax.random.PRNGKey(0)
    # env, env_params = gymnax.make("PointRobot-misc")

    # state_seq, reward_seq = [], []
    # rng, rng_reset = jax.random.split(rng)
    # obs, env_state = env.reset(rng_reset, env_params)
    # while True:
    #     state_seq.append(env_state)
    #     rng, rng_act, rng_step = jax.random.split(rng, 3)
    #     action = env.action_space(env_params).sample(rng_act) #replace this with the model params
    #     next_obs, next_env_state, reward, done, info = env.step(
    #         rng_step, env_state, action, env_params
    #     )
    #     reward_seq.append(reward)
    #     if done:
    #         break
    #     else:
    #         obs = next_obs
    #         env_state = next_env_state

    # cum_rewards = jnp.cumsum(jnp.array(reward_seq))
    # vis = Visualizer(env, env_params, state_seq, cum_rewards)
    # vis.animate("anim.gif")
    env, env_params = gymnax.make(config["ENV_NAME"])
    network = ActorCritic(env.action_space(env_params).shape[0], activation=config["ACTIVATION"])
    network_params = out['runner_state'][0].params

    # Run the visualization loop
    state_seq, reward_seq = [], []
    rng, rng_reset = jax.random.split(rng)
    obsv, env_state = env.reset(rng_reset, env_params)
    while True:
        state_seq.append(env_state)
        # Use the policy network to sample actions
        pi, _ = network.apply(network_params, obsv)
        action = pi.sample(seed=rng)  # Sample action using the policy network
        next_obs, next_env_state, reward, done, info = env.step(
            rng, env_state, action, env_params
        )
        reward_seq.append(reward)
        if done:
            break
        else:
            obsv = next_obs
            env_state = next_env_state

    # Visualize the environment
    cum_rewards = jnp.cumsum(jnp.array(reward_seq))
    vis = Visualizer(env, env_params, state_seq, cum_rewards)
    vis.animate("Aryan_anim.gif")
