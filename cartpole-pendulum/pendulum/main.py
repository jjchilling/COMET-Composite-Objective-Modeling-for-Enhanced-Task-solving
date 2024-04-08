import jax
import jax.random as jrng
import environment as environment
import model as model
import rollout
import rollout_manager
import gym
import jax.numpy as jnp
import visualization
import matplotlib.pyplot as plt

plt.rcParams['animation.writer'] = 'ffmpeg'

# Define the Policy Gradient loss function
def pg_loss(policy_params, log_probs, rewards_to_go):
    # Compute the REINFORCE loss with time-discounted rewards
    return -jnp.mean(log_probs * rewards_to_go)

def update(params, grads, learning_rate=0.01):
    # Manual parameter update based on gradients
    print(params)
    updated_params = {}
    for key, value in params.items():
        updated_params[key] = {}
        for param_key, param_value in value.items():
            updated_params[key][param_key] = {}
            for differentiable_key, differentiable_value in grads[key][param_key].items():
                updated_params[key][param_key][differentiable_key] = param_value[differentiable_key] - learning_rate * grads[key][param_key][differentiable_key]
    return updated_params

def compute_rewards_to_go(rewards, gamma=0.99):
    rewards_to_go = []
    reward_sum = 0
    for r in reversed(rewards):
        reward_sum = r + gamma * reward_sum
        rewards_to_go.append(reward_sum)
    return jnp.array(rewards_to_go[::-1])

# Main function modified to use policy gradients
def main(num_rollouts=10000):  # Specify the number of rollouts
    # Set up RNG
    rng = jrng.PRNGKey(0)

    # Create environment
    env, env_params = environment.create_environment()

    # Initialize model
    rng, rng_model = jrng.split(rng)
    m, policy_params = model.initialize_model(rng_model)

    # Plotting setup
    returns = []

    for _ in range(num_rollouts):
        # Rollout
        rng, rng_rollout = jrng.split(rng)
        obs, action, reward, next_obs, done = rollout.rollout(env, m, policy_params, env_params, rng_rollout, steps_in_episode=200) # Use jax.scan  [check single rollout and batch rolloute]

        # Compute log probabilities of actions
        log_probs = jnp.log(action)  # Assuming action is the output of the policy

        # Compute rewards-to-go
        rewards_to_go = compute_rewards_to_go(reward, gamma=0.99)

        # Compute policy gradient loss
        grads = jax.grad(pg_loss)(policy_params, log_probs, rewards_to_go)

        # Update model parameters using REINFORCE with time-discounted rewards
        policy_params = update(policy_params, grads)

        # Calculate return and append to returns list
        returns.append(jnp.sum(reward))

    # Plot returns
    plt.plot(range(num_rollouts), returns)
    plt.xlabel('Rollout')
    plt.ylabel('Return')
    plt.title('Returns over Rollouts')
    plt.show()

    # Visualization
    doc_path = "docs/anim.gif"  # Set your desired path here
    # visualization.visualize(env, env_params, rng, doc_path)
    env.close()
    gym.envs.registration.registry.env_specs.clear()

if __name__ == "__main__":
    print("Reached here 1")
    main()
    print("Reached here 2")
    

# Try and solve pendulum, pointRobot
    
    
