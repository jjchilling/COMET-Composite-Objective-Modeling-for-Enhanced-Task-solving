import jax
import jax.numpy as jnp
from collections import deque
import numpy as np
import gym
import optax

class PrioritizedReplayBuffer:
    def __init__(self, buffer_size, alpha=0.6):
        self.buffer = deque(maxlen=buffer_size)
        self.alpha = alpha
        self.priorities = np.zeros(buffer_size)

    def add(self, experience):
        max_priority = np.max(self.priorities) if self.buffer else 1.0
        self.buffer.append(experience)
        self.priorities[len(self.buffer) - 1] = max_priority ** self.alpha

    def sample(self, batch_size, beta=0.4):
        priorities = self.priorities[:len(self.buffer)]
        probs = priorities / np.sum(priorities)
        indices = np.random.choice(len(self.buffer), batch_size, p=probs, replace=False)
        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights /= np.max(weights)
        return [self.buffer[i] for i in indices], indices, weights

    def update_priorities(self, indices, td_errors):
        for i, error in zip(indices, td_errors):
            self.priorities[i] = (np.abs(error) + 1e-6) ** self.alpha

learning_rate = 0.001
discount_factor = 0.99
episodes = 10000
batch_size = 32
buffer_size = 10000  
replay_buffer = PrioritizedReplayBuffer(buffer_size)
update_target_every = 1000 
target_update_counter = 0

env = gym.make("InvertedPendulum-v4")

def scale_reward(reward):
    return reward * 10.0

class ScaledRewardEnv(gym.Wrapper):
    def step(self, action):
        next_state, reward, done, info, _ = self.env.step(action)
        return next_state, scale_reward(reward), done, info

env = ScaledRewardEnv(env)

state_size = env.observation_space.shape[0]
print(env.action_space)
action_size = env.action_space.shape[0]

def create_model_and_params(rng):
    key, _ = jax.random.split(rng)
    
    w1 = jax.random.normal(key, (state_size, 64)) 
    b1 = jnp.zeros((64,))
    w2 = jax.random.normal(key, (64, 64))  
    b2 = jnp.zeros((64,))
    w3 = jax.random.normal(key, (64, 64))  
    b3 = jnp.zeros((64,))
    w4 = jax.random.normal(key, (64, action_size))  # Output layer
    b4 = jnp.zeros((action_size,))
    
    # Define Leaky ReLU activation function
    def leaky_relu(x):
        alpha = 0.01
        return jnp.maximum(alpha * x, x)
    
    # Define layer functions with Leaky ReLU activation
    def layer1(x, params):
        w, b = params
        return leaky_relu(jnp.dot(x, w) + b)

    def layer2(x, params):
        w, b = params
        return leaky_relu(jnp.dot(x, w) + b)

    def layer3(x, params):
        w, b = params
        return leaky_relu(jnp.dot(x, w) + b)

    def layer4(x, params):
        w, b = params
        return jnp.dot(x, w) + b
    
    # Define the model function
    def model(x, params):
        x = layer1(x, params[:2])
        x = layer2(x, params[2:4])
        x = layer3(x, params[4:6])
        x = layer4(x, params[6:])
        return x
    
    params = [w1, b1, w2, b2, w3, b3, w4, b4]
    
    return model, params

rng = jax.random.PRNGKey(0)
model, params = create_model_and_params(rng)

def q_learning_policy(params, obs):
    q_values = model(obs, params)
    return jnp.argmax(q_values)

@jax.jit
def q_learning_update(params, obs, action, reward, next_obs, done):
    replay_buffer.add((obs, action, reward, next_obs, done))

    if len(replay_buffer.buffer) < batch_size:
        return params

    batch, indices, weights = replay_buffer.sample(batch_size)
    obs_batch, action_batch, reward_batch, next_obs_batch, done_batch = zip(*batch)

    def loss_fn(params):
        q_values = model(obs_batch, params)
        next_q_values = model(next_obs_batch, params)
        target_q_values_next = model(next_obs_batch, target_params)  # Using target network
        max_next_q_values = jnp.max(target_q_values_next, axis=-1)
        target_q_values = reward_batch + discount_factor * (1 - done_batch) * max_next_q_values

        target = q_values.at[jnp.arange(batch_size), action_batch].set(target_q_values)
        td_errors = target - q_values.at[jnp.arange(batch_size), action_batch]
        loss = jnp.mean(weights * td_errors ** 2)
        return loss, td_errors
    
    grads, td_errors = jax.grad(loss_fn, has_aux=True)(params)
    updated_params = [param - grad * learning_rate for param, grad in zip(params, grads)]

    replay_buffer.update_priorities(indices, td_errors)
    
    return updated_params


# Training loop
def train(params):
    global target_params, target_update_counter
    
    for e in range(episodes):
        state = env.reset()
        state = state[0]
        state = jnp.array(state)
        episode_reward = 0
        done = False
        while not done:
            # Epsilon-greedy exploration
            if jax.random.uniform(rng) < 0.1:
                action = env.action_space.sample()
            else:
                action = q_learning_policy(params, state)
                # Ensure action is within the valid range
                action = int(jnp.clip(action, 0, action_size - 1))

            if target_update_counter % update_target_every == 0:
                target_params = params  # Update target network parameters
            target_update_counter += 1
            print("action: ")
            print(action)
            print("step: ")
            print(env.step(action))
            step_result = env.step(action)
            next_state, reward, done, _ = step_result
            next_state = jnp.array(next_state)
            episode_reward += reward
            params = q_learning_update(params, state, action, reward, next_state, done)
            state = next_state
            env.render()
        print(f"Episode: {e + 1}, Reward: {episode_reward}")

# Run training
params = create_model_and_params(rng)[1]  # Initialize params
target_params = params
train(params)

# Close the environment
env.close()


# jax.conditional
# use optax for gradient update