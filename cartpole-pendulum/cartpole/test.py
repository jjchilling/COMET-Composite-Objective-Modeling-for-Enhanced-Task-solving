import gym
import numpy as np
import tensorflow as tf
from collections import deque
import random

learning_rate = 0.001
discount_factor = 0.98
episodes = 10
epsilon_initial = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
batch_size = 32
update_target_frequency = 100

env = gym.make("CartPole-v1")
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# Main Q-Network
model = tf.keras.Sequential([
    tf.keras.layers.Dense(24, input_shape=(state_size,)),
    tf.keras.layers.LeakyReLU(),
    tf.keras.layers.Dense(24),
    tf.keras.layers.LeakyReLU(),
    tf.keras.layers.Dense(action_size, activation='linear')
])

# Target Q-Network (for stability)
target_model = tf.keras.models.clone_model(model)
target_model.set_weights(model.get_weights())

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='mse')

rewards = []
epsilon = epsilon_initial
replay_buffer = deque(maxlen=10000)

print("Reached here")
for e in range(episodes):
    state = env.reset()
    state = state[0]
    if state.size != state_size:
        raise ValueError(f"Invalid state size. Expected {state_size}, but got {state.size}.")
    state = np.reshape(state, [1, state_size])
    done = False
    curr_reward = 0
    
    while not done:
        if np.random.rand() < epsilon:
            action = env.action_space.sample()  # Exploration
        else:
            q_values = model.predict(state)
            action = np.argmax(q_values[0])     # Exploitation
        print("Action taken")
        next_state, reward, done, _, _ = env.step(action)
        print("Step taken")
        next_state = np.reshape(next_state, [1, state_size])
        curr_reward += reward
        
        replay_buffer.append((state, action, reward, next_state, done))

        if len(replay_buffer) >= batch_size:
            batch = random.sample(replay_buffer, batch_size)
            for state_batch, action_batch, reward_batch, next_state_batch, done_batch in batch:
                target = reward_batch
                if not done_batch:
                    target = reward_batch + discount_factor * np.amax(target_model.predict(next_state_batch)[0])
                target_f = model.predict(state_batch)
                target_f[0][action_batch] = target
                model.fit(state_batch, target_f, epochs=1, verbose=0)
        
        state = next_state
        env.render()  # Render after each action
    
    rewards.append(curr_reward)
    
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay
    
    if e % update_target_frequency == 0:
        target_model.set_weights(model.get_weights())
    
    print(f"Episode: {e+1}, Reward: {curr_reward}, Epsilon: {epsilon}")

print("Average Reward: ", np.mean(rewards))
env.close()
