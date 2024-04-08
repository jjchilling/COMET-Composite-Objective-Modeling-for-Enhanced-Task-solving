import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.utils import set_random_seed

def make_env(env_id: str, rank: int, seed: int = 0):
    """
    Utility function for multiprocessed env.

    :param env_id: the environment ID
    :param rank: index of the subprocess
    :param seed: the initial seed for RNG
    """
    def _init():
        env = gym.make("CartPole-v1", render_mode="human")
        # env.seed(seed + rank)
        obs, info = env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init

if __name__ == "__main__":
    env_id = "CartPole-v1"
    num_cpu = 4  # Number of processes to use
    # Create the vectorized environment
    vec_env = DummyVecEnv([make_env(env_id, i) for i in range(num_cpu)])
    print("post, before PPO")

    model = PPO("MlpPolicy", vec_env, verbose=1)
    model.learn(total_timesteps=25000)  # Increased total_timesteps
    print("post PPO")


    obs = vec_env.reset()
    for _ in range(1000):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = vec_env.step(action)
        vec_env.render()
