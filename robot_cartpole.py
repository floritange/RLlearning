import gymnasium as gym
from stable_baselines3 import A2C

# 创建 Gym 环境
env = gym.make("CartPole-v1", render_mode="human")

model = A2C("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)

# 进行训练后的可视化展示
observation, info = env.reset(seed=42)
for i in range(1000):
    action, _state = model.predict(observation, deterministic=True)
    observation, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    env.render()
    if done:  # 如果环境完成，则重置环境
        observation, info = env.reset()

env.close()
