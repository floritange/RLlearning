import gymnasium as gym
import panda_gym
from stable_baselines3 import DDPG, HerReplayBuffer
import time

# 创建环境
env = gym.make("PandaReach-v3", render_mode="human")

# 加载保存的模型
loaded_model = DDPG.load("./logs/best_model/best_model", env=env)

# 进行训练后的可视化展示
observation, info = env.reset(seed=42)
for i in range(10000):  # 你可以调整循环次数以进行更长时间的可视化
    action, _state = loaded_model.predict(observation, deterministic=True)
    observation, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    env.render()
    time.sleep(0.5)
    if done:  # 如果环境完成，则重置环境
        observation, info = env.reset()

env.close()

