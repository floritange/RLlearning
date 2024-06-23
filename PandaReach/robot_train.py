import gymnasium as gym
import panda_gym
from stable_baselines3 import DDPG, HerReplayBuffer
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

log_dir = "./panda_reach_v3_tensorboard/"

# 创建环境
env = gym.make("PandaReach-v3")
eval_env = gym.make("PandaReach-v3")

# 创建并训练模型，使用 EvalCallback 进行早停
eval_callback = EvalCallback(
    eval_env, best_model_save_path="./logs/best_model/", log_path="./logs/", eval_freq=1000, deterministic=True, render=False
)

checkpoint_callback = CheckpointCallback(save_freq=1000, save_path="./logs/checkpoints/", name_prefix="ddpg_model")

model = DDPG("MultiInputPolicy", env=env, buffer_size=100000, replay_buffer_class=HerReplayBuffer, verbose=1, tensorboard_log=log_dir)

model.learn(total_timesteps=5000, callback=[eval_callback, checkpoint_callback])
model.save("ddpg_panda_reach_v3")
env.close()

