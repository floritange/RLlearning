import gymnasium as gym
from gymnasium import envs

# 获取所有注册的环境列表
env_list = envs.registry
print(f"Total environments: {len(env_list)}")
for env_id in sorted(env_list.keys()):
    print(env_id)
