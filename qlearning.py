import gym
import numpy as np

# 创建环境
env = gym.make("CartPole-v1")

# 超参数
alpha = 0.1  # 学习率
gamma = 0.99  # 折扣因子
epsilon = 1.0  # 探索概率
epsilon_decay = 0.995  # 探索概率衰减
epsilon_min = 0.01  # 最小探索概率
episodes = 1000  # 训练轮数

# 初始化Q表
state_space_size = [20, 20, 20, 20]
state_space_bins = [
    np.linspace(-4.8, 4.8, state_space_size[0] - 1),
    np.linspace(-4, 4, state_space_size[1] - 1),
    np.linspace(-0.418, 0.418, state_space_size[2] - 1),
    np.linspace(-4, 4, state_space_size[3] - 1),
]
q_table = np.random.uniform(low=-2, high=0, size=(state_space_size + [env.action_space.n]))


def discretize_state(state):
    discrete_state = []
    for i in range(len(state)):
        discrete_state.append(np.digitize(state[i], state_space_bins[i]))
    return tuple(discrete_state)


# 训练
for episode in range(episodes):
    state, _ = env.reset()
    state = discretize_state(state)
    done = False
    total_reward = 0

    while not done:
        # epsilon-greedy策略选择动作
        if np.random.rand() < epsilon:
            action = env.action_space.sample()  # 随机选择动作
        else:
            action = np.argmax(q_table[state])  # 选择最优动作

        next_state, reward, done, _, _ = env.step(action)
        next_state = discretize_state(next_state)
        total_reward += reward

        # 更新Q值
        if not done:
            best_future_q = np.max(q_table[next_state])
            q_table[state][action] = (1 - alpha) * q_table[state][action] + alpha * (reward + gamma * best_future_q)
        elif next_state[0] >= env.spec.reward_threshold:
            q_table[state][action] = 0

        state = next_state

    # 更新探索概率
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

    print(f"Episode: {episode}, Total Reward: {total_reward}, Epsilon: {epsilon}")

env.close()
