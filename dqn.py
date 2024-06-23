import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import gym
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from collections import deque
import random
import matplotlib.pyplot as plt


# 创建环境
env = gym.make("CartPole-v1")
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# 超参数
learning_rate = 0.001
gamma = 0.99
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.01
batch_size = 64
episodes = 1000
memory_size = 2000

# 经验回放缓冲区
memory = deque(maxlen=memory_size)


# 构建Q网络模型
def build_model():
    model = Sequential()
    model.add(Dense(24, input_dim=state_size, activation="relu"))
    model.add(Dense(24, activation="relu"))
    model.add(Dense(action_size, activation="linear"))
    model.compile(loss="mse", optimizer=Adam(learning_rate=learning_rate))
    return model


# 主Q网络和目标Q网络
model = build_model()
target_model = build_model()
target_model.set_weights(model.get_weights())


def update_target_model():
    target_model.set_weights(model.get_weights())


# 选择动作
def choose_action(state):
    if np.random.rand() <= epsilon:
        return env.action_space.sample()
    q_values = model.predict(state)
    return np.argmax(q_values[0])


# 经验回放训练
def replay():
    if len(memory) < batch_size:
        return
    minibatch = random.sample(memory, batch_size)
    for state, action, reward, next_state, done in minibatch:
        target = reward
        if not done:
            target = reward + gamma * np.amax(target_model.predict(next_state)[0])
        target_f = model.predict(state)
        target_f[0][action] = target
        model.fit(state, target_f, epochs=1, verbose=0)


# 记录每个episode的总奖励
rewards = []

# 训练
for episode in range(episodes):
    state, _ = env.reset()
    state = np.reshape(state, [1, state_size])
    total_reward = 0
    done = False

    while not done:
        action = choose_action(state)
        next_state, reward, done, _, _ = env.step(action)
        next_state = np.reshape(next_state, [1, state_size])
        total_reward += reward
        memory.append((state, action, reward, next_state, done))
        state = next_state

        if done:
            update_target_model()
            rewards.append(total_reward)
            break

        replay()

    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

    print(f"Episode: {episode}, Total Reward: {total_reward}, Epsilon: {epsilon}")

env.close()

# 可视化总奖励
plt.plot(rewards)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Total Reward per Episode")
plt.show()
