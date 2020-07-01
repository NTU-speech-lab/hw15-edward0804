from pyvirtualdisplay import Display
virtual_display = Display(visible=0, size=(1400, 900))
virtual_display.start()

import matplotlib.pyplot as plt

from IPython import display

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from tqdm.notebook import tqdm
import gym
import pickle
env = gym.make('LunarLander-v2')
print(env.observation_space)
print(env.action_space)
initial_state = env.reset()
print(initial_state)
class PolicyGradientNetwork(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 4)

    def forward(self, state):
        hid = torch.tanh(self.fc1(state))
        hid = torch.tanh(self.fc2(hid))
        return F.softmax(self.fc3(hid), dim=-1)

class PolicyGradientAgent():

    def __init__(self, network):
        self.network = network
        self.optimizer = optim.SGD(self.network.parameters(), lr=0.001)

    def learn(self, log_probs, rewards):
        loss = (-log_probs * rewards).sum()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def sample(self, state):
        action_prob = self.network(torch.FloatTensor(state))
        action_dist = Categorical(action_prob)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        return action.item(), log_prob

network = PolicyGradientNetwork()
agent = PolicyGradientAgent(network)

agent.network.train()  # 訓練前，先確保 network 處在 training 模式
EPISODE_PER_BATCH = 10  # 每蒐集 5 個 episodes 更新一次 agent
NUM_BATCH = 1000        # 總共更新 400 次
gamma = 0.99

avg_total_rewards, avg_final_rewards = [], []

prg_bar = tqdm(range(NUM_BATCH))
for batch in prg_bar:

    log_probs, rewards = [], []
    total_rewards, final_rewards = [], []
    # 蒐集訓練資料
    for episode in range(EPISODE_PER_BATCH):
        
        state = env.reset()
        total_reward, total_step = 0, 0
        reward_record = []
        while True:

            action, log_prob = agent.sample(state)
            next_state, reward, done, _ = env.step(action)

            log_probs.append(log_prob)
            state = next_state
            total_reward += reward
            reward_record.append(reward) 
            total_step += 1

            if done:
                final_rewards.append(reward)
                total_rewards.append(total_reward)
                game_len = len(reward_record)
                R = torch.zeros(game_len)
                R[game_len - 1] = reward_record[game_len - 1]
                for i in range(game_len - 2, -1, -1):
                    R[i] = reward_record[i] + gamma * R[i + 1]
                #total = torch.sum(R)
                rewards.append(R)
                break

    # 紀錄訓練過程
    avg_total_reward = sum(total_rewards) / len(total_rewards)
    avg_final_reward = sum(final_rewards) / len(final_rewards)
    avg_total_rewards.append(avg_total_reward)
    avg_final_rewards.append(avg_final_reward)
    print(f"num: {batch}, Total: {avg_total_reward: 4.1f}, Final: {avg_final_reward: 4.1f}")
    prg_bar.set_description(f"Total: {avg_total_reward: 4.1f}, Final: {avg_final_reward: 4.1f}")

    # 更新網路
    rewards = np.concatenate(rewards, axis=0)
    rewards = (rewards - np.mean(rewards)) / (np.std(rewards) + 1e-9)  # 將 reward 正規標準化

    agent.learn(torch.stack(log_probs), torch.from_numpy(rewards))
pickle.dump(avg_total_rewards,open('discount.pkl','wb'))
plt.plot(avg_total_rewards)
plt.title("Total Rewards")
plt.savefig("toto_rewards.png")