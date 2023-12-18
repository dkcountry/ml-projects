import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)

SEED = 42069
device = torch.device("mps")
learning_rate = 1e-4
batch_size = 32
start_e = 1
end_e = .01
exploration_fraction = .1
tau = 1.
gamma = .99
buffer_size = 10000
total_timesteps = 100000
learning_starts = 7000
target_network_frequency = 100
train_frequency = 4


env = gym.make("ALE/Tetris-v5", render_mode="rgb_array")
env = gym.wrappers.RecordEpisodeStatistics(env)
env = NoopResetEnv(env, noop_max=15)
env = MaxAndSkipEnv(env, skip=4)
env = EpisodicLifeEnv(env)
if "FIRE" in env.unwrapped.get_action_meanings():
    env = FireResetEnv(env)
env = ClipRewardEnv(env)
env = gym.wrappers.ResizeObservation(env, (84, 84))
env = gym.wrappers.GrayScaleObservation(env)
env = gym.wrappers.FrameStack(env, 4)
env.action_space.seed(SEED)

obs, _ = env.reset(seed=SEED)


class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, env.action_space.n),
        )

    def forward(self, x):
        return self.network(x)


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


q_network = QNetwork(env).to(device)
optimizer = optim.Adam(q_network.parameters(), lr=learning_rate)
target_network = QNetwork(env).to(device)
target_network.load_state_dict(q_network.state_dict())

rb = ReplayBuffer(
    buffer_size,
    env.observation_space,
    env.action_space,
    device
)