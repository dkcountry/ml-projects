import gymnasium as gym
import numpy as np
import random
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)
from torch.utils.tensorboard import SummaryWriter
import imageio


# Parameters
SEED = 42069
device = torch.device("mps")
learning_rate = 1e-4
batch_size = 32
start_e = 1
end_e = .01
exploration_fraction = 0.1
tau = 1.0
gamma = .99
buffer_size = 50000
total_timesteps = 2000000
learning_starts = 30000
target_network_frequency = 100
train_frequency = 4
dropout = 0.2


def create_env(env_id="ALE/Tetris-v5", record_video=False):
    env = gym.make(env_id, render_mode="rgb_array")
    env = NoopResetEnv(env, noop_max=15)
    env = MaxAndSkipEnv(env, skip=4)
    env = EpisodicLifeEnv(env)
    env = ClipRewardEnv(env)
    env = gym.wrappers.ResizeObservation(env, (84, 84))
    env = gym.wrappers.GrayScaleObservation(env)
    env = gym.wrappers.FrameStack(env, 4)
    if record_video:
        env = gym.wrappers.RecordVideo(env, "../tetris")
    env.action_space.seed(SEED)
    return env


class QNetwork(nn.Module):
    """Basic Q Net"""
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
        return self.network(x / 255.0)


class ComplexNetwork(nn.Module):
    """Experiment with a more complex architecture + dropout"""
    def __init__(self, env):
        super().__init__()
        self.pixel_net = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.attr_net = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, 6),
            nn.Dropout(dropout)
        )
        self.attr_net_two = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, 6),
            nn.Dropout(dropout)
        )
        self.out_net = nn.Sequential(
            nn.Linear(3148, 512),
            nn.ReLU(),
            nn.Linear(512, env.action_space.n),
        )

    def forward(self, x):
        a = self.pixel_net(x / 255.0)
        b = self.attr_net(x / 255.0)
        c = self.attr_net_two(x / 255.0)
        return self.out_net(torch.cat([a, b, c], dim=1))


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


def generate_experience(observation, global_step):
    epsilon = linear_schedule(start_e, end_e, exploration_fraction * total_timesteps, global_step)
    if random.random() < epsilon:
        action = env.action_space.sample()
    else:
        q_values = q_network(torch.Tensor(observation).to(device))
        action = torch.argmax(q_values, dim=1).cpu().numpy()[0]

    next_obs, reward, terminated, truncated, info = env.step(action)
    next_observation = torch.tensor(next_obs, dtype=torch.float32).unsqueeze(0)
    done = terminated or truncated
    rb.add(observation, next_observation, action, reward, done, [info])

    if terminated or truncated:
        print(info)
        writer.add_scalar("charts/episode_score", info["episode_frame_number"], global_step)
        env.reset(seed=SEED)
        next_obs, reward, terminated, truncated, info = env.step(action)
        next_observation = torch.tensor(next_obs, dtype=torch.float32).unsqueeze(0)

    return next_observation, reward, terminated, truncated, info, action


def train_loop(update_target_network=False, global_step=0):
    data = rb.sample(batch_size)
    with torch.no_grad():
        target_max, _ = target_network(data.next_observations).max(dim=1)
        td_target = data.rewards.flatten() + gamma * target_max * (1 - data.dones.flatten())
    old_val = q_network(data.observations).gather(1, data.actions).squeeze()
    loss = F.mse_loss(td_target, old_val)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    writer.add_scalar('Loss/train', loss, global_step)

    if update_target_network:
        for target_network_param, q_network_param in zip(target_network.parameters(), q_network.parameters()):
            target_network_param.data.copy_(tau * q_network_param.data + (1.0 - tau) * target_network_param.data)


def train(start_step=0, end_step=total_timesteps):
    observation, *_ = env.reset()
    observation = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
    for global_step in tqdm(range(start_step, end_step)):
        observation, reward, terminated, truncated, info, action = generate_experience(observation, global_step)
        if global_step > learning_starts:
            if global_step % train_frequency == 0:
                if global_step % target_network_frequency == 0:
                    train_loop(update_target_network=True, global_step=global_step)
                else:
                    train_loop(update_target_network=False, global_step=global_step)


def watch_agent(env, q_network, out_directory, fps=20):
    observation, *_ = env.reset()
    observation = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
    images = []
    done = False
    while not done:
        img = env.render()
        images.append(img)

        if random.random() < end_e:
            action = env.action_space.sample()
        else:
            q_values = q_network(torch.Tensor(observation).to(device))
            action = torch.argmax(q_values, dim=1).cpu().numpy()[0]

        next_obs, reward, terminated, truncated, info = env.step(action)
        observation = torch.tensor(next_obs, dtype=torch.float32).unsqueeze(0)
        done = terminated or truncated

    print("Frames survived:", info["episode_frame_number"])
    env.close()
    imageio.mimsave(out_directory, [np.array(img) for i, img in enumerate(images)], fps=fps)


if __name__ == "__main__":
    writer = SummaryWriter('../tetris/runs')

    env = create_env()
    q_network = ComplexNetwork(env).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=learning_rate)
    target_network = ComplexNetwork(env).to(device)
    target_network.load_state_dict(q_network.state_dict())

    rb = ReplayBuffer(
        buffer_size,
        env.observation_space,
        env.action_space,
        device
    )

    train()

    observation, *_ = env.reset()
    watch_agent(env, q_network, "../tetris.mp4")

    model_path = f"../tetris/runs/{'test'}/{'test'}.model"
    torch.save(q_network.state_dict(), model_path)
    print(f"model saved to {model_path}")
