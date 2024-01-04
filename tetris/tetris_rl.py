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
from gymnasium.spaces import Box


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
buffer_size = 5000
total_timesteps = 100000
learning_starts = 5000
target_network_frequency = 100
train_frequency = 4
dropout = 0.2
ROW_INDICES = {i for i in range(70)}


class ScaledBoolFrame(gym.ObservationWrapper):
    """Normalize pixel values in frame to 0s and 1s"""
    def observation(self, obs):
        obs_out = obs[0] if len(obs) == 2 else obs
        obs_out[obs_out == 111] = 0
        obs_out[obs_out != 0] = 1
        return (obs_out, obs[1]) if len(obs) == 2 else obs_out


class CondenseFrame(gym.ObservationWrapper):
    """isolate obs space to just tetris board"""
    def observation(self, obs, x_start=13, x_stop=33, y_start=11, y_stop=81):
        frame = obs[0] if len(obs) == 2 else obs
        condensed_frame = np.ndarray([y_stop - y_start, x_stop - x_start])
        for i in range(y_start, y_stop):
            row = frame[i][x_start:x_stop]
            condensed_frame[i - y_start] = row
        if len(obs) == 2:
            return condensed_frame, obs[1]
        else:
            return condensed_frame


def create_env(env_id="ALE/Tetris-v5", record_video=False):
    env = gym.make(env_id, render_mode="rgb_array")
    env = NoopResetEnv(env, noop_max=10)
    env = MaxAndSkipEnv(env, skip=3)
    env = EpisodicLifeEnv(env)
    env = ClipRewardEnv(env)
    env = gym.wrappers.ResizeObservation(env, (84, 84))
    env = gym.wrappers.GrayScaleObservation(env)
    env = ScaledBoolFrame(env)
    env = CondenseFrame(env)
    env = gym.wrappers.FrameStack(env, 3)
    if record_video:
        env = gym.wrappers.RecordVideo(env, "../tetris")
    env.action_space.seed(SEED)
    return env


class GlobalRewardTracker:
    def __init__(self):
        self.value = 0.0
        self.fitness = 0.0

    def add(self, new_reward):
        self.value += new_reward

    def update_fitness(self, fitness):
        self.fitness = fitness

    def reset(self):
        self.value = 0.0
        self.fitness = 0.0


def censor_moving_piece(obs_new, obs_old):
    obs_out = obs_new
    diff = obs_new - obs_old
    nonzeros = diff.nonzero()
    if len(nonzeros[0]) != 0:
        obs_out[nonzeros[0].min():nonzeros[0].max()+1, nonzeros[1].min():nonzeros[1].max()+1] = 0.

    obs_nonzeros = obs_new.nonzero()
    set_diff = ROW_INDICES - set(obs_nonzeros[0])
    if len(set_diff) != 0:
        obs_out[0:max(set_diff)+1, 0: len(obs_out[0])] = 0.

    return obs_out


def reward_fitness_func(obs, game_reward, prev_fitness):
    if game_reward == 1:
        return game_reward, prev_fitness

    img = (obs * 255).astype(int).T
    aggregate_height, bumpiness = board_stats(img)

    curr_fitness = -aggregate_height / 1400 - bumpiness / 2660
    return curr_fitness - prev_fitness, curr_fitness


def col_height(col):
    col = col[::-1]
    for i in range(len(col) - 3):
        if col[i] == 111 and col[i + 1] == 111 and col[i + 2] == 111 and col[i + 3] == 111:
            return i

    return len(col)


def board_stats(img):
    aggregate_height = 0
    bumpiness = 0
    prev_col_height = 0

    for i in range(len(img)):
        height = col_height(img[i])
        aggregate_height += height
        if i > 0:
            height_diff = np.abs(height - prev_col_height)
            bumpiness += height_diff
        prev_col_height = height

    return aggregate_height, bumpiness


class QNet(nn.Module):
    """Experiment with a more complex architecture + dropout"""
    def __init__(self, env):
        super().__init__()
        self.column_net = nn.Sequential(
            nn.Conv2d(1, 20, 2, stride=2),
            nn.ReLU(),
            nn.Conv2d(20, 40, (4, 1), stride=4),
            nn.ReLU(),
            nn.Flatten(),
            nn.Dropout(dropout)
        )

        self.out_net = nn.Sequential(
            nn.Linear(3200, 512),
            nn.ReLU(),
            nn.Linear(512, env.action_space.n),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = torch.tensor(x, dtype=torch.float32)
        if x.dim() == 2:
            x = x.unsqueeze(0)

        extracted_tensors = []
        for i in range(10):
            extracted_tensor = x[:, :, i * 2:i * 2 + 2]
            extracted_tensors.append(extracted_tensor)

        cols = [(col.unsqueeze(1)) for col in extracted_tensors]
        conv_cols = [self.column_net(col) for col in cols]
        x_out = torch.cat(conv_cols, dim=1)
        return self.out_net(x_out)


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


def generate_experience(observation, global_step):
    # determine action
    epsilon = linear_schedule(start_e, end_e, exploration_fraction * total_timesteps, global_step)
    if random.random() < epsilon:
        action = env.action_space.sample()
    else:
        q_values = q_network(torch.Tensor(observation).to(device))
        action = torch.argmax(q_values, dim=1).cpu().numpy()[0]

    # get new observation & reward
    next_observation, game_reward, terminated, truncated, info = env.step(action)
    reward, fitness = reward_fitness_func(next_observation, game_reward, global_reward.fitness)
    done = terminated or truncated
    rb.add(observation, next_observation, action, reward, done, [info])
    global_reward.add(reward)
    global_reward.update_fitness(fitness)

    if done:
        writer.add_scalar("charts/episode_length", info["episode_frame_number"], global_step)
        writer.add_scalar("charts/episode_reward", global_reward.value, global_step)
        env.reset()
        global_reward.reset()
        next_observation, reward, terminated, truncated, info = env.step(action)

    return next_observation


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

    for global_step in tqdm(range(start_step, end_step)):
        observation = generate_experience(observation, global_step)
        if global_step > learning_starts:
            if global_step % train_frequency == 0:
                if global_step % target_network_frequency == 0:
                    train_loop(update_target_network=True, global_step=global_step)
                else:
                    train_loop(update_target_network=False, global_step=global_step)


def watch_agent(env, q_network, out_directory, fps=20):
    observation, *_ = env.reset()
    global_reward.reset()
    images = []
    done = False

    while not done:
        img = env.render()
        images.append(img)

        q_values = q_network(torch.Tensor(observation).to(device))
        action = torch.argmax(q_values, dim=1).cpu().numpy()[0]
        observation, reward, terminated, truncated, info = env.step(action)

        reward, fitness = reward_fitness_func(observation, reward, global_reward.fitness)
        global_reward.update_fitness(fitness)
        global_reward.add(reward)
        done = terminated or truncated

    print("Frames survived:", info["episode_frame_number"])
    print("Reward:", global_reward.value)
    env.close()
    imageio.mimsave(out_directory, [np.array(img) for i, img in enumerate(images)], fps=fps)


if __name__ == "__main__":
    writer = SummaryWriter('../tetris/runs')

    env = create_env()
    q_network = QNet(env).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=learning_rate)
    target_network = QNet(env).to(device)
    target_network.load_state_dict(q_network.state_dict())

    rb = ReplayBuffer(
        buffer_size,
        Box(low=0, high=1, shape=(70, 20), dtype=np.uint8),
        env.action_space,
        device
    )

    global_reward = GlobalRewardTracker()
    train()

    watch_agent(env, q_network, "../tetris.mp4")

    model_path = f"../tetris/{'tetris_v2'}.model"
    torch.save(q_network.state_dict(), model_path)
    print(f"model saved to {model_path}")

    q_network.load_state_dict(torch.load(model_path))





##########
from PIL import Image
import numpy as np
image_RGB = img
image = Image.fromarray(image_RGB.astype('uint8'))
image.save('../image.jpg')

img = env.render()
imageio.mimsave("../tetris.png", [np.array(img) for i in img])


x_start = 13
x_stop = 33
y_start = 11
y_stop = 81
img = np.ndarray([y_stop - y_start, x_stop - x_start])

for i in range(y_start, y_stop):
    row = observation[i][x_start:x_stop]
    img[i-y_start] = row

condensed_frame = img.reshape(10, 2, 70)
reshape = condensed_frame.reshape(20, 70)

from PIL import Image
import numpy as np
image_RGB = t
image = Image.fromarray(image_RGB.astype('uint8'))
image.save('../image.jpg')


env = create_env()
env.reset()
observation, *_ = env.step(0)
obs = observation

x = observation
x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)

extracted_tensors = []
for i in range(10):
    extracted_tensor = x[:,:, i*2:i*2+2]
    extracted_tensors.append(extracted_tensor)


n = QNet(env)
cols = [(col.unsqueeze(1)) for col in extracted_tensors]
conv_cols = [n.column_net(col.unsqueeze(1)) for col in extracted_tensors]


class Net(nn.Module):
    """Experiment with a more complex architecture + dropout"""
    def __init__(self):
        super().__init__()
        self.column_net = nn.Sequential(
            nn.Conv2d(1, 20, 2, stride=2),
            nn.ReLU(),
            nn.Conv2d(20, 40, (4, 1), stride=4),
            nn.ReLU(),
            nn.Flatten(),
            nn.Dropout(dropout)
        )

n = Net()

v_2 = torch.cat([cols[0], cols[3]], dim=0)
conv = n.column_net(v_2)


env = create_env()
obs, *_ = env.reset()

obs = (obs * 255).astype(int)
arr = obs
arr[arr == 111] = 0
arr[arr != 0] = 1

t = torch.tensor(obs)
t = t.masked_fill(t == 111, 0)
t = t.masked_fill(t != 0, 1)

import pandas as pd
df = pd.DataFrame(t)
