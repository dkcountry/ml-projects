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


class ScaledFloatFrame(gym.ObservationWrapper):
    """Normalize pixel values in frame --> 0 to 1"""
    def observation(self, obs):
        if len(obs) == 2:
            return np.array(obs[0]).astype(np.float32) / 255.0, obs[1]
        else:
            return np.array(obs).astype(np.float32) / 255.0


class CondenseFrame(gym.ObservationWrapper):
    """isolate obs space to just tetris board"""
    def observation(self, obs, x_start=13, x_stop=33, y_start=11, y_stop=81):
        frame = obs[0] if len(obs) == 2 else obs
        condensed_frame = np.ndarray([y_stop - y_start, x_stop - x_start])
        for i in range(y_start, y_stop):
            row = frame[i][x_start:x_stop]
            condensed_frame[i - y_start] = row
        condensed_frame = condensed_frame.reshape(10, 2, 70)
        if len(obs) == 2:
            return condensed_frame, obs[1]
        else:
            return condensed_frame


def create_env(env_id="ALE/Tetris-v5", record_video=False):
    env = gym.make(env_id, render_mode="rgb_array")
    env = NoopResetEnv(env, noop_max=10)
    env = MaxAndSkipEnv(env, skip=2)
    env = EpisodicLifeEnv(env)
    env = ClipRewardEnv(env)
    env = gym.wrappers.ResizeObservation(env, (84, 84))
    env = gym.wrappers.GrayScaleObservation(env)
    env = ScaledFloatFrame(env)
    env = CondenseFrame(env)
    if record_video:
        env = gym.wrappers.RecordVideo(env, "../tetris")
    env.action_space.seed(SEED)
    return env


class GlobalReward:
    def __init__(self):
        self.value = 0.0

    def add(self, new_reward):
        self.value += new_reward

    def reset(self):
        self.value = 0.0


class QNet(nn.Module):
    """Experiment with a more complex architecture + dropout"""
    def __init__(self, env):
        super().__init__()
        self.column_net = nn.Sequential(
            nn.Conv2d(1, 20, 2, stride=2),
            nn.ReLU(),
            nn.Conv2d(20, 40, (1, 4), stride=4),
            nn.ReLU(),
            nn.Flatten(),
        )

        self.out_net = nn.Sequential(
            nn.Linear(3200, 512),
            nn.ReLU(),
            nn.Linear(512, env.action_space.n),
        )

    def forward(self, x):
        x = torch.tensor(x, dtype=torch.float32)
        if x.dim() == 3:
            x = x.unsqueeze(0)
        batch_size = x.size(0)
        extracted_tensors = []
        for i in range(x.size(1)):
            extracted_tensor = x[:, i:i + 1, :, :]
            extracted_tensors.append(extracted_tensor)

        conv_cols = [self.column_net(col) for col in extracted_tensors]
        conv_concat = torch.cat(conv_cols, dim=0)
        x_out = conv_concat.view(batch_size, 3200)
        return self.out_net(x_out)


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

    next_observation, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    rb.add(observation, next_observation, action, reward, done, [info])
    global_reward.add(reward)

    if done:
        writer.add_scalar("charts/episode_length", info["episode_frame_number"], global_step)
        writer.add_scalar("charts/episode_reward", global_reward.value, global_step)
        env.reset()
        global_reward.reset()
        next_observation, reward, terminated, truncated, info = env.step(action)

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


def watch_agent(env, q_network, out_directory, fps=15):
    observation, *_ = env.reset()
    global_reward.reset()
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

        next_observation, reward, terminated, truncated, info = env.step(action)
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
        Box(low=0, high=255, shape=(10, 2, 70), dtype=np.uint8),
        env.action_space,
        device
    )

    global_reward = GlobalReward()
    train()

    watch_agent(env, q_network, "../tetris.mp4")

    model_path = f"../tetris/{'tetris'}.model"
    torch.save(q_network.state_dict(), model_path)
    print(f"model saved to {model_path}")


q_network.load_state_dict(torch.load(model_path))


class Net(nn.Module):
    """Experiment with a more complex architecture + dropout"""
    def __init__(self, env):
        super().__init__()
        self.column_net = nn.Sequential(
            nn.Conv2d(1, 20, 2, stride=2),
            nn.ReLU(),
            nn.Conv2d(20, 20, (1, 4), stride=4),
            nn.ReLU(),
            nn.Flatten(),
        )

        self.out_net = nn.Sequential(
            nn.Linear(1600, 512),
            nn.ReLU(),
            nn.Linear(512, env.action_space.n),
        )

    def forward(self, x):
        x_reshaped = x.reshape(10, 2, 70)
        cols = [torch.tensor(col, dtype=torch.float32).unsqueeze(0) for col in x_reshaped]
        conv_cols = [self.column_net(col) for col in cols]
        conv_concat = torch.cat(conv_cols, dim=0)
        return self.out_net(conv_concat.view(1, 1600))



x_start = 13
x_stop = 33
y_start = 11
y_stop = 81

img = np.ndarray([y_stop - y_start, x_stop - x_start])

for i in range(y_start, y_stop):
    row = observation[i][x_start:x_stop]
    img[i-y_start] = row


from PIL import Image
import numpy as np
image_RGB = img
image = Image.fromarray(image_RGB.astype('uint8'))
image.save('../image.jpg')

q_network.pixel_net(torch.tensor(img[0], dtype=torch.float32))

q_network = Net()

a = torch.tensor(img[0], dtype=torch.float32).unsqueeze(-1).T
a = a.unsqueeze(1)

b = q_network.pixel_net(a)

imageio.mimsave("../tetris.png", [np.array(img) for i in img])

reshaped_array = x_T.reshape(10, 2, 70)
