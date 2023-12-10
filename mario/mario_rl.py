import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import RIGHT_ONLY
import gym
import torch.optim as optim
from tqdm.notebook import tqdm
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
from processing import MaxAndSkipEnv, ProcessFrame84, ImageToPyTorch, ScaledFloatFrame, BufferWrapper

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
JoypadSpace.reset = lambda self, **kwargs: self.env.reset(**kwargs)


def make_env(env_id):
    env = gym_super_mario_bros.make(env_id)
    env = MaxAndSkipEnv(env)
    env = ProcessFrame84(env)
    env = ImageToPyTorch(env)
    env = BufferWrapper(env, 4)
    env = ScaledFloatFrame(env)
    env = gym.wrappers.RecordVideo(env, "../mario")
    return JoypadSpace(env, RIGHT_ONLY)


env = make_env('SuperMarioBros-v0')
writer = SummaryWriter('../mario/runs')


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


def generate_experience(observation, global_step):
    epsilon = linear_schedule(start_e, end_e, exploration_fraction * total_timesteps, global_step)
    if random.random() < epsilon:
        action = env.action_space.sample()
    else:
        q_values = q_network(torch.Tensor(observation).to(device))
        action = torch.argmax(q_values, dim=1).cpu().numpy()[0]

    next_obs, reward, done, info = env.step(action)
    next_observation = torch.tensor(next_obs, dtype=torch.float32).unsqueeze(0)
    rb.add(observation, next_observation, action, reward, done, [info])

    if done:
        print(info)
        writer.add_scalar("charts/episode_score", info["score"], global_step)
        env.reset(seed=69)
        next_obs, reward, done, info = env.step(action)
        next_observation = torch.tensor(next_obs, dtype=torch.float32).unsqueeze(0)

    return next_observation, reward, done, info, action


def train(update_target_network=False, global_step=0):
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


env.reset(seed=69)
next_obs, *_ = env.step(4)
observation = torch.tensor(next_obs, dtype=torch.float32).unsqueeze(0)

for global_step in tqdm(range(total_timesteps)):
    observation, reward, done, info, action = generate_experience(observation, global_step)
    if global_step > learning_starts:
        if global_step % train_frequency == 0:
            if global_step % target_network_frequency == 0:
                train(update_target_network=True, global_step=global_step)
            else:
                train(update_target_network=False, global_step=global_step)
