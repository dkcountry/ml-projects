import numpy as np
import torch
import gym
from matplotlib import pyplot as plt
import imageio

learning_rate = 0.0035
Horizon = 500
MAX_TRAJECTORIES = 2000
gamma = 0.99
score = []

env = gym.make('CartPole-v1')
obs_size = env.observation_space.shape[0]
n_actions = env.action_space.n

HIDDEN_SIZE = 512
model = torch.nn.Sequential(
             torch.nn.Linear(obs_size, HIDDEN_SIZE),
             torch.nn.ReLU(),
             torch.nn.Linear(HIDDEN_SIZE, n_actions),
             torch.nn.Softmax(dim=0)
     )
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


for trajectory in range(MAX_TRAJECTORIES):
    curr_state = env.reset()
    done = False
    transitions = []

    for t in range(Horizon):
        act_prob = model(torch.from_numpy(curr_state).float())
        action = np.random.choice(np.array([0, 1]), p=act_prob.data.numpy())
        prev_state = curr_state
        curr_state, _, done, info = env.step(action)
        transitions.append((prev_state, action, t + 1))
        if done:
            break
    score.append(len(transitions))
    reward_batch = torch.Tensor([r for (s, a, r) in transitions]).flip(dims=(0,))
    batch_Gvals = []
    for i in range(len(transitions)):
        new_Gval = 0
        power = 0
        for j in range(i, len(transitions)):
            new_Gval = new_Gval + ((gamma ** power) * reward_batch[j]).numpy()
        power += 1
    batch_Gvals.append(new_Gval)

    expected_returns_batch = torch.FloatTensor(batch_Gvals)
    expected_returns_batch = expected_returns_batch / expected_returns_batch.max()
    state_batch = torch.Tensor([s for (s, a, r) in transitions])
    action_batch = torch.Tensor([a for (s, a, r) in transitions])
    pred_batch = model(state_batch)
    prob_batch = pred_batch.gather(dim=1, index=action_batch.long().view(-1, 1)).squeeze()

    loss = -torch.sum(torch.log(prob_batch) * expected_returns_batch)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if trajectory % 50 == 0 and trajectory > 0:
        print('Trajectory {}\tAverage Score: {:.2f}'
              .format(trajectory, np.mean(score[-50:-1])))


def running_mean(x):
    N = 50
    kernel = np.ones(N)
    conv_len = x.shape[0]-N
    y = np.zeros(conv_len)
    for i in range(conv_len):
        y[i] = kernel @ x[i:i+N]
        y[i] /= N
    return y


score = np.array(score)
avg_score = running_mean(score)
plt.figure(figsize=(15, 7))
plt.ylabel("Trajectory Duration", fontsize=12)
plt.xlabel("Training Epochs", fontsize=12)
plt.plot(score, color='gray', linewidth=1)
plt.plot(avg_score, color='blue', linewidth=3)
plt.scatter(np.arange(score.shape[0]), score, color='green', linewidth=0.3)


def watch_agent(out_directory):
    env = gym.make('CartPole-v1')
    state = env.reset()
    rewards = []
    images = []
    for t in range(2000):
        pred = model(torch.from_numpy(state).float())
        action = np.random.choice(np.array([0, 1]), p=pred.data.numpy())
        img = env.render(mode='rgb_array')
        images.append(img)
        state, reward, done, _ = env.step(action)
        rewards.append(reward)
        if done:
            print("Reward:", sum([r for r in rewards]))
            break
    env.close()
    imageio.mimsave(out_directory, [np.array(img) for i, img in enumerate(images)], fps=10)


watch_agent('../cartpole.mp4')
