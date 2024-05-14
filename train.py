import argparse

import gymnasium as gym
import torch
from gymnasium.wrappers import AtariPreprocessing
import matplotlib.pyplot as plt
import config
from utils import preprocess, get_device
from evaluate import evaluate_policy
from dqn import DQN, ReplayMemory, optimize
import os
import numpy as np

device = get_device()

parser = argparse.ArgumentParser()
parser.add_argument('--env', choices=['ALE/Pong-v5'], default='ALE/Pong-v5')
parser.add_argument('--evaluate_freq', type=int, default=25, help='How often to run evaluation.', nargs='?')
parser.add_argument('--evaluation_episodes', type=int, default=5, help='Number of evaluation episodes.', nargs='?')

# Hyperparameter configurations for different environments. See config.py.
ENV_CONFIGS = {
    'Pong': config.Pong
}

default_args = {
    'env': 'ALE/Pong-v5',
    'evaluate_freq': 25,
    'evaluation_episodes': 1    
}

def train(args=default_args, config=None):
    print(device)
    # Initialize environment and config.
    env = gym.make(args['env'])
    env = AtariPreprocessing(env, screen_size=84, grayscale_obs=True, frame_skip=1, noop_max=30)

    env_config = ENV_CONFIGS[args['env']]
    if config:
        env_config = config
    

    # Initialize deep Q-networks.
    dqn = DQN(env_config=env_config).to(device)
    # TODO: Create and initialize target Q-network.
    target_dqn = DQN(env_config=env_config).to(device)
    target_dqn.load_state_dict(dqn.state_dict())
    # Create replay memory.
    memory = ReplayMemory(env_config['replay_memory_capacity'])

    # Initialize optimizer used for training the DQN. We use Adam rather than RMSProp.
    optimizer = torch.optim.Adam(dqn.parameters(), lr=env_config['lr'])

    # Keep track of best evaluation mean return achieved so far.
    best_mean_return = -float("Inf")
    step = 0

    eval = []

    for episode in range(env_config['n_episodes']):
        terminated = False
        truncated = False
        obs, info = env.reset()
        
        obs = preprocess(obs, env=args['env']).unsqueeze(0)

        obs_stack = torch.cat(env_config['obs_stack_size'] * [obs]).unsqueeze(0).to(device)


        loss = 0
        while not terminated and not truncated:
            step += 1
            # TODO: Get action from DQN.
            action = dqn.act(obs_stack, exploit=False)

            # Act in the true environment.
            old_obs_stack = obs_stack
            obs, reward, terminated, truncated, info = env.step(action.item() + 2)

            # Preprocess incoming observation.
            #if not terminated:
            obs = preprocess(obs, env=args['env']).unsqueeze(0)
            
            obs_stack = torch.cat((obs_stack[:, 1:, ...], obs.unsqueeze(1)), dim=1).to(device)

            # TODO: Add the transition to the replay memory. Remember to convert
            #       everything to PyTorch tensors!
            reward = torch.tensor([reward], device=device)
            memory.push(old_obs_stack, action, obs_stack, reward, terminated)
            
            # TODO: Run DQN.optimize() every env_config["train_frequency"] steps.
            if step % env_config["train_frequency"] == 0:
                optimize(dqn, target_dqn, memory, optimizer)

            # TODO: Update the target network every env_config["target_update_frequency"] steps.
            if step % env_config["target_update_frequency"] == 0:
                target_dqn.load_state_dict(dqn.state_dict())


        # Evaluate the current agent.
        if episode % args['evaluate_freq'] == 0:
            mean_return = evaluate_policy(dqn, env, env_config, args, n_episodes=args['evaluation_episodes'])
            eval.append(mean_return)
            print(f'Episode {episode+1}/{env_config["n_episodes"]}: {mean_return}')

            # Save current agent if it has the best performance so far.
            if mean_return >= best_mean_return:
                best_mean_return = mean_return

                print('Best performance so far! Saving model.')
                if not os.path.exists("models"):
                    os.makedirs("models")
                torch.save(dqn, f"models/{args['env']}_best.pt")
    
    # Close environment after training is completed.
    env.close()
    return eval


if __name__ == '__main__':
    args = parser.parse_args()
    args = args.__dict__
    eval = train(args)
  