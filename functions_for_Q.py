import torch
import torch.nn as nn 
import gym
import numpy as np 
import random 
from Experience_Replay import ReplayMemory
from Q_network_relu import DQNNet_relu
from Double_Agent import Double_DQNAgent
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter


def fill_memory(env, dqn_agent, num_memory_fill_eps):

    for _ in range(num_memory_fill_eps):
        done = False
        state = env.reset()

        while not done:
            action = env.action_space.sample()
            next_state, reward, done, info = env.step(action)
            dqn_agent.memory.store(state=state, 
                                action=action, 
                                next_state=next_state, 
                                reward=reward, 
                                done=done)


def train(env, dqn_agent, num_train_eps, num_memory_fill_eps, update_frequency, batchsize, render=False):
    
    fill_memory(env, dqn_agent, num_memory_fill_eps)
    print('Memory filled. Current capacity: ', len(dqn_agent.memory))
    average_reward = []
    reward_history = []
    epsilon_history = []

        
    tb = SummaryWriter()    
    step_cnt = 0
    best_score = -np.inf

    for ep_cnt in range(num_train_eps):
        epsilon_history.append(dqn_agent.epsilon)

        done = False
        state = env.reset()

        ep_score = 0

        while not done:
            if render:
                env.render()

            action = dqn_agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            dqn_agent.memory.store(state=state, action=action, next_state=next_state, reward=reward, done=done)

            dqn_agent.learn(batchsize=batchsize)

            if step_cnt % update_frequency == 0:
                dqn_agent.update_target_net()

            state = next_state
            ep_score += reward
            step_cnt += 1

        dqn_agent.update_epsilon()

        reward_history.append(ep_score)
        current_avg_score = np.mean(reward_history[-50:]) # moving average of last 50 episodes
        average_reward.append(current_avg_score)
        
        if ep_score >= best_score:
            dqn_agent.best_net.load_state_dict(dqn_agent.policy_net.state_dict())
            best_score = ep_score

        print('Ep: {}, Total Steps: {}, Ep: Score: {},; Epsilon: {}'.format(ep_cnt, step_cnt, ep_score, epsilon_history[-1]))
        tb.add_scalar ('score', ep_score, ep_cnt)
    return reward_history, average_reward 




def test(env, dqn_agent, num_test_eps, seed, render=True):
    

    step_cnt = 0
    reward_history = []

    for ep in range(num_test_eps):
        score = 0
        done = False
        state = env.reset()
        while not done:

            if render:
                env.render()

            action = dqn_agent.select_action(state)
            next_state, reward, done, _ = env.step(action)

            score += reward
            state = next_state
            step_cnt += 1

        reward_history.append(score)
        print('Ep: {}, Score: {}'.format(ep, score))

    return reward_history