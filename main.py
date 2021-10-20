import torch
import numpy as np 
import torch.nn as nn 
import gym
import random
from Experience_Replay import ReplayMemory
from Q_network_relu import DQNNet_relu
from Double_Agent import Double_DQNAgent
from functions_for_Q import fill_memory, train, test
import os 
import torch.nn.functional as F
import argparse
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


ENV_NAME_TRAIN = 'CartPole-v0'  
RENDER_TRAIN = False

NUM_TRAIN_EPS = 1000
NUM_MEM_FILL_EPS = 50
DISCOUNT = 0.99

BATCHSIZE = 256

MEMORY_CAPACITY = 5000

UPDATE_FREQUENCY = 100

EPS_MAX = 0.01
EPS_MIN = 0.01

EPS_DECAY = 0.995


env_train = gym.make(ENV_NAME_TRAIN)
env_train._max_episode_steps = 200


if __name__ ==  '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_train_eps',type=int, default=1000, help = 'number of training episodes')
    parser.add_argument('--seed', type=int, default=10000, help='the seed')
    parser.add_argument('--lr', type=float, default=1e-4, help='the learning rate')

    
    arguments = parser.parse_args()
    
    
    
    dqn_agent_train = Double_DQNAgent(device, 
                                env_train.observation_space.shape[0], 
                                env_train.action_space.n,
                                discount=DISCOUNT, 
                                eps_max=EPS_MAX, 
                                eps_min=EPS_MIN, 
                                eps_decay=EPS_DECAY,
                                memory_capacity=MEMORY_CAPACITY,
                                lr=arguments.lr,
                                train_mode=True)
        
    
    def set_seeds(env, seed):
        os.environ['PYTHONHASHSEED']=str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        env.seed(seed)
        env.action_space.np_random.seed(seed)
    
    set_seeds(env_train, arguments.seed)

    
    rewards_training, average_reward_training= train (env_train,dqn_agent_train,arguments.num_train_eps,NUM_MEM_FILL_EPS,UPDATE_FREQUENCY,BATCHSIZE,RENDER_TRAIN)
    
    
    
    average_reward_training = np.array(average_reward_training)
    filename = 'cartpole'  + '_' + 'lr' +  str(arguments.lr) + '_' + 'seed' + str(arguments.seed) +  '.npy'
    
    np.save(filename, average_reward_training)
    
    rewards_training = np.array(rewards_training)
    k = rewards_training.size 
    plt.plot(k,rewards_training)
    plt.show()
    