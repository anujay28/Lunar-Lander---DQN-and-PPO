#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import random
import numpy as np
from collections import deque
import os
import sys

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch
import torch.nn as nn
#import torchvision.transforms as T
import torchvision
from torch.autograd import Variable
import wandb
import gym
from gym.wrappers import Monitor
from agent import Agent
from dqn_model import DQN
"""
you can import any package and define any extra function as you need
"""

torch.manual_seed(595)
np.random.seed(595)
random.seed(595)
device = torch.device("cuda" if torch.cuda.is_available() else cpu)
wandb.init(project='proj4',entity = 'aukkawut',monitor_gym = True)

class Agent_DQN(Agent):
    def __init__(self, env,args,test=False):
        super(Agent_DQN,self).__init__(env)
        self.nA = env.action_space.n
        self.A = np.arange(self.nA)
        self.mem = Replay(1000000)
        if test:
            self.Q_net = torch.load('./model/model_e9900.pth')
        else:
            self.Q_net = DQN(self.nA).to(device)
        self.Q_hat_net = DQN(self.nA).to(device)
        self.criterion = nn.MSELoss() #for empirical mean part
        #arguments
        #self.batch_size = 32 #batch size
        self.optimizer = optim.Adam(self.Q_net.parameters(),lr=0.00002, eps=0.001) #optimizer
        self.mem_init_size = 50000
        self.epsilon = 0.99
        self.epsilon_delta = (0.99 - 0.001) / 1000000
        self.P = np.zeros(self.nA, np.float32)
        self.args = args
        self.window = 100
        self.max_episodes = 50000
        self.save_freq = 100
        self.work_dir = './model/model2/'
        self.eps_min = 0.001
        self.sync_period = 100
        self.batch_size = 4
        self.learn_freq = 2
        self.gamma = 0.8
    def reset(self):
        return self.env.reset()

    def action(self,state,epsilon,ap = False):
        self.P.fill(epsilon / self.nA)
        q, argq = self.Q_net(Variable(torch.from_numpy(state).to(device), volatile=True)).data.cpu().max(0)
        self.P[argq.item()] += 1 - epsilon
        a = np.random.choice(self.A, p=self.P)
        if ap:
            return a
        ns, r, done, _ = self.env.step(a)
        self.mem.push((state, torch.LongTensor([int(a)]),
            torch.Tensor([r]), ns, torch.Tensor([done])))
        return ns, r, done, q.item()

    def make_action(self,state,current_epsilon):
        a = self.action(state,current_epsilon,ap = True)
        return a

    def test(self,  current_epsilon):
        envp = Monitor(gym.make('LunarLander-v2'),'./video2/',resume=True)
        rewards = []
        state = envp.reset()
        #agent.init_game_setting()
        done = False
        #playing one game
        while(not done):
            action = self.make_action(state,current_epsilon)
            state, reward, done, info = envp.step(action)


    def train(self):
        t = 0
        s = self.reset()
        #print(self.action(s, self.epsilon))
        for _ in range(self.mem_init_size):
            ns, _1, done, _2 = self.action(s, self.epsilon)
            s = self.reset() if done else ns
        Reward = np.zeros(self.window, np.float32)
        Qfunc = np.zeros(self.window, np.float32)
        current_epsilon = self.epsilon
        for episode in range(self.max_episodes):
            Reward[episode % self.window] = 0
            Qfunc[episode % self.window] = -1e10
            s = self.reset()
            done = False
            episode_len = 0

            if episode % self.save_freq == 0:
                model_file = os.path.join(self.work_dir, 'model_e%d.pth' % episode)
                with open(model_file, 'wb') as f:
                    torch.save(self.Q_net, f)
                self.test(current_epsilon)
                

            while not done:
                if t % self.sync_period == 0:
                    self.Q_hat_net.load_state_dict(self.Q_net.state_dict())
                current_epsilon = max(self.eps_min, self.epsilon - self.epsilon_delta * t)
                ns, r, done, q = self.action(s, current_epsilon)
                Reward[episode % self.window] += r
                Qfunc[episode % self.window] = max(Qfunc[episode % self.window], q)
                s = ns
                t += 1
                episode_len += 1
                if episode_len % self.learn_freq == 0:
                    bs, ba, br, bns, bdone = self.mem.sample(self.batch_size)
                    bq = self.Q_net(bs).gather(1, ba.unsqueeze(1)).squeeze(1)
                    bnq =  self.Q_hat_net(bns).detach().max(0)[0] * self.gamma * (1 - bdone)
                    loss = self.criterion(bq, br + bnq)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
            wandb.log({'episode':episode,'step':t,'eps':current_epsilon,'reward':Reward[episode % self.window],'q': Qfunc[episode % self.window]})
        return self.Q_net    
            
def tuple2tensor(ts):
    #print(ts)
    for x in ts:
        x = torch.from_numpy(x)
    return ts

class Replay(object):
    """ Facilitates memory replay. """
    def __init__(self, mem_size):
        self.mem_size = mem_size
        self.memory = []
        self.idx = 0

    def push(self, m):
        if len(self.memory) < self.mem_size:
            self.memory.append(None)
        self.memory[self.idx] = m
        self.idx = (self.idx + 1) % self.mem_size

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        return map(lambda x: Variable(torch.tensor(x).to(device)), zip(*batch))
