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
import torchvision.transforms as T
import torchvision
from torch.autograd import Variable
import wandb

from agent import Agent
from dqn_model import DQN
"""
you can import any package and define any extra function as you need
"""

torch.manual_seed(595)
np.random.seed(595)
random.seed(595)
device = torch.device("cuda" if torch.cuda.is_available() else cpu )

class Agent_DQN(Agent):
    def __init__(self, env,args):
        super(Agent_DQN,self).__init__(env)
        self.nA = env.action_space.n
        self.A = np.arange(self.nA)
        self.mem = Replay(args.mem_size)
        self.Q_net = DQN(self.nA).to(device)
        self.Q_hat_net = DQN(self.nA).to(device)
        self.criterion = nn.MSELoss() #for empirical mean part
        #arguments
        self.batch_size = args.bs #batch size
        self.optimizer = optim.Adam(self.Q_net.parameters(),lr=args.lr, eps=0.001) #optimizer
        
        self.epsilon = max(args.eps, args.eps_min)
        self.epsilon_delta = (self.epsilon - args.eps_min) / args.eps_decay_window
        self.P = np.zeros(self.nA, np.float32)
        self.prep = StateTransform(lambda s: s[50:, :, :], 84)
        self.args = args

    def reset(self):
        return torch.cat([self.prep.run(self.env.reset())] * 4, 1)

    def action(self,state,epsilon,a = False):
        self.P.fill(epsilon / self.nA)
        q, argq = self.Q_net(Variable(state.to(device), volatile=True)).data.cpu().max(1)
        self.P[argq.item()] += 1 - epsilon
        a = np.random.choice(self.A, p=self.P)
        ns, r, done, _ = self.env.step(a)
        ns = torch.cat([state.narrow(1, 1, 3), self.prep.run(ns)], 1)
        self.mem.push((state, torch.LongTensor([int(a)]),
            torch.Tensor([r]), ns, torch.Tensor([done])))
        if not a:
            return ns, r, done, q.item()
        else:
            return ns, r, done, q.item(), a

    def make_action(self,state,test=True):
        if test:
            ns,r,done, q,a = self.action(state,a = True)
        return a


    def train(self):
        s = self.reset()
        for _ in range(self.args.mem_init_size):
            ns, _, done, _ = self.action(s, self.epsilon)
            s = self.reset() if done else ns
        Reward = np.zeros(self.args.window, np.float32)
        Qfunc = np.zeros(self.args.window, np.float32)
        for episode in range(self.args.max_episodes):
            Reward[episode % self.args.window] = 0
            Qfunc[episode % self.args.window] = -1e10
            s = self.reset()
            done = False
            episode_len = 0

            if episode % self.args.save_freq == 0:
                model_file = os.path.join(self.args.work_dir, 'model_e%d.pth' % episode)
                with open(model_file, 'wb') as f:
                    torch.save(self.Q_net, f)

            while not done:
                if t % self.args.sync_period == 0:
                    T.load_state_dict(self.Q_net.state_dict())
                current_epsilon = max(self.args.eps_min, self.epsilon - self.epsilon_delta * t)
                ns, r, done, q = self.action(s, current_epsilon)
                Reward[episode % self.args.window] += r
                Qfunc[episode % self.args.window] = max(Qfunc[episode % self.args.window], q)
                s = ns
                t += 1
                episode_len += 1
                if episode_len % self.args.learn_freq == 0:
                    bs, ba, br, bns, bdone = self.mem.sample(self.args.batch_size)
                    bq = self.Q_net(bs).gather(1, ba.unsqueeze(1)).squeeze(1)
                    bnq = T(bns).detach().max(1)[0] * self.args.gamma * (1 - bdone)
                    loss = self.criterion(bq, br + bnq)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
            wandb.log({'episode':episode,'step':t,'eps':current_epsilon,'reward':Reward[episode % self.args.window],'q': Qfunc[episode % self.args.window]})
        return self.Q_net    
            

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
        return map(lambda x: Variable(torch.cat(x, 0).to(device)), zip(*batch))
