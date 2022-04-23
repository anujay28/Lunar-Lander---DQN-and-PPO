import gym, time
from gym.wrappers import Monitor
import numpy as np
import matplotlib.pyplot as plt
from dueling_dqn_torch import Agent
from utils import plotLearning
import wandb


def train(load_checkpoint = False):
    wandb.init(monitor_gym=True)
    agent = Agent(gamma=wandb.config.gamma, epsilon=wandb.config.epsilon, alpha=wandb.config.alpha,
                  input_dims=[8], n_actions=4, mem_size=wandb.config.mem_size, eps_min=wandb.config.eps_min,
                  batch_size=wandb.config.batch_size, eps_dec=wandb.config.eps_dec, replace=wandb.config.replace, 
                  num_layers=wandb.config.num_layers, hidden_dim=wandb.config.hidden_dim)
    if load_checkpoint:
        agent.load_models()
    env = Monitor(gym.make('LunarLander-v2'),'./video', resume = True)
    num_games = 1000
    filename = 'LunarLander-Dueling-128-128-Adam-lr0005-replace100.png'
    scores = []
    eps_history = []
    n_steps = 0

    for i in range(num_games):
        done = False
        observation = env.reset()
        score = 0

        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            n_steps += 1
            score += reward
            agent.store_transition(observation, action,
                                    reward, observation_, int(done))
            agent.learn()

            observation = observation_


        scores.append(score)
        avg_score = np.mean(scores[max(0, i-100):(i+1)])
        print('episode: ', i,'score %.1f ' % score,
             ' average score %.1f' % avg_score,
            'epsilon %.2f' % agent.epsilon)
        wandb.log({'reward':score, 'episode':i,'eps':agent.epsilon})
        if i > 0 and i % 10 == 0:
            agent.save_models()

        eps_history.append(agent.epsilon)

    x = [i+1 for i in range(num_games)]
    plotLearning(x, scores, eps_history, filename)

def manual_train(gamma=0.999,epsilon = 0.99,alpha = 1e-3,mem_size = 1000000,eps_min = 1e-3,batch_size = 8,eps_dec = 5e-5,replace=1000,hidden_dim= 64,num_layers=1,load_checkpoint = False):
    wandb.init(project='proj4',entity='aukkawut',monitor_gym=True)
    agent = Agent(gamma=gamma, epsilon=epsilon, alpha=alpha,
                  input_dims=[8], n_actions=4, mem_size=mem_size, eps_min=eps_min,
                  batch_size=batch_size, eps_dec=eps_dec, replace=replace, 
                  num_layers=num_layers, hidden_dim=hidden_dim)
    if load_checkpoint:
        agent.load_models()
    env = Monitor(gym.make('LunarLander-v2'),'./video', resume = True)
    num_games = 2000
    filename = 'LunarLander-Dueling-128-128-Adam-lr0005-replace100.png'
    scores = []
    eps_history = []
    n_steps = 0

    for i in range(num_games):
        done = False
        observation = env.reset()
        score = 0

        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            n_steps += 1
            score += reward
            agent.store_transition(observation, action,
                                    reward, observation_, int(done))
            agent.learn()

            observation = observation_


        scores.append(score)
        avg_score = np.mean(scores[max(0, i-100):(i+1)])
        print('episode: ', i,'score %.1f ' % score,
             ' average score %.1f' % avg_score,
            'epsilon %.2f' % agent.epsilon)
        wandb.log({'reward':score, 'episode':i,'eps':agent.epsilon})
        #if i > 0 and i % 10 == 0:
        #    agent.save_models()

        eps_history.append(agent.epsilon)

    x = [i+1 for i in range(num_games)]
    plotLearning(x, scores, eps_history, filename)

if __name__ == '__main__':
    sweep_config = {"name" : "param_tuning",
  "method" : "random",
  'parameters' : {
    'gamma':{
        'distribution': 'uniform',
        'min': 0,
        'max': 1,
    },
    'epsilon':{
        'distribution': 'uniform',
        'min': 0.5,
        'max': 1,
    },
    'eps_min':{
        'distribution': 'uniform',
        'min': 1e-9,
        'max': 0.0001,
    },
    'alpha':{
        'distribution': 'uniform',
        'min': 1e-10,
        'max': 1e-2,
    },
    'mem_size':{
        'values': [1e4,5e4,1e5,5e5,1e6,5e6,1e7,5e7]
    },
    'replace':{
        'values': [100,500,1000]
    },
    'batch_size':{
        'values': [2,4,8,16,32,128,256]
    },
    'eps_dec':{
        'distribution': 'uniform',
        'min': 1e-6,
        'max': 1e-4,
    },
    'num_layers':{
        'values': [1,2,3,4,5]
    },
    'hidden_dim':{
        'values': [16,32,64,128,256,512,1024]
    }
}
}
sweep_id = wandb.sweep(sweep_config)
wandb.agent(sweep_id, function=train,project='proj4',entity='aukkawut',count=100)
#manual_train()
