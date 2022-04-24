import gym
from gym.wrappers import Monitor
from simple_dqn_torch_2020 import Agent
from utils import plotLearning
import numpy as np
import wandb

if __name__ == '__main__':
    wandb.init(project='proj4',entity='aukkawut',monitor_gym=True)
    env = Monitor(gym.make('LunarLander-v2'),'./video',resume = True)
    agent = Agent(gamma=0.99, epsilon=1.0, batch_size=64, n_actions=4, eps_end=0.01,
                  input_dims=[8], lr=0.001)
    scores, eps_history = [], []
    n_games = 1000
    
    for i in range(n_games):
        score = 0
        done = False
        observation = env.reset()
        Q = []
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            agent.store_transition(observation, action, reward, 
                                    observation_, done)
            q = agent.learn()
            if q != None:
                Q.append(q)
            else:
                Q.append(0)
            observation = observation_
        scores.append(score)
        eps_history.append(agent.epsilon)
        Qval = np.mean(Q)

        avg_score = np.mean(scores[-100:])

        print('episode ', i, 'score %.2f' % score,
                'average score %.2f' % avg_score,
                'epsilon %.2f' % agent.epsilon)
        wandb.log({'episode ': i, 'reward': score,
                'Average Reward':avg_score,
                'eps' :agent.epsilon,'q':Qval})
    x = [i+1 for i in range(n_games)]
    filename = 'lunar_lander.png'
    plotLearning(x, scores, eps_history, "vanilla_dqn_1000")

