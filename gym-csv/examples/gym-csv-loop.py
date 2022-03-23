#!/usr/bin/env python

import gym
import gym_csv

import numpy as np 
import time

import sys, tty, termios
import argparse

# https://www.kaggle.com/code/charel/learn-by-example-reinforcement-learning-with-gym/notebook
# https://programmerclick.com/article/1942403954/

def str_to_bool(value):
    if value.lower() in {'false', 'n'}:
        return False
    elif value.lower() in {'true', 'y'}:
        return True

# ====================================================================================================================================================================== #
parser = argparse.ArgumentParser()
parser.add_argument("-a", "--algorithm", type=str, default="q_learning", help="vai (Value Iteration Algorithm) / q_learning")
parser.add_argument('--use_pygame', type=str_to_bool, default=False, help='Utilizar el entorno simulado en imagen')
args = parser.parse_args()
# ====================================================================================================================================================================== #

# X points down (rows)(v), Y points right (columns)(>), Z would point outwards.
LEFT = 0  # < Decrease Y (column)
DOWN = 1  # v Increase X (row)
RIGHT = 2 # > Increase Y (column)
UP = 3    # ^ Decrease X (row)

SIM_PERIOD_MS = 500.0

# Episodios
EPISODES = 5000

def main (env):
    state = env.reset()
    print("state: "+str(state))
    env.render()
    time.sleep(0.5)

    rev_list = []
    action = LEFT
    for i in range(EPISODES):
        # Reset environment
        state = env.reset()
        rAll = 0
        done = False
        #The Q-Table learning algorithm
        for t in range(100):           
            
            action = env.action_space.sample()
            print ('action: ', action)
            new_state, reward, done,_ = env.step(action)
            state = new_state
            rAll += reward
            state = new_state
            env.render()
            print("new_state: "+str(new_state)+", reward: "+str(reward)+", done: "+str(done))
            time.sleep(SIM_PERIOD_MS/1000.0)

            if done == True:
                break
        if action == 3:
            action = 0
        else:
            action += 1
        rev_list.append(rAll)
        env.render()

def q_learning (env):
    # 1. Create Q-table structure
    Q = np.zeros([env.observation_space.n,env.action_space.n])

    # 2. Parameters of Q-learning
    eta = .628
    gma = .9
    epis = 5000
    rev_list = [] # rewards per episode calculate

    # 3. Q-learning Algorithm
    print("Computing Q-Table...")
    for i in range(epis):
        # Reset environment
        s = env.reset()
        rAll = 0
        d = False
        j = 0
        #The Q-Table learning algorithm
        while j < 99:
            env.render()
            j+=1
            # Choose action from Q table
            a = np.argmax(Q[s,:] + np.random.randn(1,env.action_space.n)*(1./(i+1)))
            #Get new state & reward from environment
            #print("a",a)
            print ("action: ", a)
            s1,r,d,_ = env.step(a)
            #Update Q-Table with new knowledge
            Q[s,a] = Q[s,a] + eta*(r + gma*np.max(Q[s1,:]) - Q[s,a])
            rAll += r
            s = s1
            print("new_state: "+str(s1)+", reward: "+str(r)+", done: "+str(d))
            time.sleep(SIM_PERIOD_MS/1000.0)
            
            if d == True:
                break
            
        rev_list.append(rAll)
        env.render()
    print("Reward Sum on all episodes " + str(sum(rev_list)/epis))
    print("Final Values Q-Table")
    print(Q)

    print("Press any key to run solution...")

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    # Reset environment
    s = env.reset()
    d = False
    # The Q-Table learning algorithm
    while d != True:
        env.render()
        # Choose action from Q table
        a = np.argmax(Q[s,:] + np.random.randn(1,env.action_space.n)*(1./(i+1)))
        #Get new state & reward from environment
        s1,r,d,_ = env.step(a)
        #Update Q-Table with new knowledge
        Q[s,a] = Q[s,a] + eta*(r + gma*np.max(Q[s1,:]) - Q[s,a])
        s = s1
        time.sleep(SIM_PERIOD_MS/1000.0)
    # Code will stop at d == True, and render one state before it

def value_iteration_algorithm (env):
    s = env.reset()
    # Value iteration algorithm
    NUM_ACTIONS = env.action_space.n
    NUM_STATES = env.observation_space.n
    V = np.zeros([NUM_STATES]) # The Value for each state
    Pi = np.zeros([NUM_STATES], dtype=int)  # Our policy with we keep updating to get the optimal policy
    gamma = 0.9 # discount factor
    significant_improvement = 0.01

    def best_action_value(s):
        # finds the highest value action (max_a) in state s
        best_a = None
        best_value = float('-inf')

        # loop through all possible actions to find the best current action
        for a in range (0, NUM_ACTIONS):
            env.env.s = s
            s_new, rew, done, info = env.step(a) #take the action
            v = rew + gamma * V[s_new]
            if v > best_value:
                best_value = v
                best_a = a
        return best_a

    iteration = 0
    while True:
        # biggest_change is referred to by the mathematical symbol delta in equations
        biggest_change = 0
        for s in range (0, NUM_STATES):
            old_v = V[s]
            action = best_action_value(s) #choosing an action with the highest future reward
            env.env.s = s # goto the state
            s_new, rew, done, info = env.step(action) #take the action
            V[s] = rew + gamma * V[s_new] #Update Value for the state using Bellman equation
            Pi[s] = action
            biggest_change = max(biggest_change, np.abs(old_v - V[s]))
            env.render()
            print("new_state: "+str(s_new)+", reward: "+str(rew)+", done: "+str(done))
        iteration += 1
        if biggest_change < significant_improvement:
            print (iteration,' iterations done')
            break
    
    # Let's see how the algorithm solves the taxi game
    rew_tot=0
    obs= env.reset()
    env.render()
    done=False
    while done != True: 
        action = Pi[obs]
        obs, rew, done, info = env.step(action) #take step using selected action
        rew_tot = rew_tot + rew
        env.render()
    #Print the reward of these actions
    print("Reward: %r" % rew_tot)

# =========================================================================================================== #

if __name__ == '__main__':
    if (args.use_pygame):
        env = gym.make('csv-pygame-v0')
    else:
        env = gym.make('csv-v0')

    if ( args.algorithm == "vai" ):
        value_iteration_algorithm ( env )
    else:
        q_learning (env)
