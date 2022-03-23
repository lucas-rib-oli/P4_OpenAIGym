#!/usr/bin/env python

import gym
from paramiko import Agent
import gym_csv

import numpy as np 
import time

import sys, tty, termios
import argparse

def str_to_bool(value):
    if value.lower() in {'false', 'n'}:
        return False
    elif value.lower() in {'true', 'y'}:
        return True

# ====================================================================================================================================================================== #

## Ansi colors
class bcolors: 
    RESET = '\033[0m'
    BLACK = '\033[30m'     
    RED = '\033[31m'     
    GREEN = '\033[32m'      
    YELLOW = '\033[33m'    
    BLUE = '\033[34m'      
    MAGENTA = '\033[35m'     
    CYAN = '\033[36m'     
    WHITE = '\033[37m'      
    BOLDBLACK = '\033[1m\033[30m'      
    BOLDRED = '\033[1m\033[31m'    
    BOLDGREEN = '\033[1m\033[32m'      
    BOLDYELLOW = '\033[1m\033[33m'      
    BOLDBLUE = '\033[1m\033[34m'      
    BOLDMAGENTA = '\033[1m\033[35m'      
    BOLDCYAN = '\033[1m\033[36m'      
    BOLDWHITE = '\033[1m\033[37m'

# ====================================================================================================================================================================== #
parser = argparse.ArgumentParser()
parser.add_argument("-a", "--algorithm", type=str, default="q_learning", help="vai (Value Iteration Algorithm) / q_learning")
parser.add_argument('--use_pygame', type=str_to_bool, default=False, help='Utilizar el entorno simulado en imagen')
parser.add_argument('--episodes', type=int, default=650, help='Numero de episodios a usar en los algoritmos')
args = parser.parse_args()
# ====================================================================================================================================================================== #

# X points down (rows)(v), Y points right (columns)(>), Z would point outwards.
LEFT = 0  # < Decrease Y (column)
DOWN = 1  # v Increase X (row)
RIGHT = 2 # > Increase Y (column)
UP = 3    # ^ Decrease X (row)

SIM_PERIOD_MS = 10.0

# Episodios
EPISODES = args.episodes

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
            new_state, reward, done,_ = env.step(action)
            state = new_state
            rAll += reward
            state = new_state
            env.render()
            print(bcolors.BOLDGREEN + "new_state: " + bcolors.BOLDWHITE + str(new_state) + bcolors.BOLDGREEN + ", reward: " + bcolors.BOLDWHITE + str(reward) + bcolors.BOLDGREEN + ", done: " + bcolors.BOLDWHITE + str(done) + bcolors.BOLDGREEN + ", episode: " +  bcolors.BOLDWHITE +str (i))
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
    rev_list = [] # rewards per episode calculate

    # 3. Q-learning Algorithm
    print("Computing Q-Table...")
    for i in range(EPISODES):
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
            s1,r,d,_ = env.step(a)
            #Update Q-Table with new knowledge
            Q[s,a] = Q[s,a] + eta*(r + gma*np.max(Q[s1,:]) - Q[s,a])
            rAll += r
            s = s1
            print( bcolors.BOLDGREEN + "new_state: " + bcolors.BOLDWHITE + str(s1) + ", " + bcolors.BOLDGREEN + "reward: " + bcolors.BOLDWHITE + "{:.2f}".format ( r ) + ", " + bcolors.BOLDGREEN + "done: " + bcolors.BOLDWHITE + str(d) + ", " + bcolors.BOLDGREEN + "episode: " + bcolors.BOLDWHITE + str (i) )
            time.sleep(SIM_PERIOD_MS/1000.0)
            
            if d == True:
                break
            
        rev_list.append(rAll)
        env.render()
    print("Reward Sum on all episodes " + str(sum(rev_list)/EPISODES))
    print("Final Values Q-Table")
    print(Q)

    print ("===============================")
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
        time.sleep( 0.5 )
    # Code will stop at d == True, and render one state before it

# ====================================================================================================================================================================== #

def value_iteration_algorithm (env):
    s = env.reset()
    # Value iteration algorithm
    NUM_ACTIONS = env.action_space.n
    NUM_STATES = env.observation_space.n
    V = np.zeros([NUM_STATES]) # Valor para cada estado
    Pi = np.zeros([NUM_STATES], dtype=int)  # Politica que seguimos actualizando para conseguir la mejor política 
    gamma = 0.9 # Factor de descuento
    significant_improvement = 0.01

    def best_action_value(s):
        # Encuentra la accion de mayor valor (max_a) en el estado 's'
        best_a = None
        best_value = -9999.0

        # Recorrer todas las acciones posibles para encontrar la mejor accion actual
        for a in range (0, NUM_ACTIONS):
            env.env.s = s
            state_new, reward, done, _ = env.step(a) 
            v = reward + gamma * V[state_new]
            if v > best_value:
                best_value = v
                best_a = a
        return best_a

    iteration = 0
    while True:
        # biggest_change se denomina con el símbolo matemático delta en las ecuaciones
        biggest_change = 0
        for s in range (0, NUM_STATES):
            old_v = V[s]
            action = best_action_value(s) # elegir una accion con la mayor recompensa futura
            env.env.s = s 
            state_new, reward, done, _ = env.step(action) # Step
            V[s] = reward + gamma * V[state_new] # Actualizar el valor del estado mediante la ecuación de Bellman
            Pi[s] = action
            biggest_change = max(biggest_change, np.abs(old_v - V[s]))
            env.render()
            print(bcolors.BOLDGREEN + "new_state: " + bcolors.BOLDWHITE + str(state_new) + ", " + bcolors.BOLDGREEN + "reward: " + bcolors.BOLDWHITE + "{:.2f}".format ( reward ) + ", " + bcolors.BOLDGREEN + "done: " + bcolors.BOLDWHITE + str(done))
        iteration += 1
        if biggest_change < significant_improvement:
            print (iteration,' iterations done')
            break
    
    print ("===============================")
    print("Press any key to run solution...")
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    # Mapa resuelto
    rew_tot=0
    obs= env.reset()
    env.render()
    done=False
    while done != True: 
        action = Pi[obs]
        obs, reward, done, info = env.step(action) # Step
        rew_tot = rew_tot + reward
        env.render()
        time.sleep( 0.5 )
    # Print the reward
    print("Reward: %r" % rew_tot)

# ====================================================================================================================================================================== #

if __name__ == '__main__':
    if (args.use_pygame):
        env = gym.make('csv-pygame-v0')
    else:
        env = gym.make('csv-v0')

    if ( args.algorithm == "vai" ):
        value_iteration_algorithm ( env )
    else:
        q_learning (env)
 
