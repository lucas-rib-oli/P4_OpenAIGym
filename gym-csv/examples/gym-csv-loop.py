#!/usr/bin/env python

import gym
import gym_csv

import numpy as np 
import time

# X points down (rows)(v), Y points right (columns)(>), Z would point outwards.
LEFT = 0  # < Decrease Y (column)
DOWN = 1  # v Increase X (row)
RIGHT = 2 # > Increase Y (column)
UP = 3    # ^ Decrease X (row)

SIM_PERIOD_MS = 500.0

# Episodios
EPISODES = 5000

def main ():
    env = gym.make('csv-v0')
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
        
        


if __name__ == '__main__':
    main () 
