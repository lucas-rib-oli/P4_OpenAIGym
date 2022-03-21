#!/usr/bin/env python

import gym
import gym_csv

# env = gym.make('csv-v0')
env = gym.make('csv-pygame-v0')
env.reset()
env.render()
