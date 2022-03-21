# https://github.com/openai/gym/blob/849da90011f877853589c407c170d3c07f680d52/gym/core.py
# https://github.com/openai/gym/blob/849da90011f877853589c407c170d3c07f680d52/gym/envs/toy_text/frozen_lake.py
# https://github.com/openai/gym/blob/849da90011f877853589c407c170d3c07f680d52/gym/envs/toy_text/discrete.py
# https://www.oreilly.com/learning/introduction-to-reinforcement-learning-and-openai-gym
# https://towardsdatascience.com/reinforcement-learning-with-openai-d445c2c687d2

import gym
from gym.envs.toy_text import discrete
import numpy as np
import pygame

# X points down (rows)(v), Y points right (columns)(>), Z would point outwards.
LEFT = 0  # < Decrease Y (column)
DOWN = 1  # v Increase X (row)
RIGHT = 2 # > Increase Y (column)
UP = 3    # ^ Decrease X (row)

SCREEN_WIDTH, SCREEN_HEIGHT = 640, 480
COLOR_BACKGROUND = (0, 0, 0)
COLOR_WALL = (255, 255, 255)
COLOR_ROBOT = (255, 0, 0)

class CsvPyGameEnv(discrete.DiscreteEnv):
    """
    Has the following members
    - nS: number of states
    - nA: number of actions
    - P: transitions (*)
    - isd: initial state distribution (**)
    (*) dictionary dict of dicts of lists, where
      P[s][a] == [(probability, nextstate, reward, done), ...]
    (**) list or array of length nS
    """
    metadata = {'render.modes': ['human']}

    def __init__(self):
        # Remember: X points down, Y points right, thus Z points outwards.
        # hard-coded vars (begin)
        inFileStr = 'map1.csv'
        initX = 2
        initY = 2
        goalX = 7
        goalY = 2
        # hard-coded vars (end)
        self.inFile = np.genfromtxt(inFileStr, delimiter=',')
        self.inFile[goalX][goalY] = 3 # The goal (3) is fixed, so we paint it, but the robot (2) moves, so done at render().
        self.nrow, self.ncol = nrow, ncol = self.inFile.shape
        nS = nrow * ncol # nS: number of states
        nA = 4 # nA: number of actions
        P = {s : {a : [] for a in range(nA)} for s in range(nS)} # transitions (*), filled in at the for loop below.
        isd = np.zeros((nrow, ncol)) # initial state distribution (**)
        isd[initX][initY] = 1
        isd = isd.astype('float64').ravel() # ravel() is like flatten(). However, astype('float64') is just in case.

        def to_s(row, col):
            return row*ncol + col
        
        def inc(row, col, a): # Assures we will not go off limits.
            if a == LEFT:
                col = max(col-1,0)
            elif a == DOWN:
                row = min(row+1,nrow-1)
            elif a == RIGHT:
                col = min(col+1,ncol-1)
            elif a == UP:
                row = max(row-1,0)
            return (row, col)

        for row in range(nrow): # Fill in P[s][a] transitions and rewards
            for col in range(ncol):
                s = to_s(row, col)
                for a in range(4):
                    li = P[s][a] # In Python this is not a deep copy, therefore we are appending to actual P[s][a] !!
                    tag = self.inFile[row][col]
                    if tag == 3: # goal
                        li.append((1.0, s, 1.0, True)) # (probability, nextstate, reward, done)
                    elif tag == 1: # wall
                        li.append((1.0, s, -500.0, True)) # (probability, nextstate, reward, done) # Some algorithms fail with reward -float('inf')
                    else: # e.g. tag == 0
                        newrow, newcol = inc(row, col, a)
                        newstate = to_s(newrow, newcol)
                        li.append((1.0, newstate, 0.0, False)) # (probability, nextstate, reward, done)

        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), 0, 32)

        super(CsvPyGameEnv, self).__init__(nS, nA, P, isd)

    # DO NOT UNCOMMENT, LET 'discrete.DiscreteEnv' IMPLEMENT IT!
    #def step(self, action):
    #    print('CsvEnv.step', action)

    # DO NOT UNCOMMENT, LET 'discrete.DiscreteEnv' IMPLEMENT IT!
    #def reset(self):
    #    print('CsvEnv.reset')

    def render(self, mode='human'):
        #print('CsvEnv.render', mode)
        row, col = self.s // self.ncol, self.s % self.ncol # Opposite of ravel().
        #viewer = np.copy(self.inFile) # Force a deep copy for rendering.
        #viewer[row, col] = 2
        #print viewer
        self.screen.fill(COLOR_BACKGROUND)
        for iX in range(self.nrow):
            #print "iX:",iX
            for iY in range(self.ncol):
                #print "* iY:",iY

                pixelX = SCREEN_WIDTH/self.nrow
                pixelY = SCREEN_HEIGHT/self.ncol

                #-- Skip box if map indicates a 0
                if self.inFile[iX][iY] == 0:
                    continue
                if self.inFile[iX][iY] == 1:
                    pygame.draw.rect(self.screen, COLOR_WALL,
                                     pygame.Rect( pixelX*iX, pixelY*iY, pixelX, pixelY ))
                if self.inFile[iX][iY] == 3:
                    pygame.draw.rect(self.screen, (0,255,0),
                                     pygame.Rect( pixelX*iX, pixelY*iY, pixelX, pixelY ))
                robot = pygame.draw.rect(self.screen, COLOR_ROBOT,
                                         pygame.Rect( pixelX*row+pixelX/4.0, pixelY*col+pixelY/4.0, pixelX/2.0, pixelY/2.0 ))
        pygame.display.flip()

    def close(self):
        print('CsvEnv.close')
