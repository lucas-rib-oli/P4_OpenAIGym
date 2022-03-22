# https://github.com/openai/gym/blob/849da90011f877853589c407c170d3c07f680d52/gym/core.py
# https://github.com/openai/gym/blob/849da90011f877853589c407c170d3c07f680d52/gym/envs/toy_text/frozen_lake.py
# https://github.com/openai/gym/blob/849da90011f877853589c407c170d3c07f680d52/gym/envs/toy_text/discrete.py
# https://www.oreilly.com/learning/introduction-to-reinforcement-learning-and-openai-gym
# https://towardsdatascience.com/reinforcement-learning-with-openai-d445c2c687d2

import gym
from gym.envs.toy_text import discrete
import numpy as np

# X points down (rows)(v), Y points right (columns)(>), Z would point outwards.
LEFT = 0  # < Decrease Y (column)
DOWN = 1  # v Increase X (row)
RIGHT = 2 # > Increase Y (column)
UP = 3    # ^ Decrease X (row)

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

class CsvEnv(discrete.DiscreteEnv):
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
        initX = 1
        initY = 1
        goalX = 12
        goalY = 14
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

        def euclidean_distance (x1, y1, x2, y2):
            return np.sqrt ( np.power ( (x2 - x1), 2 ) + np.power ( (y2 - y1), 2 ) )
        
        for row in range(nrow): # Fill in P[s][a] transitions and rewards
            for col in range(ncol):
                s = to_s(row, col)
                for a in range(4):
                    li = P[s][a] # In Python this is not a deep copy, therefore we are appending to actual P[s][a] !!
                    tag = self.inFile[row][col]
                    
                    euclidean_cost = euclidean_distance ( row, col, goalX, goalY )
                    if tag == 3: # goal
                        li.append((1.0, s, 1.0, True)) # (probability, nextstate, reward, done)
                    elif tag == 1: # wall
                        li.append((1.0, s, -500.0 - euclidean_cost, True)) # (probability, nextstate, reward, done) # Some algorithms fail with reward -float('inf')
                    else: # e.g. tag == 0
                        newrow, newcol = inc(row, col, a)
                        newstate = to_s(newrow, newcol)
                        li.append((1.0, newstate, -euclidean_cost, False)) # (probability, nextstate, reward, done)

        super(CsvEnv, self).__init__(nS, nA, P, isd)

    # DO NOT UNCOMMENT, LET 'discrete.DiscreteEnv' IMPLEMENT IT!
    #def step(self, action):
    #    print('CsvEnv.step', action)

    # DO NOT UNCOMMENT, LET 'discrete.DiscreteEnv' IMPLEMENT IT!
    #def reset(self):
    #    print('CsvEnv.reset')

    def render(self, mode='human'):
        #print('CsvEnv.render', mode)
        row, col = self.s // self.ncol, self.s % self.ncol # Opposite of ravel().
        viewer = np.copy(self.inFile) # Force a deep copy for rendering.
        viewer[row, col] = 2

        for row in viewer:
            for cell in row:
                if cell == 1:
                    print (bcolors.BOLDRED + str (int(cell)) + bcolors.RESET + " ", end = '')
                elif cell == 2:
                    print (bcolors.BOLDBLUE + str (int(cell)) + bcolors.RESET + " ", end = '')
                elif cell == 3:
                    print (bcolors.BOLDYELLOW + str (int(cell)) + bcolors.RESET + " ", end = '') 
                else:
                    print (bcolors.BOLDGREEN + str (int(cell)) + bcolors.RESET + " ", end = '') 
            print ()
        # print(viewer)
    
    def close(self):
        print('CsvEnv.close')
