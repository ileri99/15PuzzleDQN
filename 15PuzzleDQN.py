#----------------------------------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

#import tensorflow as tfl
from keras.models import Sequential
from keras.layers import Dense, Flatten, Activation
from keras.callbacks import TensorBoard
from keras.optimizers import Adam
import time
#import torchvision.transforms as T
#from keras.models import Sequential

import random
import math
import numpy as np
from copy import deepcopy
#this import provides a progress bar, see how far along for loop currently is
from tqdm import tqdm
from collections import deque
import matplotlib.pyplot as plt

#----------------------------------------------------------------------------------------------------------
#dont worry about this class. it's used to prevent the network from continuously printing as its learning 

#----------------------------------------------------------------------------------------------------------


class agent:
    def __init__(self):

        self.board = [ [ (i * 4) + j for j in range(1,5)] for i in range(0,4) ]
        self.board[3][3] = 0

        binaryBoard = [ [0 for i in range(0,16)] for j in range(0,15)]
        for i in range(0,15):
            binaryBoard[i][i] = 1

        self.blankX = None
        self.blankY = None
        self.board_copy = None
        #self.board_copy = self.board.copy()
        #used to check if goal state has been reached
        self.goalState = [ [ (i * 4) + j for j in range(1,5)] for i in range(0,4) ]
        self.goalState[3][3] = 0
        self.goalState = np.array(self.goalState)
        self.nextState = []
        self.reward = 0
        self.discount = 0.95

        #used to keep track of total episode reward
        self.episodeReward = []
        self.averageReward = []

        #change the arrays to numpy arrays, needed when implementing neural network
        self.board = np.array(self.board)
        self.goalState = np.array(self.goalState)

        #once filled it'll automatically delete oldest member to add new one
        self.memory = deque(maxlen = 25000)
        self.position = 0

        #self.train = []
        self.out = []#[i for i in range(1,5)]

        #this model is training after every step 
        self.model = self.createModel()
        
        #target counter controls when its time to update targetModel
        self.targetCounter = 0
        self.target = 5

        #also need a target model. this is what we .predict against
        self.targetModel = self.createModel()
        self.targetModel.set_weights(self.model.get_weights())

        self.prevBoard = None

        #epsilon will decay after every episode
        self.epsilon = 1
        self.epsilonDecay = 0.999975
        self.minEpsilon = 0.2
        self.store = 0

        #will have a lot of print into log files slowing everything down, look for modifiedTensorBoard code to fix
        self.name = "daddys first neural network"
        self.storeSuccess = []


    #--------------------------------------------------------------------------------------------------------------
    #this section is code for the 24 puzzle game

    #function to restart the game
    def reset(self):
        self.board = [ [ (i * 4) + j for j in range(1,5)] for i in range(0,4) ]
        self.board[3][3] = 0
        self.board = np.array(self.board)
        self.reward = 0


    def printBoard(self):

        for i in range(0, len(self.board)):
            print(self.board[i][0] , "  " ,  self.board[i][1] , "  " ,  self.board[i][2] , "  " ,  self.board[i][3])# , "  " ,  self.board[i][4])
            print()
    
    #always called just before doing anything with the blank state
    def getBlankPosition(self):
        for y in range(0,4):
            for x in range(0,4):
                if self.board[y][x] == 0:
                    #must always know location of blank state before making a move
                    self.blankX = x
                    self.blankY = y
    #---------------------------------------------------------------------------
    #function to randomise the puzzle
    def makeMove(self, move):

        self.getBlankPosition()

        if move == 0:
            #first need to check if the move is legal
            if self.blankX == 0:
                #this boolean will never be true will agent is actively solving puzzle
                
                return
            self.leftMove(self.blankY, self.blankX)
            #print("leftmove")
        
        elif move == 1:
            if self.blankY == 0:
                
                return
            self.upMove(self.blankY, self.blankX)
            #print("upmove")

        elif move == 2:
            if self.blankX == 3:
                
                return
            self.rightMove(self.blankY, self.blankX)
            #print("rightmove")

        elif move == 3:
            if self.blankY == 3:
                
                return
            self.downMove(self.blankY, self.blankX)
            #print("downmove")

    def randomise(self, lb, ub):
        x = random.randint(lb, ub)
        for i in range(x):
            self.makeMove(random.randint(0,3))


    #------------------------------------------------------------------------------
    #functions have curernt position of 0 as inputs, then changes position of 0
    def leftMove(self, y, x):
        if x == 0:
            return
        
        store = self.board[y][x]
        self.board[y][x] = self.board[y][x - 1]
        self.board[y][x - 1] = store
    
    def upMove(self, y, x):
        if y == 0:
            return
        store = self.board[y][x]
        self.board[y][x] = self.board[y - 1][x]
        self.board[y - 1][x] = store

    def rightMove(self, y, x):
        if x == 3:
            return
        store = self.board[y][x]
        self.board[y][x] = self.board[y][x + 1]
        self.board[y][x + 1] = store

    def downMove(self, y, x):
        if y == 3:
            return
        store = self.board[y][x]
        self.board[y][x] = self.board[y + 1][x]
        self.board[y + 1][x] = store

    #-----------------------------------------------------------------------------
    
    def checkSolve(self):
        if (self.board == self.goalState).all():
            self.reward = 50
            return True
        self.reward = -1
        return False
    
    #function to calculate the manhattan distance. never used in DQN
    def findDistance(self):

        #function to store distances of numbers. will be summed after all loops
        distances = []

        #start finding distances of numbers, starting with 1. ignores blank tile
        for i in range(1,25):

            for y in range(0,5):
                for x in range(0,5):
                    if self.board[y][x] == i:
                        #line below finds distance between current position and goal position of specific number
                        y_dist = abs(y - ((i - 1)//5))

                        if i % 5 == 0:
                            x_dist = abs((x + 1) - 5)
                        else:
                            x_dist = abs((x + 1) - (i%5))

                        distances.append(x_dist + y_dist)

        self.nextState.append(sum(distances))

    
    #this function makes move to produce new state. if state isnt legal return -1. never used in DQN.
    def findState(self, move):

        self.getBlankPosition()
        self.board_copy = deepcopy(self.board)

        if move == 0:
            if self.blankX == 0:
                #not a legal move, so append -1
                self.nextState.append(1000) #so big this number is irrelevant
                return

            self.leftMove(self.blankY, self.blankX)
            #after making move calculate the manhatten distance of this state
            self.findDistance()
        
        elif move == 1:
            if self.blankY == 0:
                self.nextState.append(1000)
                return

            self.upMove(self.blankY, self.blankX)
            self.findDistance()

        elif move == 2:
            if self.blankX == 4:
                self.nextState.append(1000)
                return
            self.rightMove(self.blankY, self.blankX)
            self.findDistance()
            
        elif move == 3:
            if self.blankY == 4:
                self.nextState.append(1000)
                return
            self.downMove(self.blankY, self.blankX)
            self.findDistance()

    #--------------------------------------------------------------------------------------------------------------
    #this section is neural network code

    def createModel(self):

        #this line just means things go in direct order, going forward, come back to this for back tracing 
        model = Sequential()

        #this will convert 2d array to 1d array before being passed to neural network
        model.add(Flatten( input_shape = (15,16) ))#, 1) )) #self.board.shape ))

        #first hidden layer. currently has 128 units, but considering changing to 25, feels right
        model.add(Dense(256, activation = "relu" ))

        #add second hidden layer
        model.add(Dense(256, activation = "relu"))

        #model.add(Dense(512, activation = "relu"))
        
        #model.add(Dense(256, activation = "relu"))

        #model.add(Dense(256, activation = "relu"))

        #last layer, has 4 outputs for each possible move
        model.add(Dense(4, activation = "linear"))

        #try mse instead of adam
        model.compile(optimizer = Adam(lr = 0.0001), loss = 'mse', metrics = ['accuracy'])

        return model

        #model.fit(self.train, self.out, epochs = 3)

    def encode(self, board):
        
        for x in range(1, 16):
            i = (np.argwhere(self.board == x)[0][0] * 4) + np.argwhere(self.board == x)[0][1]
            board[x - 1][i] = 1

        return board

    def getQValues(self, state):

        temp = [ [0 for i in range(0,16)] for j in range(0,15)]
        temp = self.encode(temp)
        temp = np.array(temp)

        return self.model.predict(temp.reshape(-1, *temp.shape))[0]
        
    def updateMemory(self,transition):

        self.memory.append(transition)

    def train(self, terminal, step):
        
        #only start training once its collected enough starting data
        minMemorySize = 1000

        if len(self.memory) < minMemorySize:
            return

        #im pretty sure this never gets cleared, make sure youre not noting everything
        sampleSize = 50

        #takes subset from short term memory to train network
        tempMemory = random.sample(self.memory, sampleSize)

        #using these states its gonna find q values. may benefit to normalise (divide by 24).
        #print(tempMemory[0][0])
        tempStates = np.array([t[0] for t in tempMemory])
        tempQStates = self.model.predict(tempStates)

        newTempStates = np.array([t[2] for t in tempMemory])
        futureQStates = self.targetModel.predict(newTempStates)

        x = []
        y = []


        #for loop used to calculate prediction of max value of new state
        for index, (prevState, action, state, reward, terminal) in enumerate(tempMemory):
            
            if not terminal:
                maxFutureQ = np.max(futureQStates[index])
                newQ = reward + (self.discount * maxFutureQ)
            else:
                newQ = reward

            #update Q value for given state
            tempQs = tempQStates[index]
            tempQs[action] = newQ

            #add this to our training data
            x.append(prevState)
            y.append(tempQs)
        

        self.model.fit(np.array(x), np.array(y), batch_size = sampleSize, verbose=0, shuffle=False)
        #print("updating the network")

        #determines whether its time to update target model
        if terminal:
            #print("solved the game")
            self.targetCounter += 1

        if self.targetCounter > self.target:
            self.targetModel.set_weights(self.model.get_weights())
            self.targetCounter = 0

    #--------------------------------------------------------------------------------------------------------------



def train(l, u, episodes):

    for episode in tqdm(range(1, episodes)):
        
        episodeReward = 0
        step = 0
        play1.reset()
        
        #takes 80 moves max to solve. taking into account chance of collision 200
        #has higher chance to reach weirder states
        play1.randomise(l, u)
        counter = 0

        #theres a chance first randomise function will leave game solved
        while (play1.board == play1.goalState).all():
            play1.randomise(l, u)

        terminal = False

        while not terminal:

            counter +=1 
                
            #find the best action of the current state.
            if random.random() > play1.epsilon:
                #print("current board equals: ", play1.board)
                action = np.argmax(play1.getQValues(play1.board))
            else:
                action = random.randint(0,3)

            #network requires current board to be stored before performing an action
            play1.prevBoard = [ [0 for i in range(0,16)] for j in range(0,15)]
            play1.prevBoard = play1.encode(play1.prevBoard)
            play1.prevBoard = np.array(play1.prevBoard)

            play1.makeMove(action)

            #this should be one-hot encoded version
            temp = [ [0 for i in range(0,16)] for j in range(0,15)]
            temp = play1.encode(temp)
            temp = np.array(temp)
            
            
            #this if statement also updates self.reward
            if play1.checkSolve() == True:

                terminal = True

            #function checks if new state is terminal 
            reward = play1.reward
            episodeReward += reward


            play1.memory.append((play1.prevBoard, action, temp, reward, terminal))
            #after every step update the network. this requires prevState, currentState, action and reward combo
            #play1.updateMemory((play1.prevBoard, action, play1.board, play1.reward, terminal))

            #only train at certain times
            if counter % 2 == 0 or terminal == True:
                play1.train(terminal, step)


            #play1.episodeReward.append(episodeReward)

            #constantly reducing epsilon, so it will use values from network more often
            if play1.epsilon > play1.minEpsilon:
                play1.epsilon *= play1.epsilonDecay

            #after a certain number of moves give up, start again
            if counter == 250:
                break


def test():

    for episode in tqdm(range(1, 201)):
        
        episodeReward = 0
        step = 0
        play1.reset()
        
        #every state can be reached in 31 moves
        play1.randomise(190,200)
        counter = 0

        terminal = False

        while not terminal:

            counter +=1 

            action = np.argmax(play1.getQValues(play1.board))
            play1.makeMove(action)
            
            play1.checkSolve()
            episodeReward += play1.reward
            
            #this if statement also updates self.reward
            if play1.checkSolve() == True:

                play1.episodeReward.append(episodeReward)
                terminal = True

            
            #after a certain number of moves give up, start again
            if counter == 250:
                play1.episodeReward.append(episodeReward)
                
                #this will just kill the while loop, wont affect neural network
                terminal = True


        #used to produce a useful graph
        if episode%20 == 0:
            play1.averageReward.append(sum(play1.episodeReward)/len(play1.episodeReward) )
            #print("last 20 episodes explored with average reward: ", (sum(play1.episodeReward)/len(play1.episodeReward)))
            
            plt.plot([i for i in range(1, len(play1.averageReward) + 1 )], play1.averageReward)
            plt.pause(0.05)

            play1.episodeReward = []


play1 = agent()

plt.grid(True)
plt.minorticks_on()
plt.xlabel("test session")
plt.ylabel("average episode reward over 20 games")
plt.axhline(y = -30, color = 'green')
plt.axhline(y = -250, color = 'red')

#this will prob end up being in another scipt. start training easily, then scenarios get harder
for i in range(0,5):
    train(1,50, 10_001)

for i in range(0,5):
    train(1, 100, 10_001)

for i in range(0,500):
    train(1, 170, 1001)
    test()



plt.show()
