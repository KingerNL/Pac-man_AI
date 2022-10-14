# myTeam.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from captureAgents import CaptureAgent
import random, time, util
from game import Directions
import game
import numpy as np

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'ChonkyBoys', second = 'ChonkyBoys'):

  # The following line is an example only; feel free to change it.
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########

class ChonkyBoys(CaptureAgent):
  """
  A Dummy agent to serve as an example of the necessary agent structure.
  You should look at baselineTeam.py for more details about how to
  create an agent as this is the bare minimum.
  """

  def registerInitialState(self, gameState):
    # Hier wordt alles 1x gerunned
    
    # Welke teamkleur zijn we:
    My_Team_Color = gameState.isOnRedTeam(self.index)
    if (My_Team_Color == True): print("We zijn team Rood!")
    if (My_Team_Color == False): print("We zijn team Blauw!!!")
        
    grid = gameState.getCapsules()
    print(grid)
    CaptureAgent.registerInitialState(self, gameState)

    '''
    Your initialization code goes here, if you need any.
    '''


  def chooseAction(self, gameState):
    # Dit wordt herhaald gerunned
    
    # Get reward
    
    
    # Get the state of de Agent,
    # print(gameState.getAgentState(self.index))
    # print(gameState.getLegalActions(self.index))
    actions = gameState.getLegalActions(self.index)

    # Take action
    return random.choice(actions)
