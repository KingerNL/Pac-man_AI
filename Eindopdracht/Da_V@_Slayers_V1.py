
# Dit is de Reinforcement Learning agent, mogelijk kunnen we ook een Evolutionary Alghorithm maken?
# Om de file te runnen kan de de volgende command gebruiken:
# python .\capture.py -r baselineTeam -b Da_V@_Slayers
CONTACT = 'mart.veldkamp@hva.nl', 'merlijn.dascher@hva.nl'

from msilib.schema import Environment
from sre_parse import State
import string
from tkinter import Y
from captureAgents import CaptureAgent
import random, time, util
from game import Directions
import game
import numpy as np

# -=-=-=- Team creation -=-=-=-

def createTeam(firstIndex, secondIndex, isRed,
               first = 'ChonkyBoy', second = 'ChonkyBoy'):

  # The following line is an example only; feel free to change it.
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

#  -=-=-=- Agents -=-=-=-

class ChonkyBoy(CaptureAgent):
  """
  A Dummy agent to serve as an example of the necessary agent structure.
  You should look at baselineTeam.py for more details about how to
  create an agent as this is the bare minimum.
  """

  def registerInitialState(self, gameState):
    """
      NOTE: Dit wordt als init 1x gerunned. Voor ons belangrijk voor mogelijk:
        - Default parameters init
          - alpha:    learning rate
          - epsilon:  exploration rate
          - gamma:    discount factor
        - Welke kleur we zijn?
        - De algemene enviroment uitlezen?
    """

    self.epsilon = 0.05
    self.gamma = 0.8
    self.alpha = 0.2
    self.numTraining = 0
    self.observationHistory = []
    self.QValue = util.Counter()
    
    self.env = self.getEnvironment(gameState)
    
    CaptureAgent.registerInitialState(self, gameState)

  # NOTE: Dit wordt herhaald gerunned, nu zijn beide agents deze class
  def chooseAction(self, gameState): 
    
    # Dit is de stappen die je moet zetten om een Reinforcement Learning (vgm)
    """ 
      Geef een Reward / Q-value.
      indien er een punt is gepakt, terug gebracht, etc...
      of pac-man is gegeten (als ghost zijnde)
      Of General Gamestates zoals in het midden van het veld zich bevinden
    """

    """
      Krijg de state van le ChonkyBoy 
      (Dit kan alle relevante informatie over het speelveld zijn)
      Alles wat we relevante informatie kunnen vinden
    """
    new_env = self.UpdateEnvironment(gameState, self.env)
    print(new_env)
    Possible_Actions = gameState.getLegalActions(self.index)
    """
    Voorspel de beste Actie die daarna gegeven kan worden.
      Nu is dat een random actie van de list "Possible_Actions"
    """
    
    if util.flipCoin(self.epsilon):
      return random.choice(Possible_Actions)
    
    else:
      # print(self.computeActionFromQValues(gameState))
      return self.computeActionFromQValues(gameState)

  def getQValue(self, state, action):
      if self.QValue[state, action] == 0:
        return 0
      return self.QValue[state, action]

  def computeValueFromQValues(self, gameState):
      """
        Returns max_action Q(state,action)
        where the max is over legal actions.  Note that if
        there are no legal actions, which is the case at the
        terminal state, you should return a value of 0.0.
      """

      # Get the legal actions
      legal = gameState.getLegalActions(self.index)

      # Return 0 if there are no legal actions
      if len(legal) == 0:
        return 0.0

      # Set up variable that can be overwritten
      maxQ = float('-inf')

      # Loop over the legal actions
      for action in legal:

        # Overwrite if the score is better than the previously saved score
        if self.getQValue(gameState, action) > maxQ:
          maxQ = self.getQValue(gameState, action)
      
      # Return the best score
      return maxQ

  def computeActionFromQValues(self, gameState):

      # Call empty list to save best actions
      best = []

      # Get legal actions
      legal = gameState.getLegalActions(self.index)

      # Set up variable that can be overwritten
      maxQ = float('-inf')

      # Loop over legal actions
      for action in legal:

        # Save best state and action if there is an improvement
        if self.getQValue(gameState, action) > maxQ:
          maxQ = self.getQValue(gameState, action)
          best = [action]

        # Add the action if the result is the same 
        elif self.getQValue(gameState, action) == maxQ:
          best.append(action)

      # Choose random action out of the best options
      return random.choice(best)

  def update(self, gameState, action, nextState, reward):
      """
        The parent class calls this to observe a
        state = action => nextState and reward transition.
        You should do your Q-Value update here

        NOTE: You should never call this function,
        it will be called on your behalf
      """
      # Start with the partial score, based out of the reward, the discount and the computed value
      part = reward + self.discount * self.computeValueFromQValues(nextState)
      dict_key = gameState, action

      # Compute the final score
      self.QValue[dict_key] = (1.0 - self.alpha) * self.getQValue(gameState, action) + self.alpha * part

  def getEnvironment(self, gameState):
    """
      Returned 2D-numpy array (datatype: str) met alle belangrijke init environment info:
        - Walls         = 'W'
        - Food          = 'FF, EF'
        - powercapsule  = 'FP, EP'
        - Empty space   = '_'
        - Agent         = 'A, FA'
    """
    
    # Krijg de locatie van de walls en stops ze in de array
    env = gameState.getWalls()
    cop_env = []
    for r in env:       # Voor elke row in env
      cop_env.append(r)
    np_env = np.array(cop_env).astype(str)
    
    # Intereer over de np array en verrander de values
    np_env[np_env == "True"] = "W"
    np_env[np_env == "False"] = "_"
    

    # Krijg de locatie van de power capsules, en voeg toe aan np_env
    Is_red = gameState.isOnRedTeam(self.index)
    R_caps = gameState.getRedCapsules()
    B_caps = gameState.getBlueCapsules()
    
    for r in R_caps:
      if(Is_red):
        np_env[r[0]][r[1]] = "FP"
      else:
        np_env[r[0]][r[1]] = "EP"
    
    for r in B_caps:
      if(Is_red == 0):
        np_env[r[0]][r[1]] = "FP"
      else:
        np_env[r[0]][r[1]] = "EP"
    
    # Krijg de locatie van alle food
    R_food = gameState.getRedFood()
    B_food = gameState.getBlueFood()
    cop_env_R_food = []
    cop_env_B_food = []
    
    
    for r in R_food:
      cop_env_R_food.append(r)
    np_env_R_food = np.array(cop_env_R_food).astype(str)

    for r in B_food:
      cop_env_B_food.append(r)
    np_env_B_food = np.array(cop_env_B_food).astype(str)
    
    # Intereer over de np array en verrander de values
    solutions1 = np.argwhere(np_env_R_food == "True")
    for p in solutions1:
      if(Is_red):
        np_env[p[0]][p[1]] = "1"
      else:
        np_env[p[0]][p[1]] = "0"
        
    solutions2 = np.argwhere(np_env_B_food == "True")
    for p in solutions2:
      if(Is_red == 0):
        np_env[p[0]][p[1]] = "1"
      else:
        np_env[p[0]][p[1]] = "0"

    
    # Krijg positie van Agent
    pos = gameState.getAgentPosition(self.index)
    np_env[pos[0]][pos[1]] = "A"
    
    # Je kan deze print aanzetten voor debugging / check hoe de env er uit ziet.
    # print(np_env)
    
    return np_env
  
  def UpdateEnvironment(self, gameState, np_env):
    # Updates agent data in the env
    np_env[np_env == "A"] = "_"
    
    pos = gameState.getAgentPosition(self.index)
    np_env[pos[0]][pos[1]] = "A"
    
    return np_env
  