
# Dit is de Reinforcement Learning agent, mogelijk kunnen we ook een Evolutionary Alghorithm maken?
# Om de file te runnen kan de de volgende command gebruiken:
# python .\capture.py -r baselineTeam -b Da_V@_Slayers

from sre_parse import State
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
    
    env = self.getEnvironment(gameState)
    print(env)
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
    Possible_Actions = gameState.getLegalActions(self.index)
    """
    Voorspel de beste Actie die daarna gegeven kan worden.
      Nu is dat een random actie van de list "Possible_Actions"
    """
    
    if util.flipCoin(self.epsilon):
      return random.choice(Possible_Actions)
    
    else:
      print(self.computeActionFromQValues(gameState))
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
      Het liefst willen we een environment terug geven met alles erin, dus:
      - Muren
      - Power capsuls
      - Food
      - Spawn location?
      - Enemy spawn location?
    """
    
    # Welke teamkleur zijn we:
    My_Team_Color = gameState.isOnRedTeam(self.index)
    if (My_Team_Color == True): print("We zijn team Rood!")
    if (My_Team_Color == False): print("We zijn team Blauw!!!")
    
    # Krijg de locatie van de walls
    env = gameState.getWalls()      
    
    # Krijg de locatie van de power capsules, doen we nu niks mee
    grid = gameState.getCapsules()  
    
    return env    