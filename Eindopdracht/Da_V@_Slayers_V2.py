
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

  # NOTE: Dit wordt als init 1x gerunned. Voor ons belangrijk voor mogelijk:
  def registerInitialState(self, gameState):
    """
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
    
    # self.env = self.FriendlyFood(gameState)
    # print(self.env)
    # self.caps = self.FriendlyCapsules(gameState)
    # print(self.caps)
    # self.walls_temp = self.WallsNormalization(gameState)
    # print(self.walls_temp)
    
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
    # new_env = self.UpdateEnvironment(gameState, self.env)
    
    Possible_Actions = gameState.getLegalActions(self.index)
    """
      Voorspel de beste Actie die daarna gegeven kan worden.
      Nu is dat een random actie van de list "Possible_Actions"
    """
    
    return random.choice(Possible_Actions)
    
  def FriendlyFood(self, gameState):
    if gameState.isOnRedTeam(self.index):
      Friendly_food = gameState.getRedFood()
    else:
      Friendly_food = gameState.getBlueFood()
    
    Friendly_food_copy = []
    
    for row in Friendly_food:
      Friendly_food_copy.append(row)
    
    np_friendly_food = np.array(Friendly_food_copy).astype(bool)
    
    Food_coordinates = []
    row = 0
    column = 0
    
    for i in np_friendly_food:
      column = 0
      for j in i:
        if j == True:
          Food_coordinates.append((row, column))
        column += 1
      row += 1
    
    pacman_position = gameState.getAgentPosition(self.index)    
    
    Food_coordinates_norm = []

    for coordinate in Food_coordinates:
      Food_coordinates_norm.append(tuple(map(lambda i, j: i - j, coordinate, pacman_position)))

    return Food_coordinates_norm
  
  def EnemyFood(self, gameState):
    if gameState.isOnRedTeam(self.index):
      enemy_food = gameState.getBlueFood()
    else:
      enemy_food = gameState.getRedFood()
    
    Enemy_food_copy = []

    for row in enemy_food:
      Enemy_food_copy.append(row)

    np_enemy_food = np.array(Enemy_food_copy).astype(bool)

    Food_coordinates = []
    row = 0
    column = 0

    for i in np_enemy_food:
      column = 0
      for j in i:
        if j == True:
          Food_coordinates.append((row, column))
        column += 1
      row += 1
    
    pacman_position = gameState.getAgentPosition(self.index)

    Food_coordinates_norm = []

    for coordinate in Food_coordinates:
      Food_coordinates_norm.append(tuple(map(lambda i, j: i - j, coordinate, pacman_position)))

    return Food_coordinates_norm
  
  def FriendlyCapsules(self, gameState):
    # [a, Q(s,a) for a in s.getlegalactions]
  
    if gameState.isOnRedTeam(self.index):
      friendly_caps = gameState.getRedCapsules()
    else:
      friendly_caps = gameState.getBlueCapsules()
    
    pac_pos = gameState.getAgentPosition(self.index)
    
    normal_friendly_caps = []
    
    for coordinate in friendly_caps:
      normal_friendly_caps.append(tuple(map(lambda i, j: i - j, coordinate, pac_pos)))

    return normal_friendly_caps

  def EnemyCapsules(self, gameState):
    # [a, Q(s,a) for a in s.getlegalactions]
  
    if gameState.isOnRedTeam(self.index):
      enemy_caps = gameState.getBlueCapsules()
    else:
      enemy_caps = gameState.getRedCapsules()
    
    pac_pos = gameState.getAgentPosition(self.index)
    normal_enemy_caps = []
    
    for coordinate in enemy_caps:
      normal_enemy_caps.append(tuple(map(lambda i, j: i - j, coordinate, pac_pos)))
    
    return normal_enemy_caps
      
  def WallsNormalization(self, gameState):
    walls = gameState.getWalls()
    
    walls_copy = []
    
    for row in walls:
      walls_copy.append(row)
    
    np_walls = np.array(walls_copy).astype(bool)
    
    wall_coordinates = []
    row = 0
    column = 0
    
    for i in np_walls:
      column = 0
      for j in i:
        if j == True:
          wall_coordinates.append((row, column))
        column += 1
      row += 1
    
    pacman_position = gameState.getAgentPosition(self.index)    
    
    wall_coordinates_norm = []

    for coordinate in wall_coordinates:
      wall_coordinates_norm.append(tuple(map(lambda i, j: i - j, coordinate, pacman_position)))

    return wall_coordinates_norm
    
  def remember(self, state, action, reward, next_state, done):
    pass
  
  def act(self, state, action, reward, next_state, done):
    pass
  
  def replay(self, batch_size):
    pass
  
  def load(self, name):
    pass
  
  def save(self, name):
    pass