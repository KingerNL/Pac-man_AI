from game import Directions
import random, util
import numpy as np
from game import Agent

# Mijn eigen gemaakte manhatten distance
def manhattan_distance(pun1, pun2):
    afstand = 0
    punten = zip(pun1, pun2)
    for x1, x2 in punten:
        verschil = x2 - x1
        Absoluut_Verschil = abs(verschil)
        afstand = afstand + Absoluut_Verschil

    return afstand

class ReflexAgent(Agent):
  """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
  """

  def getAction(self, game_state):
      """
      You do not need to change this method, but you're welcome to.

      getAction chooses among the best options according to the evaluation function.

      Just like in the previous project, getAction takes a GameState and returns
      some Directions.X for some X in the set {North, South, West, East, Stop}
      """
      # Collect legal moves and successor states
      legalMoves = game_state.getLegalActions()

      # Choose one of the best actions
      scores = [self.evaluationFunction(game_state, action) for action in legalMoves]
      bestScore = max(scores)
      bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
      chosenIndex = random.choice(bestIndices) # Pick randomly among the best

      "Add more of your code here if you want to"

      return legalMoves[chosenIndex]

  def evaluationFunction(self, current_game_state, action):
      """
      Design a better evaluation function here.

      The evaluation function takes in the current and proposed successor
      GameStates (pacman.py) and returns a number, where higher numbers are better.

      The code below extracts some useful information from the state, like the
      remaining food (new_food) and Pacman position after moving (new_position).
      scared_time_remaining holds the number of moves that each ghost will remain
      scared because of Pacman having eaten a power pellet.

      Print out these variables to see what you're getting, then combine them
      to create a masterful evaluation function.
      """
      # Useful information you can extract from a GameState (pacman.py)
      successorGameState = current_game_state.generatePacmanSuccessor(action)
      new_position = successorGameState.getPacmanPosition()
      current_position = current_game_state.getPacmanPosition()
      new_food = successorGameState.getFood().asList()
      current_food = current_game_state.getFood().asList()
      new_ghost_position = successorGameState.getGhostPositions()

      "*** YOUR CODE HERE ***"
      food_score, ghost_score, total_score = 100, 100, 0
      for fp in current_food:
        food_score = min(food_score, manhattan_distance(fp, new_position))
        total_score += manhattan_distance(fp, new_position)
      
      for gp in new_ghost_position:
        ghost_score = min(ghost_score, manhattan_distance(new_position, gp))
      if ghost_score < 3: 
        return 100 * ghost_score - 10000
      elif len(new_food) == 0: 
        return 10000
      else: 
        return -50 * food_score - 100 * len(new_food) + 1000 * manhattan_distance(new_position, current_position) - total_score + 1000 * (len(current_food) - len(new_food))

def scoreEvaluationFunction(current_game_state):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return current_game_state.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, game_state):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total_score number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        def maxvalue(curState, curDepth, idx, numGhost):
          '''
          Max Function for pacman
          '''
          assert(idx == 0)
          if curState.isWin() or curState.isLose() or curDepth == 0:
            return self.evaluationFunction(curState)
          score = -float("inf")
          candActions = curState.getLegalActions(idx)
          for action in candActions:
            score = max(score, minvalue(curState.generateSuccessor(0, action), curDepth, 1, numGhost))
          return score

        def minvalue(curState, curDepth, idx, numGhost):
          '''
          Min Function for ghost
          '''
          if curState.isWin() or curState.isLose() or curDepth == 0:
            return self.evaluationFunction(curState)
          candActions = curState.getLegalActions(idx)
          score = float("inf")
          for action in candActions:
            if idx != numGhost:
              score = min(score, minvalue(curState.generateSuccessor(idx, action), curDepth, idx + 1, numGhost))
            else:
              score = min(score, maxvalue(curState.generateSuccessor(idx, action), curDepth - 1, 0, numGhost))
          return score

        score, optAction = -float("inf"), None
        numGhost = game_state.getNumAgents() - 1
        candActions = game_state.getLegalActions(0)
        for action in candActions:
          curScore = minvalue(game_state.generateSuccessor(0, action), self.depth, 1, numGhost)
          if curScore > score:
            score, optAction = curScore, action
        return optAction
        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, game_state):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        alpha, beta = -float("inf"), float("inf")
        numGhost = game_state.getNumAgents() - 1
        def maxvalue(curState, curDepth, idx, alpha, beta):
          '''
          Max Function for pacman
          '''          
          assert(idx == 0)
          if curState.isWin() or curState.isLose() or curDepth == 0:
            return self.evaluationFunction(curState)
          candActions = curState.getLegalActions(idx)
          v = -float("inf")
          for action in candActions:
            v = max(v, minvalue(curState.generateSuccessor(0, action), curDepth, 1, alpha, beta))
            if v > beta: return v
            alpha = max(alpha, v)
          return v

        def minvalue(curState, curDepth, idx, alpha, beta):
          '''
          Min Function for ghost
          '''          
          if curState.isWin() or curState.isLose() or curDepth == 0:
            return self.evaluationFunction(curState)
          candActions = curState.getLegalActions(idx)
          v = float("inf")
          for action in candActions:
            if idx == numGhost:
              v = min(v, maxvalue(curState.generateSuccessor(idx, action), curDepth - 1, 0, alpha, beta))
            else:
              v = min(v, minvalue(curState.generateSuccessor(idx, action), curDepth, idx + 1, alpha, beta))
            if v < alpha: return v
            beta = min(beta, v)
          return v

        optAction, score = None, -float("inf")
        candActions = game_state.getLegalActions(0)
        for action in candActions:
          curScore = minvalue(game_state.generateSuccessor(0, action), self.depth, 1, alpha, beta)
          if curScore > score:
            score = curScore
            optAction = action
          alpha = max(alpha, score)
        return optAction

class ExpectimaxAgent(MultiAgentSearchAgent):
  """
    Your expectimax agent (question 4)
  """

  def getAction(self, game_state):
      """
        Returns the expectimax action using self.depth and self.evaluationFunction
        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
      """
      "*** YOUR CODE HERE ***"
      numGhost = game_state.getNumAgents() - 1
      def expectimax(curState, curDepth, idx):
        '''
        Expected Score for ghost who acts randomly
        '''
        if curState.isWin() or curState.isLose() or curDepth == 0:
          return self.evaluationFunction(curState)
        candActions, score = curState.getLegalActions(idx), 0          
        for action in candActions:
          if idx == numGhost:
            score += maxvalue(curState.generateSuccessor(idx, action), curDepth - 1, 0)
          else:
            score += expectimax(curState.generateSuccessor(idx, action), curDepth, idx + 1)
        return float(score) / len(candActions)

      def maxvalue(curState, curDepth, idx):
        '''
        Max Score for pacman who acts optimally
        '''
        assert(idx == 0)
        if curState.isWin() or curState.isLose() or curDepth == 0:
          return self.evaluationFunction(curState)
        candActions, score = curState.getLegalActions(idx), -float("inf")
        for action in candActions:
          score = max(score, expectimax(curState.generateSuccessor(0, action), curDepth, 1))
        return score

      optAction, score = None, -float("inf")
      candActions = game_state.getLegalActions(0)
      for action in candActions:          
        curScore = expectimax(game_state.generateSuccessor(0, action), self.depth, 1)
        if curScore > score:
          score = curScore
          optAction = action
      return optAction
      util.raiseNotDefined()

def betterEvaluationFunction(current_game_state):
  """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).
    DESCRIPTION: <mart schreef hier wat, zodat je weet wat ik heb gedaan>
  """
  "*** YOUR CODE HERE ***"    
  INF = 1e6    

  foodPos = current_game_state.getFood().asList()
  ghostPos = current_game_state.getGhostPositions()
  current_position = current_game_state.getPacmanPosition()
  capsules = current_game_state.getCapsules()

  minFoodDis, minGhostDis, minCapDis, total_score = 100, 100, 100, 0
  for food in foodPos:
    minFoodDis = min(minFoodDis, manhattan_distance(food, current_position))
    total_score += manhattan_distance(food, current_position)
  for ghost in ghostPos:
    minGhostDis = min(minGhostDis, manhattan_distance(ghost, current_position))
  # for cap in capsules:
    # minCapDis = min(minCapDis, manhattan_distance(cap, current_position))

  if minGhostDis < 3: score = -1e5 - minFoodDis - total_score
  elif len(foodPos) == 0: score = INF + minGhostDis
  else:
    score = -50 * minFoodDis - total_score + minGhostDis * 2 - len(foodPos) * 2000

  # if len(foodPos) < 3: print 'Debug: ', score, 'food:', len(foodPos), 'current_position:', current_position,
  # if len(foodPos) > 0: print 'first food:', foodPos[0]
  # else: print ' '
  return score

# Abbreviation
better = betterEvaluationFunction
