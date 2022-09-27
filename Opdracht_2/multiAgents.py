from game import Directions
import random, util
import numpy as np
from game import Agent

# Mijn eigen gemaakte manhatten distance
def Manhattan_Distance(pun1, pun2):
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

  def getAction(self, gameState):
      """
      You do not need to change this method, but you're welcome to.

      getAction chooses among the best options according to the evaluation function.

      Just like in the previous project, getAction takes a GameState and returns
      some Directions.X for some X in the set {North, South, West, East, Stop}
      """
      # Collect legal moves and successor states
      legalMoves = gameState.getLegalActions()

      # Choose one of the best actions
      scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
      bestScore = max(scores)
      bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
      chosenIndex = random.choice(bestIndices) # Pick randomly among the best

      "Add more of your code here if you want to"

      return legalMoves[chosenIndex]

  def evaluationFunction(self, currentGameState, action):
      """
      Design a better evaluation function here.

      The evaluation function takes in the current and proposed successor
      GameStates (pacman.py) and returns a number, where higher numbers are better.

      The code below extracts some useful information from the state, like the
      remaining food (newFood) and Pacman position after moving (newPos).
      newScaredTimes holds the number of moves that each ghost will remain
      scared because of Pacman having eaten a power pellet.

      Print out these variables to see what you're getting, then combine them
      to create a masterful evaluation function.
      """
      # Useful information you can extract from a GameState (pacman.py)
      successorGameState = currentGameState.generatePacmanSuccessor(action)
      newPos = successorGameState.getPacmanPosition()
      curPos = currentGameState.getPacmanPosition()
      newFood = successorGameState.getFood().asList()
      curFood = currentGameState.getFood().asList()
      newGhostStates = successorGameState.getGhostStates()
      newGhostPos = successorGameState.getGhostPositions()
      newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

      "*** YOUR CODE HERE ***"        
      Food_Score, Ghost_Score, Total_Score = 100, 100, 0
      for fp in curFood:
        Food_Score = min(Food_Score, Manhattan_Distance(fp, newPos))
        Total_Score += Manhattan_Distance(fp, newPos)
      
      for gp in newGhostPos:
        Ghost_Score = min(Ghost_Score, Manhattan_Distance(newPos, gp))
      if Ghost_Score < 3: return 100 * Ghost_Score - 10000
      elif len(newFood) == 0: return 10000
      else: 
        return -50 * Food_Score - 100 * len(newFood) + 1000 * Manhattan_Distance(newPos, curPos) - Total_Score + 1000 * (len(curFood) - len(newFood))

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

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

    def getAction(self, gameState):
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
            Returns the Total_Score number of agents in the game
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
        numGhost = gameState.getNumAgents() - 1
        candActions = gameState.getLegalActions(0)
        for action in candActions:
          curScore = minvalue(gameState.generateSuccessor(0, action), self.depth, 1, numGhost)
          if curScore > score:
            score, optAction = curScore, action
        return optAction
        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        alpha, beta = -float("inf"), float("inf")
        numGhost = gameState.getNumAgents() - 1
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
        candActions = gameState.getLegalActions(0)
        for action in candActions:
          curScore = minvalue(gameState.generateSuccessor(0, action), self.depth, 1, alpha, beta)
          if curScore > score:
            score = curScore
            optAction = action
          alpha = max(alpha, score)
        return optAction

class ExpectimaxAgent(MultiAgentSearchAgent):
  """
    Your expectimax agent (question 4)
  """

  def getAction(self, gameState):
      """
        Returns the expectimax action using self.depth and self.evaluationFunction
        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
      """
      "*** YOUR CODE HERE ***"
      numGhost = gameState.getNumAgents() - 1
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
      candActions = gameState.getLegalActions(0)
      for action in candActions:          
        curScore = expectimax(gameState.generateSuccessor(0, action), self.depth, 1)
        if curScore > score:
          score = curScore
          optAction = action
      return optAction
      util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
  """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).
    DESCRIPTION: <mart schreef hier wat, zodat je weet wat ik heb gedaan>
  """
  "*** YOUR CODE HERE ***"    
  INF = 1e6    

  foodPos = currentGameState.getFood().asList()
  ghostPos = currentGameState.getGhostPositions()
  curPos = currentGameState.getPacmanPosition()
  capsules = currentGameState.getCapsules()

  minFoodDis, minGhostDis, minCapDis, Total_Score = 100, 100, 100, 0
  for food in foodPos:
    minFoodDis = min(minFoodDis, Manhattan_Distance(food, curPos))
    Total_Score += Manhattan_Distance(food, curPos)
  for ghost in ghostPos:
    minGhostDis = min(minGhostDis, Manhattan_Distance(ghost, curPos))
  # for cap in capsules:
    # minCapDis = min(minCapDis, Manhattan_Distance(cap, curPos))

  if minGhostDis < 3: score = -1e5 - minFoodDis - Total_Score
  elif len(foodPos) == 0: score = INF + minGhostDis
  else:
    score = -50 * minFoodDis - Total_Score + minGhostDis * 2 - len(foodPos) * 2000

  # if len(foodPos) < 3: print 'Debug: ', score, 'food:', len(foodPos), 'curPos:', curPos,
  # if len(foodPos) > 0: print 'first food:', foodPos[0]
  # else: print ' '
  return score

# Abbreviation
better = betterEvaluationFunction
