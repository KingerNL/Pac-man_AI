# # baselineTeam.py
# # ---------------
# # Licensing Information:  You are free to use or extend these projects for
# # educational purposes provided that (1) you do not distribute or publish
# # solutions, (2) you retain this notice, and (3) you provide clear
# # attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# # 
# # Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# # The core projects and autograders were primarily created by John DeNero
# # (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# # Student side autograding was added by Brad Miller, Nick Hay, and
# # Pieter Abbeel (pabbeel@cs.berkeley.edu).


# # baselineTeam.py
# # ---------------
# # Licensing Information: Please do not distribute or publish solutions to this
# # project. You are free to use and extend these projects for educational
# # purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# # John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# # For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

# from captureAgents import CaptureAgent
# import distanceCalculator
# import random, time, util, sys
# from game import Directions
# import game
# from util import nearestPoint

# #################
# # Team creation #
# #################

# def createTeam(firstIndex, secondIndex, isRed,
#                first = 'OffensiveReflexAgent', second = 'OffensiveReflexAgent'):
#   """
#   This function should return a list of two agents that will form the
#   team, initialized using firstIndex and secondIndex as their agent
#   index numbers.  isRed is True if the red team is being created, and
#   will be False if the blue team is being created.

#   As a potentially helpful development aid, this function can take
#   additional string-valued keyword arguments ("first" and "second" are
#   such arguments in the case of this function), which will come from
#   the --redOpts and --blueOpts command-line arguments to capture.py.
#   For the nightly contest, however, your team will be created without
#   any extra arguments, so you should make sure that the default
#   behavior is what you want for the nightly contest.
#   """
#   return [eval(first)(firstIndex), eval(second)(secondIndex)]

# ##########
# # Agents #
# ##########

# class ReflexCaptureAgent(CaptureAgent):
#   """
#   A base class for reflex agents that chooses score-maximizing actions
#   """
 
#   def registerInitialState(self, gameState):
#     self.start = gameState.getAgentPosition(self.index)
#     CaptureAgent.registerInitialState(self, gameState)

#   def chooseAction(self, gameState):
#     """
#     Picks among the actions with the highest Q(s,a).
#     """
#     actions = gameState.getLegalActions(self.index)

#     # You can profile your evaluation time by uncommenting these lines
#     # start = time.time()
#     values = [self.evaluate(gameState, a) for a in actions]
#     # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

#     maxValue = max(values)
#     bestActions = [a for a, v in zip(actions, values) if v == maxValue]

#     foodLeft = len(self.getFood(gameState).asList())

#     if foodLeft <= 2:
#       bestDist = 9999
#       for action in actions:
#         successor = self.getSuccessor(gameState, action)
#         pos2 = successor.getAgentPosition(self.index)
#         dist = self.getMazeDistance(self.start,pos2)
#         if dist < bestDist:
#           bestAction = action
#           bestDist = dist
#       return bestAction

#     return random.choice(bestActions)

#   def getSuccessor(self, gameState, action):
#     """
#     Finds the next successor which is a grid position (location tuple).
#     """
#     successor = gameState.generateSuccessor(self.index, action)
#     pos = successor.getAgentState(self.index).getPosition()
#     if pos != nearestPoint(pos):
#       # Only half a grid position was covered
#       return successor.generateSuccessor(self.index, action)
#     else:
#       return successor

#   def evaluate(self, gameState, action):
#     """
#     Computes a linear combination of features and feature weights
#     """
#     features = self.getFeatures(gameState, action)
#     weights = self.getWeights(gameState, action)
#     return features * weights

#   def getFeatures(self, gameState, action):
#     """
#     Returns a counter of features for the state
#     """
#     features = util.Counter()
#     successor = self.getSuccessor(gameState, action)
#     features['successorScore'] = self.getScore(successor)
#     return features

#   def getWeights(self, gameState, action):
#     """
#     Normally, weights do not depend on the gamestate.  They can be either
#     a counter or a dictionary.
#     """
#     return {'successorScore': 1.0}

# class OffensiveReflexAgent(ReflexCaptureAgent):
#   """
#   A reflex agent that seeks food. This is an agent
#   we give you to get an idea of what an offensive agent might look like,
#   but it is by no means the best or only way to build an offensive agent.
#   """
#   def getFeatures(self, gameState, action):
#     features = util.Counter()
#     successor = self.getSuccessor(gameState, action)
#     foodList = self.getFood(successor).asList()    
#     features['successorScore'] = -len(foodList)#self.getScore(successor)

#     # Compute distance to the nearest food

#     if len(foodList) > 0: # This should always be True,  but better safe than sorry
#       myPos = successor.getAgentState(self.index).getPosition()
#       minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
#       features['distanceToFood'] = minDistance
#     return features

#   def getWeights(self, gameState, action):
#     return {'successorScore': 100, 'distanceToFood': -1}

# class DefensiveReflexAgent(ReflexCaptureAgent):
#   """
#   A reflex agent that keeps its side Pacman-free. Again,
#   this is to give you an idea of what a defensive agent
#   could be like.  It is not the best or only way to make
#   such an agent.
#   """

#   def getFeatures(self, gameState, action):
#     features = util.Counter()
#     successor = self.getSuccessor(gameState, action)

#     myState = successor.getAgentState(self.index)
#     myPos = myState.getPosition()

#     # Computes whether we're on defense (1) or offense (0)
#     features['onDefense'] = 1
#     if myState.isPacman: features['onDefense'] = 0

#     # Computes distance to invaders we can see
#     enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
#     invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
#     features['numInvaders'] = len(invaders)
#     if len(invaders) > 0:
#       dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
#       features['invaderDistance'] = min(dists)

#     if action == Directions.STOP: features['stop'] = 1
#     rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
#     if action == rev: features['reverse'] = 1

#     return features

#   def getWeights(self, gameState, action):
#     return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -10, 'stop': -100, 'reverse': -2}







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
from util import nearestPoint
import itertools

debug = False
debug_capsule = False


#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first='OffensiveReflexAgent', second='OffensiveReflexAgent'):
  """
  This function should return a list of two agents that will form the
  team, initialized using firstIndex and secondIndex as their agent
  index numbers.  isRed is True if the red team is being created, and
  will be False if the blue team is being created.

  As a potentially helpful development aid, this function can take
  additional string-valued keyword arguments ("first" and "second" are
  such arguments in the case of this function), which will come from
  the --redOpts and --blueOpts command-line arguments to capture.py.
  For the nightly contest, however, your team will be created without
  any extra arguments, so you should make sure that the default
  behavior is what you want for the nightly contest.
  """  
  return [eval(first)(firstIndex), eval(second)(secondIndex)]


##########
# Agents #
##########


class ParticlesCTFAgent(CaptureAgent):
    """
    CTF Agent that models enemies using particle filtering.
    """

    def registerInitialState(self, gameState, numParticles=600):
        # =====original register initial state=======
        self.start = gameState.getAgentPosition(self.index)

        # =====ParticleCTFAgent init================
        self.numParticles = numParticles
        self.initialize(gameState)
        # =====Features=============
        self.numFoodToEat = len(self.getFood(gameState).asList()) - 2
        self.scaredMoves = 0
        self.defenseScaredMoves = 0
        CaptureAgent.registerInitialState(self, gameState)
        self.stopped = 0
        self.stuck = False
        self.numStuckSteps = 0
        self.offenseInitialIncentivize = True
        self.defenseInitialIncentivize = True
        self.width = self.getFood(gameState).width
        self.height = self.getFood(gameState).height
        self.halfway = self.width / 2

        self.reverse = 0
        self.flank = False
        self.numRevSteps = 0

        # ====new home
        furthest_home = None
        furthest_home_dist = 0
        myPos = gameState.getAgentPosition(self.index)

        middle = self.halfway  # already correct
        list_of_homes = [(middle, 0), (middle, 1), (middle, 2), (middle, 3), (middle, 4),
                         (middle, int(self.height / 2)), (middle, int(self.height / 2) + 1),
                         (middle, int(self.height / 2) + 2), (middle, int(self.height / 2) + 3),
                         (middle, int(self.height / 2) - 1), (middle, int(self.height / 2) - 2),
                         (middle, int(self.height / 2) - 3),
                         (middle, self.height - 1), (middle, self.height - 2), (middle, self.height - 3),
                         (middle, self.height - 4)]
        legals = set(self.legalPositions)
        legal_homes = list()
        for home in list_of_homes:
            if home in legals:
                legal_homes.append(home)

                dist = self.getMazeDistance(myPos, home)
                if dist > furthest_home_dist:
                    furthest_home_dist = dist
                    furthest_home = home
        legal_homes.append(self.start)
        self.positions_along_border_of_home = legal_homes

        # ====setting initial position to go to==========
        self.furthest_position_along_border_of_home = furthest_home
        self.go_to_furthest_position = True

    def initialize(self, gameState, legalPositions=None):
        self.legalPositions = gameState.getWalls().asList(False)
        self.initializeParticles()
        self.a, self.b = self.getOpponents(gameState)
        # for fail
        self.initialGameState = gameState

    def setEnemyPosition(self, gameState, pos, enemyIndex):
        foodGrid = self.getFood(gameState)
        halfway = foodGrid.width / 2
        conf = game.Configuration(pos, game.Directions.STOP)

        # FOR THE WEIRD ERROR CHECK
        if gameState.isOnRedTeam(self.index):
            if pos[0] >= halfway:
                isPacman = False
            else:
                isPacman = True
        else:
            if pos[0] >= halfway:
                isPacman = True
            else:
                isPacman = False
        gameState.data.agentStates[enemyIndex] = game.AgentState(conf, isPacman)

        return gameState

    def initializeParticles(self, type="both"):

        positions = self.legalPositions
        atEach = self.numParticles / len(positions)  # self.numParticles
        remainder = self.numParticles % len(positions)
        # don't throw out a particle
        particles = []
        # populate particles
        for pos in positions:
            for num in range(round(atEach)):
                particles.append(pos)
        # now populate the remainders
        for index in range(remainder):
            particles.append(positions[index])
        # save to self.particles
        if type == 'both':
            self.particlesA = particles
            self.particlesB = particles
        elif type == self.a:
            self.particlesA = particles
        elif type == self.b:
            self.particlesB = particles
        return particles

    def observeState(self, gameState, enemyIndex):

        pacmanPosition = gameState.getAgentPosition(self.index)

        if enemyIndex == self.a:
            noisyDistance = gameState.getAgentDistances()[self.a]
            beliefDist = self.getBeliefDistribution(self.a)
            particles = self.particlesA
            if gameState.getAgentPosition(self.a) != None:
                self.particlesA = [gameState.getAgentPosition(self.a)] * self.numParticles
                return
        else:
            noisyDistance = gameState.getAgentDistances()[self.b]
            beliefDist = self.getBeliefDistribution(self.b)
            particles = self.particlesB
            if gameState.getAgentPosition(self.b) != None:
                self.particlesB = [gameState.getAgentPosition(self.b)] * self.numParticles
                return

        W = util.Counter()

        for p in particles:
            trueDistance = self.getMazeDistance(p, pacmanPosition)
            W[p] = beliefDist[p] * gameState.getDistanceProb(trueDistance, noisyDistance)

        # we resample after we get weights for each ghost
        if W.totalCount() == 0:
            particles = self.initializeParticles(enemyIndex)
        else:
            values = []
            keys = []
            for key, value in W.items():
                keys.append(key)
                values.append(value)

            if enemyIndex == self.a:
                self.particlesA = util.nSample(values, keys, self.numParticles)
            else:
                self.particlesB = util.nSample(values, keys, self.numParticles)

    def elapseTime(self, gameState, enemyIndex):

        if enemyIndex == self.a:
            particles = self.particlesA
        else:
            particles = self.particlesB

        for i in range(self.numParticles):
            x, y = particles[i]

            # find all legal positions above or below it
            north = (x, y - 1)
            south = (x, y + 1)
            west = (x + 1, y)
            east = (x - 1, y)

            possibleLegalPositions = set(self.legalPositions)
            legalPositions = list()

            if north in possibleLegalPositions: legalPositions.append(north)
            if south in possibleLegalPositions: legalPositions.append(south)
            if west in possibleLegalPositions: legalPositions.append(west)
            if east in possibleLegalPositions: legalPositions.append(east)

            new_position = random.choice(legalPositions)

            particles[i] = new_position

        if enemyIndex == self.a:
            self.particlesA = particles

        else:
            self.particlesB = particles

    def getBeliefDistribution(self, enemyIndex):
        allPossible = util.Counter()
        if enemyIndex == self.a:
            for pos in self.particlesA:
                allPossible[pos] += 1
        else:
            for pos in self.particlesB:
                allPossible[pos] += 1
        allPossible.normalize()
        return allPossible

    def getEnemyPositions(self, enemyIndex):
        """
        Uses getBeliefDistribution to predict where the two enemies are most likely to be
        :return: two tuples of enemy positions
        """
        return self.getBeliefDistribution(enemyIndex).argMax()

    def getSuccessor(self, gameState, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        debug = False
        if action is None:
            successor = gameState
        else:
            successor = gameState.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()

        if pos != nearestPoint(pos):
            # Only half a grid position was covered
            return successor.generateSuccessor(self.index, action)
        else:
            return successor

    def evaluate(self, gameState, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.getFeatures(gameState, action)
        weights = self.getWeights(gameState, action)
        return features * weights

    def getFeatures(self, gameState, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        features['successorScore'] = self.getScore(successor)
        return features

    def getWeights(self, gameState, action):
        """
        Normally, weights do not depend on the gamestate.  They can be either
        a counter or a dictionary.
        """
        return {'successorScore': 1.0}

    def chooseAction(self, gameState):

        start = time.time()
        pacmanPosition = gameState.getAgentPosition(self.index)

        # elapse time
        self.elapseTime(gameState, self.a)
        self.elapseTime(gameState, self.b)
        # ----------------elapse

        self.observeState(gameState, self.a)
        self.observeState(gameState, self.b)

        actions = gameState.getLegalActions(self.index)
        values = [self.evaluate(gameState, a) for a in actions]

        maxValue = max(values)
        bestActions = [a for a, v in zip(actions, values) if v == maxValue]
        bestAction = random.choice(bestActions)

        # update scared moves
        if len(self.getCapsules(gameState)) != len(self.getCapsules(self.getSuccessor(gameState, bestAction))):
            # we ate a capsule!
            self.scaredMoves = self.scaredMoves + 40
        elif self.scaredMoves != 0:
            self.scaredMoves = self.scaredMoves - 1
        else:
            pass

        # update defense scared moves
        numCapsulesDefending = len(gameState.getBlueCapsules())
        numCapsulesLeft = len(self.getSuccessor(gameState, bestAction).getBlueCapsules())
        if self.red:
            numCapsulesLeft = len(gameState.getRedCapsules())
            numCapsulesDefending = len(self.getSuccessor(gameState, bestAction).getRedCapsules())
        if numCapsulesLeft < numCapsulesDefending:
            # enemy ate a capsule!
            self.defenseScaredMoves += 40
        elif self.defenseScaredMoves != 0:
            self.defenseScaredMoves -= 1

        return bestAction


class OffensiveReflexAgent(ParticlesCTFAgent):

    def getFeaturesFoodDefenseSide(self, myPos, foodList, features):

        features['foodScore'] = -len(foodList)

        if len(foodList) > 0:  # This should always be True,  but better safe than sorry

            minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
            features['distanceToFood'] = minDistance

    def getFeaturesFoodOffenseSide(self, myPos, numFoodEaten, foodList, shouldGoHome, features, numCarryingLimit):

        if len(foodList) <= 2 or (numFoodEaten >= numCarryingLimit and shouldGoHome) and True:
            minToHome = min([self.getMazeDistance(myPos, h) for h in self.positions_along_border_of_home])
            if minToHome == 0: minToHome = 0.000001
            # add to features
            features['distanceToHome'] = -float(minToHome)
        else:
            features['foodScore'] = -len(foodList)
            # Compute distance to the nearest food
            if len(foodList) > 0:  # This should always be True,  but better safe than sorry
                minDistances = util.PriorityQueue()
                for food in foodList:
                    minDistances.push(food, self.getMazeDistance(myPos, food))
                minDistance = self.getMazeDistance(myPos, minDistances.pop())
                secondMinDistance = self.getMazeDistance(myPos, minDistances.pop())
                thirdMinDistance = self.getMazeDistance(myPos, minDistances.pop())
                features['distanceToFood'] = minDistance
                features['seconddistanceToFood'] = secondMinDistance
                features['thirddistanceToFood'] = thirdMinDistance

    def getFeaturesCapsulesOffenseSide(self, capsuleList, myPos, features):

        if len(capsuleList) > 0:
            minCapDistance = min([self.getMazeDistance(myPos, capsule) for capsule in capsuleList])
            if minCapDistance < 5:  # CHANGED
                features['distanceToCapsule'] = -minCapDistance

    def getFeatures(self, gameState, action):

        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        myPos = successor.getAgentState(self.index).getPosition()

        foodGrid = self.getFood(successor)
        foodList = self.getFood(successor).asList()
        numCarryingLimit = int((self.numFoodToEat - 2) / 3)
        numFoodEaten = gameState.getAgentState(self.index).numCarrying

        width = foodGrid.width
        height = foodGrid.height
        halfway = foodGrid.width / 2

        when_gen_enemy_dist_matters = int(min(width, height) * 2 / 3)
        when_min_enemy_dist_matters = 30  # 10
        min_enemy_dist = 99999999999
        general_enemy_dist = 99999999999

        # go to furthest point near home
        max_furthest_point_dist = self.getMazeDistance(myPos, self.furthest_position_along_border_of_home)
        localgo_to_furthest_position = self.go_to_furthest_position

        if successor.getAgentState(
                self.index).isPacman or max_furthest_point_dist < 2:  # ((abs(myPos[0]-self.offenseGoToFurthestFoodPosition[0]) < 5 and not successor.getAgentState(self.index).isPacman) and maxfooddist < 5):
            localgo_to_furthest_position = False
        elif myPos == self.start:
            localgo_to_furthest_position = True
        if max_furthest_point_dist == 0:
            max_furthest_point_dist = 0.0001

        # distance to furthest food on our side
        if localgo_to_furthest_position:
            features['max_furthest_point_dist'] = max_furthest_point_dist
            return features

        # set default enemies value
        features['generalEnemyDist'] = when_gen_enemy_dist_matters

        if not gameState.getAgentState(self.index).isPacman:  # on defense

            # food
            self.getFeaturesFoodDefenseSide(myPos, foodList, features)

            # set default capsule value
            features['distanceToCapsule'] = -8

        else:  # on offense

            # capsules
            capsuleList = self.getCapsules(gameState)
            self.getFeaturesCapsulesOffenseSide(capsuleList, myPos, features)

            # scared time
            localScaredMoves = 0
            # when we eat a capsule
            if len(capsuleList) != len(self.getCapsules(successor)):
                # we ate a capsule!
                localScaredMoves = self.scaredMoves + 40
            elif self.scaredMoves != 0:
                localScaredMoves = self.scaredMoves - 1

            # enemies

            enemies = self.getOpponents(successor)  # returns list of enemy indices
            enemy_one_pos = successor.getAgentPosition(enemies[0])  # enemy one pos
            enemy_two_pos = successor.getAgentPosition(enemies[1])  # enemy two pos

            # check if any enemy is in viewing
            if enemy_one_pos is None and enemy_two_pos is None:

                enemy_one_loc = self.getBeliefDistribution(enemies[0]).argMax()
                enemy_two_loc = self.getBeliefDistribution(enemies[1]).argMax()
                enemy_one_dist = self.getMazeDistance(myPos, enemy_one_loc)
                enemy_two_dist = self.getMazeDistance(myPos, enemy_two_loc)

                if enemy_one_dist < general_enemy_dist and ((enemy_one_loc[0] > halfway and myPos[0] > halfway) or (
                        enemy_one_loc[0] < halfway and myPos[0] < halfway)):
                    general_enemy_dist = enemy_one_dist
                if enemy_two_dist < general_enemy_dist and ((enemy_two_loc[0] > halfway and myPos[0] > halfway) or (
                        enemy_two_loc[0] < halfway and myPos[0] < halfway)):
                    general_enemy_dist = enemy_two_dist

                if general_enemy_dist < when_gen_enemy_dist_matters:  # CAN BE MODIFIED
                    features['generalEnemyDist'] = general_enemy_dist

            else:  # at least one enemy in viewing

                if enemy_one_pos is not None and ((enemy_one_pos[0] > halfway and myPos[0] > halfway) or (
                        enemy_one_pos[0] < halfway and myPos[0] < halfway)):
                    min_enemy_dist = min(min_enemy_dist, self.getMazeDistance(myPos, enemy_one_pos))
                if enemy_two_pos is not None and ((enemy_two_pos[0] > halfway and myPos[0] > halfway) or (
                        enemy_two_pos[0] < halfway and myPos[0] < halfway)):
                    min_enemy_dist = min(min_enemy_dist, self.getMazeDistance(myPos, enemy_two_pos))

            # you only do stuff with min_enemy_dist if it is actually close
            if min_enemy_dist < when_min_enemy_dist_matters:
                # eat ghost if you can
                if localScaredMoves > 0:
                    features['eatEnemyDist'] = when_min_enemy_dist_matters - float(min_enemy_dist)
                    numEatableEnemies = len([successor.getAgentState(index) for index in enemies if
                                             not successor.getAgentState(index).isPacman and successor.getAgentState(
                                                 index).getPosition() != None])
                    features['numEatableEnemies'] = numEatableEnemies
                    # min dist feature should be 0
                elif min_enemy_dist == 0:
                    # really bad bc you are eaten!
                    features['minEnemyDist'] = 999999
                # otherwise have min dist feature
                else:
                    features['minEnemyDist'] = when_min_enemy_dist_matters - float(min_enemy_dist)

            # food
            shouldGoHome = False
            if features['minEnemyDist'] > 0 or abs(myPos[0] - halfway) < 4: shouldGoHome = True
            self.getFeaturesFoodOffenseSide(myPos, numFoodEaten, foodList, shouldGoHome, features,
                                            numCarryingLimit)

            # further from the food the better
            if min_enemy_dist == 99999999999:
                if not (len(foodList) <= 2 or (
                        numFoodEaten >= numCarryingLimit and shouldGoHome) and True):  # condition for going home
                    distance_from_border = abs(myPos[0] - halfway) + abs(myPos[1] - 0)
                    features['distance_from_border'] = float(distance_from_border)

        # punish staying in the same place
        if action == Directions.STOP: features['stop'] = 1
        # punish just doing the reverse
        rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        # stuck situation
        if self.stuck:  # didn't test stuck
            minToHome = self.getMazeDistance(myPos, self.start)
            if minToHome == 0: minToHome = 0.000001
            # add to features
            features['distanceToHome'] = -float(minToHome)
        elif self.flank and not successor.getAgentState(self.index).isPacman:
            minToHome = self.getMazeDistance(myPos, self.start)
            if minToHome == 0: minToHome = 0.000001
            # add to features
            features['distanceToHome'] = -float(minToHome)
        elif self.flank and successor.getAgentState(self.index).isPacman:
            x, y = self.start
            possibleLegalPositions = set(self.legalPositions)
            halfHome = (x, y / 2)
            up = (myPos[0], y)
            out = (myPos[0], y / 2)
            down = (myPos[0], 1)
            aims = [self.start]

            if halfHome in possibleLegalPositions: aims.append(halfHome)
            if up in possibleLegalPositions: aims.append(up)
            if out in possibleLegalPositions: aims.append(out)
            if down in possibleLegalPositions: aims.append(down)
            aim = random.choice(aims)
            minToOtherHalf = self.getMazeDistance(myPos, aim)
            if myPos == self.start: minToOtherHalf = 0.000001
            features['minToOtherHalf'] = -float(minToOtherHalf)

        if myPos == self.start:
            features['dying_punishment'] = 999999

        return features

    def getWeights(self, gameState, action):

        return {'foodScore': 100, 'distanceToFood': -3, 'distanceToHome': 1000, 'distanceToCapsule': 1.2,
                'minEnemyDist': -100, 'generalEnemyDist': 1, 'eatEnemyDist': 2.1, 'stop': -75, 'rev': -100,
                'minToOtherHalf': 1000, 'max_furthest_point_dist': -99999999999999, 'distance_from_border': 0.01,
                'seconddistanceToFood': -1, 'thirddistanceToFood': -0.8, 'numEatableEnemies': -1000,
                'dying_punishment': -1}

    def chooseAction(self, gameState):

        start = time.time()
        pacmanPosition = gameState.getAgentPosition(self.index)
        self.observeState(gameState, self.a)
        self.observeState(gameState, self.b)
        beliefs = [self.getBeliefDistribution(self.a), self.getBeliefDistribution(self.b)]
        # self.displayDistributionsOverPositions(beliefs)

        actions = gameState.getLegalActions(self.index)

        aPosition = self.getEnemyPositions(self.a)
        bPosition = self.getEnemyPositions(self.b)
        hypotheticalState = gameState.deepCopy()

        if self.getMazeDistance(aPosition,
                                pacmanPosition) <= 5:  # and self.getBeliefDistribution(self.a)[aPosition] > 0.5:
            hypotheticalState = self.setEnemyPosition(hypotheticalState, aPosition, self.a)
            order = [self.index, self.a]
            if self.getMazeDistance(aPosition, pacmanPosition) < 3:
                result = self.maxValue(hypotheticalState, order, 0, 3, -10000000, 10000000, start)
            else:
                result = self.maxValue(hypotheticalState, order, 0, 2, -10000000, 10000000, start)
            # update scared moves
            if len(self.getCapsules(gameState)) != len(self.getCapsules(self.getSuccessor(gameState, result[1]))):
                # we ate a capsule!
                self.scaredMoves = self.scaredMoves + 40
            elif self.scaredMoves != 0:
                self.scaredMoves = self.scaredMoves - 1
            else:
                pass
            bestAction = result[1]
        elif self.getMazeDistance(bPosition,
                                  pacmanPosition) <= 5:  # and self.getBeliefDistribution(self.b)[bPosition] > 0.5:
            hypotheticalState = self.setEnemyPosition(hypotheticalState, bPosition, self.b)
            order = [self.index, self.b]
            if self.getMazeDistance(bPosition, pacmanPosition) < 3:
                result = self.maxValue(hypotheticalState, order, 0, 3, -10000000, 10000000, start)
            else:
                result = self.maxValue(hypotheticalState, order, 0, 2, -10000000, 10000000, start)
            bestAction = result[1]
        else:
            values = [self.evaluate(gameState, a) for a in actions]

            maxValue = max(values)
            bestActions = [a for a, v in zip(actions, values) if v == maxValue]
            bestAction = random.choice(bestActions)

        # update scared moves
        if len(self.getCapsules(gameState)) != len(self.getCapsules(self.getSuccessor(gameState, bestAction))):
            # we ate a capsule!
            self.scaredMoves = self.scaredMoves + 40
        elif self.scaredMoves != 0:
            self.scaredMoves = self.scaredMoves - 1
        else:
            pass

        # update defense scared moves
        numCapsulesDefending = len(gameState.getBlueCapsules())
        numCapsulesLeft = len(self.getSuccessor(gameState, bestAction).getBlueCapsules())
        if self.red:
            numCapsulesLeft = len(gameState.getRedCapsules())
            numCapsulesDefending = len(self.getSuccessor(gameState, bestAction).getRedCapsules())
        if numCapsulesLeft < numCapsulesDefending:
            # enemy ate a capsule!
            self.defenseScaredMoves += 40
        elif self.defenseScaredMoves != 0:
            self.defenseScaredMoves -= 1

        if bestAction == 'Stop':
            self.stopped += 1
            if self.stopped >= 3:
                self.stuck = True
        else:
            if self.stuck and self.numStuckSteps < 6:
                self.numStuckSteps += 1
            elif self.stuck and self.numStuckSteps >= 6:
                self.stuck = False
                self.stopped = 0
                self.numStuckSteps = 0
            else:
                self.stopped = 0

        rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
        if bestAction == rev:
            self.reverse += 1
            if self.reverse >= 4:
                self.flank = True
        else:
            if self.flank and self.numRevSteps < 7:
                self.numRevSteps += 1
            elif self.flank and self.numRevSteps >= 7:
                self.flank = False
                self.reverse = 0
                self.numRevSteps = 0
            else:
                self.reverse = 0

        if gameState.getAgentPosition(self.index) == self.start:
            # reset everything
            self.stuck = False
            self.stopped = 0
            self.numStuckSteps = 0
            self.flank = 0
            self.reverse = 0
            self.numRevSteps = 0
            self.go_to_furthest_position = True

        # update weird move
        successor = self.getSuccessor(gameState, bestAction)
        myPos = successor.getAgentState(self.index).getPosition()
        max_furthest_point_dist = self.getMazeDistance(myPos, self.furthest_position_along_border_of_home)

        if successor.getAgentState(
                self.index).isPacman or max_furthest_point_dist < 2:  # ((abs(myPos[0]-self.offenseGoToFurthestFoodPosition[0]) < 5 and not successor.getAgentState(self.index).isPacman) and maxfooddist < 5):
            self.go_to_furthest_position = False
        elif self.getMazeDistance(myPos, self.start) == 0:
            self.go_to_furthest_position = True

        if bestAction is None:
            bestAction = random.choice(actions)

        return bestAction

    def maxValue(self, gameState, order, index, depth, alpha, beta, start):
        # returns a value and an action so getAction can return the best action
        action = gameState.getLegalActions(order[0])[0]
        if gameState.isOver() or depth == 0 or ((time.time() - start) > 0.9):
            return [self.evaluate(gameState, None), action]
        v = -10000000
        action = None
        for a in gameState.getLegalActions(order[0]):
            try:
                newState = gameState.generateSuccessor(order[0], a)
            except:
                return [self.evaluate(gameState, None), None]

            # eat ghost
            if newState.getAgentPosition(self.index) == newState.getAgentPosition(order[1]) and self.scaredMoves > 0:
                action = a
                break

            something = self.minValue(newState, order, index + 1, depth, alpha, beta, start)
            newScore = something[0]
            compareState = something[1]
            # don't die
            if newState.getAgentPosition(self.index) == compareState.getAgentPosition(order[1]):
                continue
            if newScore > v:
                v = newScore
                action = a
            if v > beta:
                return [v, a]
            alpha = max(alpha, v)
        return [v, action]

    def minValue(self, gameState, order, index, depth, alpha, beta, start):
        if gameState.isOver() or depth == 0 or ((time.time() - start) > 0.9):
            return [self.evaluate(gameState, None), gameState]
        v = 10000000
        bestState = gameState
        for a in gameState.getLegalActions(order[index]):
            try:
                newState = gameState.generateSuccessor(order[index], a)
            except:
                return [self.evaluate(gameState, None), gameState]
            # eat pacman
            if newState.getAgentPosition(order[1]) == gameState.getAgentPosition(self.index):
                return [-1000000, newState]
            # if pacman goes next, here is where depth is decremented
            if index + 1 >= len(order):
                newScore = self.maxValue(newState, order, 0, depth - 1, alpha, beta, start)[0]
                if newScore < v:
                    v = newScore
                    bestState = newState
            # if another enemy goes
            else:
                # change to max and [0] if using partner
                v = min(v, self.minValue(newState, order, index + 1, depth, alpha, beta, start)[0])
            if v < alpha:
                return [v, bestState]
            beta = min(beta, v)
        return [v, bestState]

