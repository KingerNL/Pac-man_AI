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
#                first = 'DefensiveReflexAgent', second = 'DefensiveReflexAgent'):
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
               first='DefensiveReflexAgent', second='DefensiveReflexAgent'):
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


class DefensiveReflexAgent(ParticlesCTFAgent):

    def getFeatures(self, gameState, action):

        features = util.Counter()
        successor = self.getSuccessor(gameState, action)

        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()

        numCapsulesDefending = len(gameState.getBlueCapsules())
        numCapsulesLeft = len(successor.getBlueCapsules())
        if self.red:
            numCapsulesDefending = len(gameState.getRedCapsules())
            numCapsulesLeft = len(successor.getRedCapsules())

        # are we scared?
        localDefenseScaredMoves = 0
        if numCapsulesLeft < numCapsulesDefending:
            localDefenseScaredMoves = self.defenseScaredMoves + 40
        elif self.defenseScaredMoves != 0:
            localDefenseScaredMoves = self.defenseScaredMoves - 1

        # Computes whether we're on defense (1) or offense (0)
        features['onDefense'] = 1
        if myState.isPacman: features['onDefense'] = 0  # lower by 100 points to discourage attacking enemy

        # Enemy
        enemyIndices = self.getOpponents(gameState)
        invaders = [successor.getAgentState(index) for index in enemyIndices if
                    successor.getAgentState(index).isPacman and successor.getAgentState(index).getPosition() != None]

        minEnemyDist = 0
        genEnemyDist = 0
        smallerGenEnemyDist = 0
        if len(invaders) > 0:
            minEnemyDist = min([self.getMazeDistance(myPos, enemy.getPosition()) for enemy in invaders])
        else:
            gen_dstr = [self.getBeliefDistribution(index) for index in enemyIndices]

            # A
            a_dist = 0
            for loc, prob in gen_dstr[0].items():
                a_dist += prob * self.getMazeDistance(loc, myPos)

            # B
            b_dist = 0
            for loc, prob in gen_dstr[1].items():
                b_dist += prob * self.getMazeDistance(loc, myPos)

            # only pursue the closest gen enemy
            genEnemyDist = b_dist
            smallerGenEnemyDist = a_dist
            if a_dist > b_dist:
                genEnemyDist = a_dist
                smallerGenEnemyDist = b_dist

        if localDefenseScaredMoves > 0:
            # we are scared of our enemies
            # we are scared of all our enemies
            if minEnemyDist > 0:
                # very scared of the closest exact enemy
                features['exactInvaderDistanceScared'] = minEnemyDist

                # need to get genEnemyDist and smallerEnemyDist
                # ============================================
                gen_dstr = [self.getBeliefDistribution(index) for index in enemyIndices]

                # A
                a_dist = 0
                for loc, prob in gen_dstr[0].items():
                    a_dist += prob * self.getMazeDistance(loc, myPos)

                # B
                b_dist = 0
                for loc, prob in gen_dstr[1].items():
                    b_dist += prob * self.getMazeDistance(loc, myPos)

                # only pursue the closest gen enemy
                genEnemyDist = b_dist
                smallerGenEnemyDist = a_dist
                if a_dist > b_dist:
                    genEnemyDist = a_dist
                    smallerGenEnemyDist = b_dist
                # ============================================
                # otherwise it is already calcualted

            features['generalInvaderDistanceScared'] = genEnemyDist
            features['smallerGeneralInvaderDistanceScared'] = smallerGenEnemyDist


        else:
            # we want to chase our enemies
            # we are only interested in chasing the closest enemy

            features['numInvaders'] = len(invaders)

            if minEnemyDist > 0:
                features['exactInvaderDistance'] = minEnemyDist
            else:
                features['generalEnemyDistance'] = genEnemyDist

        # punish staying in the same place
        if action == Directions.STOP: features['stop'] = 1
        # punish just doing the reverse
        rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        return features

    def getWeights(self, gameState, action):
        return {'numInvaders': -1000, 'onDefense': 100, 'stop': -100, 'reverse': -2,
                'exactInvaderDistance': -3.5, "generalEnemyDistance": -30,
                'exactInvaderDistanceScared': -1000, 'generalInvaderDistanceScared': -200,
                'smallerGeneralInvaderDistanceScared': -150}

