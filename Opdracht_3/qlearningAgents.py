# qlearningAgents.py
# ------------------
# Licensing Information:  You are free to use or extend these projects for 
# educational purposes provided that (1) you do not distribute or publish 
# solutions, (2) you retain this notice, and (3) you provide clear 
# attribution to UC Berkeley, including a link to 
# http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero 
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and 
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from argparse import Action
from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *
from operator import add, mul
import random,util,math

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)
        #initialize dict-counter for holding: key = (state, action) val = value
        self.qVals = util.Counter()

    def getQValue(self, state, action):
        # get the value corresponding to Q(state, action)
        # If Should return 0.0 if we never seen a state or (state,action) tuple
        if self.qVals[(state, action)] == 0:
            return 0.0
        return self.qVals[(state, action)]

    def computeValueFromQValues(self, state):
        
        legal_actions = self.getLegalActions(state)

        if legal_actions:
            return max([self.getQValue(state, a) for a in legal_actions])
        else:
            return 0.0

    def computeActionFromQValues(self, state):
        legal_actions = self.getLegalActions(state)

        if not legal_actions:
            return None

        action_QValue_pairs = [(a, self.getQValue(state, a)) for a in legal_actions]
        max_pair = max(action_QValue_pairs, key=lambda x: x[1])
        return max_pair[0]
    
    def getAction(self, state):
        legal_actions = self.getLegalActions(state)
        random_action = util.flipCoin(self.epsilon)

        if not legal_actions:
            return None

        if random_action:
            return random.choice(legal_actions)
        else:
            return self.getPolicy(state)

    def update(self, state, action, nextState, reward):
        #calculate Q(s,a)
        oldValue = self.getQValue(state, action)
        theOld = (1-self.alpha) * oldValue
        theReward = self.alpha * reward
        if not nextState:
            self.qVals[(state, action)] = theOld + theReward
        else:
            theNextState = self.alpha * self.discount * self.getValue(nextState)
            self.qVals[(state, action)] = theOld +theReward + theNextState

    def getPolicy(self, state):
        # Compute the best action to take in a state.  
        # Note that if there are no legal actions, which is the case at the terminal state,
        # you should return None.
        actions = self.getLegalActions(state)
        #return None if there are no actions from this state
        if len(actions) == 0:
            return None
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        # Note that if there are no legal actions, which is the case at the terminal state, 
        # you should return a value of 0.0.
        actions = self.getLegalActions(state)
        #return None if there are no actions from this state
        if len(actions) == 0:
            return 0.0
        return self.computeValueFromQValues(state)

class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action

class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
        Should return Q(state,action) = w * featureVector
        where * is the dotProduct operator
        """
        """Description:
            [Here we follow the formula:
            Q(s,a) = SUM ( Wi * feature(s,a)) for all features]
            """
        """ YOUR CODE HERE """
        q = 0
        currentFeatures = self.featExtractor.getFeatures(state,action)
        for feature in currentFeatures:
            score = currentFeatures[feature]
            q += self.weights[feature] * score
        return q
        # util.raiseNotDefined()
        """ END CODE """

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        #differece = reward + gamma*Q(s', a') - Q(s,a)
        difference = reward + self.discount*self.computeValueFromQValues(nextState) - self.getQValue(state, action)
        weights = self.getWeights()
        #if weight vector is empty, initialize it to zero
        if len(weights) == 0:
            weights[(state,action)] = 0
        features = self.featExtractor.getFeatures(state, action)
        #iterate over features and multiply them by the learning rate (alpha) and the difference
        for key in features.keys():
            features[key] = features[key]*self.alpha*difference
        #sum the weights to their corresponding newly scaled features
        weights.__radd__(features)
        #update weights
        self.weights = weights.copy()
        
    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            #print self.getWeights()
            pass