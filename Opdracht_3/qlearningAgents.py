# qlearningAgents.py
# ------------------
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


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

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

        # Set up counter
        self.QValue = util.Counter()

    def getQValue(self, state, action):
        if self.QValue[state, action] == 0:
          return 0
        return self.QValue[state, action]

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """

        # Get the legal actions
        legal = self.getLegalActions(state)

        # Return 0 if there are no legal actions
        if len(legal) == 0:
          return 0.0

        # Set up variable that can be overwritten
        maxQ = float('-inf')

        # Loop over the legal actions
        for action in legal:

          # Overwrite if the score is better than the previously saved score
          if self.getQValue(state, action) > maxQ:
            maxQ = self.getQValue(state, action)
        
        # Return the best score
        return maxQ

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """

        # Call empty list to save best actions
        best = []

        # Get legal actions
        legal = self.getLegalActions(state)

        # Set up variable that can be overwritten
        maxQ = float('-inf')

        # Loop over legal actions
        for action in legal:

          # Save best state and action if there is an improvement
          if self.getQValue(state, action) > maxQ:
            maxQ = self.getQValue(state, action)
            best = [action]

          # Add the action if the result is the same 
          elif self.getQValue(state, action) == maxQ:
            best.append(action)

        # Choose random action out of the best options
        return random.choice(best)

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legalActions = self.getLegalActions(state)
        action = None
        "*** YOUR CODE HERE ***"

        # Choose random action
        if util.flipCoin(self.epsilon):
          return random.choice(legalActions)
        
        # Compute action
        else:
          return self.computeActionFromQValues(state)

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        # Start with the partial score, based out of the reward, the discount and the computed value
        part = reward + self.discount * self.computeValueFromQValues(nextState)
        dict_key = state, action

        # Compute the final score
        self.QValue[dict_key] = (1.0 - self.alpha) * self.getQValue(state, action) + self.alpha * part

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
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
        "*** YOUR CODE HERE ***"

        # Get the features from the state and action
        feat = self.featExtractor
        features = feat.getFeatures(state, action)
        QValue = 0

        # Loop over all the feature keys
        for f in features.keys():

          # Add up all the weights multiplied by the corresponding features
          QValue += self.weights[f] * features[f]
        return QValue

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"

        # Get the legal actions
        next_action = self.getLegalActions(nextState)

        # Set up a variable that can be overwritten
        maxQ = float('-inf')

        # Loop over the legal actions
        for next in next_action:

          # Overwrite if improvement
          if self.getQValue(nextState, next) > maxQ:
            maxQ = self.getQValue(nextState, next)

        # If nothing happend, set to 0
        if maxQ == float('-inf'):
          maxQ = 0

        # Compute the difference between the calculated score and the QValue calculated from the function
        difference = (reward + (self.discount * maxQ)) - self.getQValue(state, action)

        # Extract features
        features = self.featExtractor.getFeatures(state, action)

        # Add up to QValue
        self.QValue[(state, action)] += self.alpha * difference

        # Loop over the feature keys
        for f in features.keys():

          # Adjust the weights
          self.weights[f] += self.alpha * difference * features[f]

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass
