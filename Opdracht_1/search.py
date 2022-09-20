# search.py
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()

def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

class Node:
    def __init__(self, state, action, cost, parent):
        self.state = state
        self.action = action
        self.cost = cost
        self.parent = parent

    def getActionPath(self):
        if self.parent is None:
            return []
        else:
            action_path = self.parent.getActionPath()
            action_path.append(self.action)
            return action_path

def depthFirstSearch(problem):

    # een container met een LIFO (Last In First Out) qeueing policy
    fringe = util.Stack()                  
    # Maak een list voor je explored Nodes     
    Explored_Nodes = []                     
    # State, action      
    Start_Node = (problem.getStartState(), [])   
    # Push de start
    fringe.push(Start_Node)
    
    # Wanneer er nodes op de fringe zitten
    while not fringe == 0:   
        # Verwijder je current state van de list (Explored)
        Huidige_Staat, acties = fringe.pop()        
        # Indien de current state nog niet in explored nodes zit, voeg hem toe aan explored nodes
        if Huidige_Staat not in Explored_Nodes:       
            print("De x,y coördinaat van pacman is: " + str(Huidige_Staat))
            # Voeg Huidige_Staat toe aan de Explored nodes list
            Explored_Nodes.append(Huidige_Staat)                    
            if problem.isGoalState(Huidige_Staat): return acties 
            for Succes_State, Succes_Action, Stepcost in problem.getSuccessors(Huidige_Staat):
                fringe.push((Succes_State, acties + [Succes_Action]))    

def breadthFirstSearch(problem):

    # een container met een FIFO qeueing policy
    fringe = util.Queue()
    # Maak een list voor je explored Nodes     
    Explored_Nodes = []                     
    # State, action      
    Start_Node = (problem.getStartState(), [])   
    # Push de start
    fringe.push(Start_Node)

    # Wanneer er nodes op de fringe zitten
    while not fringe == 0:   
        # Verwijder je current state van de list (Explored)
        Huidige_Staat, acties = fringe.pop()        
        # Indien de current state nog niet in explored nodes zit, voeg hem toe aan explored nodes
        if Huidige_Staat not in Explored_Nodes:       
            #print("De x,y coördinaat van pacman is: " + str(Huidige_Staat))
            # Voeg Huidige_Staat toe aan de Explored nodes list
            Explored_Nodes.append(Huidige_Staat)                    
            if problem.isGoalState(Huidige_Staat): return acties 
            for Succes_State, Succes_Action, Stepcost in problem.getSuccessors(Huidige_Staat):
                fringe.push((Succes_State, acties + [Succes_Action]))       

def uniformCostSearch(problem):
    """Search the node of least total cost first."""

    #to be explored (FIFO): holds (item, cost)
    fringe = util.PriorityQueue()

    #previously expanded states (for cycle checking), holds state:cost
    Explored_Nodes = {}
    
    Start_Staat = problem.getStartState()
    Start_Node = (Start_Staat, [], 0) #(state, action, cost)
    
    fringe.push(Start_Node, 0)
    
    while not fringe.isEmpty():
        #begin exploring first (lowest-cost) node on fringe
        Huidige_Staat, acties, Huidige_Cost = fringe.pop()
       
        if (Huidige_Staat not in Explored_Nodes) or (Huidige_Cost < Explored_Nodes[Huidige_Staat]):
            #put popped node's state into explored list
            Explored_Nodes[Huidige_Staat] = Huidige_Cost

            if problem.isGoalState(Huidige_Staat):
                return acties
            else:
                #list of (successor, action, stepCost)
                opvolger = problem.getSuccessors(Huidige_Staat)
                
                for Succes_State, Succes_Action, Succes_Cost in opvolger:
                    New_Action = acties + [Succes_Action]
                    New_Cost = Huidige_Cost + Succes_Cost
                    New_Node = (Succes_State, New_Action, New_Cost)

                    fringe.update(New_Node, New_Cost)

    return acties

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""

    #to be explored (FIFO): takes in item, cost+heuristic (dijkstra + heuristic)
    fringe = util.PriorityQueue()
    # Maak een list voor je explored Nodes     
    Explored_Nodes = []
    # Definieer Start_Node met state, action, cost
    Start_Node = (problem.getStartState(), [], 0)
    # Push de start Node
    fringe.push(Start_Node, 0)

    # Wanneer er nodes op de fringe zitten
    while not fringe == 0:   

        #begin exploring first (lowest-combined (cost+heuristic) ) node on fringe
        Huidige_Staat, acties, Huidige_Cost = fringe.pop()

        #put popped node into explored list
        Explored_Nodes.append((Huidige_Staat, Huidige_Cost))

        if problem.isGoalState(Huidige_Staat): return acties

        #list of (successor, action, stepCost)
        opvolger = problem.getSuccessors(Huidige_Staat)

        #loop langs elke successor
        for Succes_State, Succes_Action, Succes_Cost in opvolger:
            #Calculeer de nieuwe cost / nodes
            New_Cost = problem.getCostOfActions(acties + [Succes_Action])
            New_Node = (Succes_State, acties + [Succes_Action], New_Cost)
            already_explored = False
            
            # Loop langs voor iedere explored node tuple          
            for explored in Explored_Nodes:
                exploredState, exploredCost = explored
                if (Succes_State == exploredState) and (New_Cost >= exploredCost): already_explored = True
                
            # Indien de successor niet explored is, gooi em op de fringe en explored lijst
            if not already_explored:
                fringe.push(New_Node, New_Cost + heuristic(Succes_State, problem))
                Explored_Nodes.append((Succes_State, New_Cost))

    return acties

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
