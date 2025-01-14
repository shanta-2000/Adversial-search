# multiAgents.py
# --------------
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

from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as (1) you do not touch our method
    headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        legalMoves = gameState.getLegalActions()
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)
        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        This evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.
        """
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        return successorGameState.getScore()


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers. Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
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
        """
        action, score = self.minimax(0, 0, gameState)
        return action

    def minimax(self, curr_depth, agent_index, gameState):
        if agent_index >= gameState.getNumAgents():
            agent_index = 0
            curr_depth += 1

        if curr_depth == self.depth or gameState.isWin() or gameState.isLose():
            return None, self.evaluationFunction(gameState)

        best_score, best_action = None, None
        if agent_index == 0:  # Max player (Pacman)
            best_score = float('-inf')
            for action in gameState.getLegalActions(agent_index):
                next_state = gameState.generateSuccessor(agent_index, action)
                _, score = self.minimax(curr_depth, agent_index + 1, next_state)
                if score > best_score:
                    best_score = score
                    best_action = action
        else:  # Min player (Ghosts)
            best_score = float('inf')
            for action in gameState.getLegalActions(agent_index):
                next_state = gameState.generateSuccessor(agent_index, action)
                _, score = self.minimax(curr_depth, agent_index + 1, next_state)
                if score < best_score:
                    best_score = score
                    best_action = action

        return best_action, best_score


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Minimax agent with Alpha-Beta pruning (optimized).
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using Alpha-Beta pruning and the evaluation function.
        """

        def alpha_beta(state, depth, agentIndex, alpha, beta):
            if depth == 0 or state.isWin() or state.isLose():
                return self.evaluationFunction(state)

            if agentIndex == 0:  # Pacman turn (maximize)
                value = float('-inf')
                best_action = None
                for action in state.getLegalActions(agentIndex):
                    successor = state.generateSuccessor(agentIndex, action)
                    score = alpha_beta(successor, depth, 1, alpha, beta)
                    if score > value:
                        value = score
                        best_action = action
                    alpha = max(alpha, value)
                    if beta <= alpha:
                        break  # Beta pruning
                if depth == self.depth:
                    return best_action
                return value
            else:  # Ghost turn (minimize)
                value = float('inf')
                next_agent = (agentIndex + 1) % state.getNumAgents()
                next_depth = depth if next_agent != 0 else depth - 1
                for action in state.getLegalActions(agentIndex):
                    successor = state.generateSuccessor(agentIndex, action)
                    score = alpha_beta(successor, next_depth, next_agent, alpha, beta)
                    value = min(value, score)
                    beta = min(beta, value)
                    if beta <= alpha:
                        break  # Alpha pruning
                return value

        return alpha_beta(gameState, self.depth, 0, float('-inf'), float('inf'))


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    Expectimax agent optimized for better decision-making.
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using the evaluation function.
        """

        def expectimax(state, depth, agentIndex):
            if depth == 0 or state.isWin() or state.isLose():
                return self.evaluationFunction(state)

            if agentIndex == 0:  # Pacman turn (maximize)
                value = float('-inf')
                for action in state.getLegalActions(agentIndex):
                    successor = state.generateSuccessor(agentIndex, action)
                    value = max(value, expectimax(successor, depth - 1, 1))
                return value
            else:  # Ghost turn (expected value)
                value = 0
                actions = state.getLegalActions(agentIndex)
                prob = 1 / len(actions) if actions else 0
                for action in actions:
                    successor = state.generateSuccessor(agentIndex, action)
                    next_agent = (agentIndex + 1) % state.getNumAgents()
                    next_depth = depth - 1 if next_agent == 0 else depth
                    value += prob * expectimax(successor, next_depth, next_agent)
                return value

        return max(gameState.getLegalActions(0),
                   key=lambda action: expectimax(gameState.generateSuccessor(0, action), self.depth, 1))


def betterEvaluationFunction(currentGameState):
    """
    A better evaluation function that considers:
      - Proximity to food
      - Proximity to capsules
      - Ghost distances (scared vs active)
    """
    pacman_pos = currentGameState.getPacmanPosition()
    food = currentGameState.getFood().asList()
    ghost_states = currentGameState.getGhostStates()
    capsules = currentGameState.getCapsules()

    score = currentGameState.getScore()

    if food:
        food_distances = [manhattanDistance(pacman_pos, food_pos) for food_pos in food]
        score += 10 / (min(food_distances) + 1)

    if capsules:
        capsule_distances = [manhattanDistance(pacman_pos, cap) for cap in capsules]
        score += 5 / (min(capsule_distances) + 1)

    for ghost in ghost_states:
        ghost_distance = manhattanDistance(pacman_pos, ghost.getPosition())
        if ghost.scaredTimer == 0:
            score -= 200 / (ghost_distance + 1)
        else:
            score += 20 / (ghost_distance + 1)

    return score


# Abbreviation
better = betterEvaluationFunction
