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
import distanceCalculator
from util import nearestPoint

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'SmartAgent', second = 'SmartAgent'):
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

  # The following line is an example only; feel free to change it.
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########

class SmartAgent(CaptureAgent):    
  def registerInitialState(self, gameState):
    self.start = gameState.getAgentPosition(self.index)
    CaptureAgent.registerInitialState(self, gameState)
    
    self.length, self.width = gameState.getWalls().asList()[-1] # length and width measure how large the board is
    self.legalPositions = [p for p in gameState.getWalls().asList(False) if p[1] > 1] # positions without walls    

    self.midzone = [] # midzone will represent positions on our side of the border    
    offset = -3 if self.red else 4 # the offset from the middle our ghosts should hover at
    self.walls = list(gameState.getWalls())
    for i in range(self.width): # assign midzone
      if not self.walls[self.length/2+offset][i]:
        self.midzone.append(((self.length/2+offset), i))
        
    # two agents has a different starting position
    if self.index == max(gameState.getRedTeamIndices()) or self.index == max(gameState.getBlueTeamIndices()):
      x, y = self.midzone[3*len(self.midzone)/4]
    else:
      x, y = self.midzone[1*len(self.midzone)/4]      
    self.goalTile = (x, y) # goalTile is the location the agent is trying to get towards
    
    self.attack = 0 # after eating an enemy, our agent switch to attacker
    self.timer = 0 # timer for how long to attack
    
    global BELIEFS 
    BELIEFS = [util.Counter()] * gameState.getNumAgents() 
    for i in self.getOpponents(gameState): # inference for each enemy
      BELIEFS[i][gameState.getInitialAgentPosition(i)] = 1.0 # all BELIEFS start at agent initial positions                      
    

  def getDistribution(self, gameState, p): # the distribution for where an enemy could be
    distribution = util.Counter()
    possibleAct = [(p[0] - 1, p[1]), (p[0] + 1, p[1]), (p[0], p[1] - 1), (p[0], p[1] + 1), (p[0], p[1])]
    foodList = self.getFood(gameState).asList()
    if self.red and p[0] < self.length / 2 or not self.red and p[0] > self.length / 2:
      # Use offensive probability weights
      for act in possibleAct:
        if act in self.legalPositions:
          distribution[act] = 1.0
          if len(foodList) > 0:
            minDistance = min([self.getMazeDistance(act, food) for food in foodList])
            if minDistance == 0:
              distribution[act] += 0.2
    else:
      # Use defensive probability weights
      for act in possibleAct:
        if act in self.legalPositions:
          distribution[act] = 1.0
          myPos = gameState.getAgentState(self.index).getPosition()
          if self.getMazeDistance(act, myPos) <= 5:
            distribution[act] += 0.2
    return distribution
  
  
  def elapseTime(self, gameState): # how an agent could move from where they currently are
    for agent, belief in enumerate(BELIEFS):
      if agent in self.getOpponents(gameState):
        newBELIEFS = util.Counter()
        pos = gameState.getAgentPosition(agent) # checks to see what we can actually see
        if pos != None:
          newBELIEFS[pos] = 1.0
        else:
          for pos in belief:# look at all current BELIEFS
            if pos in self.legalPositions and belief[pos] > 0: # check all legal positions
              possiblePos = self.getDistribution(gameState, pos)
              for p in possiblePos:# iterate over these probabilities
                newBELIEFS[p] += belief[pos] * possiblePos[p] # the new chance is old chance * prob of this location from p
          if len(newBELIEFS) == 0:
            oldState = self.getPreviousObservation()
            if oldState != None and oldState.getAgentPosition(agent) != None: # just ate an enemy
              newBELIEFS[oldState.getInitialAgentPosition(agent)] = 1.0
              self.attack = 1 # switch to attacker
            else:
              for p in self.legalPositions: newBELIEFS[p] = 1.0
        BELIEFS[agent] = newBELIEFS
    # self.displayDistributionsOverPositions(BELIEFS)

  def observe(self, agent, noisyDistance, gameState): # looks for where the enemies are
    myPos = gameState.getAgentPosition(self.index)
    allPossible = util.Counter() # current rounds probabilities
    for p in self.legalPositions: # each position on board
      trueDistance = util.manhattanDistance(p, myPos)# distance between point and agent
      allPossible[p] += gameState.getDistanceProb(trueDistance, noisyDistance)
    for p in self.legalPositions:
      BELIEFS[agent][p] *= allPossible[p] # new values are product of prior probability and new probability
    BELIEFS[agent].normalize() # normalize the new multiplied probs
          
  def chooseAction(self, gameState): # chooses action for next turn
    actions = gameState.getLegalActions(self.index)
    opponents = self.getOpponents(gameState)
    noisyD = gameState.getAgentDistances() # each index in noisyD is an observation
    myPos = gameState.getAgentPosition(self.index)

    for agent in opponents:
      self.observe(agent, noisyD[agent], gameState) # observe each opponenet

    self.locations = [self.midzone[len(self.midzone)/2]] * gameState.getNumAgents() # prepare in the center of the board
    for i, belief in enumerate(BELIEFS):
      maxLoc = 0
      checkForAllEq = 0
      for val in BELIEFS[i]:
        if belief[val] == maxLoc and maxLoc > 0: # checks if there are lots of equal probability places enemy is
          checkForAllEq += 1 # if there are, we want to ignore this inference as it is inaccurate
        elif belief[val] > maxLoc: #locations has predicted
          maxLoc = belief[val]
          self.locations[i] = val          
      if checkForAllEq > 5: # if we don't know where they are, continue on as planned
        self.locations[i] = self.goalTile 	 
      
    self.elapseTime(gameState)
            
    if self.timer > 100: # controls how long we pillage their side
      self.attack = 0
      self.timer = 0

    if self.attack:
      actions = max(actions, key=lambda action: self.attacker(gameState, action))	
    else:
      actions = max(actions, key=lambda action: self.defender(gameState, action))
    return actions

  def getSuccessor(self, gameState, action):
    """
    Finds the next successor which is a grid position (location tuple).
    """
    successor = gameState.generateSuccessor(self.index, action)
    pos = successor.getAgentState(self.index).getPosition()
    if pos != nearestPoint(pos):
      # Only half a grid position was covered
      return successor.generateSuccessor(self.index, action)
    else:
      return successor
  
  def attacker(self, gameState, action):
    self.timer += 1    
    successor = self.getSuccessor(gameState, action)

    features = util.Counter()
    features['getScore'] = self.getScore(successor)
    if action == Directions.STOP: features['stop'] = 1
    foodList = self.getFood(successor).asList() 
    features['eatFood'] = -len(foodList)
    capsuleList = self.getCapsules(successor)
    features['eatCapsule'] = -len(capsuleList)
    myPos = successor.getAgentPosition(self.index)
    # Compute distance to the nearest food
    if len(foodList) > 0: # This should always be True,  but better safe than sorry      
      features['closestFood'] = min([self.getMazeDistance(myPos, food) for food in foodList])
    
    utility = 100*features['getScore'] + 10*features['eatFood'] + 20*features['eatCapsule'] - features['closestFood'] - 2*features['stop']    
    for agent in self.getOpponents(gameState):
      pos = gameState.getAgentPosition(agent) # checks to see what we can actually see
      if pos != None and (self.red and pos[0] > self.length / 2 or not self.red and pos[0] < self.length / 2) and not gameState.getAgentState(agent).scaredTimer > 0:
        dist = self.getMazeDistance(myPos, pos)
        if dist < 4:
          utility -= 1000 / (dist + 1)
        else: 
          utility += (len(successor.getLegalActions(self.index))) * dist
    return utility

  def defender(self, gameState, action):
    successor = gameState.generateSuccessor(self.index, action)
    myPos = successor.getAgentPosition(self.index)
    foodList = self.getFood(successor).asList()
    opponents = self.getOpponents(gameState) 
    
    for agent in self.getOpponents(gameState):
      minDist = 999
      dist = self.getMazeDistance(myPos, self.locations[agent])
      intercept = 0
      for pos in self.midzone: # identify intercept location 
        if self.getMazeDistance(pos, self.locations[agent]) < dist:
          dist = self.getMazeDistance(pos, self.locations[agent])
          if dist < minDist:
            minDist = dist
          intercept = pos
      if dist < 10 and intercept != 0 and dist <= minDist:
        self.goalTile = intercept # whether it's worth it to try and intercept yet
            
    for agent in self.getOpponents(gameState): 
      x, y = self.locations[agent]
      # if on my home side
      if (self.red and x < self.length / 2) or (not self.red and x > self.length / 2):
        self.goalTile = self.locations[agent] # chasing after an invader

    distToGoal = self.getMazeDistance(myPos, self.goalTile)		

    if self.getMazeDistance(myPos, self.goalTile) == 0: # if we are at a goal, return to patrol positions
      if self.index == max(gameState.getRedTeamIndices()) or self.index == max(gameState.getBlueTeamIndices()):
        self.goalTile = self.midzone[3*len(self.midzone)/4]
      else:
        self.goalTile = self.midzone[1*len(self.midzone)/4]
    
    return -1 * distToGoal
