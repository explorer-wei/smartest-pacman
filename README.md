# Smartest Pacman

![](/pacman.png)

This is the Pacman AI written by Wei Huang (yes, it's me!) for the [Contest: Pacman Capture the Flag](https://inst.eecs.berkeley.edu/~cs188/sp10/projects/contest/contest.html). It got the first place in "CS 5804: Introduction to Artificial Intelligence" @ Virginia Tech 2015 Spring ✌️

In this game each team has one half of the board with food pellets that the other team will try to eat. Each team is made up of two agents. When an agent is on "home turf", it is a Ghost, and can eat the enemy Pac-man. When an agent is on "enemy territory", it is a Pac-man, and can be eaten by the enemy Ghosts. The goal is to eat more than the other team. There are various other nuances to the game, like the fact that your agent is not given the position of the other agents - only the noisy distance to them.

You could download the contest platform codes and read detailed rules in the original course page.

## AI algorithms used

### Reflex Agent
The agents I built are based on the idea of reflex agent, which will choose an action at each step by examining its alternatives via a state evaluation function.

#### Attacker

This is an agent that seeks food and avoids being eaten by the ghosts. The evaluation function includes the following components:

- 'getScore': +100 points; Because our final goal is to optimize the game score.
- 'eatFood': +10 points; Eating something is better than nothing.
- 'eatCapsule': +20 points; Scaring the ghosts is really useful – it can help avoid lots of trouble. Thus, I subtract 20 from evaluation score for each existing capsule, to encourage pacman eat nearby capsule.
- 'closestFood': +1 point; By subtracting the distance to nearest food, I make sure that the score will be higher when the pacman goes closer towards a food.
- 'stop': -2 points; This is used to encourage pacman keep moving and exploring.
- Escape from the ghosts (only when scaredTimer==0): If the ghost is within 4 squares (I tried 5 squares first, but it cannot win all 35 matches), then the evaluation score will be subtracted 1000 / (distance + 1) to urge the pacman to run away; otherwise, this component will take both “distance to the ghosts” and “available next steps” (i.e. avoid running into a dead end) into account.

#### Defender
This is an agent that keeps my side Pacman-free. The evaluation function is inversely proportional to the distance to the “goal”. The “goal” can be either an invader or the agent’s patrol point, depending on the BELIEFS.

### Bayesian Network

#### Inference with Time Elapse: 

Since my agents can only observe an opponent if it (or its teammate) is within 5 squares (and the distance reading is noisy!), I need to infer the probability distribution of where they could be located most of the time. 

More specifically, the BELIEFS are updated based on the current beliefs and the probability distribution of next actions:
```
newBELIEFS[pos] = Sum(oldBELIEFS[p] * probability of action(p->pos)), where p is a legal neighbor of pos.
```

#### Probability distribution of next actions

First, the next position should be somewhere legal (not a wall) and neighboring.

Then I adjust the probability distribution based on some simple assumptions:

- If the opponent is on my "home turf", then its next action is more likely to go towards a nearest food.
- If the opponent is on the other side, then its next action is more likely to chase “me” (if within 5 squares).
- Last but not least, if the opponent has just been eaten, then we know exactly that it will re-spawn at its starting position.

### Other strategies

#### Patrol area

I designate the area (I call it “mid-zone”) which is 3 squares behind the boundary as my defense line. If the enemies are too far away, or if we don’t know where they are (more than 5 positions hold the same value of BELIEFS), then my defender(s) will hover in this area to prepare for an interception.

#### Role Switch

A defender will switch to an attacker after eating an enemy. The basic idea is that: after “throwing” an enemy back to its starting position, the defender doesn’t have much work to do until the enemy reach the boundary again. So it can make use of this period of time to pillage the opponent side.

An attacker will switch to a defender after pillaging enemy territory for 100 moves. By this way, I want to make sure that the attacker can periodically carry back some food dots. A too greedy attacker (e.g.: it spends 300 moves to eat 80% of the food dots but not able to successfully return back) can only lower my game score.
