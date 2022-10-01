import numpy as np
from torch.distributions import Categorical

from emp.policy import Policy


class Robot:
    def __init__(self, graph, index, opt):
        pi = Policy(graph, index, opt)
        self.policy = pi.to(pi.device)
        del pi
        print(f'The optimizer is using {self.policy.device} for policy {self.policy.idx}!')

        self.Experience = []
        self.visitHistory = []
        self.stateVector = np.zeros(shape=(self.policy.numS,), dtype=int)

    def initPosition(self, opt):
        if opt.env == 'HOUSE':b = 3
        elif opt.env == 'OFFICE': b = 43
        elif opt.env == 'MUSEUM': b = 1
        else: b = int(input('\nPlease assign the starting node:'))
        self.stateVector[b - 1] = 1
        print(f'Robot {self.policy.idx} starts at node {b}!')

    def selectAction(self, actionDistribution, currentNode):
        action = int(Categorical(actionDistribution).sample())
        while self.policy.mask[currentNode - 1, action] == 0.0:
            action = int(Categorical(actionDistribution).sample())
        return action + 1

    def executeAction(self, chosenAction):
        self.stateVector = np.zeros(shape=(self.policy.numS,), dtype=int)
        self.stateVector[chosenAction - 1] = 1

    def updateVisitHistory(self, currentNode):
        self.visitHistory.append(currentNode)

    def getVisitFrequency(self):
        visitFrequency = []
        for i in range(self.policy.numS):
            visitFrequency.append(self.visitHistory.count(i + 1))

        return visitFrequency

    def getCurrentNode(self):
        return list(self.stateVector).index(1) + 1

    def reset(self):
        self.Experience.clear()
        self.visitHistory.clear()
        self.stateVector = np.zeros(shape=(self.policy.numS,), dtype=int)
