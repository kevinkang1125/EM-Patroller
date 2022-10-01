import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Policy(nn.Module):
    def __init__(self, graph, index, opt):
        super(Policy, self).__init__()
        self.graph = graph
        self.idx = index
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.numS = len(list(self.graph.nodes()))
        self.numA = self.numS
        self.mask = np.array(nx.adjacency_matrix(self.graph).todense(), dtype=float)

        self.affine1 = nn.Linear(self.numS, 4 * self.numS)
        self.affine2 = nn.Linear(4 * self.numS, self.numA)

        self.optimizer = optim.Adam(self.parameters(), lr=opt.lr)
        self.epsilon = opt.epsilon

    def forward(self, x):
        # Identify current state
        idx = int(torch.nonzero(x))

        # Use GPU
        x = x.to(self.device)
        mask = torch.from_numpy(self.smoothMask(self.mask[idx])).to(self.device)

        # Get action distribution
        x = self.affine1(x)
        x = F.relu(x)
        x = self.affine2(x)
        scores = F.softmax(x, dim=0)

        # Generate legal action distribution
        maskedActionScores = scores * mask
        output = maskedActionScores / torch.sum(maskedActionScores)

        return output

    def generateStateVector(self, i):
        stateVector = np.zeros(shape=(self.numS,), dtype=float)
        stateVector[i] = 1.0

        return torch.from_numpy(stateVector).float().unsqueeze(0)[0]

    def smoothMask(self, maskVector):
        count = list(maskVector).count(1.0)
        smoothedMaskVector = np.zeros(shape=maskVector.shape, dtype=float)
        for i in range(len(maskVector)):
            if maskVector[i] == 0.0:
                smoothedMaskVector[i] = count * self.epsilon / (self.numS - count)
            else:
                smoothedMaskVector[i] = 1.0 - self.epsilon

        return smoothedMaskVector
