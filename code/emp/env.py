import numpy as np
import networkx as nx

nodes_house = [1,2,3,4,5,6,7,8,9]
edges_house = [(1,4),(1,8),(2,8),(2,9),(3,4),(4,5),(4,6),(4,9),(5,7),(6,7)]

nodes_office = [1+i for i in range(60)]
edges_office = [(1,56),(2,56),(3,56),(4,56),(5,58),(6,58),(7,58),(8,45),(9,45),(10,45),(11,46),(12,47),(13,47),(14,47),
                (15,48),(16,59),(17,60),(18,60),(19,59),(20,59),(21,47),(22,23),(22,45),(23,44),(24,45),(25,26),(25,44),
                (25,50),(26,53),(27,54),(28,54),(29,54),(30,54),(31,55),(32,55),(33,55),(34,56),(35,56),(36,56),(37,54),
                (38,54),(39,54),(40,54),(41,58),(42,58),(43,44),(44,45),(44,49),(44,58),(45,46),(46,47),(47,48),(48,59),
                (49,50),(49,60),(50,51),(50,53),(51,52),(53,54),(53,58),(54,55),(55,56),(56,57),(57,58),(59,60)]

nodes_museum = [1+i for i in range(70)]
edges_museum = [(1,2),(1,5),(1,7),(1,9),(2,3),(2,4),(6,7),(7,8),(9,10),(10,11),(10,12),(10,13),(10,18),(10,19),(10,20),
                (10,27),(11,14),(11,28),(12,15),(12,16),(13,17),(14,15),(16,17),(18,19),(18,26),(18,70),(19,20),(19,25),
                (19,26),(20,21),(20,24),(21,22),(22,23),(23,24),(24,25),(25,26),(27,49),(28,29),(29,34),(30,31),(30,33),
                (31,32),(32,33),(32,37),(33,36),(34,35),(35,36),(35,40),(35,49),(36,37),(36,39),(37,38),(38,42),(39,40),
                (39,41),(40,43),(41,42),(43,44),(43,47),(44,45),(45,46),(46,47),(47,48),(48,50),(49,50),(49,58),(50,51),
                (50,52),(52,53),(52,56),(53,54),(53,55),(53,56),(54,55),(56,64),(57,58),(57,63),(58,59),(58,62),(59,60),
                (59,61),(60,61),(60,70),(61,68),(61,69),(62,67),(63,64),(63,66),(64,65),(65,66),(66,67),(67,68),(68,69)]


class Environment:
    def __init__(self, opt):
        # Topology
        if opt.env == 'HOUSE': V, E = nodes_house, edges_house
        elif opt.env == 'OFFICE': V, E = nodes_office, edges_office
        elif opt.env == 'MUSEUM': V, E = nodes_museum, edges_museum
        else:
            V, E = None, None
            Exception(f'{opt.env} not implemented!')
        self.nodes, self.edges = V, E
        self.graph = self.getGraph()
        # States and actions
        self.stateSpace = set([1 + i for i in range(len(self.nodes))])
        self.actionSpace = set([1 + i for i in range(len(self.nodes))])
        self.visitFrequency = np.zeros(shape=(len(self.nodes),), dtype=float)
        # Epsilon (avoid NaN from logarithmic operations)
        self.epsilon = opt.epsilon

    def getGraph(self):
        graph = nx.Graph()
        graph.add_nodes_from(self.nodes)
        graph.add_edges_from(self.edges)

        return graph

    def updateVisitFrequency(self, visitRecord_t):
        self.visitFrequency += visitRecord_t

    def getVisitFrequency(self, opt):
        visitFrequency = self.visitFrequency
        visitFrequency /= opt.step
        for i in range(len(self.nodes)):
            if visitFrequency[i] == 0.0: visitFrequency[i] += self.epsilon

        return visitFrequency

    def isAllNodesVisited(self):
        return np.all(self.visitFrequency != 0)

    def reset(self):
        self.visitFrequency = np.zeros(shape=(len(self.nodes),), dtype=int)