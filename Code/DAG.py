import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from itertools import product
from pyvis.network import Network


class DAG:
    def __init__(self, adjacency_matrix = None, biass = None, n = 5, strength = 2, roots = 1, precalculate_paths = False):
        assert n > 0, "n must be greater than 0"
        assert roots > 0, "roots must be greater than 0"
        if biass is not None:
            if adjacency_matrix is not None: 
                assert len(biass) == adjacency_matrix.shape[0], "biass must be of same size as adjacency matrix"
            else:
                assert len(biass) == n, "biass must be of size n"

        self.strength = strength
        self.roots = roots

        if adjacency_matrix is not None:
            self.adjacency_matrix = adjacency_matrix
            self.size = adjacency_matrix.shape[0]
        else:
            self.adjacency_matrix = self.random_dag(n = n, strength = strength, roots = roots)
            self.size = n

        if biass is not None:
            self.biass = biass
        else:
            self.biass = np.ones(n)

        self.paths = []

        self.precalculated_paths = precalculate_paths
        if precalculate_paths:
            self.precalculate_paths()

    def precalculate_paths(self):
        pths = np.empty((self.size, self.size), dtype = object)
        for i in range(self.size):
            for j in range(self.size):
                if i == j:
                    pths[i,j] = []
                    continue

                pths[i,j] = self.all_paths_between(i, j)

        self.paths = pths.copy()

    def random_dag(self, n = 5, strength = 2, roots = 1):
        adjacency_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(i+1, n):
                if np.random.randint(0, 2) == 0:
                    edge = np.random.uniform(-strength, strength)
                else:
                    edge = 0

                if j < roots and i < roots:
                    edge = 0
                adjacency_matrix[i, j] = edge
                
        # make sure each node has at least one parent
        for i in range(roots, n):
            if np.sum(adjacency_matrix[:, i]) == 0:
                adjacency_matrix[np.random.randint(0, i), i] = 1
                
        # make sure roots have no parents
        for i in range(roots):
            adjacency_matrix[:, i] = 0

        return adjacency_matrix

    def get_varsortability(self, analytical = False, simulated = False, N = 1000):
        assert analytical or simulated, "must calculate at least one of analytical or simulated"
        _return = {}
        if analytical:
            ana = self.get_analytical_var()
            analytical = self.varsortability(ana)
            _return["analytical"] = analytical
        if simulated:
            sim = self.get_simulated_var(N)
            simulated = self.varsortability(sim)
            _return["simulated"] = simulated

        return _return

    def get_continous_varsortability(self, analytical = False, simulated = False, N = 1000):
        assert analytical or simulated, "must calculate at least one of analytical or simulated"
        _return = {}
        if analytical:
            ana = self.get_analytical_var()
            analytical = self.continous_varsortability(ana)
            _return["analytical"] = analytical
        if simulated:
            sim = self.get_simulated_var(N)
            simulated = self.continous_varsortability(sim)
            _return["simulated"] = simulated

        return _return

    def varsortability(self, variances):
        N = 0
        sortable = 0
        if not self.precalculated_paths:
            self.precalculate_paths()

        for i in range(self.size):
            for j in range(self.size):
                if i == j:
                    continue
                for p in self.paths[i,j]:
                    N += 1
                    if variances[p[0][0]] < variances[p[-1][1]]:
                        sortable += 1
                    elif variances[p[0][0]] == variances[p[-1][1]]:
                        sortable += 0.5

        return sortable / N

    def continous_varsortability(self, variances):
        numerator = 0
        for i in range(self.size):
            for j in range(self.size):
                if i == j:
                    continue

                if self.adjacency_matrix[i, j] == 0:
                    continue
                if (variances[i] + variances[j]) == 0:
                    print("WTH")
                numerator += int(np.sign(variances[j] - variances[i]) > 0) + (variances[j] - variances[i]) / np.max([variances[j] + variances[i], 1e-10])

        return numerator 
        
    def plot(self):
        plt.figure(figsize=(8,8))
        G = nx.DiGraph(self.adjacency_matrix)
        edge_labels = nx.get_edge_attributes(G, 'weight')

        pos=nx.spring_layout(G)
        fs = 15
        nx.draw(G, pos = pos, node_size=1000, node_color="skyblue", edge_color="black", width=3, font_size=fs, font_weight='bold', arrowsize=10, with_labels=True)
        nx.draw_networkx_edge_labels(G, pos = pos, edge_labels=edge_labels, font_size=fs, font_weight='bold')

        plt.show()

    def adj2edges(self):
        edges = []
        for i in range(self.adjacency_matrix.shape[0]):
            for j in range(self.adjacency_matrix.shape[1]):
                if abs(self.adjacency_matrix[i, j]) > 0:
                    edges.append((i, j))
        return np.array(edges).copy()


    def all_paths_between(self, a, b):
        edges = self.adj2edges()
        allpths = self.find_all_paths(edges, a, b)
        return allpths
    
    
    def analytical_var_node(self, node):
        assert node < self.size and node >= 0, "node out of bounds"

        if not self.precalculated_paths:
            self.precalculate_paths()

        val = 0
        for i in range(0, self.size):
            allpaths = self.paths[i, node]
            
            for path1, path2 in product(allpaths, allpaths):
                if len(path1) == 0 or len(path2) == 0:
                    continue

                p1 = np.prod([self.adjacency_matrix[edge[0], edge[1]] for edge in path1])  
                p2 = np.prod([self.adjacency_matrix[edge[0], edge[1]] for edge in path2])
                p_total = p1 * p2 * self.biass[i]             

                val += p_total

        val += self.biass[node]

        return val

    def get_analytical_var(self):
        return np.array([self.analytical_var_node(i) for i in range(self.size)])

    def find_all_paths(self, edges, src, dest):
        if (src == dest):
            return [[]]
        else:
            paths = []
            for adjnode in filter(lambda x: x[0] == src, edges):
                for path in self.find_all_paths(edges, adjnode[1], dest):
                    paths.append([adjnode] + path)
            return paths

    def get_simulated_var(self, N = 100):
        return self.get_simulated_data(N).var(axis = 1)

    def mutate(self):
        _adja = self.adjacency_matrix.copy().astype(float)
        # mutate edges
        for i in range(self.size):
            for j in range(self.size):
                if i == j:
                    continue
                if self.adjacency_matrix[i, j] == 0:
                    continue

                if np.random.uniform(0, 1) < 0.5:
                    edge = np.random.uniform(0.5, self.strength) * np.random.choice([-1, 1])
                    _adja[i, j] = edge
                else:
                    edge = self.adjacency_matrix[i, j]
                    _adja[i, j] = self.adjacency_matrix[i, j]

                if edge == 0:
                    print("edge is 0", i, j)

        # make child
        child = DAG(n = self.size, adjacency_matrix = _adja, biass = self.biass, strength = self.strength)
        return child

    def get_simulated_data(self, N = 100):
        adj = self.adjacency_matrix.copy()
        values = np.zeros((self.size, N))
        visited = []

        while True:
            # find nodes without parents
            wh =  np.where(np.sum(adj, axis = 0) == 0)[0]
            # print("wh",np.where(np.sum(adj, axis = 0) == 0)[0])
            roots = list(filter(lambda x: x not in visited, wh))
            if len(roots) == 0:
                break
            for root in roots:
                visited.append(root)
                # add bias
                values[root] += np.random.normal(0, self.biass[root], N)

                # propagate values
                for node, weigth in enumerate(adj[root, :]):
                    # if there is a connection
                    if abs(weigth) > 0:
                        values[node] += values[root] * weigth
                    
                # remove connection
                adj[root, :] = 0

        return values