import numpy as np
from DAG import DAG
from itertools import combinations



def all_combinations(_list, up_to = 1):
    for i in range(1, up_to + 1):
        for c in combinations(_list, i):
            yield c


n = 6

# # upper triangular matrix
fully_connected_adj = np.triu(np.ones((n,n)), 1)
# random_fully_connected_adj = np.random.uniform(0.5, 2, size = (n,n))
random_fully_connected_adj = np.random.randint(0, 3, size = (n,n))

for i in range(n):
    for j in range(0, i+1):
        random_fully_connected_adj[i,j] = 0



VERBOSE = False


dag = DAG(n = n, roots = 1, strength = 5, precalculate_paths = False, adjacency_matrix=random_fully_connected_adj, integer = True)
flip_up_to = 5

mat = dag.adjacency_matrix.copy()
mat = (mat != 0).astype(int)


print(dag.adjacency_matrix)
# print(dag.get_simulated_var(1000000).astype(int))
print(dag.get_analytical_var())

# print(dag.get_analytical_var().astype(int))
print(dag.get_varsortability(smart = True)["smart"])


final_adj = dag.adjacency_matrix.copy()


for TOP in range(1, n):
    TOP_N_TO_MAX = TOP
    print("TOP_N_TO_MAX", TOP_N_TO_MAX)

    start_vars = dag.get_analytical_var()

    for node in range(dag.size):
        if VERBOSE:
            print("node", node)
        ways_into_node = np.where(mat[:, node] != 0)[0]
        if len(ways_into_node) < 2:
            continue

        ways = np.empty((len(ways_into_node), dag.size))

        for i, way_into_node in enumerate(ways_into_node):
            mat2 = mat.copy()
            mat2[:, node] = 0
            mat2[way_into_node, node] = 1
            
            exp = np.zeros_like(mat2)
            for n_exp in range(dag.size):
                exp += np.linalg.matrix_power(mat2, n_exp)
            exp[node, node] = 0

            exp = exp != 0

            ways[i] = exp[:, node]

        n_ways = ways.sum(axis = 0)

        usable_ways = set([])

        for parent in range(node):
            if n_ways[parent] < 2:
                continue
            
            wh = np.where(ways[:, parent] > 0)[0]
            
            usable_ways |= set(ways_into_node[wh])

        if VERBOSE:
            print("for node",node)
            print("the ways that has covariance are:", usable_ways)


        usable_ways = list(usable_ways)

        # try all combinations using itertools combinations
        all_combinations_up_to = list(all_combinations(usable_ways, flip_up_to))

        vars = [start_vars[node]]
        for _all in all_combinations_up_to:
            matrix_w_flip = final_adj.copy()

            for uw in _all:
                matrix_w_flip[uw, node] *= -1

            new_dag = DAG(n = dag.size, roots = 1, strength = 15, precalculate_paths = False, adjacency_matrix = matrix_w_flip, integer = True)
            var = new_dag.get_simulated_data_smart(100000).var(axis = 0)[node]
            # var = new_dag.get_simulated_var(100000)[node]

            vars.append(var)

        # find index of smallest var

        if node > TOP_N_TO_MAX:
            index = np.argmin(vars)
        else:
            index = np.argmax(vars)

        if index != 0:
            index -= 1
            if VERBOSE:
                print("flipping the way from", all_combinations_up_to[index], "to", node)
            for way in all_combinations_up_to[index]:
                final_adj[way, node] *= -1

    if VERBOSE:
        print("final adj:")
        print(final_adj)
        print("final var:")

    newdag = DAG(n = dag.size, roots = 1, strength = 15, precalculate_paths = False, adjacency_matrix = final_adj, integer = True)

    # print(newdag.get_analytical_var().astype(int))
    # print(newdag.get_simulated_var(1000000).astype(int))

    a = newdag.get_analytical_var()
    a = newdag.get_varsortability(smart = True, simulated=True, analytical=True)

    print(newdag.get_analytical_var())
    print("final varsortability:", newdag.get_varsortability(smart = True, simulated=True, analytical=True, N = 1000000)["smart"])
    