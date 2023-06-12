import numpy as np
from DAG import DAG
from itertools import combinations
from tqdm import tqdm

def all_combinations(_list, up_to = 1):
    for i in range(1, up_to + 1):
        for c in combinations(_list, i):
            yield c


def run_loco(dag, update_by_varsort=False,maximize_genghisnodes=False, maximize_top_n=None , VERBOSE=False, check_ways=False, flip_up_to=None, N=5000):

    mat = dag.adjacency_matrix.copy() # copy adjacency matrix to be safe
    mat = (mat != 0).astype(int) #convert to int
    final_adj = dag.adjacency_matrix.copy() #final adjacency matrix, which will be returned

    # If no maximal flips are given all flips are allowed
    if not flip_up_to:
        flip_up_to = mat.shape[0]

    start_varsort = dag.get_varsortability( analytical = False, simulated = False, smart = True, N = N)['smart']
    start_vars = dag.get_smart_var()

    for node in range(dag.size):
        if VERBOSE:
            print("node", node)
        ways_into_node = np.where(mat[:, node] != 0)[0]
        if len(ways_into_node) < 2:
            continue

        if check_ways:
            ways = np.empty((len(ways_into_node), dag.size))
            for i, way_into_node in enumerate(ways_into_node):
                mat2 = mat.copy()
                mat2[:, node] = 0 #Setting all ways into node 0
                mat2[way_into_node, node] = 1 #Setting 1 way into node again
                
                # Finding all ancestors to the node which has a path through way_into_node 
                exp = np.zeros_like(mat2) 
                for n_exp in range(dag.size):
                    exp += np.linalg.matrix_power(mat2, n_exp)
                exp[node, node] = 0

                exp = exp != 0 #Converting to boolean

                ways[i] = exp[:, node] #Noting the ancestors

            n_ways = ways.sum(axis = 0) #Number of paths from ancestors that terminate in the node through different ways in 

        usable_ways = set([])

            for parent in range(node):
                if n_ways[parent] < 2:
                    continue
                
                wh = np.where(ways[:, parent] > 0)[0]
                
                usable_ways |= set(ways_into_node[wh]) #finding set of ways that can be flipped for minimizing variance

        if VERBOSE:
            print("for node",node)
            print("the ways that has covariance are:", usable_ways)


            usable_ways = list(usable_ways)
        else:
            usable_ways = list(set(ways_into_node))

        # try all combinations using itertools combinations
        all_combinations_up_to = list(all_combinations(usable_ways, flip_up_to))

        vars = [start_vars[node]]
        for _all in all_combinations_up_to:
            matrix_w_flip = final_adj.copy()

            #Do the flippy-flip 
            for uw in _all:
                matrix_w_flip[uw, node] *= -1

            # Calculate new variance of node
            new_dag = DAG(n = dag.size, biass=dag.biass, adjacency_matrix = matrix_w_flip, integer = dag.integer) #maybe implement function without initializing DAG
            if update_by_varsort:
                var = new_varsort = new_dag.get_varsortability( analytical = False, simulated = False, smart = True, N = N)['smart']
            else:
                var = new_dag.get_smart_var(N)[node]

            vars.append(var)

        # Minimizing or maximizing variance
        if maximize_genghisnodes:
            ways_outof_node = np.where(mat[node] != 0)[0]
            if len(ways_outof_node) > len(ways_into_node):
                index = np.argmax(vars)
            else:
                index = np.argmin(vars)

        elif maximize_top_n and not(update_by_varsort):
            if node > maximize_top_n:
                index = np.argmin(vars)
            else:
                index = np.argmax(vars)

        else:
            index = np.argmin(vars)

        if index != 0:
            index -= 1
            if VERBOSE:
                print("flipping the way from", all_combinations_up_to[index], "to", node)
            for way in all_combinations_up_to[index]:
                final_adj[way, node] *= -1

    result_dag = DAG(adjacency_matrix = final_adj, biass=dag.biass, integer = dag.integer)

    new_varsort = result_dag.get_varsortability( analytical = False, simulated = False, smart = True, N = 10000)['smart']

    if VERBOSE:
        print(f'Old varsortability: {start_varsort:.2f}')
        print(f'New varsortability: {new_varsort:.2f}')

    return result_dag, new_varsort


def bruteforce_flip(dag):
    n = dag.size
    tot_n = n*(n-1)//2

    getpos = lambda x: list(zip(*np.where(np.triu(np.ones((n,n)), 1) - np.eye(n)== 1)))[x]


    best = dag.get_varsortability(smart = True, N=1000)["smart"]
    best_adj = dag.adjacency_matrix.copy()
    best_vars = dag.get_smart_var(N=1000)


    ac = list(all_combinations(range(tot_n), tot_n))
    for c in tqdm(ac, total = len(ac)):
        adj = dag.adjacency_matrix.copy()
        for pos in c:
            i,j = getpos(pos)
            adj[i,j] *= -1

        newdag = DAG(adjacency_matrix=dag.adjacency_matrix, integer = dag.integer)
        vs = newdag.get_varsortability(smart = True, N = 1000)["smart"]

        if vs < best:
            best = vs
            best_vars = newdag.get_smart_var(N=1000)
            best_adj = adj.copy()
            print("new best:", best)
            print(c)
            print("")

    print("best:", best)
    print(best_adj)
    print(best_vars)

    return None

if __name__ == '__main__':
    np.random.seed(43)
    test_dag = DAG(n = 10, roots=1, precalculate_paths=False, integer=False, connectivity=.8)
    run_loco(test_dag, check_ways=False)
