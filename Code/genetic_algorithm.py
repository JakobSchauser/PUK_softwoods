import numpy as np
from DAG import DAG


# genetic algorithm

def run_genetic_algorithm(dag, population = 50, generations = 200, VERBOSE = False, continous = False):
    # start_adj = np.array([[0,1,1,1,1,1],[0,0,1,1,1,1],[0,0,0,1,1,1],[0,0,0,0,1,1],[0,0,0,0,0,1], [0,0,0,0,0,0]]).astype(float)
    start_adj = dag.adjacency_matrix.copy()
    mkdag = lambda: DAG(n = start_adj.shape[0], roots = 1, strength = 15, precalculate_paths = False, adjacency_matrix = start_adj, integer = True)

    start_dags = [mkdag() for _ in range(population)]


    # thomas siger lav grupper 
    true = True

    N = generations

    for i in range(N):
        if VERBOSE:
            print("")
            print("generation", i)
        if continous:
            start_dags = sorted(start_dags, key = lambda x: x.get_continous_varsortability(smart = True, simulated = False, N = 1000000)["simulated"], reverse = False)
        else:
            start_dags = sorted(start_dags, key = lambda x: x.get_varsortability(smart = True, simulated = False, N = 1000000)["smart"], reverse = False)

        start_dags = start_dags[:(population // 2)]

        for j in range(population // 2):
            start_dags.append(start_dags[j].mutate(1 - i/N))

        if i < N//2:
            new = mkdag()
            start_dags.append(new.mutate())

        if VERBOSE and i % 10 == 0:
            print("varsortability:", start_dags[j].get_varsortability(analytical = False, simulated = False, smart = True,  N = 10000000))
            print("continous_varsortability:", start_dags[j].get_continous_varsortability(analytical = False, simulated = False,smart = True, N = 10000000))

    start_dags = sorted(start_dags, key = lambda x: x.get_varsortability(smart = True, simulated = False, N = 1000000)["smart"], reverse = False)

    return start_dags