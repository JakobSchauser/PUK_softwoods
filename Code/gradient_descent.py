import numpy as np
from DAG import DAG
from tqdm.notebook import tqdm
from time import time
from numba import jit


# ignore all NumbaDeprecationWarnings
import warnings
from numba import NumbaDeprecationWarning, NumbaPendingDeprecationWarning, NumbaWarning
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaWarning)


@jit
def add_to_arr(x, arr):
    ad = np.zeros_like(arr).astype(float)
    k = 0

    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            if arr[i,j] != 0:
                ad[i,j] = x[k]
                k += 1
    return ad


def loss_fn(x, bool_adj):
    
    ad = add_to_arr(x,bool_adj)
    
    d = DAG(n = bool_adj.shape[0], strength=2, precalculate_paths = False, adjacency_matrix = ad)
    return d.get_continous_varsortability(simulated = True, N = 100000)["simulated"]


def varsortability(x, bool_adj):
    ad = add_to_arr(x, bool_adj)
    
    d = DAG(n = bool_adj.shape[0],  strength=2, precalculate_paths = False, adjacency_matrix = ad)
    return d.get_varsortability(smart = True, N = 100000)["smart"]




def gradient_descent(dag, VERBOSE = False, num_iterations = 100, lr = 0.1, adaptive = True):
    """
    Performs gradient descent on the loss function to find the optimal values for the parameters.
    dag is a DAG object
    """
    start_adj = dag.adjacency_matrix.copy()
    bool_adj = start_adj != 0
    n = np.sum(bool_adj)
    # Initialize the parameters
    learning_rate = lr
    start_lr = learning_rate
    num_iterations = num_iterations

    # Initial guess for the parameters
    initial_guess = np.random.randint(0, 2, size = n) * 2 - 1
    # Gradient descent loop
    current_guess = initial_guess.copy()

    for i in range(num_iterations):
        if adaptive:
            learning_rate = start_lr * (1 - (i/num_iterations)/10)
        gradient = np.zeros(n)  # Initialize gradient to zero

        # Compute the gradient by finite differences
        for j in range(len(current_guess)):
            h = 1e-5  # Step size for finite differences
            delta = np.zeros(n)
            delta[j] = h
            gradient[j] = (loss_fn(current_guess + delta, bool_adj) - loss_fn(current_guess - delta, bool_adj)) / (2 * h)

        # Update the parameters using gradient descent
        current_guess = current_guess - learning_rate * gradient

        # make sure we are within our constraints
        
        for j in range(len(current_guess)):
            mag = 0.0
            if current_guess[j] < mag and current_guess[j] > 0:
                current_guess[j] = -mag
            if current_guess[j] > -mag and current_guess[j] < 0:
                current_guess[j] = mag
            if current_guess[j] > 2:
                current_guess[j] = 2
            if current_guess[j] < -2:
                current_guess[j] = -2

        if VERBOSE:
            print("Iteration %d: loss = %.3f" % (i, loss_fn(current_guess, bool_adj)))
            
            if i % 50 == 0:
                print("current_guess:", current_guess)
                print("varsortability:", varsortability(current_guess, bool_adj))

    final_adj = np.zeros_like(bool_adj).astype(float)
    k = 0
    for i in range(bool_adj.shape[0]):
        for j in range(bool_adj.shape[1]):
            if bool_adj[i,j] != 0:
                final_adj[i,j] = current_guess[k]
                k += 1

    return final_adj


if __name__ == "__main__":
    # Perform gradient descent to minimize the black-box function
    t = time()
    result = gradient_descent(DAG(n = 5), VERBOSE = False, num_iterations = 100, lr = 0.1)
    print("Time:", time() - t)
    # print("Optimal values:", result)

