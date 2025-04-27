import numpy as np
from time import time
import matplotlib.pyplot as plt


# 0.1
def gen_balanced_patterns(P,N) :
    '''
    Inputs :
    - P : number of patterns
    - N : Size of each pattern (must be odd !)
    
    Outputs :
    - patterns : np.ndarray of size (P,N)
    '''

    patterns = np.zeros((P,N), dtype=int)

    for i in np.arange(P) :
        pattern = np.array([1] * (N // 2) + [-1] * (N // 2))
        np.random.shuffle(pattern)
        patterns[i] = pattern

    return patterns


# 0.2
def next_state(S : np.ndarray, patterns : np.ndarray) :

    ''' 
    Inputs :
    - S : np.ndarray of size (N,1) (state of the network at time i)
    - patterns : np.ndarray of size (P,N) (patterns that have to be compared to the state)

    Output :
    - next_state : np.ndarray of size (N,1) (state of the network at time i+1)
    - w : np.ndarray of size (N,N) (Hebbian values matrix)
    '''
    
    N = patterns.shape[1]

    # Compute weights array (Eq. 1.1), a p x p symmetric array
    w = 1/N * patterns.T @ patterns

    # Diagonal has to be 0, as the sum has to be over odd numbers to avoid getting 0 as a result
    # Irrelevant as long as you do more than one step
    # np.fill_diagonal(w, 0)
    # Follow Eq. 1.2 to update state
    h = w @ S
    next_state = np.sign(h)
    
    return next_state, w


# 0.3
def acc_next_state(S : np.ndarray, patterns : np.ndarray) :

    ''' 
    Inputs :
    - S : np.ndarray of size (N,1) (state of the network at time i)
    patterns : np.ndarray of size (P,N) (patterns that have to be compared to the state)

    Output :
    - next_state : np.ndarray of size (N,1) (state of the network at time i+1)
    - m : np.ndarray of size (P,1) (overlap variables for each pattern, useful for ex. 1.1)
    '''
    N = patterns.shape[1]

    # Compute overlap variables
    m = 1/N * patterns @ S # (pxN) x (Nx1)
    h = patterns.T @ m  # (Nxp) x (px1)
    next_state = np.sign(h)
    
    return next_state, m 

# Ex 1 in general
def plot_overlap_variables(T : int, m_list : np.ndarray) :
    '''
    General plotting function that is used in exercise 1.
    
    Inputs :
    - T : number of time steps
    - m_list : np.ndarray containing each overlap variable for every pattern at each time step,
    meaning that m_list[i][j] corresponds to the overlap variable of pattern j 
    at time step i
    
    Output :
    None (plots a figure, but does not return anything)
    '''
    plt.figure()
    for i in np.arange(m_list.shape[1]) :
        if i == 0 :
            plt.plot(np.arange(T), m_list[:,i], label='First pattern',color='black')
        else :
            plt.plot(np.arange(T), m_list[:,i])
    plt.grid()
    plt.title('Overlap variables evolution over time')
    plt.legend()
    plt.show()


# 2.1
def gen_dilution_mask(N : int):

    """
    Generate a dilution mask such that each post-synaptic neuron has 0.5N connections.

    Input :
    - N : int (number of neurons)

    Output :
    - C : np.ndarray (dilation mask)
    """

    K = N // 2
    C = np.zeros((N, N), dtype=int)
    
    for j in range(N):
        # Randomly select K presynaptic neurons to connect to neuron j
        presynaptic_indices = np.random.choice(N, K, replace=False)
        C[presynaptic_indices, j] = 1
        
    # np.fill_diagonal(C, 0)
    
    return C

