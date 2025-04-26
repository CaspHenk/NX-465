import numpy as np
from time import time


# 0.1
def gen_balanced_patterns(p,N) :
    '''
    Inputs :
    p : number of patterns
    N : Size of each pattern (must be odd !)
    
    Outputs :
    patterns : np.ndarray of size (p,N)
    '''

    patterns = np.zeros((p,N), dtype=int)

    for i in np.arange(p) :
        pattern = np.array([1] * (N // 2) + [-1] * (N // 2))
        np.random.shuffle(pattern)
        patterns[i] = pattern

    return patterns


# 0.2
def next_state(S : np.ndarray, patterns : np.ndarray) :

    ''' 
    Inputs :
    S : np.ndarray of size (N,1) (state of the network at time i)
    patterns : np.ndarray of size (p,N) (patterns that have to be compared to the state)

    Output :
    next_state : np.ndarray of size (N,1) (state of the network at time i+1)
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
    
    return next_state


# 0.3
def acc_next_state(S : np.ndarray, patterns : np.ndarray) :

    ''' 
    Inputs :
    S : np.ndarray of size (N,1) (state of the network at time i)
    patterns : np.ndarray of size (p,N) (patterns that have to be compared to the state)

    Output :
    next_state : np.ndarray of size (N,1) (state of the network at time i+1)
    '''
    
    p = patterns.shape[0]
    N = patterns.shape[1]

    # Compute overlap variables
    m = 1/N * patterns @ S # (pxN) x (Nx1)
    h = patterns.T @ m  # (Nxp) x (px1)
    next_state = np.sign(h)
    
    return next_state


# 1.1
def overlap_next_state(S : np.ndarray, patterns : np.ndarray) :

    ''' 
    Exactly the same function as for 0.3, but here it also returns m
    Not the most clever way to put it, but at least it separates the
    two functions for the two exercises (we can rewrite it later if necessary)



    Inputs :
    S : np.ndarray of size (N,1) (state of the network at time i)
    patterns : np.ndarray of size (p,N) (patterns that have to be compared to the state)

    Output :
    next_state : np.ndarray of size (N,1) (state of the network at time i+1)
    m : np.ndarray of size (p,1) (overlap variables for each pattern)
    '''
    
    p = patterns.shape[0]
    N = patterns.shape[1]

    # Compute overlap variables
    m = 1/N * patterns @ S # (pxN) x (Nx1)
    h = patterns.T @ m  # (Nxp) x (px1)
    next_state = np.sign(h)
    
    return next_state, m