# Author : Rishabh Pomaje
# Email  : rishabhpomaje11@gmail.com

# Dependencies
import numpy as np 
import cvxpy as cp

# Custom function definitions        
def QPdataSetGen(set_size=1, dim_x=2, numIneq=1, numEq=1, epsilon=1e-6, scale=1):
    """
    Generates random valid problem parameters for a standard form Quadratic Program (QP).

    Parameters:
    set_size : int
        Number of QP instances to generate.
    dim_x : int
        Dimension of the vector space (number of variables).
    numIneq : int
        Number of inequality constraints.
    numEq : int
        Number of equality constraints.
    epsilon : float
        Small value to ensure matrix P is positive semidefinite.
    scale : float
        Scaling factor for randomly generated elements.

    Returns:
    data_set : np.ndarray
        QP problem parameters (P, q, r, G, h, A, b) for each valid instance.
    solutionSet : np.ndarray
        Solutions, including the optimal value, variable values, dual values, and problem status.
    """
    vecSpaceLim = 1 * scale     # Limits of the field 
    
    