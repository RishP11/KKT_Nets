# Author : Rishabh Pomaje
# Email  : rishabhpomaje11@gmail.com

# Dependencies
import os
import numpy as np 
import cvxpy as cp

# Custom function definitions        
def QPdataSetGen(file_path, set_size=1, dim_x=2, numIneq=1, numEq=1):
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
    vecSpaceLim = 1             # Limits of the field 
    count_explored = 0          # Number of explored problems 
    count_accepted = 0          # Number of problems accepted (so far)    

    # Ensure the directory exists
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

    # Problem Parameters
    P_set, q_set, r_set, G_set, h_set, A_set, b_set = [], [], [], [], [], [], [] 
    # Solutions 
    x_opt_set, lambda_opt_set, nu_opt_set = [], [], []

    while(count_accepted < set_size):
        # Generate random P (positive semidefinite matrix), q, r :: Objective Function 
        P = np.random.uniform(-vecSpaceLim, vecSpaceLim, (dim_x, dim_x))
        P = P @ P.T                                                     # Make symmetric and Positive semidefinite
        q = np.random.uniform(-vecSpaceLim, vecSpaceLim, (dim_x))         
        r = np.random.uniform(-vecSpaceLim, vecSpaceLim) 

        # Generate random matrices G, h :: Inequality Constraints
        G = np.random.uniform(-vecSpaceLim, vecSpaceLim, (numIneq, dim_x)) if numIneq > 0 else None
        h = np.random.uniform(-vecSpaceLim, vecSpaceLim, (numIneq)) if numIneq > 0 else None

        # Generate random matrices A, b :: Equality Constraints 
        A = np.random.uniform(-vecSpaceLim, vecSpaceLim, (numEq, dim_x)) if numEq > 0 else None 
        b = np.random.uniform(-vecSpaceLim, vecSpaceLim, (numEq)) if numEq > 0 else None

        # Complete Problem Normalization Process : 
        normFactor = np.max([np.max(np.abs(P)), np.max(np.abs(q)), np.max(np.abs(r)), np.max(np.abs(G)), np.max(np.abs(h)), np.max(np.abs(A)), np.max(np.abs(b))])
        P   /= normFactor
        q   /= normFactor 
        r   /= normFactor 
        G   /= normFactor
        h   /= normFactor
        A   /= normFactor
        b   /= normFactor  

        # Instantiation of the problem
        x = cp.Variable(dim_x)

        objFnc = cp.Minimize((1 / 2) * cp.quad_form(x, P) + q.T @ x)

        constFncs = [G @ x <= h]
        if numEq > 0:
            constFncs.append(A @ x == b)

        prob = cp.Problem(objFnc, constFncs)
        count_explored += 1 
        try:
            prob.solve()    
            # If the problem is solvable, store the data
            if prob.status == cp.OPTIMAL:
                x_opt = np.array(x.value)
                p_opt = prob.value
                lambda_opt = np.array(prob.constraints[0].value)
                nu_opt = np.array(prob.constraints[1].value) if numEq > 0 else None
                # Store the problem data and solution
                P_set.append(P)
                q_set.append(q)
                r_set.append(r)
                G_set.append(G)
                h_set.append(h)
                A_set.append(A)
                b_set.append(b)
                x_opt_set.append(x_opt)
                lambda_opt_set.append(lambda_opt)
                nu_opt_set.append(nu_opt)
                count_accepted += 1
                print(f'# of Explored Problems = {count_explored} :: # of Accepted Instances = {count_accepted}', end='\r') 
        except cp.error.SolverError:
                # Skip the problem instance if the solver fails
                print(f"Solver failed for problem {count_explored}, skipping...")
    
    np.savez_compressed(file=file_path, 
                        P_set=np.array(P_set),
                        q_set=np.array(q_set),
                        r_set=np.array(r_set),
                        G_set=np.array(G_set),
                        h_set=np.array(h_set),
                        A_set=np.array(A_set),
                        b_set=np.array(b_set),
                        x_opt_set=np.array(x_opt_set),
                        lambda_opt_set=np.array(lambda_opt_set),
                        nu_opt_set=np.array(nu_opt_set))