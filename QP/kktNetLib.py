# Author : Rishabh Pomaje
# Email  : rishabhpomaje11@gmail.com

# Dependencies
import numpy as np 
import cvxpy as cp

# Custom function definitions

def generate_Data(set_size=100, training=True):
    """
    Function description here
    """
    X, y = QPdataSetGen(set_size)  # Generate 100 instances
    # Save the generated data to a file using np.savez
    if training:
        np.savez('training_data.npz', x_train=X, y_train=y)
    else:
        np.savez('testing_data.npz', x_train=X, y_train=y)

def QPdataSetGen(set_size=1, dim_x=2, numIneq=2, numEq=2, epsilon=1e-6, scale=1):
    """ 
    Generate random but valid problem parameters for a standard form of Quadratic Program.
    """
    vecSpaceLim = 1 * scale
    data_set = [None] * set_size
    solutionSet = [None] * set_size
    valid_count = 0
    total_count = 0 
    while valid_count < set_size:
        # Generate random matrix M and compute symmetric positive semidefinite matrix P
        M = np.random.uniform(-vecSpaceLim, vecSpaceLim, (dim_x, dim_x))
        P = np.dot(M.T, M) + epsilon * np.eye(dim_x)

        # Generate q, r, G, h, A, b in one go
        q = np.random.uniform(-vecSpaceLim, vecSpaceLim, dim_x)
        r = np.random.uniform(-vecSpaceLim, vecSpaceLim)
        G = np.random.uniform(-vecSpaceLim, vecSpaceLim, (numIneq, dim_x))
        h = np.random.uniform(-vecSpaceLim, vecSpaceLim, numIneq)
        A = np.random.uniform(-vecSpaceLim, vecSpaceLim, (numEq, dim_x))
        b = np.random.uniform(-vecSpaceLim, vecSpaceLim, numEq)

        # Define the variable for solving
        x = cp.Variable(dim_x)
        objFnc = cp.Minimize(0.5 * cp.quad_form(x, P) + q.T @ x + r)
        constrs = [G @ x <= h, A @ x == b]

        try:
            prob = cp.Problem(objective=objFnc, constraints=constrs)
            prob.solve()  # You can specify other solvers if needed

            if prob.status == 'optimal':
                # Valid problem instance; store data
                data_set[valid_count] = [P, q, r, G, h, A, b]
                solutionSet[valid_count] = [prob.value, x.value, constrs[0].dual_value, constrs[1].dual_value, prob.status]
                valid_count += 1
            else:
                total_count += 1
        except cp.SolverError:
            # Ignore and continue if solver fails
            total_count += 1
            pass
        
        # Progress display (updated less frequently)
        # if valid_count % (set_size // 10) == 0: 
        print(f'Progress: {valid_count}/{set_size}')

    print(f'STATUS :: Data labels ready.')
    print(f'Total Problem Searched = {total_count}')
    print(f'% problems accepted = {100 * valid_count / total_count} %')
    return np.array(data_set, dtype=object), np.array(solutionSet, dtype=object)