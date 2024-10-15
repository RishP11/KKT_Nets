# Author : Rishabh Pomaje
# Email  : rishabhpomaje11@gmail.com

# Dependencies
import os
import numpy as np 
import cvxpy as cp
import tensorflow as tf 

import matplotlib.pyplot as plt 
# Update plot settings to use LaTeX for text rendering
plt.rcParams.update({
    'text.usetex': True,        # Use LaTeX for text rendering
    'font.family': 'serif',     # Set font family to serif
    'axes.labelsize': 12,       # Font size for axis labels
    'font.size': 12,            # General font size
    'legend.fontsize': 10,      # Font size for legends
    'xtick.labelsize': 10,      # Font size for x-tick labels
    'ytick.labelsize': 10,      # Font size for y-tick labels
})

# Custom function definitions 

def QP_nn_routine(dim_x, num_Eq, num_Ineq, Epchs, training_data, testing_data):
    # Input formatting 
    P_train = training_data['P_set']
    q_train = training_data['q_set']
    r_train = training_data['r_set']

    G_train = training_data['G_set']
    h_train = training_data['h_set']

    A_train = training_data['A_set']
    b_train = training_data['b_set']

    x_opt_train = training_data['x_opt_set']
    lambda_opt_train = training_data['lambda_opt_set']
    nu_opt_train = training_data['nu_opt_set']

    y_train =  output_preprocessing(x_opt_train, lambda_opt_train, nu_opt_train)

    # Building the neural network
    input_P = tf.keras.layers.Input(shape=(dim_x, dim_x, ), name="input_P")
    input_q = tf.keras.layers.Input(shape=(dim_x, ), name="input_q")
    input_r = tf.keras.layers.Input(shape=(1, ), name="input_r")  
    input_G = tf.keras.layers.Input(shape=(num_Ineq, dim_x, ), name="input_G")
    input_h = tf.keras.layers.Input(shape=(num_Ineq, ), name="input_h")
    input_A = tf.keras.layers.Input(shape=(num_Eq, dim_x, ), name="input_A")
    input_b = tf.keras.layers.Input(shape=(num_Eq, ), name="input_b")
    
    input_x_opt = tf.keras.layers.Input(shape=(dim_x, ), name='x_opt_label')
    input_lambda_opt = tf.keras.layers.Input(shape=(num_Ineq, ), name='lambda_opt_label')
    input_nu_opt = tf.keras.layers.Input(shape=(num_Ineq, ), name='nu_opt_label')

    P_flat = tf.keras.layers.Flatten()(input_P)
    G_flat = tf.keras.layers.Flatten()(input_G)
    A_flat = tf.keras.layers.Flatten()(input_A)

    concatenated_input = tf.keras.layers.Concatenate()([P_flat, input_q, input_r, G_flat, input_h, A_flat, input_b]) 
    common_hidden = tf.keras.layers.Dense(16 * dim_x, activation='relu')(concatenated_input)
    common_hidden = tf.keras.layers.Dense(32 * dim_x, activation='relu')(common_hidden)
    common_hidden = tf.keras.layers.Dense(32 * dim_x, activation='relu')(common_hidden)
    common_hidden = tf.keras.layers.Dense(16 * dim_x, activation='relu')(common_hidden)
    
    # Branch out common --> x, lambda, nu
    branch_x = tf.keras.layers.Dense(8*dim_x, activation='relu')(common_hidden)
    branch_x = tf.keras.layers.Dense(16*dim_x, activation='relu')(branch_x)
    branch_x = tf.keras.layers.Dense(16*dim_x, activation='relu')(branch_x)
    branch_x = tf.keras.layers.Dense(16*dim_x, activation='relu')(branch_x)
    branch_x = tf.keras.layers.Dense(dim_x, activation='linear')(branch_x)
    
    branch_lambda = tf.keras.layers.Dense(8*num_Ineq, activation='relu')(common_hidden)
    branch_lambda = tf.keras.layers.Dense(16*num_Ineq, activation='relu')(branch_lambda)
    branch_lambda = tf.keras.layers.Dense(16*num_Ineq, activation='relu')(branch_lambda)
    branch_lambda = tf.keras.layers.Dense(16*num_Ineq, activation='relu')(branch_lambda)
    branch_lambda = tf.keras.layers.Dense(num_Ineq, activation='relu')(branch_lambda)

    branch_nu = tf.keras.layers.Dense(8*num_Eq, activation='relu')(common_hidden)
    branch_nu = tf.keras.layers.Dense(16*num_Eq, activation='relu')(branch_nu)
    branch_nu = tf.keras.layers.Dense(16*num_Eq, activation='relu')(branch_nu)
    branch_nu = tf.keras.layers.Dense(16*num_Eq, activation='relu')(branch_nu)
    branch_nu = tf.keras.layers.Dense(num_Eq, activation='linear')(branch_nu)

    # Combine 
    common_output = tf.keras.layers.Concatenate()([branch_x, branch_lambda, branch_nu])
    
    kktnet = tf.keras.Model(
        inputs = [input_P, input_q, input_r, input_G, input_h, input_A, input_b],
        outputs = [common_output]
    )

    # Compiling the model :
    kktnet.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=tf.keras.losses.mse, metrics=['accuracy'])
    kktnet.summary()

    # Visualize the block diagram of the model
    tf.keras.utils.plot_model(
        kktnet,
        to_file='plots/QP_model.png',
        show_shapes=True,
        show_dtype=True,
        show_layer_names=True,
        rankdir='LR',
        expand_nested=True,
        dpi=200,
        show_layer_activations=True,
        show_trainable=True,
    )

    # Fitting the model by using the training set :
    history = kktnet.fit([P_train, q_train, r_train, G_train, h_train, A_train, b_train], y_train, batch_size=1000, epochs=Epchs)
    training_loss = history.history['loss']
    training_accuracy = history.history['accuracy']

    # Plot training loss and accuracy
    epochs = range(1, len(training_loss) + 1)

    fig, axes = plt.subplots()
    # Plotting loss
    # axes.plot(epochs, training_loss, 'r', label='Training Loss')
    axes.plot(epochs, training_accuracy, 'b', label='Training Accuracy')
    axes.set_title('Training history')
    axes.set_xlabel('Epochs')
    axes.set_ylabel('Accuracy/ Loss')
    axes.legend()
    fig.savefig(f'plots/training_history.svg', transparent=True)
    fig.savefig(f'plots/training_history.png')

def output_preprocessing(x_arr, lmbda_arr, nu_arr):
    """
    Concatenates corresponding elements from x_arr, lmbda_arr, and nu_arr into a 2D NumPy array.

    Parameters:
    ----------
    x_arr, lmbda_arr, nu_arr : array-like
        Input arrays of equal length. Each element of the input arrays is itself an array or list.

    Returns:
    -------
    np.ndarray
        A 2D NumPy array where each row contains the concatenated elements from 
        the corresponding elements of x_arr, lmbda_arr, and nu_arr.
    
    Example:
    --------
    >>> x_arr = [np.array([1, 2]), np.array([3, 4])]
    >>> lmbda_arr = [np.array([5]), np.array([6])]
    >>> nu_arr = [np.array([7]), np.array([8])]
    >>> output_preprocessing(x_arr, lmbda_arr, nu_arr)
    array([[1, 2, 5, 7],
           [3, 4, 6, 8]])
    """
    return np.array([np.concatenate([x, lmbda, nu]) for x, lmbda, nu in zip(x_arr, lmbda_arr, nu_arr)])


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
        P = np.random.uniform(-vecSpaceLim, vecSpaceLim, (dim_x, dim_x, ))
        P = P @ P.T                                                     # Make symmetric and Positive semidefinite
        q = np.random.uniform(-vecSpaceLim, vecSpaceLim, (dim_x, ))         
        r = np.random.uniform(-vecSpaceLim, vecSpaceLim, (1, )) 

        # Generate random matrices G, h :: Inequality Constraints
        G = np.random.uniform(-vecSpaceLim, vecSpaceLim, (numIneq, dim_x, )) if numIneq > 0 else None
        h = np.random.uniform(-vecSpaceLim, vecSpaceLim, (numIneq, )) if numIneq > 0 else None

        # Generate random matrices A, b :: Equality Constraints 
        A = np.random.uniform(-vecSpaceLim, vecSpaceLim, (numEq, dim_x, )) if numEq > 0 else None 
        b = np.random.uniform(-vecSpaceLim, vecSpaceLim, (numEq, )) if numEq > 0 else None

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
        constrFncs = [G @ x <= h, A @ x == b]
        
        prob = cp.Problem(objFnc, constrFncs)
        count_explored += 1 
        try:
            prob.solve()    
            # If the problem is solvable, store the data
            if prob.status == cp.OPTIMAL:
                x_opt = np.array(x.value)
                lambda_opt = np.array(constrFncs[0].dual_value)
                nu_opt = np.array(constrFncs[1].dual_value) 
                # Store the problem data and solution
                P_set.append(P)
                q_set.append(q)
                r_set.append([r])
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