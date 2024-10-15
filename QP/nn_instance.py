import numpy as np 
import kktNetLib as knl

# Problem specification 
dim_x = 2       # Dimension of the objective variable
numEq = 1       # Number of Equality constraints
numIneq = 1     # Number of Inequality constraints

# Load the training data 
training_data = np.load('E:/Research/KKTNet/Source_Code_Simulations/QP/dataset/qp_train.npz')

dim_x = 2               # dimension of the objective variable
num_Eq = 1              # number of equality constraints
num_Ineq = 1            # number of inequality constraints

knl.QP_nn_routine(2, 1, 1, 1000, training_data, 0)