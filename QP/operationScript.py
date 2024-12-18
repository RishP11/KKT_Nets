import numpy as np 
import kktNetLib as kktNN

number_problem_instances = 10 ** 5

kktNN.QPdataSetGen('dataset/qp_test.npz', number_problem_instances)

# Load the saved .npz file
data = np.load('dataset/qp_test.npz')
print(np.shape(data))
# Access guide to the problem_set and solution_set
test = data['G_set']
