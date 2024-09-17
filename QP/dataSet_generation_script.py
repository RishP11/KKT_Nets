import kktNetLib as knl
import matplotlib.pyplot as plt 
import numpy as np 

# Run this only once to generate the training and the testing data set. And then just load the data every time you train to save some compute time.
knl.generate_Data(int(input(f'Enter # of problems = ')), False)
