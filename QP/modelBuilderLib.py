import tensorflow as tf 
import numpy as np 

def QP_nn_(tf.keras.Model):
    def __init__(self, dim_x, num_Eq, num_Ineq):
        super(QP_nn, self).__init__()

        # Input layers
        self.input_P = tf.keras.layers.Input(shape=(dim_x, dim_x, ), name="input_P")
        self.input_q = tf.keras.layers.Input(shape=(dim_x, ), name="input_q")
        self.input_r = tf.keras.layers.Input(shape=(1, ), name="input_r")  
        self.input_G = tf.keras.layers.Input(shape=(num_Ineq, dim_x, ), name="input_G")
        self.input_h = tf.keras.layers.Input(shape=(num_Ineq, ), name="input_h")
        self.input_A = tf.keras.layers.Input(shape=(num_Eq, dim_x, ), name="input_A")
        self.input_b = tf.keras.layers.Input(shape=(num_Eq, ), name="input_b")

        # Flatten layers
        self.flatten_P = tf.keras.layers.Flatten()
        self.flatten_A = tf.keras.layers.Flatten()
        self.flatten_G = tf.keras.layers.Flatten()

        # Hidden layers
        self.hidden_1 = tf.keras.layers.Dense(4 * dim_x, activation='relu')
        self.hidden_2 = tf.keras.layers.Dense(4 * dim_x, activation='relu')
        # self.hidden_3 = tf.keras.layers.Dense(4 * dim_x, activation='relu')
        # self.hidden_4 = tf.keras.layers.Dense(4 * dim_x, activation='relu')

        # Output layers for each branch
        self.output_branch_1_relu0 = tf.keras.layers.Dense(dim_x, activation='relu')
        self.output_branch_1_relu1 = tf.keras.layers.Dense(dim_x, activation='relu')
        self.output_branch_1_linear = tf.keras.layers.Dense(dim_x)

        self.output_branch_2_relu0 = tf.keras.layers.Dense(num_Eq, activation='relu')
        self.output_branch_2_relu1 = tf.keras.layers.Dense(num_Eq, activation='relu')
        self.output_branch_2_linear = tf.keras.layers.Dense(num_Eq)

        self.output_branch_3_relu0 = tf.keras.layers.Dense(num_Ineq, activation='relu')
        self.output_branch_3_relu1 = tf.keras.layers.Dense(num_Ineq, activation='relu')
        self.output_branch_3_linear = tf.keras.layers.Dense(num_Ineq)

    def call(self, inputs):
        input_P, input_q, input_r, input_G, input_h, input_A, input_b = inputs
        
        # Flatten the inputs
        P_flat = self.flatten_P(input_P)
        A_flat = self.flatten_A(input_A)
        G_flat = self.flatten_G(input_G)

        # Concatenate all flattened layers
        concatenated = tf.keras.layers.Concatenate()([P_flat, input_q, input_r, G_flat, input_h, A_flat, input_b])

        # Pass through hidden layers
        hidden_1 = self.hidden_1(concatenated)
        hidden_2 = self.hidden_2(hidden_1)

        # Splitting into three branches
        output_branch_1 = self.output_branch_1_linear(self.output_branch_1_relu(hidden_2))
        output_branch_2 = self.output_branch_2_linear(self.output_branch_2_relu(hidden_2))
        output_branch_3 = self.output_branch_3_linear(self.output_branch_3_relu(hidden_2))

        return [output_branch_1, output_branch_2, output_branch_3]
    
    def model_summary(self, dim_x, num_Eq, num_Ineq):
        inputs = [
            tf.keras.Input(shape=(dim_x, dim_x,)), 
            tf.keras.Input(shape=(dim_x,)), 
            tf.keras.Input(shape=(1,)), 
            tf.keras.Input(shape=(num_Ineq, dim_x,)), 
            tf.keras.Input(shape=(num_Ineq,)), 
            tf.keras.Input(shape=(num_Eq, dim_x,)), 
            tf.keras.Input(shape=(num_Eq,))
        ]
        model = tf.keras.Model(inputs=inputs, outputs=self.call(inputs))
        model.summary()