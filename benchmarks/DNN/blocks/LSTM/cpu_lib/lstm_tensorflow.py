# For convenience, disable warnings of type FutureWarning
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import tensorflow as tf
import numpy as np
import time, sys

from tensorflow.contrib import rnn

FEATURE_SIZE = 1024
BATCH_SIZE = 1
NUM_LAYERS = 4
SEQ_LENGTH = 100

NB_TESTS = 101

def LSTM(X):
    # List of LSTM cells (there are NUM_LAYERS cells)
    lstm_cells = [rnn.LSTMBlockFusedCell(FEATURE_SIZE, forget_bias=1.0) for _ in range(NUM_LAYERS)]

    # We manually stack our LSTM cells as MultiRNNCell doesn't work with LSTMBlockFusedCell
    outputs = X
    for cell in lstm_cells:
        outputs, states = cell(outputs, dtype=tf.float32)

    return outputs
    
# Init buffers
np.random.seed(0)
batch_X = np.random.rand(SEQ_LENGTH, BATCH_SIZE, FEATURE_SIZE)

# Create the network
X = tf.compat.v1.placeholder(tf.float32, [SEQ_LENGTH, BATCH_SIZE, FEATURE_SIZE], name="X")
activations = LSTM(X)

# Initialize the variables (i.e. assign their default value)
init = tf.compat.v1.global_variables_initializer()

# Start execution session
with tf.compat.v1.Session() as sess:
    sess.run(init)
    durations = []
    
    # Run the network
    for i in range(NB_TESTS):
        start = time.perf_counter()
        batch_y = sess.run(activations, feed_dict={X: batch_X})
        end = time.perf_counter()
        
        durations.append((end - start)*1000)
    
    print("Duration : ", np.median(durations))

