from __future__ import print_function

import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
from tensorflow.python.client import timeline

FEATURE_SIZE = 16 
BATCH_SIZE = 8
NUM_LAYERS = 4
SEQ_LENGTH = 10 

# tf Graph input
X = tf.placeholder("float", [FEATURE_SIZE, BATCH_SIZE, SEQ_LENGTH])
Y = tf.placeholder("float", [FEATURE_SIZE, BATCH_SIZE, SEQ_LENGTH])

# Define weights
weights = {
    'out': tf.Variable( tf.random_normal([FEATURE_SIZE, 4 * FEATURE_SIZE, 2, NUM_LAYERS]))
}
biases = {
    'out': tf.Variable(tf.random_normal([4 * FEATURE_SIZE, NUM_LAYERS]))
}

def RNN(x, weights, biases):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, timesteps, n_input)
    # Required shape: 'timesteps' tensors list of shape (batch_size, n_input)

    # Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)
    x = tf.unstack(x, BATCH_SIZE, 1)

    # Define a lstm cell with tensorflow
    lstm_cell = rnn.BasicLSTMCell(NUM_LAYERS, forget_bias=1.0)

    # Get lstm cell output
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return outputs #tf.matmul(outputs[-1], weights['out']) + biases['out']

logits = RNN(X, weights, biases)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.compat.v1.Session() as sess:
    # Run the initializer
    sess.run(init)
    np.random.seed(0)
    batch_x = np.random.rand(FEATURE_SIZE, BATCH_SIZE, SEQ_LENGTH)
    
        # add additional options to trace the session execution
    options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()

    # Run optimization op (backprop)
    batch_y = sess.run(logits, feed_dict={X: batch_x}, options=options, run_metadata=run_metadata)
    
    # Create the Timeline object, and write it to a json file
    fetched_timeline = timeline.Timeline(run_metadata.step_stats)
    chrome_trace = fetched_timeline.generate_chrome_trace_format()
    with open('timeline_01.json', 'w') as f:
        f.write(chrome_trace)

print (batch_y)
