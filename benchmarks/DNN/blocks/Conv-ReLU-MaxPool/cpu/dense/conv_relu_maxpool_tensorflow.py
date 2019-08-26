from __future__ import print_function

# For convenience, disable warnings of type FutureWarning
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import tensorflow as tf
import numpy as np
import time, sys

from tensorflow import nn

BATCH_SIZE = 8
N = 112
FIN = 3
FOUT = 32

K_Y = 3
K_X = 3

NB_TESTS = 101

def ConvReluMaxPool(X, weights, bias):
    conv = nn.conv2d(X, weights, strides=[1, 1, 1, 1], padding="VALID", data_format="NHWC")
    conv_bias = nn.bias_add(conv, bias, data_format="NHWC")

    relu = nn.relu(conv_bias)
    maxpool = nn.max_pool2d(relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID", data_format="NHWC")
    
    return maxpool
    
# Init buffers
np.random.seed(0)

batch_X = np.random.rand(BATCH_SIZE, N + 2, N + 2, FIN)
weights = np.random.rand(K_Y, K_X, FIN, FOUT)
bias = np.random.rand(FOUT)

# Create the network
X = tf.compat.v1.placeholder(tf.float32, [BATCH_SIZE, N + 2, N + 2, FIN], name="X")
activations = ConvReluMaxPool(X, weights, bias)

# Save the graph and exit if requested
if len(sys.argv) > 1 and sys.argv[1] == "0":
    tf.io.write_graph(tf.compat.v1.get_default_graph(), "", "tf_model.pb", as_text=False)
    exit()

# Start execution session
with tf.compat.v1.Session() as sess:
    durations = []
    
    # Run the network
    for _ in range(NB_TESTS):
        start = time.perf_counter()
        batch_y = sess.run(activations, feed_dict={X: batch_X})
        end = time.perf_counter()
        
        durations.append((end - start)*1000)
    
    print("Duration : ", np.median(durations))

