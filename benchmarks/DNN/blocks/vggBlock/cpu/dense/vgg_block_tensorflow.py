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

def vggBlock(X, weights1, bias1, weights2, bias2):
    conv1 = nn.conv2d(X, weights1, strides=[1, 1, 1, 1], padding="VALID", data_format="NHWC")
    conv1_bias = nn.bias_add(conv1, bias1, data_format="NHWC")
    relu1 = nn.relu(conv1_bias)
    
    conv2 = nn.conv2d(relu1, weights2, strides=[1, 1, 1, 1], padding="SAME", data_format="NHWC")
    conv2_bias = nn.bias_add(conv2, bias2, data_format="NHWC")

    relu2 = nn.relu(conv2_bias)
    maxpool = nn.max_pool2d(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID", data_format="NHWC")
    
    return maxpool
    
# Init buffers
np.random.seed(0)

batch_X = np.random.rand(BATCH_SIZE, N + 2, N + 2, FIN)
weights1 = np.random.rand(K_Y, K_X, FIN, FOUT)
weights2 = np.random.rand(K_Y, K_X, FOUT, FOUT)

bias1 = np.random.rand(FOUT)
bias2 = np.random.rand(FOUT)

# Create the network
X = tf.compat.v1.placeholder(tf.float32, [BATCH_SIZE, N + 2, N + 2, FIN], name="X")
activations = vggBlock(X, weights1, bias1, weights2, bias2)

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

