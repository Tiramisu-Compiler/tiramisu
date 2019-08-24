from __future__ import print_function

# For convenience, disable warnings of type FutureWarning
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import tensorflow as tf
import numpy as np
import time, sys

from tensorflow import nn
from tensorflow import image

IMG_WIDTH = 600
IMG_HEIGHT = 400

BATCH_SIZE = 32
N = 224
FIN = 3
FOUT = 32

K_Y = 3
K_X = 3

NB_TESTS = 101

def ResizeConvReluMaxPool(X, weights, bias):
    resize = image.resize(X, [N + 2, N + 2])
    conv = nn.conv2d(resize, weights, strides=[1, 1, 1, 1], padding="VALID", data_format="NHWC")
    conv_bias = nn.bias_add(conv, bias, data_format="NHWC")

    relu = nn.relu(conv_bias)
    maxpool = nn.max_pool2d(relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID", data_format="NHWC")
    
    return maxpool
    
# Init buffers
np.random.seed(0)

batch_X = np.random.rand(BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, FIN)
weights = np.random.rand(K_Y, K_X, FIN, FOUT)
bias = np.random.rand(FOUT)

# Create the network
X = tf.compat.v1.placeholder(tf.float32, [BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, FIN], name="X")
activations = ResizeConvReluMaxPool(X, weights, bias)

# Save the graph and exit if requested
if len(sys.argv) > 1 and sys.argv[1] == "0":
    tf.io.write_graph(tf.compat.v1.get_default_graph(), "", "tf_model.pb", as_text=False)
    exit()

# Start execution session
with tf.compat.v1.Session() as sess:
    durations = []
    
    # Run the network
    for i in range(NB_TESTS):
        start = time.perf_counter()
        batch_y = sess.run(activations, feed_dict={X: batch_X})
        end = time.perf_counter()
        
        durations.append((end - start)*1000)
    
    print("Duration : ", np.median(durations))

