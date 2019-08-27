# For convenience, disable warnings of type FutureWarning
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import tensorflow as tf
import numpy as np
import time, sys

from tensorflow import nn

BATCH_SIZE = 32
N = 224
FIN = 32
FOUT = 32

K_Y = 3
K_X = 3

EPSILON = 1e-05

NB_TESTS = 101

def resnetBlock(X, weights1, bias1, weights2, bias2, mean1, variance1, offset1, scale1, mean2, variance2, offset2, scale2):
    conv1 = nn.conv2d(X, weights1, strides=[1, 1, 1, 1], padding="VALID", data_format="NHWC")
    conv1_bias = nn.bias_add(conv1, bias1, data_format="NHWC")

    bn1 = nn.batch_normalization(conv1_bias, mean1, variance1, offset1, scale1, EPSILON)
    relu1 = nn.relu(bn1)
    
    conv2 = nn.conv2d(relu1, weights2, strides=[1, 1, 1, 1], padding="SAME", data_format="NHWC")
    conv2_bias = nn.bias_add(conv2, bias2, data_format="NHWC")

    bn2 = nn.batch_normalization(conv2_bias, mean2, variance2, offset2, scale2, EPSILON)
    
    return bn2
    
# Init buffers
np.random.seed(0)

batch_X = np.random.rand(BATCH_SIZE, N + 2, N + 2, FIN)
weights1 = np.random.rand(K_Y, K_X, FIN, FOUT)
weights2 = np.random.rand(K_Y, K_X, FOUT, FOUT)

bias1 = np.random.rand(FOUT)
bias2 = np.random.rand(FOUT)

mean1 = np.random.rand(FOUT)
variance1 = np.random.rand(FOUT)
offset1 = np.random.rand(FOUT)
scale1 = np.random.rand(FOUT)

mean2 = np.random.rand(FOUT)
variance2 = np.random.rand(FOUT)
offset2 = np.random.rand(FOUT)
scale2 = np.random.rand(FOUT)

# Create the network
X = tf.compat.v1.placeholder(tf.float32, [BATCH_SIZE, N + 2, N + 2, FIN], name="X")
activations = resnetBlock(X, weights1, bias1, weights2, bias2, mean1, variance1, offset1, scale1, mean2, variance2, offset2, scale2)

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
