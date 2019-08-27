import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import time, os

import tvm
from tvm import relay
from tvm import autotvm
from tvm.relay import testing
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner
from tvm.autotvm.graph_tuner import DPTuner, PBQPTuner
import tvm.contrib.graph_runtime as runtime

import tensorflow as tf
import tvm.relay.testing.tf as tf_testing

from tensorflow import nn
from tensorflow import image

np.random.seed(0)

""" 
    Network parameters
"""
IMG_WIDTH = 600
IMG_HEIGHT = 400

BATCH_SIZE = 32
N = 224
FIN = 3
FOUT = 32

K_Y = 3
K_X = 3

NB_TESTS = 101

""" 
    Target settings
"""
target = "llvm -mcpu=core-avx2"
target_host = "llvm"
layout = None

input_shape = (BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, FIN)
dtype = "float32"

""" 
    Create the graph in TensorFlow 
"""
def ResizeConvReluMaxPool(X, weights, bias):
    resize = image.resize(X, [N + 2, N + 2])
    
    conv = nn.conv2d(resize, weights, strides=[1, 1, 1, 1], padding="VALID", data_format="NHWC")
    conv_bias = nn.bias_add(conv, bias, data_format="NHWC")

    relu = nn.relu(conv_bias)
    maxpool = nn.max_pool2d(relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID", data_format="NHWC", name="output")
    
    return maxpool

weights = np.random.rand(K_Y, K_X, FIN, FOUT)
bias = np.random.rand(FOUT)

X = tf.compat.v1.placeholder(tf.float32, [BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, FIN], name="X")
activations = ResizeConvReluMaxPool(X, weights, bias)

model_path = "tf_model.pb"
tf.io.write_graph(tf.compat.v1.get_default_graph(), "", model_path, as_text=False)

""" 
    Create the graph in TVM and compile it 
"""
with tf.io.gfile.GFile(model_path, "rb") as f:
    graph_def = tf.compat.v1.GraphDef()
    graph_def.ParseFromString(f.read())
    graph = tf.import_graph_def(graph_def, name="")

    graph_def = tf_testing.ProcessGraphDefParam(graph_def)

    # Add shapes to the graph.
    with tf.Session() as sess:
        graph_def = tf_testing.AddShapesToGraphDef(sess, "output")

# Import TF graph definition to Relay frontend
shape_dict = {"X": input_shape}
mod, parameters = relay.frontend.from_tensorflow(graph_def, layout=layout, shape=shape_dict)

# Compile the graph
with relay.build_config(opt_level=3):
    graph, lib, params = relay.build_module.build(
        mod, target=target, params=parameters)

""" Execute and evaluate the graph """
ctx = tvm.cpu()
data_tvm = tvm.nd.array((np.random.uniform(size=input_shape)).astype(dtype))
module = runtime.create(graph, lib, ctx)
module.set_input("X", data_tvm)
module.set_input(**params)

# evaluate
print("Evaluate inference time cost...")
ftimer = module.module.time_evaluator("run", ctx, number=NB_TESTS, repeat=1)
prof_res = np.array(ftimer().results) * 1000  # convert to millisecond

print("Network execution time : ", np.median(prof_res))