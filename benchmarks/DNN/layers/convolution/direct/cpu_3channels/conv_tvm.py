import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import time, os

import tvm
import tvm.contrib.graph_runtime as runtime
from tvm import relay

import tensorflow as tf
import tvm.relay.testing.tf as tf_testing

from tensorflow import nn

np.random.seed(0)

""" 
    Network parameters
"""
BATCH_SIZE = 8
N = 112
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

input_shape = (BATCH_SIZE, N + 2, N + 2, FIN)
dtype = "float32"

""" 
    Create the graph in TensorFlow 
"""
def Convolution(X, weights, bias):
    conv = nn.conv2d(X, weights, strides=[1, 1, 1, 1], padding="VALID", data_format="NHWC")
    conv_bias = nn.bias_add(conv, bias, data_format="NHWC")

    return conv_bias

weights = np.random.rand(K_Y, K_X, FIN, FOUT)
bias = np.random.rand(FOUT)

X = tf.compat.v1.placeholder(tf.float32, [BATCH_SIZE, N + 2, N + 2, FIN], name="X")
activations = Convolution(X, weights, bias)

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