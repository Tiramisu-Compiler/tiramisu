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

""" 
    Create the graph in TensorFlow 
"""
def ResizeConvReluMaxPool(X, weights, bias):
    resize = image.resize(X, [N + 2, N + 2])
    resize_transposed = tf.transpose(resize, [0, 3, 1, 2])

    conv = nn.conv2d(resize_transposed, weights, strides=[1, 1, 1, 1], padding="VALID", data_format="NCHW")
    conv_bias = nn.bias_add(conv, bias, data_format="NCHW")

    relu = nn.relu(conv_bias)
    maxpool = nn.max_pool2d(relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID", data_format="NCHW", name="output")
    
    return maxpool

weights = np.random.rand(K_Y, K_X, FIN, FOUT)
bias = np.random.rand(FOUT)

X = tf.compat.v1.placeholder(tf.float32, [BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, FIN], name="X")
activations = ResizeConvReluMaxPool(X, weights, bias)

model_path = "tf_model.pb"
tf.io.write_graph(tf.compat.v1.get_default_graph(), "", model_path, as_text=False)

""" 
    Create TF graph definition 
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

""" 
    AutoTune the network (taken from https://docs.tvm.ai/tutorials/autotvm/tune_relay_x86.html) 
"""
log_file = "tvm_autotuning.log"

tuner_type = "random"
input_name = "X"
dtype = "float32"

tuning_option = {
    "log_filename": log_file,
    "tuner": tuner_type,
    "early_stopping": None,

    "measure_option": autotvm.measure_option(
        builder=autotvm.LocalBuilder(),
        runner=autotvm.LocalRunner(number=10, repeat=1,
                                   min_repeat_ms=1000),
    ),
}

# Set number of threads used for tuning
num_threads = 4
os.environ["TVM_NUM_THREADS"] = str(num_threads)

# Tune a set of convolutions
def tune_kernels(tasks,
                 measure_option,
                 tuner="gridsearch",
                 early_stopping=None,
                 log_filename="tuning.log"):

    for i, tsk in enumerate(tasks):
        prefix = "[Task %2d/%2d] " % (i+1, len(tasks))

        # converting conv2d tasks to conv2d_NCHWc tasks
        op_name = tsk.workload[0]
        if op_name == "conv2d":
            func_create = "topi_x86_conv2d_NCHWc"
        elif op_name == "depthwise_conv2d_nchw":
            func_create = "topi_x86_depthwise_conv2d_NCHWc_from_nchw"
        else:
            raise ValueError("Tuning {} is not supported on x86".format(op_name))

        task = autotvm.task.create(func_create, args=tsk.args,
                                   target=target, template_key="direct")
        task.workload = tsk.workload

        # create tuner
        if tuner == "xgb" or tuner == "xgb-rank":
            tuner_obj = XGBTuner(task, loss_type='rank')
        elif tuner == "ga":
            tuner_obj = GATuner(task, pop_size=50)
        elif tuner == "random":
            tuner_obj = RandomTuner(task)
        elif tuner == "gridsearch":
            tuner_obj = GridSearchTuner(task)
        else:
            raise ValueError("Invalid tuner: " + tuner)

        # do tuning
        n_trial = len(task.config_space)
        tuner_obj.tune(n_trial=n_trial,
                       early_stopping=early_stopping,
                       measure_option=measure_option,
                       callbacks=[
                           autotvm.callback.progress_bar(n_trial, prefix=prefix),
                           autotvm.callback.log_to_file(log_filename)])

# Call this function with tuning options to start tuning
def tune_and_evaluate(tuning_opt):
    # extract workloads from relay program
    print("Extract tasks...")
    tasks = autotvm.task.extract_from_program(mod["main"], target=target,
                                              params=parameters, ops=(relay.op.nn.conv2d,))

    # run tuning tasks
    print("Tuning...")
    tune_kernels(tasks, **tuning_opt)

    # compile kernels with graph-level best records
    with autotvm.apply_history_best(log_file):
        print("Compile...")
        with relay.build_config(opt_level=3):
            graph, lib, params = relay.build_module.build(
                mod, target=target, params=parameters)

        # upload parameters to device
        ctx = tvm.cpu()
        data_tvm = tvm.nd.array((np.random.uniform(size=input_shape)).astype(dtype))
        module = runtime.create(graph, lib, ctx)
        module.set_input(input_name, data_tvm)
        module.set_input(**params)

        # evaluate
        print("Evaluate inference time cost...")
        ftimer = module.module.time_evaluator("run", ctx, number=NB_TESTS, repeat=1)
        prof_res = np.array(ftimer().results) * 1000  # convert to millisecond

        print("Tuned network execution time : ", np.median(prof_res))

tune_and_evaluate(tuning_option)