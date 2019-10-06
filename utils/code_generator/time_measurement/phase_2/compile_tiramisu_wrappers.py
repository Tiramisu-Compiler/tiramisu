"""
This compiles the wrappers. 

It supposes that the wrappers are in the good format (see rewrite_tiramisu_wrappers.py for more details)

Note that the wrappers must take nb_tests as a command line argument

This script accepts two parameters : start end
It processes the programs in the range progs_list[start, end)

"""

import os, sys, pickle, subprocess
from pathlib import Path
from shutil import copyfile
from multiprocessing import Pool

# Parameters to multiprocessing.Pool (see last lines of the script)
nb_threads = 48
chunksize = 100

# Path to the directory containing the programs
data_path = Path("/data/scratch/henni-mohammed/data3/programs/")

# Path to the list of programs
progs_list_path = Path("progs_list.pickle")

# Path to where to store the compiled wrappers
dst_path = Path("wrappers")

# Path to Tiramisu root directory
TIRAMISU_ROOT = "/data/scratch/k_abdous/software/tiramisu"

# prog is a tuple of format (function_id, schedule_id)
# Compile the wrapper of the given program
def rewrite_wrapper(prog):
    func_id, sched_id = prog
    
    # Respectively path to the wrapper, and path to the compiled wrapper
    wrap_src_path = data_path / func_id / sched_id
    wrap_dst_path = dst_path / func_id / sched_id
    
    os.makedirs(wrap_dst_path, exist_ok=True)
                
    # Compile wrapper
    subprocess.run(["g++", "-std=c++11", "-fno-rtti", "-I%s/include" % TIRAMISU_ROOT, "-I%s/3rdParty/Halide/include" % TIRAMISU_ROOT, "-I%s/3rdParty/isl/include/" % TIRAMISU_ROOT, "-I%s/benchmarks" % TIRAMISU_ROOT, "-L%s/build" % TIRAMISU_ROOT, "-L%s/3rdParty/Halide/lib/" % TIRAMISU_ROOT, "-L%s/3rdParty/isl/build/lib" % TIRAMISU_ROOT, "-ltiramisu", "-lHalide", "-ldl", "-lpthread", "-lz", "-lm", "-Wl,-rpath,%s/build" % TIRAMISU_ROOT, "-o", str(wrap_dst_path / sched_id), str(wrap_src_path / (sched_id + "_wrapper.cpp")), str(wrap_src_path / (sched_id + ".o")), "-ltiramisu", "-lHalide", "-ldl", "-lpthread", "-lz", "-lm"])
    
    # Make sure that the wrapper is executable
    subprocess.run(["chmod", "a+x", str(wrap_dst_path / sched_id)])
    
if __name__ == "__main__":
    start = int(sys.argv[1])
    end = int(sys.argv[2])

    with open(progs_list_path, "rb") as f:
        progs_list = pickle.load(f)

    progs_list = progs_list[start:end]
    
    # We use multiprocessing.Pool to parallelize inside a node
    with Pool(nb_threads) as pool:
        pool.map(rewrite_wrapper, progs_list, chunksize=chunksize)
