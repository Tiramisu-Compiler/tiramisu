"""
This script compiles the Tiramisu codes and generates the object files.

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

# Path to where to store the results
dst_path = Path("results/")

# Path to Tiramisu root directory
TIRAMISU_ROOT = "/data/scratch/k_abdous/software/tiramisu"

# prog is a tuple of format (function_id, schedule_id)
# Compile the Tiramisu code of the given program and generate the object file.
def compile_tiramisu_code(prog):
    func_id, sched_id = prog
    
    src_folder_path = data_path / func_id / sched_id
    dst_folder_path = dst_path / func_id / sched_id
    
    os.makedirs(dst_folder_path, exist_ok=True)
                
    # Compile tiramisu code generator
    subprocess.run(["clang++", "-std=c++11", "-fno-rtti", "-DHALIDE_NO_JPEG", "-I%s/include" % TIRAMISU_ROOT, "-I%s/3rdParty/isl/include/" % TIRAMISU_ROOT, "-I%s/3rdParty/Halide/include" % TIRAMISU_ROOT, "-I%s/benchmarks" % TIRAMISU_ROOT, "-I%s/build" % TIRAMISU_ROOT, "-L%s/build" % TIRAMISU_ROOT, "-L%s/3rdParty/isl/build/lib" % TIRAMISU_ROOT, "-L%s/3rdParty/Halide/lib/" % TIRAMISU_ROOT, "-o", str(dst_folder_path / (sched_id + "_generator")), "-ltiramisu", "-lisl", "-lHalide", "-ldl", "-lpthread", "-lz", "-lm", "-Wl,-rpath,%s/build" % TIRAMISU_ROOT, str(src_folder_path / (sched_id + ".cpp")), "-ltiramisu", "-lisl", "-lHalide", "-ldl", "-lpthread", "-lz", "-lm"], check=True)
    
    # Make sure that the code generator is executable
    subprocess.run(["chmod", "a+x", str(dst_folder_path / (sched_id + "_generator"))], check=True)
    
    # Execute code generator
    # Remark : we deactivate stdout because Tiramisu prints a lot of messages when generating an object file
    # When using sbatch, those messages will be redirected to a file, so it may make the script very slow
    subprocess.run([str(dst_folder_path / (sched_id + "_generator"))], stdout=subprocess.DEVNULL, check=True)
    
if __name__ == "__main__":
    start = int(sys.argv[1])
    end = int(sys.argv[2])

    with open(progs_list_path, "rb") as f:
        progs_list = pickle.load(f)

    progs_list = progs_list[start:end]
    
    # We use multiprocessing.Pool to parallelize inside a node
    with Pool(nb_threads) as pool:
        pool.map(compile_tiramisu_code, progs_list, chunksize=chunksize)

