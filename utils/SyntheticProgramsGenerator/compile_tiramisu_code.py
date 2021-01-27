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
batchName = sys.argv[3]

# Path to the directory containing the programs
data_path = Path("./data/"+batchName+"/programs/")

# Path to the list of programs
progs_list_path = Path("./time_measurement/progs_list_"+batchName+".pickle")

# Path to where to store the results
dst_path = Path("./time_measurement/results_"+batchName+"/")
dst_path.mkdir(parents=True, exist_ok=True)

# Path to Tiramisu root directory
TIRAMISU_ROOT = "/data/scratch/mhleghettas/Work/tiramisu"


errors_compile_tiramisu = "./time_measurement/results_"+batchName+"/"+"errors_compile_tiramisu.txt"
errors_execute_generator = "./time_measurement/results_"+batchName+"/"+"errors_execute_generator.txt"


# prog is a tuple of format (function_id, schedule_id)
# Compile the Tiramisu code of the given program and generate the object file.
def compile_tiramisu_code(prog):
    func_id, sched_id = prog

    src_folder_path = data_path / func_id / sched_id
    dst_folder_path = dst_path / func_id / sched_id
    
    os.makedirs(dst_folder_path, exist_ok=True)
                
    
    try:
        subprocess.run(["g++", "-std=c++11", "-fno-rtti", "-DHALIDE_NO_JPEG", "-I%s/include" % TIRAMISU_ROOT, "-I%s/3rdParty/isl/include/" % TIRAMISU_ROOT, "-I%s/3rdParty/Halide/include" % TIRAMISU_ROOT, "-I%s/benchmarks" % TIRAMISU_ROOT, "-I%s/build" % TIRAMISU_ROOT, "-L%s/build" % TIRAMISU_ROOT, "-L%s/3rdParty/isl/build/lib" % TIRAMISU_ROOT, "-L%s/3rdParty/Halide/lib/" % TIRAMISU_ROOT, "-o", str(dst_folder_path / (sched_id + "_generator")), "-ltiramisu", "-lisl", "-lHalide", "-ldl", "-lpthread", "-lz", "-lm", "-Wl,-rpath,%s/build" % TIRAMISU_ROOT, str(src_folder_path / (sched_id + ".cpp")), "-ltiramisu", "-lisl", "-lHalide", "-ldl", "-lpthread", "-lz", "-lm"], check=True)
        subprocess.run(["chmod", "a+x", str(dst_folder_path / (sched_id + "_generator"))], check=True)
    except:
        with open(errors_compile_tiramisu,"a") as f:
            f.write(sched_id+",\n")
    # Execute code generator
    # Remark : we deactivate stdout because Tiramisu prints a lot of messages when generating an object file
    # When using sbatch, those messages will be redirected to a file, so it may make the script very slow
#     subprocess.run(["echo", "%s - running gen" % sched_id], check=True)
    
    try:
        subprocess.run([str(dst_folder_path / (sched_id + "_generator"))], check=True)
    except:
        with open(errors_execute_generator,"a") as f:
            f.write(sched_id+",\n")
    
    
    
#     subprocess.run([str(dst_folder_path / (sched_id + "_generator"))], stdout=subprocess.DEVNULL, check=True)
#     subprocess.run(["time", str(dst_folder_path / (sched_id + "_generator"))], check=True)

    
if __name__ == "__main__":
    start = int(sys.argv[1])
    end = int(sys.argv[2])
    
    

    with open(progs_list_path, "rb") as f:
        progs_list = pickle.load(f)

    progs_list = progs_list[start:end]
    
    # We use multiprocessing.Pool to parallelize inside a node
    with Pool(nb_threads) as pool:
        pool.map(compile_tiramisu_code, progs_list, chunksize=chunksize)
    print("finished")

