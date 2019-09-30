"""
This script rewrites the wrapper files and compiles them.

The current wrappers are of the form :

1. measure execution time of one execution
2. save the execution time to a file

This script generates wrappers of the form:

1. for (int i = 0; i < nb_tests; ++i)
       measure execution time
2. Send execution times to the calling process by writing to stdout.

The generated wrappers take nb_tests as a command line argument

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
progs_list_path = Path("../results/progs_list.pickle")

# Path to where to store the new wrappers
dst_path = Path("../results/wrappers")

# Path to Tiramisu root directory
TIRAMISU_ROOT = "/data/scratch/k_abdous/software/tiramisu"

# prog is a tuple of format (function_id, schedule_id)
# Rewrite the wrapper of the given program and compile it.
def rewrite_wrapper(prog):
    func_id, sched_id = prog
    
    # Respectively path to the original and the new wrapper
    wrap_src_path = data_path / func_id / sched_id
    wrap_dst_path = dst_path / func_id / sched_id
    
    os.makedirs(wrap_dst_path, exist_ok=True)
    
    # Copy wrapper.h
    copyfile(wrap_src_path / (sched_id + "_wrapper.h"), wrap_dst_path / (sched_id + "_wrapper.h"))
    
    # Rewrite wrapper.cpp
    with open(wrap_src_path / (sched_id + "_wrapper.cpp"), "r") as f_in:
        with open(wrap_dst_path / (sched_id + "_wrapper.cpp"), "w") as f_out:
            lines = f_in.readlines()

            # Didn't find a better way to edit the wrappers
            for line in lines:
                if line.strip().startswith("#include \"function"):
                    f_out.write("#include \"" + sched_id + "_wrapper.h\"\n")
                    continue
                    
                elif line.strip().startswith("int main"):
                    f_out.write(line.replace("char **", "char ** argv"))
                    continue

                elif line.strip().startswith("auto t1"):
                    f_out.write("\tint nb_tests = atoi(argv[1]);\n\n")
                    f_out.write("\tfor (int i = 0; i < nb_tests; ++i) {\n")

                elif line.strip().startswith("std::c"):
                    f_out.write("\t" + line)
                    f_out.write("\t\tstd::cout << diff.count() * 1000000 << \" \";\n\t}\n")
                    continue

                elif line.strip().startswith("std::o"):
                    f_out.write("\treturn 0; \n}\n")
                    break

                f_out.write(line)
                
    # Compile wrapper
    subprocess.run(["g++", "-std=c++11", "-fno-rtti", "-I%s/include" % TIRAMISU_ROOT, "-I%s/3rdParty/Halide/include" % TIRAMISU_ROOT, "-I%s/3rdParty/isl/include/" % TIRAMISU_ROOT, "-I%s/benchmarks" % TIRAMISU_ROOT, "-L%s/build" % TIRAMISU_ROOT, "-L%s/3rdParty/Halide/lib/" % TIRAMISU_ROOT, "-L%s/3rdParty/isl/build/lib" % TIRAMISU_ROOT, "-ltiramisu", "-lHalide", "-ldl", "-lpthread", "-lz", "-lm", "-Wl,-rpath,%s/build" % TIRAMISU_ROOT, "-o", str(wrap_dst_path / sched_id), str(wrap_dst_path / (sched_id + "_wrapper.cpp")), str(wrap_src_path / (sched_id + ".o")), "-ltiramisu", "-lHalide", "-ldl", "-lpthread", "-lz", "-lm"])
    
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
