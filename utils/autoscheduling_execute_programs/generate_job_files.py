"""
Generate the job files needed by the sbatch command.
Two type of job files are generated :

- One for editing and compiling the wrappers.
- The other for executing the compiled wrappers and measuring execution time.

Here's an example of a job file of type execute :

#!/bin/bash
#SBATCH --job-name=exec17
#SBATCH --output=/data/scratch/k_abdous/autoscheduling_tiramisu/execute_all/log/log_exec_17_1096959_1161486
#SBATCH -N 1
#SBATCH --exclusive
srun python3 /data/scratch/k_abdous/autoscheduling_tiramisu/execute_all/execute_programs.py 1096959 1161486 17
"""

import pickle
from pathlib import Path

# Number of nodes in the cluster
# Each node will do the job on a portion of the programs
nb_nodes = 19

# Path to the list of programs
progs_list_path = Path("/data/scratch/k_abdous/autoscheduling_tiramisu/execute_all/results/progs_list.pickle")

# Path where to store the job files
# This script will use two subdirectories (don't forget to create them first) : wrappers and execute
dst_path = Path("/data/scratch/k_abdous/autoscheduling_tiramisu/execute_all/job_files")

# Path to the script that edits and compiles the wrappers
wrappers_script = Path("/data/scratch/k_abdous/autoscheduling_tiramisu/execute_all/rewrite_tiramisu_wrappers.py")

# Path to the script that execute the compiled wrappers
execute_script = Path("/data/scratch/k_abdous/autoscheduling_tiramisu/execute_all/execute_programs.py")

# Path to where to store the logs of the jobs
log_path = Path("/data/scratch/k_abdous/autoscheduling_tiramisu/execute_all/log/")

# Content of the job files of type wrappers
wrappers_job = "\
#!/bin/bash\n\
#SBATCH --job-name=wrap{2}\n\
#SBATCH --output=%s/log_wrap_{2}_{0}_{1}\n\
#SBATCH -N 1\n\
#SBATCH --exclusive\n\
srun python3 %s {0} {1} {2}" % (str(log_path), str(wrappers_script)) # This replaces the %s
    
# Content of the job files of type execute
execute_job = "\
#!/bin/bash\n\
#SBATCH --job-name=exec{2}\n\
#SBATCH --output=%s/log_exec_{2}_{0}_{1}\n\
#SBATCH -N 1\n\
#SBATCH --exclusive\n\
srun python3 %s {0} {1} {2}" % (str(log_path), str(execute_script)) # This replaces the %s

with open(progs_list_path, "rb") as f:
    progs_list = pickle.load(f)
    
nb_progs = len(progs_list)
progs_per_node = nb_progs // nb_nodes

for i in range(nb_nodes):
    # Each node will process the programs in the range progs_list[start, end)
    start = i * progs_per_node
    
    if i < nb_nodes - 1:
        end = (i + 1) * progs_per_node
    else:
        end = nb_progs
    
    with open(dst_path / "wrappers" / ("wrappers_job_%s_%s.batch" % (start, end)), "w") as f:
        f.write(wrappers_job.format(start, end, i))
        
    with open(dst_path / "execute" / ("execute_job_%s_%s.batch" % (start, end)), "w") as f:
        f.write(execute_job.format(start, end, i))
