"""
Generate the job files needed by the sbatch command.

Here's an example of a job file :

#!/bin/bash
#SBATCH --job-name=comp2
#SBATCH --output=log/log_comp_2_6842_10263
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH -p lanka-v3
srun python3 compile_tiramisu_code.py 6842 10263 2
"""

import pickle
from pathlib import Path

# Number of nodes in the cluster
# Each node will do the job on a portion of the programs
nb_nodes = 19

# Path to the list of programs
progs_list_path = Path("progs_list.pickle")

# Path where to store the job files
dst_path = Path("job_files")

# Path to the script that will be distributed
script_path = Path("compile_tiramisu_code.py")

# Path to where to store the logs of the jobs
log_path = Path("log/")

# Content of the job files
job_file_content = "\
#!/bin/bash\n\
#SBATCH --job-name=comp{2}\n\
#SBATCH --output=%s/log_comp_{2}_{0}_{1}\n\
#SBATCH -N 1\n\
#SBATCH --exclusive\n\
#SBATCH -p lanka-v3\n\
srun python3 %s {0} {1} {2}" % (str(log_path), str(script_path)) # This replaces the %s

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
    
    with open(dst_path / ("compile_job_%s_%s.batch" % (start, end)), "w") as f:
        f.write(job_file_content.format(start, end, i))

