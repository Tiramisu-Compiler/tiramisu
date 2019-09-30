Here's a description of each script (there are scripts that generate python lists, we use pickle to save them to disk) :

### create_progs_list.py

The first script to call : it generates a list of programs that will be executed. This list will be a more convenient way to distribute tasks across multiple nodes.

Each entry of the list is of format : (function_id, schedule_id).

**IMPORTANT :** There are some directories that contain only a program, with no schedule. Those programs won't be included in the final list as they don't need to be executed : they just have a speedup of 1, so if you want them in the final dataset, just include them with a speedup of 1.

### rewrite_tiramisu_wrappers.py

This script reads the list of programs, and edits and compiles each Tiramisu wrapper. See the script comments for the reason why we need to do that.

This script generates .cpp, .h and executable files in the specified directory.

### execute_programs.py

This is the script that measures execution times (`rewrite_tiramisu_wrappers.py` must be executed first). It generates a set of lists that contain the execution times. Each entry of a list is of format (function_id, schedule_id, list of execution times).

### generate_job_files.py

This script generates the job files that will be submitted to SLURM. See the script comments for more information.

### preprocess_programs.py

At the end of `execute_programs.py`, this script must be executed to calculate medians and speedups, and to fuse all the generated files to a single one. The final file is a list, where each entry has the following format :

(function_id, schedule_id, list of execution times, median, speedup)

### job_files/submit_execute_jobs.sh job_files/submit_wrapper_jobs.sh

Submit the jobs to SLURM.

### How to work with the scripts

Before executing a script, you need to configure some parameters. See each script for more details.

First execute `create_progs_list.py` to generate the list of programs that will be executed. After that, configure `rewrite_tiramisu_wrappers.py` and `execute_programs.py` (note that if your wrappers are already in the good format, you don't need to use `rewrite_tiramisu_wrappers.py`). Then execute `generate_job_files.py` to get the job files that will be submitted to SLURM. You will get two types of job files : 

1. Wrapper job files : these job files will execute the script `rewrite_tiramisu_wrappers.py`.
2. Execute job files : these job files will execute the script `execute_programs.py`.

When this is done, submit the wrapper jobs to SLURM by using `job_files/submit_wrapper_jobs.sh`. Wait for the jobs to finish, and then submit the executon jobs by using `job_files/submit_execute_jobs.sh`.

When all the jobs are done, you will have a lot of files that contain execution times. Execute `preprocess_programs.py` to calculate medians and speedups, and to fuse everything to a single file.