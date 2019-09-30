## Phase 2 : compile wrappers and execute them

Here's a description of each script (there are scripts that generate python lists, we use pickle to save them to disk) :

### rewrite_tiramisu_wrappers.py

This script reads the list of programs, and edits and compiles each Tiramisu wrapper. See the script comments for the reason why we need to do that.

This script generates .cpp, .h and executable files in the specified directory.

### compile_tiramisu_wrappers.py

If your wrappers are already in the good format (see `rewrite_tiramisu_wrappers.py` for more details), use this script instead of `rewrite_tiramisu_wrappers.py`.

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

Before executing a script, you need to configure some parameters. See each script for more details. Note that to execute those scripts, you need to have generated :

- the list of programs with `create_progs_list.py`.
- The tiramisu object files (see phase 1).

First, configure `rewrite_tiramisu_wrappers.py` and `execute_programs.py` (note that if your wrappers are already in the good format, use `compile_tiramisu_wrappers.py` instead of `rewrite_tiramisu_wrappers.py`). Then execute `generate_job_files.py` to get the job files that will be submitted to SLURM (if you use `compile_tiramisu_wrappers.py`, don't forget to configure `generate_job_files.py` so that it will use the correct script). You will get two types of job files : 

1. Wrapper job files : these job files will execute the script `rewrite_tiramisu_wrappers.py` or `compile_tiramisu_wrappers.py`.
2. Execute job files : these job files will execute the script `execute_programs.py`.

When this is done, submit the wrapper jobs to SLURM by using `submit_wrapper_jobs.sh`. Wait for the jobs to finish, and then submit the executon jobs by using `submit_execute_jobs.sh`.

When all the jobs are done, you will have a lot of files that contain execution times. Execute `preprocess_programs.py` to calculate medians and speedups, and to fuse everything to a single file.
