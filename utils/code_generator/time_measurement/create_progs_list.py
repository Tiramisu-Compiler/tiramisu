"""
Create the list of programs that will be executed.
The result is a list of tuples of the following format :

(function_id, schedule_id)

Example :
(function524, function524_schedule_125)
"""

import pickle
from pathlib import Path

# Path to the directory containing the programs
data_path = Path("/data/scratch/henni-mohammed/data3/programs/")

# Path to where to store the list of programs
dst_path = Path("progs_list.pickle")

progs_list = []

for func_path in data_path.iterdir():
    # We discard programs that have no schedule.
    # We don't need to execute those programs as they just have a speedup of 1,
    # and they have no programs with schedules.
    # If you want them is the dataset, just include them with speedup = 1.
    if len(list(func_path.iterdir())) <= 2:
        continue
    
    for sched_path in func_path.iterdir():
        if not sched_path.is_dir():
            continue
            
        func_id = func_path.parts[-1]
        sched_id = sched_path.parts[-1]
        
        progs_list.append((func_id, sched_id))
        
with open(dst_path, "wb") as f:
    pickle.dump(progs_list, f)
