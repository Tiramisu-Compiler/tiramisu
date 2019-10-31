import pickle
import json

from pathlib import Path
from multiprocessing import Pool

MAX_NB_ITERATORS = 4

nb_threads = 48
chunksize = 100

# Path to the programs folder
progs_path = Path("programs")

# Path where to store the list of filtered programs
# Each element of the list has the format (func_id, sched_id)
dst_path = Path("progs_list.pickle")

# Process a program folder containing schedules
def process_programs(func_path):
    output = []
    
    func_id = func_path.name
    with open(str(func_path / (func_id + ".json"))) as f:
        prog_desc = json.load(f)
    
    # A dictionary containing the pairs iterator_id : iterator_extent
    iterators = dict()
    
    # Get information about loop iterators
    for i, loop_it in enumerate(prog_desc["loops"]["loops_array"]):
        for it in prog_desc["iterators"]["iterators_array"]:
            if loop_it["loop_it"] == it["it_id"]:
                iterators[it["it_id"]] = it["upper_bound"] - it["lower_bound"]
    
    # If true, do not apply respectively interchange, tiling to every schedule
    no_interchange = True    
    no_tiling = True
         
    for comp in prog_desc["computations"]["computations_array"]:
        if not no_interchange and not no_tiling:
            break
            
        # If this computation is a reduction, we try to apply tiling
        if "reduction_axes" in comp.keys() and len(comp["reduction_axes"]) != 0:
            no_tiling = False
        
        accesses = comp["rhs_accesses"]["accesses"]

        # Check if it can be useful to apply interchange
        if no_interchange:
            for el in accesses:
                access = el["access"]
                
                # Check the number of access iterators in each dimension
                # If it's more than one, we try interchange
                for i, line in enumerate(access):
                    if line[:-1].count(1) > 1:
                        no_interchange = False
                        break
                        
                if not no_interchange:
                    break
                    
                # Check the order of access iterators
                cur = 0
                for line in access:
                    i = line[:-1].index(1)
                    if i < cur:
                        no_interchange = False
                        break
                        
                    cur = i
                    
                if not no_interchange:
                    break
        
        # Check if it can be useful to apply tiling
        if no_tiling:
            # Check if every buffer is used at most once
            buf_set = set()
            for el in accesses:
                if el["comp_id"] in buf_set:
                    no_tiling = False
                    break

                buf_set.add(el["comp_id"])

            if no_tiling:
                # Check the access matrices
                for el in accesses:
                    acc_matrix = el["access"]

                    # The number of iterators is less than the number of dimensions of the buffer
                    # Thus, this can't be an element-wise operation
                    if len(acc_matrix) < len(iterators):
                        no_tiling = False
                        break

                    # Check if every dimension contains only one iterator
                    # If a dimension is accessed with two iterators (for example i + j),
                    # this can't be an element-wise operation.
                    for line in acc_matrix:
                        if sum(line[:-1]) > 1:
                            no_tiling = False
                            break

                    if not no_tiling:
                        break

                    # Check if there's not an iterator that is used in two dimensions or more.
                    # If an iterator is used in multiple dimensions, it might be useful to try tiling.
                    for i in range(len(acc_matrix[0])):
                        sum_var = 0
                        for j in range(len(acc_matrix)):
                            sum_var = sum_var + acc_matrix[j][i]

                        if sum_var > 1:
                            no_tiling = False
                            break

                    if not no_tiling:
                        break
    
    # Process schedules
    for sched_path in func_path.iterdir():
        if not sched_path.is_dir():
            continue
            
        sched_id = sched_path.name
        with open(str(sched_path / (sched_id + ".json"))) as f:
            sched_desc = json.load(f)
        
        # Discard this schedule if it applies interchange and we don't want it.
        if len(sched_desc["interchange_dims"]) != 0 and no_interchange:
            continue
            
        if sched_desc["tiling"] != None:
            # Discard this schedule if it applies tiling and we don't want it.
            if no_tiling:
                continue
                
            # Check the tiling factors (must be at least half of the iterators extent)
            tiling_facts_big = False
            
            tdims = sched_desc["tiling"]["tiling_dims"]
            tfacts = sched_desc["tiling"]["tiling_factors"]
            
            for i in range(len(tdims)):
                if iterators[tdims[i]] / tfacts[i] < 2.0:
                    tiling_facts_big = True
                    break
                    
            if tiling_facts_big:
                continue
            
        output.append((func_id, sched_id))
        
    return output

if __name__ == "__main__":
    progs = list(progs_path.iterdir())
    with Pool(nb_threads) as p:
        map_ret = p.map(process_programs, progs, chunksize=chunksize)

    scheds_list = []
    
    # Flatten map_ret
    for i1 in map_ret:
        scheds_list.extend(i1)
        
    print(len(scheds_list))
        
    # Save the filtered programs list
    with open(dst_path, "wb") as f:
        pickle.dump(scheds_list, f)