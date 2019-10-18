
import numpy as np 
from itertools import permutations
from tqdm import tqdm 
import h5py
import dill as pickle
from src.data.stats import *

def data_to_h5(programs, schedules, exec_times, filename="speedup_dataset.h5"):
    #create file
    f = h5py.File(filename, mode="w")

    #get dimensions of data
    n_cols_progs = len(programs[0].__array__())
    n_cols_scheds = len(schedules[0][0].__array__())

    #get data
    programs_array, schedules_array, speedup_array, times_array, prog_names, sched_names = get_speedup_data(programs, schedules, exec_times)

    assert programs_array.shape[0] == schedules_array.shape[0]
    assert schedules_array.shape[0] == times_array.shape[0]
    assert schedules_array.shape[1] == n_cols_scheds
    assert programs_array.shape[1] == n_cols_progs 

    #create datasets 
    f.create_dataset('programs', data=programs_array, dtype="int32")
    f.create_dataset('schedules', data=schedules_array, dtype="int16") 
    f.create_dataset('times', data=times_array) 
    f.create_dataset('speedup', data=speedup_array)
    f.create_dataset('programs_names', data=prog_names)
    f.create_dataset('schedules_names', data=sched_names)
    f.close()



def get_speedup_data(programs, schedules, exec_times):

    assert len(programs) == len(schedules) 
    assert len(schedules) == len(exec_times)

   
    prog_names = []
    schedules_array = []
    sched_names = []
    times_array = []
    duplicated_programs = []
    exec_times_array = []

    for i in range(len(programs)):
        
        try:
            
            assert "no_schedule" in schedules[i][0].name 
        except AssertionError:
            print(schedules[i][0].name)
            exit(1)
   

        for j in range(len(schedules[i])):
            duplicated_programs.append(np.array(programs[i]))
            schedules_array.append(np.array(schedules[i][j]))
            sched_names.append(schedules[i][j].name)
            prog_names.append(programs[i].name)

            speedup = exec_times[i][0] / exec_times[i][j] 
            times_array.append(speedup)
            exec_times_array.append(exec_times[i][j])



    schedules_array = np.array(schedules_array)
    times_array = np.array(times_array)
    exec_times_array = np.array(exec_times_array)
    duplicated_programs = np.array(duplicated_programs)
    prog_names = np.array(prog_names, dtype=h5py.special_dtype(vlen=str))
    sched_names = np.array(sched_names, dtype=h5py.special_dtype(vlen=str))


    return (duplicated_programs, schedules_array, times_array, exec_times_array, prog_names, sched_names)


def get_data(programs, schedules, exec_times):
    
    assert len(programs) == len(schedules) 
    assert len(schedules) == len(exec_times)

   
    programs_array = np.array([np.array(program) for program in programs])

    schedules_array = []
    times_array = []

    for program_schedules in schedules:
        program_schedules = [np.array(schedule) for schedule in program_schedules]
        schedules_array.extend(program_schedules)

    for program_times in exec_times:
        times_array.extend(program_times)

    schedules_array = np.array(schedules_array)
    times_array = np.array(times_array)

    indexes_array = []
    schedule_offset = 0
    permutation_offset = 0 

    for i in range(len(programs_array)):
        num_schedules = len(schedules[i]) #number of schedules for prog i

        schedule_offset += num_schedules 
        permutation_offset += num_schedules*(num_schedules - 1)

        indexes_array.append([schedule_offset, permutation_offset])

        
        
    indexes_array = np.array(indexes_array)


    return (programs_array, indexes_array, schedules_array, times_array)


def serialize(programs, schedules, exec_times, filename='speedup_dataset.pkl'):

    assert len(programs) == len(schedules) 
    assert len(schedules) == len(exec_times)

    speedup_array = []
    program_indexes = []
    schedules_array = []
    exec_times_array = []
    
    for i in tqdm(np.random.RandomState(seed=42).permutation(range(len(programs)))):
        #try:
        #    assert "no_schedule" in schedules[i][0].name 
         
       # except AssertionError:
       #     print(schedules[i][0].name)
       #     exit(1)
        if "no_schedule" not in schedules[i][0].name:
            continue

        for j in range(len(schedules[i])):
            
            program_indexes.append(i)
            schedules_array.append(schedules[i][j])

            speedup = exec_times[i][0] / exec_times[i][j] 
            speedup_array.append(speedup)
            exec_times_array.append(exec_times[i][j])


    save  = {'programs':programs, 'program_indexes':program_indexes,
             'schedules':schedules_array, 'exec_times':exec_times_array, 'speedup': speedup_array}

    f = open(filename, 'wb')

    pickle.dump(save, f)

    f.close()





if __name__=='__main__':
    st = Stats('/data/scratch/mmerouani/data/batch2501-3000/')

    print("loading data")
#     programs, schedules, exec_times = st.load_data()
    programs, schedules, exec_times = st.load_data_separate_exec_times('/data/scratch/mmerouani/data/batch2501-3000/final_exec_times_batch2501-3000.pickle')
    print("data loaded")
    print("calculating model input")
    #data_to_h5(programs, schedules, exec_times)
    serialize(programs, schedules, exec_times,filename='speedup_dataset_research_batch2501-3000.pkl')
    print("done")


