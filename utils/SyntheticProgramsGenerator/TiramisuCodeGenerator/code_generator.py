import json
from functools import reduce
from os import path, mkdir, makedirs
from random import sample
from random import randint
from shutil import rmtree
from schedule import *
# from classes import *

from tqdm import tqdm

#research nodes
# seed = 455000
# number_of_functions = 10000
# nb_schedule_per_function = 32


#bigram node
# seed = 400000
# number_of_functions = 500
# nb_schedule_per_function = 32
# batchName = 'bigram_batch'+'{:06d}'.format(seed)+'-'+'{:06d}'.format(seed+number_of_functions-1)




def generate_programs(seed=455000, number_of_functions=10000, nb_schedule_per_function=32, batchName=None, output_dir=None):
    

    if batchName == None:
        batchName ='batch'+'{:06d}'.format(seed)+'-'+'{:06d}'.format(seed+number_of_functions-1)
    if output_dir == None:
        output_dir= './'+batchName+'/programs/'
        
    output_dir = str(output_dir)
    
    if path.exists(output_dir):
        rmtree(output_dir)

    makedirs(output_dir)
    schedule_numbers = []
    
    print('Generating ', number_of_functions, ' programs to ', output_dir, flush=True )
    
    
    for i in tqdm(range(seed, seed + number_of_functions)):
        i = '{:06d}'.format(i)  # use fixed size ids
        scheduledF = ScheduledFunction(i, batch_name=batchName, output_dir=output_dir)
        scheduledF.f.apply_schedule(None)
        schedule_numbers.append(len(scheduledF.schedules))
        function_dir = path.join(output_dir, 'function' + str(i))
        mkdir(function_dir)
        # j = json.dumps(scheduledF.f.get_representation(), indent=4)
        # file = open(path.join(function_dir, 'function' + str(i) + ".json"), 'w')
        # print(j, file=file)

        function_schedule_dir = path.join(function_dir, 'function' + str(i) + '_no_schedule')
        mkdir(function_schedule_dir)
        file = open(path.join(function_schedule_dir, 'function' + str(i) + '_no_schedule' + ".cpp"), 'w')
        print(scheduledF.f.get_program_cpp("_no_schedule", scheduledF.f.comps_ordering), file=file)

    #    d = {"interchange_dims": None, "tiling": None, "unrolling_factor": None}
    #    j = json.dumps(d, indent=4)
    #    file = open(path.join(function_schedule_dir, 'function' + str(i) + '_no_schedule' + ".json"), 'w')
    #    print(j, file=file)

        file = open(path.join(function_schedule_dir, 'function' + str(i) + '_no_schedule_wrapper' + ".cpp"), 'w')
        print(scheduledF.f.get_wrapper_cpp("_no_schedule"), file=file)
        file = open(path.join(function_schedule_dir, 'function' + str(i) + '_no_schedule_wrapper' + ".h"), 'w')
        print(scheduledF.f.get_wrapper_h("_no_schedule"), file=file)

        nb_schedule = 0

        if (len(scheduledF.schedules)<nb_schedule_per_function): #remove function that have low number of schedules
            continue
        if nb_schedule_per_function > 0:
            schedules_list = sample(scheduledF.schedules, nb_schedule_per_function) 


        for schedule in schedules_list:
    #        print(nb_schedule)
            scheduledF.update_function(i)
            scheduledF.f.apply_schedule(schedule)
            if not schedule['repeated']:
                schedule_string = '_schedule_' + '{:04d}'.format(nb_schedule)  # use fixed size schedule id
                nb_schedule = nb_schedule + 1
                function_schedule_dir = path.join(function_dir, 'function' + str(i) + schedule_string)
                mkdir(function_schedule_dir)
                file = open(path.join(function_schedule_dir, 'function' + str(i) + schedule_string + ".cpp"), 'w')
                print(scheduledF.f.get_program_cpp(schedule_string,schedule['comp_ordering']), file=file)
                j = 0
    #            file = open(path.join(function_schedule_dir, 'function' + str(i) + schedule_string + ".json"), 'w')
    #            print(schedule, file=file)
                file = open(path.join(function_schedule_dir, 'function' + str(i) + schedule_string + '_wrapper' + ".cpp"), 'w')
                print(scheduledF.f.get_wrapper_cpp(schedule_string), file=file)
                file = open(path.join(function_schedule_dir, 'function' + str(i) + schedule_string + '_wrapper' + ".h"), 'w')
                print(scheduledF.f.get_wrapper_h(schedule_string), file=file)


#     n = reduce(lambda a, b: a + b, schedule_numbers)
    print("Done")
#     print("Number of data-points : ", n)
#     print("Number of functions : ", len(schedule_numbers))
#     print("Max schedule per function : ", max(schedule_numbers))
#     print("Min schedule per function : ", min(schedule_numbers))
#     print("Average schedules : ", int(n/len(schedule_numbers)-1))
#     print(schedule_numbers.index(max(schedule_numbers)))
