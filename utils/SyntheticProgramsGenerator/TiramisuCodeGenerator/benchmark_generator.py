
import json
from functools import reduce
from os import path, mkdir, makedirs
from random import randint
from shutil import rmtree
from schedule import *
from tqdm import tqdm

BENCHMARKS = [
              "_matmul",
              "_cvtcolor", 
              "_cvtcolor_orig",
              "_heat3d",
              "_blur",
              "_blur_i", 
              "_doitgen_r", 
              "_gemm",
              "_convolution", 
              "_conv_relu",
              "_atax",
              "_bicg",              
              "_heat2d",
              "_heat2d_noinit",
              "_jacobi2d_r",
              "_jacobi1d_r", 
              "_seidel2d",
              "_gemver",
              "_mvt", 
              ]
# TODO: "_heat3d" not shared iterators
# TODO: variables stat always from 0!
# TODO:
batchName = 'Benchmark_batch10'
SAMPLE_DIR = './' +batchName +'/programs'

if path.exists(SAMPLE_DIR):
    rmtree(SAMPLE_DIR)

makedirs(SAMPLE_DIR)
schedule_numbers = []

frequent_sizes = dict()
for nb_dims in range(2,8):
    frequent_sizes[nb_dims]=[]
    with open('most_frequent_'+str(nb_dims)+'d_dims.txt','r') as f:
        for line in f:
            sizes_list = [int(i) for i in line.split()]
            assert len(sizes_list)==nb_dims
            frequent_sizes[nb_dims].append(sizes_list)



for i in tqdm(BENCHMARKS):
    benchmark_name = i
    if i in ["_matmul","_cvtcolor","_cvtcolor_orig","_heat3d","_blur","_blur_i","_doitgen_r","_gemm"]: 
        dim_extents = frequent_sizes[3][:256]
    elif i in ["_atax","_bicg","_heat2d","_heat2d_noinit","_jacobi2d_r"]:
        dim_extents = frequent_sizes[2][:256]
    elif i in ['_convolution','_conv_relu']:
        dim_extents = frequent_sizes[7][:256]
    elif i in ["_jacobi1d_r","_seidel2d","_gemver","_mvt"]:
        dim_extents = [size[0] for size in frequent_sizes[2][:256]]
        dim_extents = [[s] for s in set(dim_extents)]
    for sizes in dim_extents:
        i = benchmark_name+'_'+'x'.join([str(s) for s in sizes])
        scheduledF = ScheduledFunction(i, batch_name=batchName, benchmark=True, sizes=sizes)
        scheduledF.f.apply_schedule(None)
        schedule_numbers.append(len(scheduledF.schedules))
        function_dir = path.join(SAMPLE_DIR, 'function' + str(i))
        mkdir(function_dir)
        # j = json.dumps(scheduledF.f.get_representation(), indent=4)
        # file = open(path.join(function_dir, 'function' + str(i) + ".json"), 'w')
        # print(j, file=file)

        function_schedule_dir = path.join(function_dir, 'function' + str(i) + '_no_schedule')
        mkdir(function_schedule_dir)
        file = open(path.join(function_schedule_dir, 'function' + str(i) + '_no_schedule' + ".cpp"), 'w')
        print(scheduledF.f.get_program_cpp("_no_schedule", scheduledF.f.comps_ordering), file=file)
        d = {"interchange_dims": None, "tiling": None, "unrolling_factor": None}
    #     j = json.dumps(d, indent=4)
    #     file = open(path.join(function_schedule_dir, 'function' + str(i) + '_no_schedule' + ".json"), 'w')
    #     print(j, file=file)
        file = open(path.join(function_schedule_dir, 'function' + str(i) + '_no_schedule_wrapper' + ".cpp"), 'w')
        print(scheduledF.f.get_wrapper_cpp("_no_schedule"), file=file)
        file = open(path.join(function_schedule_dir, 'function' + str(i) + '_no_schedule_wrapper' + ".h"), 'w')
        print(scheduledF.f.get_wrapper_h("_no_schedule"), file=file)

        nb_schedule = 0
        for schedule in scheduledF.schedules:
    #         print(nb_schedule)
            scheduledF.update_function(i)
            scheduledF.f.apply_schedule(schedule)
            if not schedule['repeated']:
                schedule_string = '_schedule_' + '{:04d}'.format(nb_schedule)  # use fixed size schedule id
                nb_schedule = nb_schedule + 1
                function_schedule_dir = path.join(function_dir, 'function' + str(i) + schedule_string)
                mkdir(function_schedule_dir)
                file = open(path.join(function_schedule_dir, 'function' + str(i) + schedule_string + ".cpp"), 'w')
                print(scheduledF.f.get_program_cpp(schedule_string, schedule['comp_ordering']), file=file)
    #             j = 0
    #             file = open(path.join(function_schedule_dir, 'function' + str(i) + schedule_string + ".json"), 'w')
    #             print(schedule, file=file)
                file = open(path.join(function_schedule_dir, 'function' + str(i) + schedule_string + '_wrapper' + ".cpp"), 'w')
                print(scheduledF.f.get_wrapper_cpp(schedule_string), file=file)
                file = open(path.join(function_schedule_dir, 'function' + str(i) + schedule_string + '_wrapper' + ".h"), 'w')
                print(scheduledF.f.get_wrapper_h(schedule_string), file=file)


n = reduce(lambda a, b: a + b, schedule_numbers)
print("Number of data-points : ", n)
print("Number of functions : ", len(schedule_numbers))
print("Max schedule per function : ", max(schedule_numbers))
print("Min schedule per function : ", min(schedule_numbers))
print("Average schedules : ", int( n /len(schedule_numbers ) -1))
print(schedule_numbers.index(max(schedule_numbers)))

for i, n in zip(BENCHMARKS, schedule_numbers):
    print(i[1:], " : ", n)
