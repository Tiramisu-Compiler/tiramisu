from os import listdir
import subprocess
import fire
import matplotlib.pyplot as plt
import random
import pandas as  pd 
import re
import json 
import numpy as np
from functools import reduce
from tqdm import tqdm

from src.data.loop_ast import *
from src.data.schedule import *

class Stats():
    def __init__(self, data_path, programs_folder="programs", stats_folder="programs"):

        #self.tiramisu_root = tiramisu_root +'/'
        self.data_path = data_path+'/'
        self.programs_folder = programs_folder + '/'
        self.stats_folder = stats_folder + '/'
        self.stats_folder_created = ('stats' in listdir(self.data_path))

        self.computation_types_dict = {
                                        0: "arithmetic exp",
                                        1: "inputs",
                                        2: "stencil"
                                      }

    def get_all_programs_schedules(self):

        for program in self.get_all_programs():    
            for schedule in self.get_all_schedules(program):
                yield (program, schedule)

    def get_all_programs(self):
        folder = self.stats_folder if self.stats_folder_created else self.programs_folder

        programs = listdir(self.data_path + folder)

        for program in programs:
            yield program

    def get_all_schedules(self, program):
        folder = self.stats_folder if self.stats_folder_created else self.programs_folder

        schedules = listdir(self.data_path + folder +'/'+ program)
        schedules = filter(lambda x:not x.endswith('.json'), schedules)
        
        for schedule in schedules:
            yield schedule
            


    def create_stats_dir(self):
        full_path = self.data_path + self.stats_folder + '/'

        for program,schedule in self.get_all_programs_schedules(): 
            self.exec_command("mkdir -p " + full_path + program + "/" + schedule)

        self.stats_folder_created = True

    def copy_results(self):
        
        for program,schedule in self.get_all_programs_schedules():
            program_json = self.data_path + self.programs_folder + "/" + program + "/" + program + '.json'
            schedule_json = self.data_path + self.programs_folder + "/" + program \
                            + "/" + schedule + '/' + schedule + '.json'

            exec_times = self.data_path + self.programs_folder + "/" + program \
                            + "/" + schedule + '/' + 'exec_times.txt'


            self.exec_command("cp " + program_json + ' ' \
                    + self.data_path + self.stats_folder + "/" + program + "/" + program + '.json')

            self.exec_command("cp " + schedule_json + ' ' \
                    + self.data_path + self.stats_folder + "/" + program \
                            +"/" + schedule + '/' + schedule + '.json')

            try:
                self.exec_command("cp " + exec_times + ' ' \
                        + self.data_path + self.stats_folder + "/" + program \
                                + "/" + schedule + '/exec_times.txt')
            except FileNotFoundError:
                print("didnt find exec times of : " + program + '/' + schedule)
                self.exec_command("touch " +self.data_path + self.stats_folder \
                                    + '/' + program + '/' + schedule + '/exec_times.txt')

    def read_times(self):
        full_path = self.data_path + self.stats_folder + '/'

        results = {}
       
        for program,schedule in tqdm(self.get_all_programs_schedules(), total=1277934):
            exec_times = np.array(self.read_exec_times(program, schedule), dtype='float64')

            results[(program, schedule)] = {
                                                'Min': min(exec_times),
                                                'Max': max(exec_times),
                                                'Mean': np.mean(exec_times),
                                                'Median': np.median(exec_times),
                                                'Std': np.std(exec_times),
                                                'N_samples':len(exec_times),
                                                'Times':exec_times

                                            }

        keys, vals = list(zip(*results.items()))

        index = pd.MultiIndex.from_tuples(keys, names=("program", "schedule"))
        

        
        return pd.DataFrame(list(vals), index=index)
        

        

        


    def read_results(self):
        results = {}
        full_path = self.data_path + self.stats_folder + '/'

        for program,schedule in self.get_all_programs_schedules():
            
            #Getting representations of the program and the schedule
            program_json =  self.read_program_json(program)
            schedule_json = self.read_schedule_json(program, schedule)

            exec_time = self.read_exec_time(program, schedule)

            
            type_program = self.type_of_program(program_json)
            type_schedule = self.type_of_schedule(schedule_json)
            comp_size = self.computation_size(program_json)

            interchange_params = self.interchange_params(schedule_json)
            tiling_params = self.tiling_params(schedule_json)
            unrolling_params = self.unrolling_params(schedule_json)

            results[program, schedule] = {
                                            "exec_time": exec_time,
                                            "program_type": type_program,
                                            "schedule_type": type_schedule,
                                            "comp_size": comp_size,
                                            "interchange": interchange_params,
                                            "tiling":tiling_params,
                                            "unrolling":unrolling_params
                                         }
        
        keys, vals = list(zip(*results.items()))

        index = pd.MultiIndex.from_tuples(keys, names=("program", "schedule"))
        

        order_of_cols = ['exec_time', 'no_schedule', 'speedup', 
                        'program_type', 'schedule_type', 
                        'interchange', 'tiling', 'unrolling']

        self.results = pd.DataFrame(list(vals), index=index)
        
        self.results = self.calculate_stats()[order_of_cols]

        return self.results

    def computation_size(self, program_repr):
        loops = program_repr['loops']['loops_array']

        iterators = [loop['loop_it'] for loop in  loops]

        iterators = [it for it in program_repr['iterators']['iterators_array'] if it['it_id'] in iterators]

        iterators = map(lambda x: x['upper_bound'] - x['lower_bound'], iterators)

        return reduce(lambda x, y: x*y, iterators)



    def type_of_program(self, program_representation):
        return self.computation_types_dict[program_representation['type']]
    
    def type_of_schedule(self, schedule_representation):
        
        interchange = int((len(schedule_representation['interchange_dims']) > 0))

        tiling = int((schedule_representation['tiling'] is not None))

        unrolling = int((schedule_representation['unrolling_factor'] is not None))
        
        return str((interchange, tiling, unrolling))

    def read_program_json(self, program):

        full_path = self.data_path + self.stats_folder + '/' + program + '/' + program + '.json'

        try:
            json_dict = json.load(open(full_path, 'r'))
        except Exception:
            print(program)
            exit(1)

        return json_dict

    def read_schedule_json(self, program, schedule):

        full_path = self.data_path + self.stats_folder + '/' + program + '/' + schedule \
                    + '/' + schedule + '.json'

        json_dict = json.load(open(full_path, 'r'))

        return json_dict


    # def patch(self):
    #     full_path = self.data_path + self.programs_folder

    #     for program,schedule in self.get_all_programs_schedules():
    #         if not self.check(program, schedule):
    #             self.exec_command("rm -rf " + full_path + program +'/' +schedule +'/')




    # def check(self, program, schedule):

    #     full_path = self.data_path + self.programs_folder + '/' + program + '/' + schedule \
    #                 + '/' + schedule + '.json'

    #     json_dict = json.load(open(full_path, 'r'))

    #     if 0 in json_dict['interchange_dims']:
    #         if not json_dict['tiling']:
    #             return True

    #         if json_dict['tiling']['tiling_dims'][0] != json_dict['interchange_dims'][1]:
    #             return True 


    #     return False


    
    def calculate_stats(self):

        df = self.results

        df['no_schedule'] = 0.0

        for program in df.index.levels[0]:
            #get no_schedule exec time
            no_sched = float(df.loc[program] [df.loc[program].index.str.endswith('no_schedule')].exec_time)
            df.loc[program, 'no_schedule'] = no_sched
           # df.loc[program, 'speedup'] = (df.loc[program, 'exec_time'] / no_sched).values 

        df['speedup'] = df.no_schedule / df.exec_time

        df = df.sort_values(by=["program", "speedup"], ascending=[True, False])

        self.results = df

        return df

    def unrolling_params(self, schedule_repr):
        unrolling = schedule_repr['unrolling_factor'] 
        result = None

        if unrolling is not None:
            result = str(unrolling)
        
        return result

    def interchange_params(self, schedule_repr):
        interchange = schedule_repr['interchange_dims'] 
        result = None

        if len(interchange) > 0:
           
            result = str(tuple(interchange))
        
        return result

    def tiling_params(self, schedule_repr):
        tiling = schedule_repr['tiling'] 
        result = None

        if tiling is not None:
            dims = tiling['tiling_dims']
            factors = tiling['tiling_factors']

            result = str(dict(zip(dims, factors)))
        
        return result

    def show_random_func(self):

        func = random.choice(list(self.results.values()))
        x, y = zip(*func.items())

        index = [i for i in range(len(x)) if x[i].endswith("no_schedule")][0]

        bar_list = plt.bar(x, y)
        bar_list[index].set_color('r')
        plt.xticks(x, rotation=20)
        plt.ylabel("Execution time (ms)")
        plt.show()

    def exec_command(self, command):  
        ret_code = subprocess.call(command.split())
        
        if ret_code == 1:
            print(command)
            exit(1)
        if ret_code != 0:
            print(f"Return code {ret_code}")
            print(f"Error executing command {command} \n")

    def program_to_ast(self, program):
        #read json in dictionary
        program_dict = self.read_program_json(program)

        #transform to ast
        program = Loop_AST(program, program_dict)

        return program

    def schedule_repr(self, program, schedule):
        #read json in dict
        schedule_dict = self.read_schedule_json(program, schedule)

        #get representation
        schedule = Schedule(schedule, schedule_dict)

        return schedule

    def read_exec_time(self, program, schedule):
        full_path = self.data_path + self.stats_folder + '/'

        exec_time = np.NaN
        #Getting the execution time of the schedule
        with open(full_path + '/'+program + '/' + schedule +'/exec_times.txt', 'r') as f:

            exec_times = f.readlines()
            
            if len(exec_times) > 0:
                r = re.compile(r"[0-9]+(.[0-9]+)?(e\+[0-9]+)?") 
                
                exec_times = [r.match(value) for value in exec_times]

                exec_times = np.array([val.group(0) for val in exec_times
                                if val is not None], dtype='float64')
                
                exec_time = np.median(exec_times)/1000
                
            f.close()

        return exec_time

    def read_exec_times(self, program, schedule):
        full_path = self.data_path + self.stats_folder + '/'

        exec_times = []
        #Getting the execution time of the schedule
        with open(full_path + '/'+program + '/' + schedule +'/exec_times.txt', 'r') as f:

            exec_times = f.readlines()
            r = re.compile(r"[0-9]+(.[0-9]+)?(e\+[0-9]+)?")
            exec_times = [r.match(value) for value in exec_times]
            exec_times = np.array([val.group(0) for val in exec_times
                                if val is not None], dtype='float64')
            
            f.close()

        return exec_times


    def load_data(self):
        '''
            Returns (programs, schedules, times)
        '''

        progs_array = []
        schedules_array = []
        exec_times_array = []

        programs = sorted(self.get_all_programs())
        for program in programs:

            prog_ast = self.program_to_ast(program)
            progs_array.append(prog_ast)

            progs_schedules = []
            progs_times = []

            schedules = sorted(list(self.get_all_schedules(program)))

            for schedule in schedules:
                #get schedule representation
                schedule_repr = self.schedule_repr(program, schedule)
                progs_schedules.append(schedule_repr)

                t = self.read_exec_time(program, schedule)
                progs_times.append(t)

            
            schedules_array.append(progs_schedules)
            exec_times_array.append(progs_times)

        
        return (progs_array, schedules_array, exec_times_array)



    


if __name__=="__main__":

    fire.Fire(Stats)


    
