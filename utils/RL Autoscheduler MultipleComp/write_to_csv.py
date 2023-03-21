import os, csv, json

def write_results_to_csv(testing_dataset_file, csv_file_path):

    f=open(testing_dataset_file,'r')
    progs_dict= json.load(f)

    with open(csv_file_path, 'a+', encoding='UTF8', newline='') as file:
        writer = csv.writer(file)

        if os.stat(csv_file_path).st_size == 0:#if file is empty then write header
            header = ['function name', 'search time', 'initial exec time', 'best schedule', 'speedup']
            writer.writerow(header)

        for func in progs_dict.keys():
            #print('function: ',func)
            initial_exec_time=progs_dict[func]['initial_execution_time']
            if 'schedules_list' in progs_dict[func]:
                comp_name=list(progs_dict[func]['schedules_list'][0].keys())[0]
                schedule_str=progs_dict[func]['schedules_list'][0][comp_name]['schedule_str']
                new_exec_time=progs_dict[func]['schedules_list'][0][comp_name]['execution_times'][0]
                if initial_exec_time >=  new_exec_time:                
                    speedup = initial_exec_time / new_exec_time
                else:
                    speedup = - new_exec_time / initial_exec_time

                search_time=progs_dict[func]['schedules_list'][0][comp_name]['search_time']

                data=[func,search_time, initial_exec_time, schedule_str,speedup]
            else:
                data=[func, 0, initial_exec_time, 0, 0]

            writer.writerow(data)

if __name__ == "__main__":
    write_results_to_csv("../test_batch10_checkpoint60.json", "../new_datastet_batch10_csv_checkpoint60.csv")
