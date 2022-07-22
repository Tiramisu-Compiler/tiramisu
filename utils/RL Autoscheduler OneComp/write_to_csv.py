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
            best_speedup = 0
            best_schedule_str=""
            best_search_time=0
            #print('function: ',func)
            initial_exec_time=progs_dict[func]['initial_execution_time']
            if "schedules_list" in progs_dict[func]:
                comp_name=list(progs_dict[func]['schedules_list'][0].keys())[0]
                for i in range(len(progs_dict[func]['schedules_list'])):
                    schedule_str=progs_dict[func]['schedules_list'][i][comp_name]['schedule_str']
                    new_exec_time=progs_dict[func]['schedules_list'][i][comp_name]['execution_times'][0]                
                    speedup = initial_exec_time / new_exec_time
                    #search_time=progs_dict[func]['schedules_list'][0][comp_name]['search_time']
                    if speedup > best_speedup:
                        best_schedule_str=schedule_str
                        best_speedup =speedup
                        #best_search_time= search_time
    

                

                data=[func,best_search_time, initial_exec_time, best_schedule_str,best_speedup]
            else:
                data=[func, 0, initial_exec_time, 0, 1]

            writer.writerow(data)

if __name__ == "__main__":
    write_results_to_csv("../benchmarks_no_full_fetch_enhanced_hyper_checkpoint45.json", "../benchmarks_no_full_fetch_search_time_enhanced_hyper_checkpoint45.csv")
    #write_results_to_csv("../test_enhanced_hyper_true_checkpoint45.json", "../test_enhanced_hyper_true_checkpoint45.csv")

