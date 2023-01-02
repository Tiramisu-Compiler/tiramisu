from os import environ

#training and data storing devices: 'cpu' or 'cuda:X'
environ['train_device'] = 'cuda:6'
environ['store_device'] = 'cuda:6'

#initializing dataset files
dataset_file_MC = './datasets/train_set_new_representation.pkl'
test_dataset_file_MC = './datasets/test_set_new_representation.pkl'
benchmark_dataset_file ='./datasets/benchmarks_ds3_modified.json'
train_val_dataset_file_SC = './datasets/train_set_old_representation.pkl'
test_dataset_file_SC = './datasets/test_set_old_representation.pkl'

# import and define some utils functions
from utils import *

#Loading the test dataset
print("Starting loading data ...")
test_dataset, test_bl, test_indices, _, _ = load_merge_data(test_dataset_file_MC, test_dataset_file_SC, split_ratio=1, filter_func_MC=filter_schedule_MC, filter_func_SC=filter_schedule_SC)

print()

#Loading the benchmark
bench_ds, bench_bl, bench_indices, _, _ = load_merge_data(None, benchmark_dataset_file, 1, filter_func_MC=filter_schedule_MC, filter_func_SC=filter_schedule_SC)


print()
print()

#Loading the model
input_size = 970
model = Model_Recursive_LSTM_v2(input_size,comp_embed_layer_sizes=[600, 900, 600, 400, 200], drops=[0.275, 0.4, 0.275, 0.175, 0.175], output_size=one_output * k)
model.load_state_dict(torch.load('loop_interchange_model.pkl',map_location=train_device))
model.to(train_device)

######################################## Accuracy Test ########################################
print("Accuracy testing")

#getting accuracy on the test set and the benchmark
test_df = get_results_df(test_dataset, test_bl, test_indices, model)

print()

bench_df = get_results_df(bench_ds, bench_bl, bench_indices, model)

print()


test_accs = accuracy(test_df)
benchmark_accs = accuracy(bench_df)

print("On the test set:")
print("\tOne-Shot accuracy is: ", test_accs[0])
print("\t2-Shot accuracy is: ", test_accs[1])
print("\t5-Shot accuracy is: ", test_accs[-1])

print()

print("On the banchmark:")
print("\tOne-Shot accuracy is: ", benchmark_accs[0])
print("\t2-Shot accuracy is: ", benchmark_accs[1])
print("\t5-Shot accuracy is: ", benchmark_accs[-1])


######################################## Search Performance Test ########################################
print()
print()
print("Search Performance testing")

#loading data for the cost model
bench_ds_CM, bench_bl_CM, bench_indices_CM, _, _ = load_data_CM(benchmark_dataset_file, 
                                             split_ratio = 1,
                                             max_batch_size = 1,
                                             drop_sched_func = drop_schedule, 
                                             drop_prog_func = None,
                                             default_eval = default_eval,
                                             speedups_clip_func = speedup_clip)

print()


#loading cost model
input_size = 776
CM = Model_Recursive_LSTM_v2_CM(input_size,drops=[0.250, 0.250, 0.250, 0.250])
CM.load_state_dict(torch.load('./MAPE_base_13+4+2.6.pkl',map_location=train_device))
CM.to(train_device)

bench_df_CM = get_results_df_CM(bench_ds_CM, bench_bl_CM, bench_indices_CM, CM)

print()


#get the predictions made by the LI model to use them in the Simulation of beam search.
enforced_scheds_df = pd.DataFrame(columns=['name','sched_str', 'depth','priority'])
enf = 0
for ind in bench_df.index:
    name = bench_df["name"][ind]
    LIs = get_LI_ordered(bench_df["prediction"][ind])
    if LIs == []:
        print("--")
    i = 0
    for li in LIs:
        enforced_scheds_df.loc[enf] = [name, li, 1, i]# 1 is the level
        i += 1
        enf += 1
# enforced_scheds_df

print()

print("Presenting the search performance results as schedules...")
#return the results of beam search, with the speedups of the final schedules
pd.set_option('display.max_rows',100)
df = simulate_BeamSearch_on_Dataset(bench_ds_CM,bench_df_CM, enforced_scheds_df, true_beam_search=False, get='schedules')
print(df.iloc[:, :7])
df.to_csv(r'search_perfs_schedules.csv')


print()
print()



print("Presenting the search performance results as speedups...")
#return the results of beam search with its final schedules
pd.set_option('display.max_rows',100)
df = simulate_BeamSearch_on_Dataset(bench_ds_CM,bench_df_CM, enforced_scheds_df, true_beam_search=False, get='speedups')
print(df.iloc[:, :7])
df.to_csv(r'search_perfs_speedups.csv')

print()
print()


print("Presenting the search performance statistics compared to the perfect auto-scheduler...")
#statistics of the search performance
search_perf_df = get_search_performance(bench_ds_CM,bench_df_CM, enforced_scheds_df,true_beam_search=False, tira = False) 
search_perf = np.mean([min(100,i) for i in search_perf_df['bs=3']])
print("\tWith beam size = 1:", np.mean(search_perf_df['bs=1']))
print("\tWith beam size = 2:", np.mean(search_perf_df['bs=2']))
print("\tWith beam size = 3:", search_perf)
print("\tWith beam size = 5:", np.mean(search_perf_df['bs=4']))

print()
print()

print("Script completed successfully")