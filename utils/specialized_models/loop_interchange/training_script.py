from os import environ

#training and data storing devices: 'cpu' or 'cuda:X'
environ['train_device'] = 'cuda:6'
environ['store_device'] = 'cuda:6'

# import and define some utils functions
# os.system("python utils.py")  
from utils import *

#initializing dataset files
dataset_file_MC = './datasets/train_set_new_representation.pkl'
test_dataset_file_MC = './datasets/test_set_new_representation.pkl'
benchmark_dataset_file ='./datasets/benchmarks_ds3.json' 
train_val_dataset_file_SC = './datasets/train_set_old_representation.pkl'
test_dataset_file_SC = './datasets/test_set_old_representation.pkl'

#Loading the train/val dataset
print("Starting loading data ...")
train_val_dataset, val_bl, val_indices, train_bl, train_indices = load_merge_data(dataset_file_MC, train_val_dataset_file_SC, 0.2, max_batch_size=128, filter_func_MC=filter_schedule_MC, filter_func_SC=filter_schedule_SC)

print()
#verification of the data loading
print("Train set contains", len(train_bl),"batch.")
print("Val set contains", len(val_bl),"batch.")

#Initializing the model
print("Starting Training model ...")
input_size = 970 #can be confirmed through: train_val_dataset.X[0][1].size(2) ##To confirm
model = Model_Recursive_LSTM_v2(input_size,comp_embed_layer_sizes=[600, 900, 600, 400, 200], drops=[0.275, 0.4, 0.275, 0.175, 0.175], output_size=one_output * k)
model.to(train_device) 

#training modules
criterion = nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(),weight_decay=0.375e-2) 
bl_dict={'train':train_bl, 'val':val_bl}

#Logging file (to track training)
log_file = 'log.txt'
NB_EPOCHS = 500

losses, best_model = train_model(model, criterion, optimizer , max_lr=0.001, dataloader=bl_dict,
                                 num_epochs=NB_EPOCHS, logFile=log_file, log_every=100)

print()
print()

#getting results on the training and the validation sets
train_df = get_results_df(train_val_dataset, train_bl, train_indices, model)
val_df = get_results_df(train_val_dataset, val_bl, val_indices, model)

print()
print()

train_accs = accuracy(train_df)
val_accs = accuracy(val_df)

print("On the training set:")
print("\tOne-Shot accuracy is: ", train_accs[0])
print("\t2-Shot accuracy is: ", train_accs[1])
print("\t5-Shot accuracy is: ", train_accs[-1])


print("On the val set:")
print("\tOne-Shot accuracy is: ", val_accs[0])
print("\t2-Shot accuracy is: ", val_accs[1])
print("\t5-Shot accuracy is: ", val_accs[-1])

#save the best model
torch.save(best_model.state_dict(), 'loop_interchange_model.pkl')
print("Model saved.")

print("Script completed successfully")