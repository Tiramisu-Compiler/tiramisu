import torch.utils.data as data 
import torch
import h5py 
from bisect import bisect_right
import numpy as np
from src.data import load_data
import src.data.stats as stats
import dill
from tqdm import tqdm

class DatasetFromHdf5(data.Dataset):

    def __init__(self, filename, normalized=True, log=False, maxsize=30000):
        super().__init__()

        self.maxsize = maxsize
        self.f = h5py.File(filename, mode='r', swmr=True)
        
        self.schedules = self.f.get('schedules')
        self.programs = self.f.get('programs')
        self.speedups = self.f.get('speedup')
        self.times = self.f.get('times')
        self.prog_names = self.f.get('programs_names')
        self.sched_names = self.f.get('schedules_names')

        self.X = np.concatenate((np.array(self.programs), np.array(self.schedules)), axis=1).astype('float32')
        self.Y = np.array(self.speedups, dtype='float32').reshape(-1, 1)

        if log:
            self.Y = np.log(self.Y)
            self.mean = np.mean(self.Y)
            self.std = np.std(self.Y)

            self.Y = (self.Y - self.mean)/self.std
        
        
    def __len__(self):
        if self.maxsize is None:
            return len(self.Y)

        return self.maxsize
    

    def __getitem__(self, index):
        return self.X[index], self.Y[index]

    def get_prog_name(self, index):
        return self.prog_names[index]
    
    def get_sched_name(self, index):
        return self.sched_names[index]

    def normalize_min_max(self, data):
        data = np.array(data)

        denominator = data.max(axis=0) - data.min(axis=0) 
        denominator[denominator == 0] = 1

        data = (data - data.min(axis=0))/denominator

        return data

    def normalize_dataset(self):
        #reopen file in write mode
        filename = self.f.filename
        self.f.close()
        self.f = h5py.File(filename, mode='a')

        self.programs = self.f.get('programs')
        self.schedules = self.f.get('schedules')
        #normalize programs 
        normalized_progs = self.normalize_min_max(self.programs)
        self.f.create_dataset('normalized_programs', data=normalized_progs, dtype="float32")
        #normalize schedules
        normalized_scheds = self.normalize_min_max(self.schedules)
        self.f.create_dataset('normalized_schedules', data=normalized_scheds, dtype="float32")

        #go back to read mode
        self.f.close()
        self.__init__(filename)

class DatasetFromPkl(data.Dataset):
    def __init__(self, filename, normalized=False, log=False, maxsize=100000):
        super().__init__()

        self.maxsize = maxsize
        self.dataset = filename
        
        #read dataset
        f = open(filename, 'rb')
        dataset_dict = dill.load(f)
        f.close()

        self.programs = dataset_dict['programs']
        self.program_indexes = dataset_dict['program_indexes']
        self.schedules = dataset_dict['schedules']
        self.exec_times = dataset_dict['exec_times']
        self.speedups = dataset_dict['speedup']

        
        self.X = []
        self.Y = []
        self.restricted_program_indexes = []
        self.restricted_schedules = []
        for i in tqdm(range(len(self.schedules))):      
            program = self.programs[self.program_indexes[i]]
            self.X.append(program.add_schedule(self.schedules[i]).__array__())
            self.Y.append(self.speedups[i])
            self.restricted_program_indexes.append(self.program_indexes[i])
            self.restricted_schedules.append(self.schedules[i])
        self.X = np.array(self.X).astype('float32')
        self.Y = np.array(self.Y, dtype='float32').reshape(-1, 1)
        self.Y_speedups = np.array(self.Y, dtype='float32')



        if log:
            self.Y = np.log(self.Y)
            
            self.mean = np.mean(self.Y)
            self.std = np.std(self.Y)

            self.Y = (self.Y - self.mean)/self.std

        
    def __getitem__(self, index):
        return self.X[index], self.Y[index] 

    def __len__(self):
        if self.maxsize is None:
            return len(self.Y)

        return self.maxsize

    @staticmethod
    def pickle_data(data_path='data/training_data/', dataset_path='data/speedup_dataset.pkl'):
        st = stats.Stats(data_path)

        print("Reading data")
        programs, schedules, exec_times = st.load_data()
        print("data loaded")
        print("Serializing")
        load_data.serialize(programs, schedules, exec_times, filename=dataset_path)
        print("done")
        

#loads data using filter on speedup and excluding some functions        
class DatasetFromPkl_Filter(data.Dataset):
    def __init__(self, filename, normalized=False, log=False, maxsize=100000, speedup_lo_bound=0, speedup_up_bound=np.inf, exlude_funcs=[]):
        super().__init__()

        self.maxsize = maxsize
        self.dataset = filename
        
        #read dataset
        f = open(filename, 'rb')
        dataset_dict = dill.load(f)
        f.close()

        self.programs = dataset_dict['programs']
        self.program_indexes = dataset_dict['program_indexes']
        self.schedules = dataset_dict['schedules']
        self.exec_times = dataset_dict['exec_times']
        self.speedups = dataset_dict['speedup']

        
        self.X = []
        self.Y = []
        self.restricted_program_indexes = []
        self.restricted_schedules = []
        for i in tqdm(range(len(self.schedules))):
            if((self.speedups[i]>=speedup_lo_bound) & (self.speedups[i]<=speedup_up_bound) & (self.programs[self.program_indexes[i]].name not in exlude_funcs)):
                program = self.programs[self.program_indexes[i]]
                self.X.append(program.add_schedule(self.schedules[i]).__array__())
                self.Y.append(self.speedups[i])
                self.restricted_program_indexes.append(self.program_indexes[i])
                self.restricted_schedules.append(self.schedules[i])
            else:
                pass
        self.X = np.array(self.X).astype('float32')
        self.Y = np.array(self.Y, dtype='float32').reshape(-1, 1)
        self.Y_speedups = np.array(self.Y, dtype='float32')



        if log:
            self.Y = np.log(self.Y)
            
            self.mean = np.mean(self.Y)
            self.std = np.std(self.Y)

            self.Y = (self.Y - self.mean)/self.std

        
    def __getitem__(self, index):
        return self.X[index], self.Y[index] 

    def __len__(self):
        if self.maxsize is None:
            return len(self.Y)

        return self.maxsize
    
#loads data using filter function and transformation on speedups      
class DatasetFromPkl_Transform(data.Dataset):
    def __init__(self, filename, normalized=False, log=False, maxsize=100000, filter_func=None, transform_func=None):
        super().__init__()

        self.maxsize = maxsize
        self.dataset = filename
        
        #read dataset
        f = open(filename, 'rb')
        dataset_dict = dill.load(f)
        f.close()

        self.programs = dataset_dict['programs']
        self.program_indexes = dataset_dict['program_indexes']
        self.schedules = dataset_dict['schedules']
        self.exec_times = dataset_dict['exec_times']
        self.speedups = dataset_dict['speedup']

        
        self.X = []
        self.Y = []
        self.restricted_program_indexes = []
        self.restricted_schedules = []
        
        if (filter_func==None):
            filter_func = lambda x : True
        if (transform_func==None):
            transform_func = lambda x : x
            
        for i in tqdm(range(len(self.schedules))):
            if(filter_func(self)):
                program = self.programs[self.program_indexes[i]]
                self.X.append(program.add_schedule(self.schedules[i]).__array__())
                self.Y.append(self.speedups[i])
                self.restricted_program_indexes.append(self.program_indexes[i])
                self.restricted_schedules.append(self.schedules[i])
            else:
                pass
        self.X = np.array(self.X).astype('float32')
        self.Y = np.array(self.Y, dtype='float32').reshape(-1, 1)
        self.Y_speedups = np.array(self.Y, dtype='float32')
        self.Y = transform_func(self.Y)
        self.Y = np.array(self.Y, dtype='float32')


        if log:
            self.Y = np.log(self.Y)
            
            self.mean = np.mean(self.Y)
            self.std = np.std(self.Y)

            self.Y = (self.Y - self.mean)/self.std

        
    def __getitem__(self, index):
        return self.X[index], self.Y[index] 

    def __len__(self):
        if self.maxsize is None:
            return len(self.Y)

        return self.maxsize

       
class DatasetFromPkl_old(data.Dataset):
    def __init__(self, filename, normalized=False, log=False, maxsize=100000):
        super().__init__()

        self.maxsize = maxsize
        self.dataset = filename
        
        #read dataset
        f = open(filename, 'rb')
        dataset_dict = dill.load(f)
        f.close()

        self.programs = dataset_dict['programs']
        self.program_indexes = dataset_dict['program_indexes']
        self.schedules = dataset_dict['schedules']
        self.exec_times = dataset_dict['exec_times']
        self.speedups = dataset_dict['speedup']

        programs = [program.__array__() for program in self.programs]
        schedules = [schedule.__array__() for schedule in self.schedules]
       
        self.X = np.concatenate((np.array(programs)[self.program_indexes], np.array(schedules)), axis=1).astype('float32')
        self.Y = np.array(self.speedups, dtype='float32').reshape(-1, 1)

        if log:
            self.Y = np.log(self.Y)
            self.mean = np.mean(self.Y)
            self.std = np.std(self.Y)

            self.Y = (self.Y - self.mean)/self.std

        
    def __getitem__(self, index):
        return self.X[index], self.Y[index] 

    def __len__(self):
        if self.maxsize is None:
            return len(self.Y)

        return self.maxsize

   



    @staticmethod
    def pickle_data(data_path='data/training_data/', dataset_path='data/speedup_dataset.pkl'):
        st = stats.Stats(data_path)

        print("Reading data")
        programs, schedules, exec_times = st.load_data()
        print("data loaded")
        print("Serializing")
        load_data.serialize(programs, schedules, exec_times, filename=dataset_path)
        print("done")

