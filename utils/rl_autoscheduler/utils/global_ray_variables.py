from typing import List
import ray
import json
import os


@ray.remote
class GlobalVarActor:

    def __init__(self, programs_file, dataset_path, num_workers=7):
        self.index = -1
        self.num_workers = num_workers
        self.progs_list = self.get_dataset(dataset_path)
        self.programs_file = programs_file
        self.progs_dict = dict()
        self.lc_data = []
        if os.path.isfile(programs_file):
            try:
                with open(programs_file) as f:
                    self.progs_dict = json.load(f)
            except:
                self.progs_dict = dict()
        else:
            self.progs_dict = dict()
            with open(programs_file,"w+") as f:
                f.write(json.dumps(self.progs_dict))

        if os.path.isfile("lc_data.json"):
            try:
                with open("lc_data.json") as f:
                    self.lc_data = json.load(f)
            except:
                self.lc_data = []
        else:
            self.lc_data = []
            with open("lc_data.json","w+") as f:
                f.write(json.dumps(self.lc_data))

    def get_dataset(self, path):
        os.getcwd()
        print("***************************", os.getcwd())
        prog_list = os.listdir(path)
        return prog_list

    def set_progs_list(self, v):
        self.progs_list = v
        return True

    def get_progs_list(self, id):
        return [
            item for (index, item) in enumerate(self.progs_list)
            if (index % self.num_workers) == (id % self.num_workers)
        ]

    def update_lc_data(self, v: List):
        self.lc_data.extend(v)
        return True

    def get_lc_data(self) -> List:
        return self.lc_data
    
    def write_lc_data(self):
        print("Saving lc_data to disk")
        with open("lc_data.json", "w") as f:
            json.dump(self.lc_data, f)
        return True

    def update_progs_dict(self, v):
        self.progs_dict.update(v)
        return True

    def write_progs_dict(self):
        print("Saving progs_dict to disk")
        with open(self.programs_file, "w") as f:
            json.dump(self.progs_dict, f)
        return True

    def get_progs_dict(self):
        return self.progs_dict

    def increment(self):
        self.index += 1
        return self.index


@ray.remote
class Actor:

    def __init__(self, data_registry):
        self.data_registry = data_registry

    def get_progs_list(self, id):
        return ray.get(self.data_registry.get_progs_list.remote(id))

    def update_lc_data(self, v: List):
        return ray.get(self.data_registry.update_lc_data.remote(v))

    def get_lc_data(self) -> List:
        return ray.get(self.data_registry.get_lc_data.remote())
    
    def write_lc_data(self):
        return ray.get(self.data_registry.write_lc_data.remote())
    
    def get_progs_dict(self):
        return ray.get(self.data_registry.get_progs_dict.remote())

    def write_progs_dict(self):
        return ray.get(self.data_registry.write_progs_dict.remote())

    def update_progs_dict(self, v):
        return ray.get(self.data_registry.update_progs_dict.remote(v))

    def increment(self):
        return ray.get(self.data_registry.increment.remote())
