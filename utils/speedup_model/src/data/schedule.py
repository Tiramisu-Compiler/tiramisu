import numpy as np

class Schedule():

    def __init__(self, name, dict_repr):
        self.name = name 
        self.type_dict = {
                            'interchange':0,
                            'tiling':1,
                            'unrolling':2
                        }
        self.binary_repr = None
        self.load_schedules(dict_repr)


    def add_interchange(self, interchange):
        if interchange:
            self.schedule_list.append({
                                        'type':'interchange',
                                        'params':interchange,
                                        'factors': None
                                    })
        else:
            self.schedule_list.append({
                                        'type':'interchange',
                                        'params':[-1, -1],
                                        'factors':None
                                    })

    def add_tiling(self, tiling):
        if tiling:
            dims = tiling['tiling_dims']
            factors = tiling['tiling_factors']

            # if tiling['tiling_depth'] == 2:
            #     dims.append(-1)
            #     factors.append(-1)

            self.schedule_list.append({
                                    'type':'tiling',
                                    'params':dims,
                                    'factors': tiling['tiling_factors']
                                })
        else:
            self.schedule_list.append({
                                    'type':'tiling',
                                    'params':[-1, -1, -1],
                                    'factors': [-1, -1, -1]
                                })


    def add_unrolling(self, unrolling):
        if unrolling:
            self.schedule_list.append({
                                    'type':'unrolling',
                                    'params': None,
                                    'factors': [unrolling]
                                })
        else:
            self.schedule_list.append({
                                    'type':'unrolling',
                                    'params': None,
                                    'factors': [1]
                                })



    def load_schedules(self, dict_repr):
        self.schedule_list = []
        
        interchange = dict_repr['interchange_dims']
        self.add_interchange(interchange)

        
        tiling = dict_repr['tiling']
        self.add_tiling(tiling)
        
        unrolling_factor = dict_repr['unrolling_factor']
        self.add_unrolling(unrolling_factor)

        self.binary_repr = (+(len(interchange) > 0), +(tiling is not None), +(unrolling_factor is not None))

    def __eq__(self, other, binary=True):

        if self.binary_repr == other.binary_repr:
            if not binary:
                return self.schedule_list == other.schedule_list

            return True
        return False
    def __array__(self):

        arr = []
        #sort by type
        self.schedule_list.sort(key=lambda x: self.type_dict[x['type']])


        for schedule in self.schedule_list:
            type_ = self.type_dict[schedule['type']]
            params = schedule['params']
            factors = schedule['factors']

            arr.append(type_)
            if params:
                arr.extend(params)
            if factors:
                arr.extend(factors)

        return np.array(arr)

            



