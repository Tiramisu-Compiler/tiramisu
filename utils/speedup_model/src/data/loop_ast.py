import numpy as np 
import pprint

class Loop_Iterator():
        def __init__(self, it_id, dict_repr, depth=0):
            self.depth=depth
            #get loop iterator
            iterator = next(it for it in dict_repr['iterators']['iterators_array'] if it['it_id'] == it_id)

            self.id = it_id
            self.lower_bound = iterator['lower_bound']
            self.upper_bound = iterator['upper_bound']

        def __repr__(self):
            return f"({self.lower_bound}, {self.upper_bound})"

        def __array__old(self):
            return [self.id, self.lower_bound, self.upper_bound]

        def __array__(self):
            return [self.lower_bound, self.upper_bound]

class Input():
    def __init__(self, input_id, dict_repr, depth=0):
        self.id = input_id
        self.depth = depth
        
        #search for input_id
        input_ = next(i for i in dict_repr['inputs']['inputs_array']
                                if i['input_id'] == input_id)

        self.dtype = input_['data_type']
    
    def __repr__(self):
        return f"Input {self.id}"

    def __array__(self):
        return [self.id]

class Access_pattern():
    def __init__(self, access_matrix, depth=0):
        self.access_matrix = np.array(access_matrix)
        self.max_shape = (4, 5)
    
    def __repr__(self):
        return repr(self.access_matrix)
    
    def __array__(self):
        rows = self.max_shape[0] - self.access_matrix.shape[0]
        cols = self.max_shape[1] - self.access_matrix.shape[1]

        for _ in range(cols):
            self.access_matrix = np.insert(self.access_matrix, -1, 0, axis=1)

        for _ in range(rows):
            self.access_matrix = np.insert(self.access_matrix, len(self.access_matrix), 0, axis=0)

        
        return self.access_matrix.flatten()

        
        





class Computation():
    def __init__(self, comp_id, dict_repr, depth=0):
        self.depth = depth
        self.id = comp_id
        self.max_children = 17 #max accesses
        self.max_comp_len = 21*self.max_children

        #search for comp_i
        computation = next(c for c in dict_repr['computations']['computations_array']
                                if c['comp_id'] == comp_id)

        self.dtype = computation['lhs_data_type']

        self.op_histogram = computation['operations_histogram'][0] #take only first row for now

        self.children = []

        mem_accesses = computation['rhs_accesses']['accesses']

        for mem_access in mem_accesses:
            inp = Input(mem_access['comp_id'], dict_repr, self.depth+1)
            access_pattern = Access_pattern(mem_access['access'], self.depth+1)

            self.children.append((inp, access_pattern))


    def __repr__(self):
        sep = '\n' +  (self.depth+1)*'\t'

        children_repr = [repr(child) for child in self.children]
        children_repr = sep + sep.join(children_repr)

        return f"Computation {self.id}:" + children_repr

    def __array__(self):
        children_arr = []

        for inp, access in self.children[:self.max_children]:
            children_arr.extend(inp.__array__() + access.__array__())

        children_arr.extend([0] * (self.max_comp_len - len(children_arr)))
        
        return children_arr
        
        


        
class Loop():
    def __init__(self, loop_repr, dict_repr, depth=0):
        self.tiled = False
        self.tile_factor = 0 

        self.interchanged = False


        self.depth = depth
        self.id = loop_repr['loop_id']

        it_id = loop_repr['loop_it']
        self.iterator = Loop_Iterator(it_id, dict_repr)

        #search and create all children of loop (other loops, and computation assignments)
        self.children_dict = {}

        #add assignments to children
        comps = loop_repr['assignments']['assignments_array']
        for comp in comps:
            comp_id = comp['id']
            position = comp['position']

            self.children_dict[position] = Computation(comp_id, dict_repr, self.depth+1)

        #add loops to children
        loops = dict_repr['loops']['loops_array']

        for loop in loops: 
            if loop['parent'] == self.id:
                self.children_dict[loop['position']] = Loop(loop, dict_repr, self.depth+1)

        self.children = self.sort_children()
        
    def sort_children(self):
        #sort children by position 
        return list(list(zip(*sorted(self.children_dict.items(), key=lambda x: int(x[0]))))[1])  

    def tile(self, factor):
        self.tiled = True
        self.tile_factor = factor
    
    def interchange(self):
        self.interchanged = True 

    def __repr__(self):
        children_repr = [repr(child) for child in self.children]
        children_repr = '\n' + (self.depth+1)*'\t'  + "\n".join(children_repr)
        #print(children_repr)

        return  f"Loop {self.id} {repr(self.iterator    )}:" + children_repr

    def __array__old(self):
        loop_arr = []
        loop_arr.extend(self.iterator.__array__())

        if not isinstance(self.children[0], Loop): 
            #fill loop space with -1
            loop_arr_len = len(loop_arr)
            loop_arr.extend([-1]*loop_arr_len * (3 - self.depth))
        
       
        loop_arr.extend(self.children[0].__array__())

        return loop_arr

    def __array__(self):
        arr = []
        arr.extend(self.iterator.__array__())
        
        #loop interchanged
        arr.extend([+self.interchanged])

        #loop tiled
        arr.extend([+self.tiled, self.tile_factor])

        if not isinstance(self.children[0], Loop): 
            #fill loop space with 0
            loop_arr_len = len(arr)
            arr.extend([0]*loop_arr_len * (3 - self.depth))
        
       
        arr.extend(self.children[0].__array__())

        return arr
        
        

    

class Loop_AST():

    def __init__(self, name, dict_repr=None, schedule=None):
        self.name = name
        self.root_loop = None
        self.dtype_int_dict = {"p_int": 199}
        self.dict_repr = dict_repr
        self.load_from_dict(dict_repr)
        self.schedule = schedule

        self.unrolled = False
        self.unroll_factor = 0

        if self.schedule:
            self.apply_schedule()

    
    def apply_schedule(self):
        self.name = self.schedule.name 
        binary_schedule = self.schedule.binary_repr

        for command in self.schedule.schedule_list:
            type_ = command['type']
            params = command['params']
            factors = command['factors']

            if type_ == 'tiling' and binary_schedule[1] == 1:
                for loop_id, factor in zip(params, factors):
                    self.tile(loop_id, factor)
            
            elif type_ == 'interchange' and binary_schedule[0] == 1:
                self.interchange(params[0])
                self.interchange(params[1])

            elif type_ == 'unrolling' and binary_schedule[2] == 1:
                self.unroll(factors[0])

    def unroll(self, factor):
        self.unrolled = True
        self.unroll_factor = factor

    def interchange(self, loop_id):
        loop = self.root_loop

       
        while loop.iterator.id != loop_id:
            loop = loop.children[0]

       
        
        loop.interchange() 

    def tile(self, loop_id, factor):
        loop = self.root_loop
        try:
            while loop.iterator.id != loop_id:
                loop = loop.children[0]
         

            loop.tile(factor)
        except AttributeError:
            print(self.name)
            print(loop_id)
            print(self.root_loop)
            from pprint import pprint
            pprint(self.dict_repr)
            exit(1)
     
    def add_schedule(self, schedule):
        
        return Loop_AST(self.name, self.dict_repr, schedule)
    
    def dtype_to_int(self, dtype):
        return self.dtype_int_dict[dtype]

    def load_from_dict(self, dict_repr):
        if not dict_repr:
            return
            
        self.dict_repr = dict_repr

        loops = dict_repr['loops']['loops_array']

        #find root loop
        root = next(l for l in loops if l['parent'] == -1)

        self.root_loop = Loop(root, dict_repr)

    def __array__old(self):
        return np.array(self.root_loop.__array__())

    def __array__(self):

        arr = self.root_loop.__array__()

        #loop unrolling
        arr.extend([+self.unrolled, self.unroll_factor])

        return arr