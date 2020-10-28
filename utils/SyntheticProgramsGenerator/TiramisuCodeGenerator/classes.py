import random as r
from pprint import pprint
from random import random, randint, uniform, randrange, choice, shuffle
import function
import functools

def split_for_reduction(l):
    assert len(l) > 2
    axe = [l[-1]]
    mid = int((len(l)-1)/2)
    left = l[:mid] + axe
    right = axe + l[mid:-1]
    return axe, left, right


def choose_elements_from_list(l, k, order=True):
    if len(l) == 0:
        raise Exception("can not choose from empty list")
    if k <= len(l):
        l_indices = list(range(len(l)))
        r_indices = []
        for _ in range(k):
            c = choice(l_indices)
            l_indices.remove(c)
            r_indices.append(c)
        if order:
            r_indices.sort()
        res = [l[i] for i in r_indices]
        return res
    else:
        larger_list = l
        while k > len(larger_list):
            larger_list = larger_list + l
        return larger_list[:k]


def matrix0(size):
    x = 11  # TODO : map to function.py constant
    M = [[0 for i in range(x)] for j in range(x)]
    for ii in range(size):
        M[ii][ii] = 1
    for l in M:
        l.append(randint(0, 4))

    return M


def matrix(h, w):
    M = [[1 if i == j else 0 for i in range(w + 1)] for j in range(h)]
    return M


def access_matrix():
    x = 11  # TODO : map to function.py constant
    M = [[0 for i in range(x + 1)] for j in range(x)]
    return M


class Constant:
    MIN_CONSTANT_VALUE = 5  # 2**7=256
    MAX_CONSTANT_VALUE = 11 # 2**10=

    def __init__(self, id):
        self.id = id
        self.name = "c" + str(id).zfill(2)
        #self.value = 2 ** randint(self.MIN_CONSTANT_VALUE, self.MAX_CONSTANT_VALUE)
        self.value = 0
        self.relative = False

    def __str__(self):
        return self.name + '("' + self.name + '", ' + str(self.value) + ")"


class Variable:

    def __init__(self, id, parent, sup, schedulable, reference_variables=[], inf=0):
        self.correction = 0
        self.id = id
        self.parent = parent
        self.name = "i" + str(self.id).zfill(2)
        self.inf = inf
        self.sup = sup
        self.schedulable = schedulable
        self.tiled = False
        self.level = -1
        self.reference_variables = reference_variables
        if len(self.reference_variables) > 0:
            self.sup.relative = True

    def __str__(self):
        if self.sup is not None:
            if self.correction == 0:
                correction = ""
            else:
                correction = " - " + str(2 * self.correction)
            return self.name + '("' + self.name + '", ' + str(self.inf) + ", " + self.sup.name + correction + ") "
        else:
            return self.name + '("' + self.name + '") '

    def update_constant(self):
        if self.sup.relative:
            self.sup.value = self.reference_variables[0].sup.value + self.reference_variables[1].sup.value - 1

    def correct(self, offset):
        if self.correction < offset:
            self.correction = offset




class Tensor:
    id = None
    variables = []
    data_type = "p_int32"

    def get_volume(self):
        if self.data_type == "p_int32":
            size_unit = 4
        for v in self.variables:
            v.update_constant()
        number_units = functools.reduce(lambda a, b: a * b, [vv.sup.value for vv in self.variables])
        return number_units * size_unit


class Input(Tensor):

    def __init__(self, id, variables, unstorable_variables=[], data_type="p_int32"):
        self.id = id
        self.name = "input" + str(id).zfill(2)
        self.variables = variables
        self.dataType = data_type
        self.unstorable_variables = unstorable_variables

    def __str__(self):
        res = "input " + self.name + '("' + self.name + '", {'
        for v in self.variables:
            res = res + v.name + ", "
        res = res[:-2]
        res = res + "}, " + self.dataType + ");"

        return res


class Computation(Tensor):
    MAX_NUMBER_TERMS = 3  #  3*3 , 1*3
    MIN_NUMBER_TERMS = 1  # for stencils only

    def __init__(self, id, variables, used_tensors, input_assignment=False, simple_expression=False,
                 stencil=False, reduction=False, reduction_axe=[], convolution=False,
                 unstorable_variables=[], expression="", dataType="p_int32"):
        self.id = id
        self.name = "comp" + str(id).zfill(2)
        self.variables = variables
        self.dataType = dataType
        self.used_tensors = used_tensors
        self.expression = expression
        self.reduction = reduction
        self.convolution = convolution
        self.reduction_axe = reduction_axe
        self.unstorable_variables = unstorable_variables
        self.separate_expression = False

        self.simple_expression = simple_expression
        self.stencil = stencil
        self.input_assignment = input_assignment
        self.reduction = reduction
        self.convolution = convolution

        if reduction or convolution:
            self.separate_expression = True
        if reduction:
            self.unstorable_variables = self.reduction_axe[:]
        if convolution:
            self.unstorable_variables = list(filter(lambda v: not v.schedulable, self.variables))

        if self.expression == "":
            if simple_expression:
                self.expression += str(randint(1, 100))
                self.expression += choice([" + ", " - ", " * "])
                self.expression += str(randint(1, 100))

            elif stencil:
                number_terms = randint(self.MIN_NUMBER_TERMS, len(used_tensors.variables))
                stencil_variables = choose_elements_from_list(used_tensors.variables, number_terms,
                                                              order=True)
                random_number = choice([1, 1, 1, 1, 1])
                for stencil_variable in stencil_variables:
                    stencil_variable.correct(random_number)
                    for roundp in range(3):
                        self.expression += used_tensors.name + "("
                        for v in used_tensors.variables:
                            if v == stencil_variable and roundp == 0:
                                self.expression += v.name + ", "
                            elif v == stencil_variable and roundp == 1:
                                self.expression += v.name + " + " + str(random_number) + ", "
                            elif v == stencil_variable and roundp == 2:
                                self.expression += v.name + " + " + str(2*random_number) + ", "
                            elif v in stencil_variables:
                                self.expression += v.name + " + " + str(random_number) + ", "
                            else:
                                self.expression += v.name + ", "
                        self.expression = self.expression[:-2] + ")" + choice([" + ", " - ", " * "])
                self.expression = self.expression[:-3]

            elif input_assignment:
                for tensor in used_tensors:
                    self.expression += tensor.name + "("
                    for v in tensor.variables:
                        self.expression += v.name + ", "
                self.expression = self.expression[:-2] + ")" + choice([" + ", " - ", " * "])
                self.expression = self.expression[:-3]

            elif reduction:
                self.expression += self.name + "("
                for v in self.variables:
                    self.expression += v.name + ", "
                self.expression = self.expression[:-2] + ")" + choice([" + ", " - ", " * "])

                for tensor in used_tensors:
                    self.expression += tensor.name + "("
                    for v in tensor.variables:
                        self.expression += v.name + ", "
                    self.expression = self.expression[:-2] + ")" + choice([" + ", " - ", " * "])
                self.expression = self.expression[:-3]

            elif convolution:
                self.expression = self.name + "("
                for v in variables:
                    self.expression += v.name + ", "
                self.expression = self.expression[:-2] + ")" + choice([" + ", " - ", " * "])
                for tensor in used_tensors:
                    self.expression += tensor.name + "("
                    for v in tensor.variables:
                        if len(v.reference_variables) > 0:
                            self.expression += v.reference_variables[0].name + " + " + v.reference_variables[1].name + ", "
                        else:
                            self.expression += v.name + ", "
                    self.expression = self.expression[:-2] + ")" + choice([" + ", " - ", " * "])
                self.expression = self.expression[:-3]

    def __str__(self):
        res = "computation " + self.name + '("' + self.name + '", {'
        for v in self.variables:
            res = res + v.name + ", "
        res = res[:-2]
        if not self.separate_expression:
            res = res + "}, " + self.expression + ");"

        else:
            res = res + "}, " + self.dataType + ");"

        return res
    def get_setexpression(self):
        if self.separate_expression:
            return self.name + ".set_expression(" + self.expression + ");\n"
        else:
            return ""
    def tensor_from_id(self, id):
        tensor = None
        try:
            for tensor in self.usedTensors:
                if tensor.id == id:
                    return tensor
        except TypeError:
            return self.usedTensors

    def variable_from_id(self, id):
        v = None
        for v in self.variables:
            if v.id == id:
                return v


class Schedule:
    def __init__(self, tiles, interchanges, unrolls, parallelizes):
        self.tiles = tiles
        self.interchanges = interchanges
        self.unrolls = unrolls
        self.parallelizes = parallelizes


class Buffer:
    def __init__(self, tensor):
        self.id = tensor.id
        self.name = "buf" + str(self.id).zfill(2)
        self.tensor = tensor
        self.variables = []
        self.ignore_correction = False
        for v in self.tensor.variables:
            if v not in self.tensor.unstorable_variables:
                self.variables.append(v)

    def __str__(self):
        if isinstance(self.tensor, Input):
            res = "buffer " + self.name + '("' + self.name + '", {'
            for v in self.variables:
                res = res + str(v.sup.value) + ", "
            res = res[:-2]
            x = ", a_input);"
            res = res + "}, " + self.tensor.dataType + x
        else:
            res = "buffer " + self.name + '("' + self.name + '", {'
            for v in self.variables:
                if self.ignore_correction:
                    size = v.sup.value
                else:
                    size = v.sup.value - 2 * v.correction
                res = res + str(size) + ", "
            res = res[:-2]
            x = ", a_output);"
            res = res + "}, " + self.tensor.dataType + x

        return res

    def write_store(self):
        dimentions = ""
        if len(self.tensor.unstorable_variables) > 0:
            dimentions = ", {"
            for v in self.variables:
                dimentions = dimentions + v.name + ", "
            dimentions = dimentions[:-2] + "}"

        return self.tensor.name + ".store_in(&" + self.name + dimentions + ");"

    def write_for_wrapper(self):
        if isinstance(self.tensor, Input):
            inversed_variables = self.variables[:]
            inversed_variables.reverse()
            res = "int *c_" + self.name + " = (int*)malloc("
            for v in inversed_variables:
                res += str(v.sup.value) + " * "
            res += "sizeof(int));\n"
            res += "    parallel_init_buffer(c_"+self.name+", "
            for v in inversed_variables:
                res += str(v.sup.value) + " * "
            res = res[:-3] + ",  (int32_t)" + str(randint(0, 100)) + ");\n"
            res += "    Halide::Buffer<int32_t> " + self.name + "(c_"+self.name+", "
            for v in inversed_variables:
                res = res + str(v.sup.value) + ", "
            res = res[:-2] + ");"
        else:
            res = "Halide::Buffer<int32_t> " + self.name + "("
            inversed_variables = self.variables[:]
            inversed_variables.reverse()
            for v in inversed_variables:
                size = v.sup.value - 2 * v.correction
                res = res + str(size) + ", "
            res = res[:-2] + ");"

        return res


class Tile:
    TILE_SIZES = [32, 64, 128, ]

    def __init__(self, originVariables, generatedVariables, sizes):
        self.originVariables = originVariables
        self.generatedVariables = generatedVariables
        self.sizes = sizes


class Unroll:
    UNROLL_SIZES = [4, 8, 16, ]

    def __init__(self, variable, factor):
        self.variable = variable
        self.factor = factor


class Interchange:
    def __init__(self, variables):
        self.variables = variables


class Parallelize:
    def __init__(self, variable):
        self.variable = variable
