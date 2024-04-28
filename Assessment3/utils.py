###MODULES###
from functools import reduce, partial
from collections import defaultdict
import re
import numpy as np
from itertools import product
###FUCNTIONS###
###TASK1###
class Task1:
    def __init__(self, min, max):
        self._min = min
        self._max = max

    # @staticmethod
    def mapper(self, x):
        return self._min <= x <= self._max
    # @staticmethod
    def reducer(self, x, y):
        return (x[0], sum([x[1], y[1]])) 
    
    def chunk_mapper(self, chunk):
        mapped = map(self.mapper, chunk)
        mapped = zip(chunk, mapped)
        return reduce(self.reducer, mapped)
    
    def triv2_fun(self, row):
        return row[(self._max >= row) &  (row >= self._min)].shape[0]
    
    def triv1_fun(self, row):
        bools =  [self._min <= x <= self._max for x in row]
        return sum(bools)


class Task2:
    def __init__(self):
        pass

    @staticmethod
    def triv_a_fun(books):
        count = 0
        for book in books:
            with open(book, 'r') as bk:
                    text = bk.read()
            text = re.sub(r'[^a-zA-Z]+', ' ', text)
            text = text.upper()
            tokens = [txt for txt in text.split(" ") if txt != ""]
            count += len(tokens)
        return count
    
    @staticmethod
    def prepare_corpus(text):
            text = re.sub(r'[^a-zA-Z]+', ' ', text)
            text = text.upper()
            #TODO: normalization
            return text

    @staticmethod
    def tokenize(text):
         return [txt for txt in text.split(" ") if txt != ""]

    @staticmethod
    def mapper_a(file):
        with open(file, 'r') as book:
            book_text = book.read()
        corpus = Task2.prepare_corpus(book_text)
        tokenized = Task2.tokenize(corpus)
        return len(tokenized)
    
    @staticmethod
    def reducer_a(x, y):
        return (x[0], sum([x[1], y[1]]))

    @staticmethod
    def chunk_mapper_a(chunk):
        mapped = map(Task2.mapper_a, chunk)
        mapped = zip(chunk, mapped)
        return reduce(Task2.reducer_a, mapped)


    @staticmethod
    def add_to_dict(dict, item):
        dict[item] = dict.get(item, 0) + 1
         
    @staticmethod
    def triv_b_fun(books):
        count = {}
        for book in books:
            with open(book, 'r') as bk:
                book_text = bk.read()
            #Prepare corpus
            text = re.sub(r'[^a-zA-Z]+', ' ', book_text)
            text = text.upper()
            tokens = [txt for txt in text.split(" ") if txt != ""]
            [Task2.add_to_dict(count, item) for item in tokens]
        return count
    
    @staticmethod
    def triv_b_reducer1(dict1, dict2):
        dict2 = {k:[v] for k, v in dict2.items()}
        d = defaultdict(list, dict1)
        for k, v in dict2.items():
            d[k].extend(v)
        return d
    
    def triv_b_reducer2(kv):
        key_, val = kv
        return (key_, sum(val))
        
    

class Task4:
    @staticmethod
    def fun_a(bounds, mat):
        return (mat[bounds[0]:bounds[1], :])

    @staticmethod
    def get_split_rows_idx(m : np.ndarray) -> list:
        #Set batch size and no. of batches
        l = m.shape[0]
        batches = int(np.sqrt(l) * .25)
        batches_size = int(l / batches)
        #Spilt data to batches
        chunks_idx = []
        for idx in range(batches - 1):
            chunks_idx.append((batches_size * idx, batches_size * (idx + 1)))
        chunks_idx.append((batches_size * (batches - 1), l))
        return chunks_idx
    
    @staticmethod
    def get_split_dict(dict_):
        #Set batch size and no. of batches
        l = len(dict_)
        if l < 10:
            return [dict_]
        batches = int(np.sqrt(l) * .25)
        batches_size = int(l / batches)
        #Spilt data to batches
        chunks_idx = []
        for idx in range(batches - 1):
            chunks_idx.append((batches_size * idx, batches_size * (idx + 1)))
        chunks_idx.append((batches_size * (batches - 1), l))
        d_list = list(dict_.items())
        batches = [{k : v for k, v in d_list[beg:end]} for beg, end in chunks_idx]
        return batches


    @staticmethod
    def mp_mapper1_inner(row):
        num, row, mat = row
        mapped = map(lambda x: (*x, mat), enumerate(row))
        mapped = zip([num] * len(row), mapped)
        return [*mapped]
    
    @staticmethod
    def extender(x, y):
        x.extend(y)
        return x
        
    @staticmethod
    def mp_mapper1(beg_rows, mat : str):
        beg, rows = beg_rows
        nums = np.arange(rows.shape[0]) + beg
        rows = zip(nums, rows, [mat] * rows.shape[0])
        mapped = map(Task4.mp_mapper1_inner, rows)
        reduced = reduce(Task4.extender, mapped)
        return reduced

    @staticmethod
    def group(data):
        dict_ = defaultdict(list)
        for k, v in data:
            dict_[k].append(v)
        return dict_

    @staticmethod
    def mapper2_inner(triples):    
        ms= [x[:2] for x in triples if x[-1] == "M"]
        ns = [x[:2] for x in triples if x[-1] == "N"]
        combs = product(ms, ns)
        row = map(lambda x: ((x[0][0], x[1][0]), x[0][1] * x[1][1]) , combs)
        return [*row]

    @staticmethod
    def mapper2(chunk):
        mapped = map(Task4.mapper2_inner, chunk.values())
        reduced = reduce(Task4.extender, mapped)
        grouped = Task4.group(reduced)
        reduced = map(Task4.reducer, grouped.items())
        return [*reduced]
        
    @staticmethod
    def reducer(x):
        return (x[0], sum(x[1]))
    


# p1mapper = lambda x: _min <= x <= _max
p1reducer = lambda x, y: (x[0], sum([x[1], y[1]])) 


class Task3:
    @staticmethod
    def inner_mapper1(k, d, j):
        return pow(16, d - k, 8 * k + j) / (8 * k + j)
    
    @staticmethod
    def inner_reducer1(x, y):
        return (x + y) % 1
    
    @staticmethod
    def inner_mapper2(k, d, j):
        return pow(16, d - k) / (8 * k + j)
    
    @staticmethod
    def inner_reducer2(x, y):
        return x + y

    @staticmethod
    def mapper_sj(input_, eps = 1000):
        d, j = input_
        vals1 = range(d + 1)
        vals2 = range(d + 1, eps + d + 2)
        #Series1            
        mapped1 = map(partial(Task3.inner_mapper1, d = d, j = j), vals1)
        reduced1 = reduce(Task3.inner_reducer1, mapped1)
        #Series2
        mapped2 = map(partial(Task3.inner_mapper2, d = d, j = j), vals2)
        reduced2 = reduce(Task3.inner_reducer2, mapped2)
        #sj
        reduced =  reduce(Task3.inner_reducer2, [reduced1, reduced2])
        return reduced
    
    @staticmethod
    def reducer_num1(x, y):
        return  (1, x[0] * x[1] + y[0] * y[1])

    @staticmethod
    def reducer_num2(x):
        #Fixing modulo for negative
        val = x[1] - int(x[1])
        if val < 0:
            val = 1 + val
        return "%x" % int(val * 16)
    
    @staticmethod
    def mapper(x):
        mapped = zip([x] * 4, (1, 4, 5, 6))
        mapped = map(Task3.mapper_sj, mapped)
        mapped = zip((4, -2, -1, -1), mapped)
        reduced = reduce(Task3.reducer_num1, mapped)
        #Interesting fact: after creating tuples  operation % 1 no longer works correctly
        reduced = Task3.reducer_num2(reduced)
        return reduced

    @staticmethod
    def reducer_outer(x, y):
        return "".join([x,y])

    @staticmethod
    def mapper_outter(chunks):
        mapped = map(Task3.mapper, chunks)
        reduced = reduce(Task3.reducer_outer, mapped)
        return reduced
    
    @staticmethod
    def pi_basic2_sj(j, d, eps = 1000):
        series_1 = (pow(16, d - k, 8 * k + j) / (8 * k + j) for k in range(d + 1))
        series_1 = reduce(lambda x, y: (x + y) % 1, series_1)
        series_2 = sum(pow(16, d - k) / (8 * k + j) for k in range(d + 1, d + 2 + eps))
        
        return series_1 + series_2
    
    @staticmethod
    def pi_basic_sj(j, digs, eps = 1000):
        total = 0.0
        for idx in range(digs + eps):
            k = 8 * idx + j
            term = pow(16, digs - idx, k) if idx < digs else pow(16, digs - idx)
            total += term / k
        return total 
    
    @staticmethod
    def pi_basic(digits, eps = 1000):
        digits_ = digits - 1
        res = (
            4 * Task3.pi_basic2_sj(1, digits_, eps) - 2 * Task3.pi_basic2_sj(4, digits_, eps) -
            Task3.pi_basic2_sj(5, digits_, eps) - Task3.pi_basic2_sj(6, digits_, eps)
        ) % 1
        return "%x" % int(res * 16)
    
    def pi_basic1(digits, eps = 1000):
        digits_ = digits - 1
        res = (
            4 * Task3.pi_basic_sj(1, digits_, eps) - 2 * Task3.pi_basic_sj(4, digits_, eps) -
            Task3.pi_basic_sj(5, digits_, eps) - Task3.pi_basic_sj(6, digits_, eps)
        ) % 1
        return "%x" % int(res * 16)