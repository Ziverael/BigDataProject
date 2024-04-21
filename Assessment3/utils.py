###MODULES###
from functools import reduce
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
            [Task2.add(item) for item in tokens]
    

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