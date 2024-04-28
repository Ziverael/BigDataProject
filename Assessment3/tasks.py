"""
I realized that using MapReduce can be very useful approach. Even without multiprocessing chunkifying 
 can reduced execution time (watch results for task4). Even though  specialized methods from libs like numpy, scipy, etc 
 are always more usefull approach. 

Executing task1 for matrix shape 500x500 
    Function: problem1_triv Total execution time: 0.0367 sec
    Function: problem1_triv2 Total execution time: 0.0027 sec
    Function: problem1_seq Total execution time: 0.0891 sec
Executing task1 for matrix shape 9000x9000
    Function: problem1_triv Total execution time: 11.5825 sec
    Function: problem1_triv2 Total execution time: 0.4409 sec
    Function: problem1_seq Total execution time: 30.3945 sec
Executing task1 for matrix shape 10000x10000
    Function: problem1_triv Total execution time: 13.8705 sec
    Function: problem1_triv2 Total execution time: 0.5180 sec
    Function: problem1_seq Total execution time: 37.0761 sec
Executing task1 for matrix shape 10000x20000
    Function: triv1_seq Total execution time: 31.1570 sec
    Function: triv2_seq Total execution time: 1.2926 sec
    Function: triv2_mp Total execution time: 4.5410 sec
    Function: mapreduce_seq Total execution time: 80.4974 sec
    Function: mapreduce_mp Total execution time: 25.1892 sec
Executing task1 for matrix shape 20000x20000
    Function: triv1_seq Total execution time: 65.6387 sec
    Function: triv2_seq Total execution time: 5.0075 sec
    Function: triv2_mp Total execution time: 8.6527 sec
    Function: triv1_mp Total execution time: 20.3244 sec
    Function: mapreduce_seq Total execution time: 163.9033 sec
    Function: mapreduce_mp Total execution time: 49.8929 sec

Executing task2 for 303 books
    Function: triv_b_seq Total execution time: 9.0372 sec
    Function: triv_b_mp Total execution time: 4.6523 sec
    Function: mapreduce_b_seq Total execution time: 37.8650 sec
    Function: mapreduce_b_mp Total execution time: 4.7699 sec

Executing task3 for 1,000th 50 digits
    Function: triv_seq1 Total execution time: 0.1660 sec
    Function: triv_seq2 Total execution time: 0.1707 sec
    Function: mapreduce_seq Total execution time: 0.2181 sec
    Function: triv1_mp Total execution time: 0.0572 sec
    Function: triv2_mp Total execution time: 0.0602 sec
    Function: mapreduce_mp Total execution time: 0.0880 sec
Executing task3 for 1,000,000th 50 digits
    Function: triv_seq1 Total execution time: 194.1920 sec
    Function: triv_seq2 Total execution time: 221.8192 sec 
    

Executing task4 for matrices shaped (200, 400) x (400, 300)
    Function: triv_seq Total execution time: 0.0171 sec
    Function: mapreduce_seq Total execution time: 83.6241 sec
    Function: mapreduce_seq2 Total execution time: 31.3015 sec
    Function: mapreduce_mp Total execution time: 13.3028 sec

    Actually  it looks like, the numpy approach is much better, however now we considered only sequential cases.
I have notice that even producing matrix with size 50000x50000 generate the following error
    Traceback (most recent call last):
    File "/home/zive_bewise/Documents/ML/BigData/Course/Assessment3/tasks.py", line 123, in <module>
        x = np.random.randint(-100,100, (50000, 50000))
    File "numpy/random/mtrand.pyx", line 781, in numpy.random.mtrand.RandomState.randint
    File "numpy/random/_bounded_integers.pyx", line 1343, in numpy.random._bounded_integers._rand_int64
    numpy.core._exceptions._ArrayMemoryError: Unable to allocate 18.6 GiB for an array with shape (50000, 50000) and data type int64

Step2 consider chunkifying for and parallel for all cases.
"""
###MODULES###
from typing import List, Tuple
from functools import wraps, reduce, partial
from itertools import chain
import time
import numpy as np
import os
from pathlib import Path
import re
from collections import defaultdict
from multiprocessing import Pool
from utils import *
from itertools import product
from decimal import getcontext, Decimal

###VARIABLES###
DATA_DIR = '/home/zive_bewise/Documents/ML/BigData/Course/Data'


###FUNCTIONS###
###WRAPPER###
def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f'Function: {func.__name__} Total execution time: {total_time:.4f} sec')
        return result
    return timeit_wrapper


class Problem1:
    def __init__(self, matrix : List[List[float]], range_ : Tuple[float, float]) -> None:
        self._matrix = matrix
        self._min, self._max =  range_
        self._func = Task1(self._min, self._max)
    
    @timeit
    def triv1_seq(self) -> int:
        """
        "Given a 2D matrix (or list of lists), count how many numbers are present in a given range"
        Performance without MapReduce
        matrix      List of lists filled with numbers
        range       Tuple [min, max] declaring numeric interval. We consider enclosed interval.
        """
        return sum([self._min <= x <= self._max for x in chain.from_iterable(self._matrix)])

    @timeit
    def triv2_seq(self) -> int:
        """
        "Given a 2D matrix (or list of lists), count how many numbers are present in a given range"
        Performance without MapReduce
        matrix      List of lists filled with numbers
        range       Tuple [min, max] declaring numeric interval. We consider enclosed interval.
        """
        mat = np.array(self._matrix)
        return mat[(self._max >= mat) &  (mat >= self._min)].shape[0]
    
    @timeit
    def triv2_mp(self) -> int:
        """
        "Given a 2D matrix (or list of lists), count how many numbers are present in a given range"
        Performance without MapReduce
        matrix      List of lists filled with numbers
        range       Tuple [min, max] declaring numeric interval. We consider enclosed interval.
        """
        with Pool(processes = 8) as pl:
            res = pl.map_async(self._func.triv2_fun, self._matrix)
            res = res.get()
        return sum(res)

    @timeit
    def triv1_mp(self) -> int:
        """
        "Given a 2D matrix (or list of lists), count how many numbers are present in a given range"
        Performance without MapReduce
        matrix      List of lists filled with numbers
        range       Tuple [min, max] declaring numeric interval. We consider enclosed interval.
        """
        with Pool(processes = 8) as pl:
            res = pl.map_async(self._func.triv1_fun, self._matrix)
            res = res.get()
        return sum(res)

    @timeit
    def mapreduce_seq(self) -> int:
        """
        "Given a 2D matrix (or list of lists), count how many numbers are present in a given range"

        We consider the following mapper and reducer tasks:
        Map -   return boolean value explaining if certain value is in a given range

        matrix      List of lists filled with numbers
        range       Tuple [min, max] declaring numeric interval. We consider enclosed interval.
        """
        mapper = lambda x: self._min <= x <= self._max
        reducer = lambda x, y: (x[0], sum([x[1], y[1]])) 

        reduce_all = []
        
        for chunk in self._matrix:
            mapped = map(mapper, chunk)
            mapped = zip(chunk, mapped)

            reduced = reduce(reducer, mapped)
            reduce_all.append(reduced)
        reduced = reduce(reducer, reduce_all)
        return reduced[1]
    
    @timeit
    def mapreduce_mp(self) -> int:
        task = Task1(self._min, self._max)
        with Pool(processes = 8) as pl:
            mapped = pl.map_async(task.chunk_mapper, self._matrix)
            mapped = mapped.get()
        return reduce(task.reducer, mapped)[1]

    



class Problem2:
    def __init__(self, dir_):
        self._books = Problem2.load_books(dir_)

    @staticmethod
    def load_books(path):
        #Load data
        if not os.path.isdir(path):
                raise IOError("Directory does not exist.")
        return [str(bk) for bk in Path(path).rglob('*.txt')]
    
    @staticmethod
    def split_list2chunks(lst):
        #Set batch size and no. of batches
        l = len(lst)
        batches = int(np.sqrt(l) * .25)
        batches_size = int(l / batches)
        #Spilt data to batches
        chunks = []
        for idx in range(batches - 1):
            chunks.append(lst[batches_size * idx : (idx + 1) * batches_size])
        chunks.append(lst[batches_size * (batches - 1) : ])
        return chunks

    def split_books2chunks(self):
        #Set batch size and no. of batches
        l = len(self._books)
        batches = int(np.sqrt(l) * .25)
        batches_size = int(l / batches)
        #Spilt data to batches
        chunks = []
        for idx in range(batches - 1):
            chunks.append(self._books[batches_size * idx : (idx + 1) * batches_size])
        chunks.append(self._books[batches_size * (batches - 1) : ])
        return chunks

    @timeit
    def triv_a_seq(self):
        count = 0
        for book in self._books:
            with open(book, 'r') as bk:
                text = bk.read()
            text = re.sub(r'[^a-zA-Z]+', ' ', text)
            text = text.upper()
            tokens = [txt for txt in text.split(" ") if txt != ""]
            count += len(tokens)
        return count
    
    @timeit
    def triv_a_mp(self):
        count = 0
        books_split = Problem2.split_list2chunks(self._books)
        with Pool(processes = 8) as pl:
            res = pl.map_async(Task2.triv_a_fun, books_split)
            res = res.get()
        return sum(res)
        # for book in self._books:
        #     with open(book, 'r') as bk:
        #         text = bk.read()
        #     text = re.sub(r'[^a-zA-Z]+', ' ', text)
        #     text = text.upper()
        #     tokens = [txt for txt in text.split(" ") if txt != ""]
        #     count += len(tokens)
        # return count
    
    

    @timeit
    def mapreduce_a_seq(self):
        """
        Download as many books as possible from the Gutenberg project and:
         count the words in all books,
         compute a histogram of the words
        """
        #Perform cleaning
        def prepare_corpus(text):
            text = re.sub(r'[^a-zA-Z]+', ' ', text)
            text = text.upper()
            #TODO: normalization
            return text
        
        tokenize = lambda text: [txt for txt in text.split(" ") if txt != ""]

        def mapper(file):
            with open(file, 'r') as book:
                book_text = book.read()
            corpus = prepare_corpus(book_text)
            tokenized = tokenize(corpus)
            return len(tokenized)
        
        reducer = lambda x, y: (x[0], sum([x[1], y[1]]))

        #MapReduce
        mapped = map(mapper, self._books)
        mapped = zip(self._books, mapped)
        reduced = reduce(reducer, mapped)
        return reduced[1]
    

    @timeit
    def mapreduce_a_mp(self):
        """
        Download as many books as possible from the Gutenberg project and:
         count the words in all books,
         compute a histogram of the words
        """
        
        #MapReduce
        split_books = self.split_books2chunks()
        with Pool(processes = 8) as pl:
            res = pl.map_async(Task2.chunk_mapper_a, split_books)
            res = res.get()
        
        return reduce(Task2.reducer_a, res)[1]
    
    
    @timeit
    def triv_b_seq(self):
        counts = {}
        def add(item):
            counts[item] = counts.get(item, 0) + 1
        #Loop over books    
        for bk_path in self._books:
            with open(bk_path, 'r') as book:
                book_text = book.read()
            #Prepare corpus
            text = re.sub(r'[^a-zA-Z]+', ' ', book_text)
            text = text.upper()
            tokens = [txt for txt in text.split(" ") if txt != ""]
            [add(item) for item in tokens]
        return counts
    
    @timeit
    def triv_b_mp(self):
        #Loop over books    
        split_books = self.split_books2chunks()
        with Pool(processes = 8) as pl:
            # res = pl.map_async(partial(Task2.triv_b_fun, dict_ = counts), split_books)
            res = pl.map_async(Task2.triv_b_fun, split_books)
            res = res.get()
        result = reduce(Task2.triv_b_fun2, res)
        return result

    @timeit
    def mapreduce_b_seq(self):
        """
        Download as many books as possible from the Gutenberg project and:
         count the words in all books,
         compute a histogram of the words
        Here 
        """
        #Load data
        def prepare_corpus(text):
            text = re.sub(r'[^a-zA-Z]+', ' ', text)
            text = text.upper()
            #TODO: normalization
            return text

        tokenize = lambda text: [txt for txt in text.split(" ") if txt != ""]

        def mapper1(file):
            with open(file, 'r') as book:
                book_text = book.read()
            corpus = prepare_corpus(book_text)
            tokenized = tokenize(corpus)
            return tokenized

        def reducer1(x, y):
            x[1].extend(y[1])
            return x
        
        def reducer2(x, y):
            x.extend(y)
            return x

        #Mapper&reducer for book unit        
        mapper2 = lambda _ : [1]
        mapper3 = lambda x: (x[0], len(x[1]))
        mapper4 = lambda x: (x[0], [x[1]])
        mapper5 = lambda x: (x[0], sum(x[1]))

        def grouping(mapped):
            grouped = defaultdict(list)
            for key, val in mapped:
                grouped[key].extend(val)
            return grouped

        def chunk_mapper(chunk):
            mapped = map(mapper1, chunk)
            mapped = zip(chunk, mapped)
            return reduce(reducer1, mapped)
        
        def chunk_mapper2(chunk):
            mapped = map(mapper2, chunk)
            mapped = zip(chunk, mapped)
            #Grouping
            grouped = grouping(mapped)
            mapped = map(mapper3, grouped.items())
            return [*mapped]
        
        #MapReduce
        #Rduce list of books to copruses and reduce to word list
        books_split = self.split_books2chunks()
        mapped = map(chunk_mapper, books_split)
        reduced = reduce(reducer1, mapped)
        #Count words
        words_split = Problem2.split_list2chunks(reduced[1])
        mapped = map(chunk_mapper2, words_split)
        reduced = reduce(reducer2, mapped)
        mapped = map(mapper4, reduced)
        grouped = grouping(mapped)
        reduced = map(mapper5, grouped.items())
        return dict(reduced)
                
    @timeit
    def mapreduce_b_mp(self):
        #Loop over books    
        split_books = self.split_books2chunks()
        with Pool(processes = 8) as pl:
            res = pl.map_async(Task2.chunk_mapper_b, split_books)
            res = res.get()
        result = reduce(Task2.reducer2, res)
        return result











class Problem4:
    def __init__(self, m : np.ndarray, n : np.ndarray):
        if len(m.shape) != (2,2) != len(n.shape) != 2:
            raise ValueError("We consider only 2d matrices.")
        self.M = m
        self.N = n

    @staticmethod
    def get_split_rows_idx(m : np.ndarray) -> list:
        #Set batch size and no. of batches
        l = m.shape[0]
        if l < 10:
            return [(0, l)]
        batches = int(np.sqrt(l) * .25)
        batches_size = int(l / batches)
        #Spilt data to batches
        chunks_idx = []
        for idx in range(batches - 1):
            chunks_idx.append((batches_size * idx, batches_size * (idx + 1)))
        chunks_idx.append((batches_size * (batches - 1), l))
        return chunks_idx
    
    @staticmethod
    def get_split_cols_idx(m : np.ndarray) -> list:
        #Set batch size and no. of batches
        l = m.shape[1]
        batches = int(np.sqrt(l) * .25)
        batches_size = int(l / batches)
        #Spilt data to batches
        chunks_idx = []
        for idx in range(batches - 1):
            chunks_idx.append((batches_size * idx, batches_size * (idx + 1)))
        chunks_idx.append((batches_size * (batches - 1), l))
        return chunks_idx
    
    @staticmethod
    def arr_from_dict(data):
        rows = max(index[0] for index, _ in data) + 1
        cols = max(index[1] for index, _ in data) + 1
        arr = np.zeros((rows, cols))
        # Fill the array with values from the list of tuples
        for (x, y), val in data:
            arr[x, y] = val
        return arr


    @staticmethod
    def idxs_mat(mat, reverse = False):
        """If reverse, then build for columns as first indexes."""
        rows, cols = mat.shape
        if reverse:
            cols, rows = mat.shape
        col = np.arange(cols)
        mapped = map(lambda x: [*zip([x]*rows, np.arange(rows))], col)
        reduced = [*mapped]
        return reduced

    @timeit
    def triv_seq(self):
        return self.M @ self.N

    @timeit
    def mapreduce_seq(self):
        # chunks_m_idx = Problem4.get_split_cols_idx(m)
        # chunks_n_idx = Problem4.get_split_rows_idx(n)

        def extender(x, y):
            x.extend(y)
            return x

        def inner_mapper(row, mat : str):
            num, row = row
            mapped = map(lambda x: (*x, mat), enumerate(row))
            mapped = zip([num] * len(row), mapped)
            return [*mapped]

        def mapper1(triples):
            ms= [x[:2] for x in triples if x[-1] == "M"]
            ns = [x[:2] for x in triples if x[-1] == "N"]
            combs = product(ms, ns)
            row = map(lambda x: ((x[0][0], x[1][0]), x[0][1] * x[1][1]) , combs)
            return [*row]

        def reducer(x):
            return (x[0], sum(x[1]))

        #Map M
        mapped = map(partial(inner_mapper, mat = "M"), enumerate(self.M.T))
        reduced1 = reduce(extender, mapped)
        #Map N
        mapped = map(partial(inner_mapper, mat = "N"), enumerate(self.N))
        reduced2 = reduce(extender, mapped)
        reduced = reduce(extender, [reduced1, reduced2])
        dict_ = defaultdict(list)
        for k, v in reduced:
            dict_[k].append(v)
        mapped = map(mapper1, dict_.values())
        reduced = reduce(extender, mapped)
        dict_ = defaultdict(list)
        for k, v in reduced:
            dict_[k].append(v)
        reduced = map(reducer, dict_.items())
        return [*reduced]
    
    @staticmethod
    def mapper_1(chunks_idx, mat):
        mapped = map(partial(Task4.fun_a, mat = mat), chunks_idx)
        begs, _ = [*zip(*chunks_idx)]
        mapped = zip(begs, mapped)
        return mapped
    

    @timeit
    def mapreduce_seq2(self):
        def mapper_2(beg_rows, mat : str):
            beg, rows = beg_rows
            nums = np.arange(rows.shape[0]) + beg
            rows = zip(nums, rows, [mat] * rows.shape[0])
            mapped = map(inner_mapper, rows)
            reduced = reduce(extender, mapped)
            return reduced

        def mapper3(chunk):
            #V1
            # mapped = map(mapper3_inner, chunk.values())
            # reduced = reduce(extender, mapped)
            # return reduced 
        
            #V2
            mapped = map(mapper3_inner, chunk.values())
            reduced = reduce(extender, mapped)
            grouped = group(reduced)
            reduced = map(reducer, grouped.items())
            return [*reduced]
            return grouped

        def mapper3_inner(triples):
            ms= [x[:2] for x in triples if x[-1] == "M"]
            ns = [x[:2] for x in triples if x[-1] == "N"]
            combs = product(ms, ns)
            row = map(lambda x: ((x[0][0], x[1][0]), x[0][1] * x[1][1]) , combs)
            return [*row]

        def inner_mapper(row):
            num, row, mat = row
            mapped = map(lambda x: (*x, mat), enumerate(row))
            mapped = zip([num] * len(row), mapped)
            return [*mapped]

        def extender(x, y):
            x.extend(y)
            return x
        
        def group(data):
            dict_ = defaultdict(list)
            for k, v in data:
                dict_[k].append(v)
            return dict_

        def reducer(x):
            return (x[0], sum(x[1]))
        #Prepare data chunks
        chunks_m_idx = Problem4.get_split_rows_idx(self.M.T)
        m_chunks = Problem4.mapper_1(chunks_m_idx, self.M.T)
        chunks_n_idx = Problem4.get_split_rows_idx(self.N)
        n_chunks = Problem4.mapper_1(chunks_n_idx, self.N)
        
        #MapReduce 1
        m_mapped = map(partial(mapper_2, mat = "M"), m_chunks)
        n_mapped = map(partial(mapper_2, mat = "N"), n_chunks)
        reduced = reduce(extender, [*m_mapped, *n_mapped])
        grouped = group(reduced)
        #MapReduce 2
        chunks = Task4.get_split_dict(grouped)
        mapped = map(mapper3, chunks)
        reduced = reduce(extender, mapped)
        grouped = group(reduced)
        reduced = map(reducer, grouped.items())
        return [*reduced]
        

    @timeit
    def mapreduce_mp(self):
        #Prepare data chunks
        chunks_m_idx = Problem4.get_split_rows_idx(self.M.T)
        m_chunks = Problem4.mapper_1(chunks_m_idx, self.M.T)
        chunks_n_idx = Problem4.get_split_rows_idx(self.N)
        n_chunks = Problem4.mapper_1(chunks_n_idx, self.N)
        
        with Pool(processes = 8) as pl:
            res = pl.map_async(partial(Task4.mp_mapper1, mat = "M"), m_chunks)
            m_mapped = res.get()
            res = pl.map_async(partial(Task4.mp_mapper1, mat = "N"), n_chunks)
            n_mapped = res.get()
        reduced = reduce(Task4.extender, [*m_mapped, *n_mapped])
        grouped = Task4.group(reduced)
        chunks = Task4.get_split_dict(grouped)
        with Pool(processes = 8) as pl:
            res = pl.map_async(Task4.mapper2, chunks)
            mapped = res.get()
        reduced = reduce(Task4.extender, mapped)
        grouped = Task4.group(reduced)
        reduced = map(Task4.reducer, grouped.items())
        return [*reduced] 

        

class Problem3:
    @staticmethod
    def split_list2chunks(lst):
        #Set batch size and no. of batches
        l = len(lst)
        batches = int(l * .125)
        batches = batches if batches else 1
        batches_size = int(l / batches)
        #Spilt data to batches
        chunks = []
        for idx in range(batches - 1):
            chunks.append(lst[batches_size * idx : (idx + 1) * batches_size])
        chunks.append(lst[batches_size * (batches - 1) : ])
        return chunks

    def __init__(self, d, n, eps = 1000):
        """
        n   [int]   number of digits to compute
        d   [int]   starting digit
        """
        self.n = n
        self.d = d
        self.eps = eps
    
    def _pi_basic_(self, digits):
        def sj(digs, j):
            total = 0.0
            
            for idx in range(digs + self.eps):
                k = 8 * idx + j
                term = pow(16, digs - idx, k) if idx < digs else pow(16, digs - idx)
                total += term / k
            return total
        digits_ = digits - 1
        res = (
            4 * sj(digits_, 1) -
            2 * sj(digits_, 4) -
            sj(digits_, 5) -
            sj(digits_, 6)
        ) % 1
        return "%x" % int(res * 16)

    @timeit
    def triv_seq1(self):
        return "".join(
            self._pi_basic_(i) for i in range(self.d , self.d + self.n))

    @timeit
    def triv1_mp(self):
        vals = range(self.d, self.d + self.n)
        with Pool(processes = 8) as pl:
            res = pl.map_async(partial(Task3.pi_basic1, eps = self.eps), vals)
            res = res.get()
        return "".join(res)

    def _pi_basic2_(self, digits):
        def sj(j, d):
            series_1 = (pow(16, d - k, 8 * k + j) / (8 * k + j) for k in range(d + 1))
            series_1 = reduce(lambda x, y: (x + y) % 1, series_1)
            series_2 = sum(pow(16, d - k) / (8 * k + j) for k in range(d + 1, d + 2 + self.eps))
            
            return series_1 + series_2
        digits_ = digits - 1
        res = (
            4 * sj(1, digits_) -
            2 * sj(4, digits_) -
            sj(5, digits_) -
            sj(6, digits_)
        ) % 1
        return "%x" % int(res * 16)

    @timeit    
    def triv_seq2(self):
        return "".join(
            self._pi_basic2_(i) for i in range(self.d , self.d + self.n))
    
    @timeit
    def triv2_mp(self):
        vals = range(self.d, self.d + self.n)
        with Pool(processes = 8) as pl:
            res = pl.map_async(partial(Task3.pi_basic, eps = self.eps), vals)
            res = res.get()
        return "".join(res)

    @timeit
    def mapreduce_seq(self):
        inner_mapper1 = lambda k, d, j: pow(16, d - k, 8 * k + j) / (8 * k + j)
        inner_reducer1 = lambda x, y: (x + y) % 1
        inner_mapper2 = lambda k, d, j: pow(16, d - k) / (8 * k + j)
        inner_reducer2 = lambda x,y: x + y

        def mapper_sj(input_):
            d, j = input_
            vals1 = range(d + 1)
            vals2 = range(d + 1, self.eps + d + 2)
            #Series1            
            mapped1 = map(partial(inner_mapper1, d = d, j = j), vals1)
            reduced1 = reduce(inner_reducer1, mapped1)
            #Series2
            mapped2 = map(partial(inner_mapper2, d = d, j = j), vals2)
            reduced2 = reduce(inner_reducer2, mapped2)
            #sj
            reduced =  reduce(inner_reducer2, [reduced1, reduced2])
            return reduced
        
        def reducer_num1(x, y):
            return  (1, x[0] * x[1] + y[0] * y[1])

        def reducer_num2(x):
            #Fixing modulo for negative
            val = x[1] - int(x[1])
            if val < 0:
                val = 1 + val
            return "%x" % int(val * 16)
        

        def mapper(x):
            mapped = zip([x] * 4, (1, 4, 5, 6))
            mapped = map(mapper_sj, mapped)
            mapped = zip((4, -2, -1, -1), mapped)
            reduced = reduce(reducer_num1, mapped)
            #Interesting fact: after creating tuples  operation % 1 no longer works correctly
            reduced = reducer_num2(reduced)
            return reduced

        def reducer_outer(x, y):
            return "".join([x,y])

        def mapper_outter(chunks):
            mapped = map(mapper, chunks)
            reduced = reduce(reducer_outer, mapped)
            return reduced
        

        chunks = Problem3.split_list2chunks([*range(self.d -1, self.d + self.n -1)])
        mapped = map(mapper_outter, chunks)
        reduced = reduce(reducer_outer, mapped)
        return reduced
    
    @timeit
    def mapreduce_mp(self):
        chunks = Problem3.split_list2chunks([*range(self.d -1, self.d + self.n -1)])
        with Pool(processes = 8) as pl:
            res = pl.map_async(Task3.mapper_outter, chunks)
            mapped = res.get()
        reduced = reduce(Task3.reducer_outer, mapped)
        return reduced


    
#Task1
#Test accuracy solution 
# k,l = 20000, 20000
k,l = 5, 3
min_, max_ = 5, 200
# k,l = 200, 4
m = np.random.rand(k, l) * 500
# m = np.random.rand(k, l) * 1000 
n = np.random.rand(l, k) * 1000
p1 = Problem1(m, (min_, max_))
# print(p1.triv1_seq())
# print(p1.triv2_seq())
# print(p1.triv2_mp())
# print(p1.triv1_mp())
# print(p1.mapreduce_seq())
# print(p1.mapreduce_mp())

#Task2
# print("Task 2")
# p2 = Problem2(f'{DATA_DIR}/Guthenberg')
# print(p2.triv_a_seq())
# print(p2.triv_a_mp())
# print(p2.mapreduce_a_seq())
# print(p2.mapreduce_a_mp())

# print(p2.triv_b_seq())
# print(p2.triv_b_mp())

# x = p2.triv_b_seq()
# print(x == p2.triv_b_mp())
# print(x == p2.mapreduce_b_seq())
# print(x == p2.mapreduce_b_mp())


print("Task 3")
p3 = Problem3(int(1e3), 50)
# p3 = Problem3(int(1e3), 4)
x = p3.triv_seq1()
print(x)
print(p3.triv_seq2() == x)
# print(p3.triv_seq4())
print(p3.mapreduce_seq() == x)
print(p3.mapreduce_mp() == x)
print(p3.triv2_mp() == x)
print(p3.triv1_mp() == x)

# print(pi2(0,25))
    

print("Task 4")
k,l,i = 200, 400, 300
# k,l, i= 200, 60, 300
m = np.random.randint(-4, 200,(k, l))
n = np.random.randint(-4, 200, (l, i))

# p4 = Problem4(m, n)
# x = p4.triv_seq()
# print((Problem4.arr_from_dict(p4.mapreduce_seq()) == x).all())
# print((Problem4.arr_from_dict(p4.mapreduce_seq2()) == x).all())
# print((Problem4.arr_from_dict(p4.mapreduce_mp()) == x).all())


#   // Start at digit d and compute n digits
#   return function piBBP(d, n) {
#     // Seems to be the convention for including the leading 3
#     d -= 1;
#     // Shift n digits to the left of the radix point to obtain our final
#     // result as a integer
#     return Math.floor(
#       16 ** n * mod(4 * S(1, d) - 2 * S(4, d) - S(5, d) - S(6, d), 1)
#     );
#   };