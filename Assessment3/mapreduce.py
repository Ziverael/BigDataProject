###MODULES###
import multiprocessing
from enum import IntEnum
from functools import reduce
from typing import Callable, Any
# from dataclasses import dataclass


###CLASSES###
class TaskStatus(IntEnum):
    PROCESSING = 1
    FAILURE = 2
    DONE = 3
    READY = 4



# @dataclass
class Worker():
    FAIL_TRESHOLD = 3
    """
    Attrs
    -----
    status      TaskStatus
    task        Callable | None
    """
    def __init__(self):
        #Init status variables
        self._status = TaskStatus.READY
        self.task = None
        #Init thread
        #TODO

    @property
    def status(self):
        return self._status
    
    def _exec_(self, func : Callable, input, *args,  **kwargs):
        return func(input, *args, **kwargs)

    def __call__(self, func : Callable, input, *args,  **kwargs) -> Any:
        #TODO execute in thread
        fails = 0
        while fails < Worker.FAIL_TRESHOLD:
            try:
                self._status = TaskStatus.PROCESSING
                self.task = self._exec_(func, input, *args, **kwargs)
                self._status = TaskStatus.DONE
                return self._status
            except Exception as e:
                print("Exception: ",e)
                fails += 1
                self._status = TaskStatus.FAILURE
        return self._status
            

    def pop_result(self):
        """Send result and Clear result variable."""
        res = self.task
        self.task = None
        self._status = TaskStatus.READY
        return res 


class MapperWorker(Worker):
    def _exec_(self, func: Callable[..., Any], input, *args, **kwargs):
        return map(func, input)

class ReducerWorker(Worker):
    def _exec_(self, func: Callable[..., Any], input, *args, **kwargs):
        return reduce(func, input)

#Test
w = Worker()
fun = lambda x: 2 * x ** 2
x = [*range(100)]
x = 10
print(w(fun, x))
print(w.pop_result())
w= MapperWorker()
fun = lambda x: 2 * x ** 2
x = [*range(100)]
print(w(fun, x))
print(w.pop_result())
