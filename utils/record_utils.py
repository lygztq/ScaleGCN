from typing import Callable
from time import process_time

def record_time(func:Callable, *args, **kwargs):
    start_time = process_time()
    ret = func(*args, **kwargs)
    end_time=  process_time()
    return ret, end_time - start_time

def update_avg(inc_value, value, count):
    return (count * value + inc_value) / (count + 1)
