import threading
from typing import Dict

class TimeAndCount(object):
    def __init__(self, time=0, count=0):
        self.time: float = time
        self.count: int = count

    def add_time(self, new_time):
        self.time += new_time
        self.count += 1

    @property
    def mean_time(self):
        return self.time / self.count

# Singleton
class Profiler(object):
    _instance_lock = threading.Lock()
    def __new__(cls, *args, **kwargs):
        with cls._instance_lock:
            if not hasattr(cls, "_instance"):
                cls._instance = object.__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        self.time_stats : Dict[str, TimeAndCount] = {}

    def __getitem__(self, key):
        item = self.time_stats.get(key, None)
        if item is None:
            item = TimeAndCount()
            self.time_stats[key] = item
        return item
    
    def dump(self):
        res = {}
        for k, v in self.time_stats.items():
            res[k] = v.mean_time
        return res
