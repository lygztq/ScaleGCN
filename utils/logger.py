import os
import logging
import pickle

class Logger(object):
    """Logger class for training process record

    Examples::

        >>> l = Logger("./output", logname="trainlog")
        >>> l.info("Epoch: {}, loss: {}".format(0, 0.32))
        >>> l.record("loss", 0.32, mode=Logger.APPEND, in_log=True)
        >>> l.dump_records()

    Args:
        output_dir (str): Directory path to the output files.
        logname (str, optional): Log file name. (default: "log")
        log_level (int, optional): Default log level. (default: :obj:`logging.INFO`)
        to_file (bool, optional): Output log to files. (default: :obj:`True`)
        to_std (bool, optional): Output log to std stream. (default: :obj:`True`)
    """
    REPLACE = 0
    APPEND  = 1
    MIN     = 2
    MAX     = 3
    SUM     = 4
    def __init__(self, output_dir, logname="log", log_level=logging.INFO, to_file=True, to_std=True):
        # output path
        self.output_dir = os.path.normpath(os.path.expanduser(output_dir))
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # logging setting
        self.logger = logging.Logger(logname)
        self.logger.setLevel(log_level)
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

        if to_file:
            path = os.path.join(self.output_dir, logname)
            file_ch = logging.FileHandler(filename=path, mode='w')
            file_ch.setLevel(log_level)
            file_ch.setFormatter(formatter)
            self.logger.addHandler(file_ch)
        if to_std:
            std_ch = logging.StreamHandler()
            std_ch.setLevel(log_level)
            std_ch.setFormatter(formatter)
            self.logger.addHandler(std_ch)
        
        self.records = {}

    def log(self, level, *msg):
        msg = " ".join(msg)
        self.logger.log(level, msg)

    def debug(self, *msg):
        self.log(logging.DEBUG, *msg)

    def info(self, *msg):
        self.log(logging.INFO, *msg)

    def warning(self, *msg):
        self.log(logging.WARNING, *msg)

    def error(self, *msg):
        self.log(logging.ERROR, *msg)
    
    def fatal(self, *msg):
        self.log(logging.FATAL, *msg)
    
    def record(self, key, value, mode=None, in_log=False):
        if mode is None: mode=Logger.REPLACE
        
        if key not in self.records or mode == Logger.REPLACE:
            self.records[key] = value
        elif mode == Logger.APPEND:
            if isinstance(self.records[key], list):
                self.records[key].append(value)
            else:
                self.records[key] = [self.records[key], value]
        elif mode == Logger.MAX:
            if isinstance(self.records[key], list):
                raise ValueError("Cannot record {0}={1} in MAX mode because {0} is a list".format(key, value))
            self.records[key] = max(self.records[key], value)
        elif mode == Logger.MIN:
            if isinstance(self.records[key], list):
                raise ValueError("Cannot record {0}={1} in MIN mode because {0} is a list".format(key, value))
            self.records[key] = min(self.records[key], value)
        elif mode == Logger.SUM:
            if isinstance(self.records[key], list):
                raise ValueError("Cannot record {0}={1} in SUM mode because {0} is a list".format(key, value))
            self.records[key] += value
        
        if in_log:
            logging.log(logging.INFO, "New Record: {}:{}".format(key, value))

    def dump_records(self):
        path = os.path.join(self.output_dir, "records.pkl")
        with open(path, 'wb') as f:
            pickle.dump(self.records, f)
