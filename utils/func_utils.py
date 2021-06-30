import logging


def func_with_parameters(func, *args, **kwargs):
    def partial_func(data):
        return func(data, *args, **kwargs)
    return partial_func

def delete_func(func):
    logging.warning("This function '{}' is deleted...".format(func.__name__))
    return None
