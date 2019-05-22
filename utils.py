""" File that contains method useful not only for the img autoencoder, but
    for any related sub-project
"""

import numpy as np
import math
import psutil


def estimate_queue_size(shapes_list, numpy_types_list, mem_percentage):
    """ Method that calculates the queue size based on percentual of
    memory reserved to the queue. It receives the shape of data that will
    be put as each element of the queue. The references used are the numpy
    types and numpy arrays.
    """
    available_mem = psutil.virtual_memory().total
    mem_limit = available_mem * mem_percentage
    bytes_per_batch = 0
    for shape, dtype in zip(shapes_list, numpy_types_list):
        bytes_per_batch += np.empty(shape, dtype=dtype).nbytes
    queue_size = math.floor(mem_limit / bytes_per_batch)
    return queue_size
