import numpy as np
from pycuda import gpuarray
from ..utils import bmath as bm
from . import gpu_butils_wrap as gpu_utils

def fill(self, value):
    # from .gpu_butils_wrap import set_zero_int, set_zero_double, set_zero_complex, set_zero_float
    if (self.dtype in [np.int, np.int32]):
        gpu_utils.set_zero_int(self)
    elif (self.dtype in [np.float, np.float64]):
        gpu_utils.set_zero_double(self)
    elif self.dtype in [np.float32]:
        gpu_utils.set_zero_float(self)
    elif self.dtype in [np.complex64]:
        gpu_utils.set_zero_complex64(self)
    elif self.dtype in [np.complex128]:
        gpu_utils.set_zero_complex128(self)


   
gpuarray.GPUArray.fill = fill

dtype_to_bytes_dict = {np.float64: 64,
                       np.float32: 32, 
                       np.complex64: 64, 
                       np.complex128: 128, 
                       np.int32: 32}


class gpuarray_cache:
    """ this class is a software implemented cache for our gpuarrays, 
    in order to avoid unnecessary memory allocations in the gpu"""

    def __init__(self, capacity):
        self.gpuarray_dict = {}
        self.enabled = False
        self.capacity = 0
        self.curr_capacity = 0

    def add_array(self, key):

        self.gpuarray_dict[key] = gpuarray.empty(key[0], dtype=key[1])
        self.curr_capacity += dtype_to_bytes_dict[key[1]]*key[0]

    def get_array(self, key, zero_fills):
        if (self.enabled):
            if (not (key) in self.gpuarray_dict):
                self.add_array(key)
            else:
                if (zero_fills):
                    self.gpuarray_dict[key].fill(0)
            return self.gpuarray_dict[key]
        else:
            to_ret = gpuarray.empty(key[0], dtype=key[1])
            to_ret.fill(0)
            return to_ret
    # def free_space(self)

    def enable(self):
        self.enabled = True

    def disable(self):
        self.enabled = False


gpu_cache = gpuarray_cache(10000)
#gpu_cache_fft = gpuarray_cache(1000)


def get_gpuarray(key, zero_fills=False):
    return gpu_cache.get_array(key, zero_fills=zero_fills)


# def get_gpuarray_fft(key):
#     return gpu_cache_fft.get_array(key)


def enable_cache():
    gpu_cache.enable()


def disable_cache():
    gpu_cache.disable()
