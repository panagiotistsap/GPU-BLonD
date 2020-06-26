import numpy as np
from pycuda.compiler import SourceModule
from pycuda import gpuarray, driver as drv, tools
import atexit
from ..utils import bmath as bm

drv.init()
#assert ( driver.Device.count() >= 1)
dev = drv.Device(bm.gpuId())
# ctx = dev.make_context()
# atexit.register(ctx.pop)

dtype_to_bytes_dict = {np.float64: 64,
                       np.float32: 32, np.complex128: 128, np.int32: 32}


class gpuarray_cache:
    """ this class is a software implemented cache for our gpuarrays, 
    in order to avoid unnecessary memory allocations in the gpu"""

    def __init__(self, capacity):
        self.gpuarray_dict = {}
        self.enabled = False
        self.capacity = 0
        self.curr_capacity = 0

    def add_array(self, key):
        # if (not dtype_to_bytes_dict[key[1]]*key[0] + self.curr_capacity <= self.capacity):
        #    print("need to free array")
        #    pass
        self.gpuarray_dict[key] = gpuarray.zeros(key[0], dtype=key[1])
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
            return gpuarray.zeros(key[0], dtype=key[1])
    # def free_space(self)

    def enable(self):
        self.enabled = True

    def disable(self):
        self.enabled = False


gpu_cache = gpuarray_cache(10000)
#gpu_cache_fft = gpuarray_cache(1000)


def get_gpuarray(key, zero_fills=False):
    return gpu_cache.get_array(key, zero_fills=zero_fills)


def get_gpuarray_fft(key):
    return gpu_cache_fft.get_array(key)


def enable_cache():
    gpu_cache.enable()


def disable_cache():
    gpu_cache.disable()
