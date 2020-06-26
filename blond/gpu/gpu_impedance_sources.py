from __future__ import division, print_function
from builtins import range, object
import numpy as np
from scipy.constants import e
from ..utils import bmath as bm
from types import MethodType
from ..utils.cucache import get_gpuarray
from ..gpu.gpu_butils_wrap import gpu_copy_d2d,set_zero, increase_by_value,\
                                        increase_by_value, add_array, complex_mul, gpu_mul, gpu_interp

import pycuda.reduction as reduce
import pycuda.cumath as cm
from pycuda import gpuarray, driver as drv, tools


drv.init()
dev = drv.Device(bm.gpu_num)