from __future__ import division
from builtins import object
import numpy as np
from types import MethodType
from ..utils import bmath as bm
from ..utils.cucache import get_gpuarray
from pycuda.compiler import SourceModule
import pycuda.reduction as reduce
from pycuda.elementwise import ElementwiseKernel
from ..gpu.gpu_butils_wrap import gpu_diff, cugradient, gpu_copy_d2d, gpu_interp

from pycuda import gpuarray, driver as drv, tools

drv.init()
dev = drv.Device(bm.gpuId())


def gpu_slice(self, reduce=True):
    """
    Constant space slicing with a constant frame.
    """

    bm.slice(self.cut_left, self.cut_right, self.Beam, self)
    self.n_macroparticles_obj.invalidate_cpu()


def funcs_update(prof):
    prof.beam_profile_derivative = MethodType(gpu_beam_profile_derivative, prof)
    old_slice = prof._slice
    prof._slice = MethodType(gpu_slice, prof)
    prof.beam_spectrum_generation = MethodType(
        gpu_beam_spectrum_generation, prof)

    for i in range(len(prof.operations)):
        if (prof.operations[i] == old_slice):
            prof.operations[i] = prof._slice


def gpu_beam_spectrum_generation(self, n_sampling_fft):
    """
    Beam spectrum calculation
    """
    self.dev_beam_spectrum = bm.rfft(self.dev_n_macroparticles, n_sampling_fft)


def gpu_beam_profile_derivative(self, mode='gradient', caller_id=None):
    """
    The input is one of the three available methods for differentiating
    a function. The two outputs are the bin centres and the discrete
    derivative of the Beam profile respectively.*
    """
    x = self.bin_centers
    dist_centers = x[1] - x[0]
    if mode == 'filter1d':
        raise RuntimeError('filted1d mode is not supported in GPU.')
    elif mode == 'gradient':
        if (caller_id):
            derivative = get_gpuarray(
                (x.size, np.float64, caller_id, 'der'), True)
        else:
            derivative = gpuarray.zeros(x.size, dtype=np.float64)
        cugradient(np.float64(dist_centers), self.dev_n_macroparticles,
                   derivative, np.int32(x.size), block=(1024, 1, 1), grid=(16, 1, 1))
    elif mode == 'diff':
        if (caller_id):
            derivative = get_gpuarray(
                (x.size, np.float64, caller_id, 'der'), True)
        else:
            derivative = gpuarray.zeros(
                self.dev_n_macroparticles.size-1, np.float64)
        gpu_diff(self.dev_n_macroparticles, derivative, dist_centers)
        diffCenters = get_gpuarray(
            (self.dev_bin_centers.size-1, np.float64, caller_id, 'dC'))
        gpu_copy_d2d(diffCenters, self.dev_bin_centers, slice=slice(0, -1))

        diffCenters = diffCenters + dist_centers/2
        derivative = gpu_interp(self.dev_bin_centers, diffCenters, derivative)
    else:
        # ProfileDerivativeError
        raise RuntimeError('Option for derivative is not recognized.')

    return x, derivative
