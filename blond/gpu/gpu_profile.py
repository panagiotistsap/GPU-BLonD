# from __future__ import division
# from builtins import object
import numpy as np
from types import MethodType
from ..utils import bmath as bm
from ..gpu.cucache import get_gpuarray
from ..gpu.gpu_butils_wrap import gpu_diff, cugradient, gpu_copy_d2d, gpu_interp

from pycuda import gpuarray
from ..gpu import grid_size, block_size
try:
    from pyprof import timing
except ImportError:
    from ..utils import profile_mock as timing

# drv.init()
# dev = drv.Device(bm.gpuId())

def funcs_update(prof):
        prof.beam_profile_derivative = MethodType(gpu_beam_profile_derivative, prof)
        old_slice = prof._slice
        prof._slice = MethodType(gpu_slice, prof)
        prof.beam_spectrum_generation = MethodType(
            gpu_beam_spectrum_generation, prof)

        for i in range(len(prof.operations)):
            if (prof.operations[i] == old_slice):
                prof.operations[i] = prof._slice

from ..beam.profile import Profile

class gpu_Profile(Profile):

    ## bin_centers

    @property
    def bin_centers(self):
        return self.bin_centers_obj.my_array

    @bin_centers.setter
    def bin_centers(self, value):
        self.bin_centers_obj.my_array = value


    @property
    def dev_bin_centers(self):
        return self.bin_centers_obj.dev_my_array


    @dev_bin_centers.setter
    def dev_bin_centers(self, value):
        self.bin_centers_obj.dev_my_array = value

    ## n_macroparticles

    @property
    def n_macroparticles(self):
        return self.n_macroparticles_obj.my_array

    @n_macroparticles.setter
    def n_macroparticles(self, value):
        self.n_macroparticles_obj.my_array = value


    @property
    def dev_n_macroparticles(self):
        return self.n_macroparticles_obj.dev_my_array


    @dev_n_macroparticles.setter
    def dev_n_macroparticles(self, value):
        self.n_macroparticles_obj.dev_my_array = value

    ## beam_spectrum

    @property
    def beam_spectrum(self):
        return self.beam_spectrum_obj.my_array

    @beam_spectrum.setter
    def beam_spectrum(self, value):
        self.beam_spectrum_obj.my_array = value


    @property
    def dev_beam_spectrum(self):
        return self.beam_spectrum_obj.dev_my_array


    @dev_beam_spectrum.setter
    def dev_beam_spectrum(self, value):
        self.beam_spectrum_obj.dev_my_array = value

    ## beam_spectrum_freq

    @property
    def beam_spectrum_freq(self):
        return self.beam_spectrum_freq_obj.my_array

    @beam_spectrum_freq.setter
    def beam_spectrum_freq(self, value):
        self.beam_spectrum_freq_obj.my_array = value


    @property
    def dev_beam_spectrum_freq(self):
        return self.beam_spectrum_freq_obj.dev_my_array


    @dev_beam_spectrum_freq.setter
    def dev_beam_spectrum_freq(self, value):
        self.beam_spectrum_freq_obj.dev_my_array = value

    @timing.timeit(key='comp:histo')
    def _slice(self, reduce=True):
        """
        Constant space slicing with a constant frame.
        """

        bm.slice(self.cut_left, self.cut_right, self.Beam, self)
        self.n_macroparticles_obj.invalidate_cpu()
   

    @timing.timeit(key='serial:beam_spectrum_gen')
    def beam_spectrum_generation(self, n_sampling_fft):
        """
        Beam spectrum calculation
        """
        temp = bm.rfft(self.dev_n_macroparticles, n_sampling_fft)
        self.dev_beam_spectrum = temp


    def beam_profile_derivative(self, mode='gradient', caller_id=None):
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
                derivative = gpuarray.empty(x.size, dtype=np.float64)
            cugradient(np.float64(dist_centers), self.dev_n_macroparticles,
                    derivative, np.int32(x.size), block=block_size, grid=(16, 1, 1))
        elif mode == 'diff':
            if (caller_id):
                derivative = get_gpuarray(
                    (x.size, np.float64, caller_id, 'der'), True)
            else:
                derivative = gpuarray.empty(
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


    def reduce_histo(self, dtype=np.uint32):
        if not bm.mpiMode():
            raise RuntimeError(
                'ERROR: Cannot use this routine unless in MPI Mode')

        from ..utils.mpi_config import worker
        worker.sync()
        if self.Beam.is_splitted:

            with timing.timed_region('serial:conversion'):
                # with mpiprof.traced_region('serial:conversion'):
                my_n_macroparticles = self.n_macroparticles.astype(
                    np.uint32, order='C')

            worker.allreduce(my_n_macroparticles, dtype=np.uint32, operator='custom_sum')

            with timing.timed_region('serial:conversion'):
                # with mpiprof.traced_region('serial:conversion'):
                self.n_macroparticles = my_n_macroparticles.astype(dtype=np.float64, order='C', copy=False)


    @timing.timeit(key='serial:scale_histo')
    # @mpiprof.traceit(key='serial:scale_histo')
    def scale_histo(self):
        if not bm.mpiMode():
            raise RuntimeError(
                'ERROR: Cannot use this routine unless in MPI Mode')

        from ..utils.mpi_config import worker
        if self.Beam.is_splitted:
            bm.mul(self.n_macroparticles, worker.workers, self.n_macroparticles)
            self.n_macroparticles_obj.invalidate_gpu()