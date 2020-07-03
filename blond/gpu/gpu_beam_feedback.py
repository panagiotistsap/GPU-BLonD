# from __future__ import division, print_function

# from builtins import range, object
import numpy as np
from ..toolbox.next_regular import next_regular
from ..utils import bmath as bm
from types import MethodType
from ..gpu.gpu_butils_wrap import gpu_copy_one, set_zero, triple_kernel,\
    first_kernel_x, second_kernel_x, third_kernel_x, indexing_double, indexing_int, sincos_mul_add, mul_d, gpu_trapz_2, sincos_mul_add_2
from ..gpu.cucache import get_gpuarray
from ..utils.bphysics_wrap import beam_phase as cpu_beamphase
import pycuda.cumath as cm
from pycuda import gpuarray
# from ..gpu import grid_size, block_size
# , driver as drv, tools
try:
    from pyprof import timing
except ImportError:
    from ..utils import profile_mock as timing


# drv.init()
# dev = drv.Device(bm.gpuId())


# str1 = drv.Stream()
# str2 = drv.Stream()

@timing.timeit(key='serial:beam_phase')
def gpu_beam_phase(self):
    '''
    *Beam phase measured at the main RF frequency and phase. The beam is 
    convolved with the window function of the band-pass filter of the 
    machine. The coefficients of sine and cosine components determine the 
    beam phase, projected to the range -Pi/2 to 3/2 Pi. Note that this beam
    phase is already w.r.t. the instantaneous RF phase.*
    '''

    if self.time_offset is None:
        coeff = bm.beam_phase(self.profile.dev_bin_centers,
                              self.profile.dev_n_macroparticles,
                              self.alpha, self.rf_station.dev_omega_rf,
                              self.rf_station.dev_phi_rf,
                              self.rf_station.counter[0],
                              self.profile.bin_size)

    else:
        indexes = self.profile.bin_centers >= self.time_offset
        indexing = np.where(indexes)

        gpu_indexing = gpuarray.to_gpu(indexing[0])
        bin_centers_indexed = get_gpuarray(
            (gpu_indexing.size, np.float64, 0, "bc"))
        indexing_double(bin_centers_indexed,
                        self.profile.dev_bin_centers, gpu_indexing)
        n_macroparticles_indexed = get_gpuarray(
            (gpu_indexing.size, np.float64, 0, "mc"))
        indexing_int(n_macroparticles_indexed,
                     self.profile.dev_n_macroparticles, gpu_indexing)

        time_offset = self.time_offset

        sin_result = get_gpuarray((gpu_indexing.size, np.float64, 0, "sin"))
        cos_result = get_gpuarray((gpu_indexing.size, np.float64, 0, "cos"))

        sincos_mul_add(bin_centers_indexed, omega_rf,
                       phi_rf, sin_result, cos_result)

        # Convolve with window function
        scoeff = np.trapz(np.exp(self.alpha*(self.profile.bin_centers[indexes] -
                                             time_offset)) *
                          np.sin(omega_rf*self.profile.bin_centers[indexes] +
                                 phi_rf) *
                          self.profile.n_macroparticles[indexes],
                          dx=self.profile.bin_size)
        ccoeff = np.trapz(np.exp(self.alpha*(self.profile.bin_centers[indexes] -
                                             time_offset)) *
                          np.cos(omega_rf*self.profile.bin_centers[indexes] +
                                 phi_rf) *
                          self.profile.n_macroparticles[indexes],
                          dx=self.profile.bin_size)
        coeff = scoeff/ccoeff

    self.phi_beam = np.arctan(coeff) + np.pi


def gpu_track(self):
    '''
    Calculate PL correction on main RF frequency depending on machine.
    Update the RF phase and frequency of the next turn for all systems.
    '''

    # Calculate PL correction on RF frequency
    # with region_timer('getattr','gpu') as rtn:
    getattr(self, self.machine)()

    # Update the RF frequency of all systems for the next turn
    counter = self.rf_station.counter[0] + 1

    # first_elementwise_kernel,
    triple_kernel(self.rf_station.dev_omega_rf, self.rf_station.dev_harmonic,
                  self.rf_station.dev_dphi_rf, self.rf_station.dev_omega_rf_d,
                  self.rf_station.dev_phi_rf, np.float64(np.pi),
                  np.float64(self.domega_rf), np.int32(
                      self.rf_station.n_turns+1),
                  np.int32(counter), np.int32(self.rf_station.n_rf), block=(32, 1, 1), grid=(1, 1, 1))

    # CPU CODE
    # self.rf_station.omega_rf[:, counter] += self.domega_rf * \
    #         self.rf_station.harmonic[:, counter] / \
    #         self.rf_station.harmonic[0, counter]

    # self.rf_station.dphi_rf += 2.*np.pi*self.rf_station.harmonic[:, counter] * \
    #         (self.rf_station.omega_rf[:, counter] -
    #         self.rf_station.omega_rf_d[:, counter]) / \
    #         self.rf_station.omega_rf_d[:, counter]

    # self.rf_station.phi_rf[:, counter] += self.rf_station.dphi_rf

    self.rf_station.omega_rf_obj.invalidate_cpu()
    self.rf_station.dphi_rf_obj.invalidate_cpu()
    self.rf_station.phi_rf_obj.invalidate_cpu()


@timing.timeit(key='serial:beam_phase_sharpWindow')
def gpu_beam_phase_sharpWindow(self):
    '''
    *Beam phase measured at the main RF frequency and phase. The beam is
    averaged over a window. The coefficients of sine and cosine components
    determine the beam phase, projected to the range -Pi/2 to 3/2 Pi.
    Note that this beam phase is already w.r.t. the instantaneous RF phase.*
    '''
    # Main RF frequency at the present turn
    turn = self.rf_station.counter[0]
    dummy = gpuarray.zeros(1, dtype=np.float64)
    gpu_copy_one(dummy, self.rf_station.dev_omega_rf,
                 self.rf_station.counter[0], slice=slice(0, 1))
    # self.rf_station.omega_rf[0, self.rf_station.counter[0]]
    omega_rf = dummy.get()[0]
    gpu_copy_one(dummy, self.rf_station.dev_phi_rf,
                 self.rf_station.counter[0], slice=slice(0, 1))
    # self.rf_station.phi_rf[0, self.rf_station.counter[0]]
    phi_rf = dummy.get()[0]
    needs_indexing = True

    if self.alpha != 0.0:
        indexes = np.logical_and((self.time_offset - np.pi / omega_rf)
                                 <= self.profile.bin_centers,
                                 self.profile.bin_centers
                                 <= (-1/self.alpha + self.time_offset -
                                     2 * np.pi / omega_rf))
        indexing = np.where(indexes)
    else:
        indexes = np.ones(self.profile.n_slices, dtype=bool)
        needs_indexing = False

    # GPU VERSION CODE

    # Convolve with window function
    # if (needs_indexing):
    #         gpu_indexing = gpuarray.to_gpu(indexing[0].astype(np.int32))
    #         bin_centers_indexed = get_gpuarray((gpu_indexing.size, np.float64, 0, "bc"))
    #         indexing_double(bin_centers_indexed, self.profile.dev_bin_centers, gpu_indexing)
    #         n_macroparticles_indexed = get_gpuarray((gpu_indexing.size, np.float64, 0, "mc"))
    #         indexing_int(n_macroparticles_indexed,self.profile.dev_n_macroparticles,gpu_indexing)
    # else:
    #         bin_centers_indexed = self.profile.dev_bin_centers
    #         n_macroparticles_indexed = self.profile.dev_n_macroparticles

    # np.testing.assert_allclose(n_macroparticles_indexed.get(), self.profile.n_macroparticles[indexes])
    # np.testing.assert_allclose(bin_centers_indexed.get(), self.profile.bin_centers[indexes])

    # sin_result = get_gpuarray((gpu_indexing.size,np.float64, 0, "sin")).fill(0)
    # cos_result = get_gpuarray((gpu_indexing.size,np.float64, 0, "cos")).fill(0)

    # sincos_mul_add(bin_centers_indexed, omega_rf, phi_rf, sin_result, cos_result)
    # mul_d(sin_result, n_macroparticles_indexed)
    # mul_d(cos_result, n_macroparticles_indexed)

    #np.testing.assert_allclose(sin_result.get(), np.sin(omega_rf*self.profile.bin_centers[indexes]+ phi_rf) * self.profile.n_macroparticles[indexes])
    #np.testing.assert_allclose(cos_result.get(), np.cos(omega_rf*self.profile.bin_centers[indexes]+ phi_rf) * self.profile.n_macroparticles[indexes])

    sin_cpu = np.sin(
        omega_rf*self.profile.bin_centers[indexes] + phi_rf) * self.profile.n_macroparticles[indexes]
    cos_cpu = np.cos(
        omega_rf*self.profile.bin_centers[indexes] + phi_rf) * self.profile.n_macroparticles[indexes]

    scoeff = np.trapz(sin_cpu, dx=self.profile.bin_size)
    ccoeff = np.trapz(cos_cpu, dx=self.profile.bin_size)

    # gpu_scoeff = gpu_trapz_2(sin_result, self.profile.bin_size, sin_result.size)
    # gpu_ccoeff = gpu_trapz_2(cos_result, self.profile.bin_size, sin_result.size)

    # gpu_phi_beam = np.arctan(gpu_scoeff/gpu_ccoeff) + np.pi

    # print(gpu_scoeff,scoeff)

    #np.testing.assert_allclose(gpu_scoeff/gpu_ccoeff, scoeff/ccoeff)

    # Project beam phase to (pi/2,3pi/2) range
    cpu_phi_beam = np.arctan(scoeff/ccoeff) + np.pi

    #np.testing.assert_allclose(gpu_phi_beam, cpu_phi_beam)

    self.phi_beam = cpu_phi_beam


def LHC(self):
    counter = self.rf_station.counter[0]
    dphi_rf = self.rf_station.dev_dphi_rf[0].get()

    self.beam_phase()
    self.phase_difference()
    # Frequency correction from phase loop and synchro loop
    self.domega_rf = - self.gain*self.dphi \
        - self.gain2*(self.lhc_y + self.lhc_a[counter]
                      * (dphi_rf + self.reference))

    # Update recursion variable
    self.lhc_y = (1 - self.lhc_t[counter])*self.lhc_y + \
        (1 - self.lhc_a[counter])*self.lhc_t[counter] * \
        (dphi_rf + self.reference)


def bf_funcs_update(obj):
    if (bm.gpuMode()):
        obj.track = MethodType(gpu_track, obj)
        obj.beam_phase = MethodType(gpu_beam_phase, obj)
        obj.LHC = MethodType(LHC, obj)
        obj.beam_phase_sharpWindow = MethodType(gpu_beam_phase_sharpWindow, obj)
