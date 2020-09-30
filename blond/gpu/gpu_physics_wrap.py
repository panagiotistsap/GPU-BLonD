'''
BLonD physics wrapper functions

@author Konstantinos Iliakis
@date 20.10.2017
'''

import ctypes as ct
import numpy as np
# from setup_cpp import libblondphysics as __lib
from .. import libblond as __lib
import numpy as np
import pycuda.cumath as cm
import traceback
from ..gpu.cucache import get_gpuarray
from ..gpu.gpu_butils_wrap import bm_phase_exp_times_scalar, bm_phase_mul_add, bm_sin_cos, d_multiply, d_multscalar
from pycuda import gpuarray
from ..utils.butils_wrap import trapz
from ..utils import bmath as bm
from ..gpu import block_size, grid_size
from .gpu_butils_wrap import set_zero_int
# drv.init()

my_gpu = bm.gpuDev()
ker = bm.getMod()


drift_simple = ker.get_function("drift_simple")
drift_legacy_0 = ker.get_function("drift_legacy_0")
drift_legacy_1 = ker.get_function("drift_legacy_1")
drift_legacy_2 = ker.get_function("drift_legacy_2")
drift_else = ker.get_function("drift_else")
kick_kernel = ker.get_function("simple_kick")
rvc = ker.get_function("rf_volt_comp")
hybrid_histogram = ker.get_function("hybrid_histogram")
sm_histogram = ker.get_function("sm_histogram")
gm_linear_interp_kick_help = ker.get_function("lik_only_gm_copy")
gm_linear_interp_kick_comp = ker.get_function("lik_only_gm_comp")
gm_linear_interp_kick_drift_comp = ker.get_function("lik_drift_only_gm_comp")
halve_edges = ker.get_function("halve_edges")
beam_phase_v2 = ker.get_function("beam_phase_v2")
beam_phase_sum = ker.get_function("beam_phase_sum")

synch_rad = ker.get_function("synchrotron_radiation")
synch_rad_full = ker.get_function("synchrotron_radiation_full")

# beam phase kernels


def gpu_rf_volt_comp(dev_voltage, dev_omega_rf, dev_phi_rf, dev_bin_centers, dev_rf_voltage, f_rf=0):
    rvc(dev_voltage, dev_omega_rf, dev_phi_rf, dev_bin_centers,
        np.int32(dev_voltage.size), np.int32(
            dev_bin_centers.size), np.int32(f_rf), dev_rf_voltage,
        block=block_size, grid=grid_size, shared=3*dev_voltage.size*64, time_kernel=True)


def gpu_kick(dev_voltage, dev_omega_rf, dev_phi_rf, charge, n_rf, acceleration_kick, beam):
    dev_voltage_kick = get_gpuarray((dev_voltage.size, bm.precision.real_t, 0, 'vK'))

    #dev_voltage_kick  = np.float64(charge)*dev_voltage
    d_multscalar(dev_voltage_kick, dev_voltage, charge)

    kick_kernel(beam.dev_dt,
                beam.dev_dE,
                np.int32(n_rf),
                dev_voltage_kick,
                dev_omega_rf,
                dev_phi_rf,
                np.int32(beam.dev_dt.size),
                bm.precision.c_real_t(acceleration_kick),
                block=block_size, grid=grid_size, time_kernel=True)
    beam.dE_obj.invalidate_cpu()


def gpu_drift(solver_utf8, t_rev, length_ratio, alpha_order, eta_0,
              eta_1, eta_2, alpha_0, alpha_1, alpha_2, beta, energy, beam):
    solver = solver_utf8.decode('utf-8')
    T = bm.precision.c_real_t(t_rev)*bm.precision.c_real_t(length_ratio)
    n_macroparticles = len(beam.dev_dt)
    if (solver == "simple"):
        ##### simple solver #####
        # coeff =  np.float64(eta_0) / (np.float64(beta)* np.float64(beta)* np.float64(energy))
        # beam.dev_dt += T*coeff*beam.dev_dE
        drift_simple(beam.dev_dt, beam.dev_dE,
                     (t_rev), bm.precision.c_real_t(length_ratio),
                     (eta_0), (beta),
                     (energy), np.int32(n_macroparticles),
                     grid=grid_size, block=block_size, time_kernel=True)

    elif (solver == "legacy"):
        ##### legacy solver #####

        coeff = 1. / (beta * beta * energy)
        eta0 = eta_0 * coeff
        eta1 = eta_1 * coeff * coeff
        eta2 = eta_2 * coeff * coeff * coeff
        if (alpha_order == 0):
            drift_legacy_0(beam.dev_dt, beam.dev_dE,
                           bm.precision.c_real_t(T), bm.precision.c_real_t(eta0),
                           np.int32(n_macroparticles),
                           block=block_size, grid=grid_size, time_kernel=True)
        elif (alpha_order == 1):
            drift_legacy_1(beam.dev_dt, beam.dev_dE,
                           bm.precision.c_real_t(T), bm.precision.c_real_t(eta0),
                           bm.precision.c_real_t(eta1), np.int32(n_macroparticles),
                           block=block_size, grid=grid_size, time_kernel=True)
        else:
            drift_legacy_2(beam.dev_dt, beam.dev_dE,
                           bm.precision.c_real_t(T), bm.precision.c_real_t(eta0),
                           bm.precision.c_real_t(eta1), bm.precision.c_real_t(eta2),
                           np.int32(n_macroparticles),
                           block=block_size, grid=grid_size, time_kernel=True)

    else:
        ##### other solver  #####
        invbetasq = 1. / (beta * beta)
        invenesq = 1. / (energy * energy)
        drift_else(beam.dev_dt, beam.dev_dE,
                   bm.precision.c_real_t(invbetasq), bm.precision.c_real_t(invenesq),
                   bm.precision.c_real_t(T), bm.precision.c_real_t(alpha_0), bm.precision.c_real_t(alpha_1),
                   bm.precision.c_real_t(alpha_2), bm.precision.c_real_t(energy),
                   np.int32(n_macroparticles),
                   block=block_size, grid=grid_size, time_kernel=True)
    beam.dt_obj.invalidate_cpu()


def gpu_linear_interp_kick(dev_voltage,
                           dev_bin_centers, charge,
                           acceleration_kick, beam=None):
    macros = beam.dev_dt.size
    slices = dev_bin_centers.size

    dev_voltageKick = get_gpuarray((slices-1, bm.precision.real_t, 0, 'vK'))
    dev_factor = get_gpuarray((slices-1, bm.precision.real_t, 0, 'dF'))

    gm_linear_interp_kick_help(beam.dev_dt,
                               beam.dev_dE,
                               dev_voltage,
                               dev_bin_centers,
                               bm.precision.c_real_t(charge),
                               np.int32(slices),
                               np.int32(macros),
                               bm.precision.c_real_t(acceleration_kick),
                               dev_voltageKick,
                               dev_factor,
                               grid=grid_size, block=block_size,
                               time_kernel=True)
    gm_linear_interp_kick_comp(beam.dev_dt,
                               beam.dev_dE,
                               dev_voltage,
                               dev_bin_centers,
                               bm.precision.c_real_t(charge),
                               np.int32(slices),
                               np.int32(macros),
                               bm.precision.c_real_t(acceleration_kick),
                               dev_voltageKick,
                               dev_factor,
                               grid=grid_size, block=block_size,
                               time_kernel=True)
    beam.dE_obj.invalidate_cpu()


def gpu_linear_interp_kick_drift(dev_voltage,
                                 dev_bin_centers, charge,
                                 acceleration_kick, 
                                 T0, length_ratio, eta0, beta, energy, 
                                 beam=None):
    macros = beam.dev_dt.size
    slices = dev_bin_centers.size

    dev_voltageKick = get_gpuarray((slices-1, bm.precision.real_t, 0, 'vK'))
    dev_factor = get_gpuarray((slices-1, bm.precision.real_t, 0, 'dF'))

    gm_linear_interp_kick_help(beam.dev_dt,
                               beam.dev_dE,
                               dev_voltage,
                               dev_bin_centers,
                               bm.precision.c_real_t(charge),
                               np.int32(slices),
                               np.int32(macros),
                               bm.precision.c_real_t(acceleration_kick),
                               dev_voltageKick,
                               dev_factor,
                               grid=grid_size, block=block_size,
                               time_kernel=True)
    gm_linear_interp_kick_drift_comp(beam.dev_dt,
                                     beam.dev_dE,
                                     dev_voltage,
                                     dev_bin_centers,
                                     bm.precision.c_real_t(charge),
                                     np.int32(slices),
                                     np.int32(macros),
                                     bm.precision.c_real_t(acceleration_kick),
                                     dev_voltageKick,
                                     dev_factor,
                                     bm.precision.c_real_t(T0),
                                     bm.precision.c_real_t(length_ratio),
                                     bm.precision.c_real_t(eta0),
                                     bm.precision.c_real_t(beta),
                                     bm.precision.c_real_t(energy),
                                     grid=grid_size, block=block_size,
                                     time_kernel=True)
    beam.dE_obj.invalidate_cpu()
    beam.dt_obj.invalidate_cpu()


def gpu_slice(cut_left, cut_right, beam, profile):

    n_slices = profile.dev_n_macroparticles.size
    set_zero_int(profile.dev_n_macroparticles)
    # find optimal block and grid parameters
    # max_num_of_blocks_per_sm = my_gpu.MAX_SHARED_MEMORY_PER_MULTIPROCESSOR // min(4*n_slices, my_gpu.MAX_SHARED_MEMORY_PER_BLOCK)
    # threads_per_block = max(min(my_gpu.MAX_THREADS_PER_MULTIPROCESSOR // max_num_of_blocks_per_sm, my_gpu.MAX_THREADS_PER_BLOCK),64)
    # num_of_blocks_per_sm = max_num_of_blocks_per_sm
    # threads_per_block = max(filter(lambda x: x <= threads_per_block , possible_threads))
    # num_of_blocks_per_sm = min(my_gpu.MAX_THREADS_PER_MULTIPROCESSOR // threads_per_block , max_num_of_blocks_per_sm)

    # while (threads_per_block*num_of_blocks_per_sm < my_gpu.MAX_THREADS_PER_MULTIPROCESSOR):
    #     threads_per_block = threads_per_block + 32
    #     num_of_blocks_per_sm = my_gpu.MAX_THREADS_PER_MULTIPROCESSOR // threads_per_block
    # #print(n_slices,"-",threads_per_block, num_of_blocks_per_sm)
    # grid = (my_gpu.MULTIPROCESSOR_COUNT * num_of_blocks_per_sm,1,1)
    # block  = (threads_per_block , 1, 1)
    # print(threads_per_block,num_of_blocks_per_sm)
    # print(my_gpu.MAX_SHARED_MEMORY_PER_BLOCK)
    if (4*n_slices < my_gpu.MAX_SHARED_MEMORY_PER_BLOCK):
        sm_histogram(beam.dev_dt, profile.dev_n_macroparticles, bm.precision.c_real_t(cut_left),
                     bm.precision.c_real_t(cut_right), np.uint32(n_slices),
                     np.uint32(beam.dev_dt.size),
                     grid=grid_size, block=block_size, shared=4*n_slices, time_kernel=True)
    else:
        hybrid_histogram(beam.dev_dt, profile.dev_n_macroparticles, bm.precision.c_real_t(cut_left),
                         bm.precision.c_real_t(cut_right), np.uint32(n_slices),
                         np.uint32(beam.dev_dt.size), np.int32(
                             my_gpu.MAX_SHARED_MEMORY_PER_BLOCK/4),
                         grid=grid_size, block=block_size, shared=my_gpu.MAX_SHARED_MEMORY_PER_BLOCK, time_kernel=True)
    profile.n_macroparticles_obj.invalidate_cpu()
    return profile.dev_n_macroparticles


def gpu_synchrotron_radiation(dE, U0, n_kicks, tau_z):
    synch_rad(dE, bm.precision.c_real_t(U0), np.int32(dE.size), bm.precision.c_real_t(tau_z),
              np.int32(n_kicks), block=block_size, grid=(my_gpu.MULTIPROCESSOR_COUNT, 1, 1))


def gpu_synchrotron_radiation_full(dE, U0, n_kicks, tau_z, sigma_dE, energy):
    # print("Entering")
    synch_rad_full(dE, bm.precision.c_real_t(U0), np.int32(dE.size), bm.precision.c_real_t(sigma_dE), bm.precision.c_real_t(energy),
                   np.int32(n_kicks), np.int32(1), block=block_size, grid=grid_size)


def gpu_beam_phase(bin_centers, profile, alpha, omega_rf, phi_rf, ind, bin_size):

    n_bins = bin_centers.size

    array1 = get_gpuarray((bin_centers.size, bm.precision.real_t, 0, 'ar1'))
    array2 = get_gpuarray((bin_centers.size, bm.precision.real_t, 0, 'ar2'))

    dev_scoeff = get_gpuarray((1, bm.precision.real_t, 0, 'sc'))
    dev_coeff = get_gpuarray((1, bm.precision.real_t, 0, 'co'))

    beam_phase_v2(bin_centers,
                  profile, bm.precision.c_real_t(alpha), omega_rf, phi_rf, np.int32(
                      ind), bm.precision.c_real_t(bin_size),
                  array1, array2, np.int32(n_bins),
                  block=block_size)  # , grid=grid_size)

    beam_phase_sum(array1, array2, dev_scoeff, dev_coeff, np.int32(
        n_bins), block=block_size, grid=(1, 1, 1), time_kernel=True)
    to_ret = dev_scoeff[0].get()

    return to_ret
