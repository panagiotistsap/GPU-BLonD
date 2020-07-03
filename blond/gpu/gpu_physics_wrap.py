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
from pycuda.compiler import SourceModule
import pycuda.cumath as cm
import traceback
from pycuda.reduction import ReductionKernel
from skcuda.misc import diff as cuda_diff
from ..gpu.cucache import get_gpuarray
from ..gpu.gpu_butils_wrap import ElementwiseKernel
from pycuda import gpuarray
# , driver as drv, tools
import atexit
from ..utils.butils_wrap import trapz
from ..utils import bmath as bm
from ..gpu import block_size, grid_size

# drv.init()
my_gpu = bm.gpuDev()

try:

    ker_with_atomicAdd = SourceModule("""
            __device__ double atomicAdd(double* address, double val)
        {
            unsigned long long int* address_as_ull = (unsigned long long int*)address;
            unsigned long long int old = *address_as_ull, assumed;
            do {
                assumed = old;
                old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val + __longlong_as_double(assumed)));
            } while (assumed != old);
            return __longlong_as_double(old);
        }

        __global__ void beam_phase_sum(
            const double *ar1,
            const double *ar2,
            double *scoeff,
            double *coeff,
            int n_bins)
        {   
            int tid = threadIdx.x + blockDim.x * blockIdx.x;

            if (tid==0){
                scoeff[0]=0;
                coeff[0] =0;
            }
            double my_sum_1 = 0;
            double my_sum_2 = 0;
            if (tid==0){
                my_sum_1 += ar1[0]/2+ar1[n_bins-1]/2;
                my_sum_2 += ar2[0]/2+ar2[n_bins-1]/2;
            }
            for (int i = tid+1; i<n_bins-1; i+=gridDim.x*blockDim.x){
                my_sum_1 += ar1[i];
                my_sum_2 += ar2[i];
            }
            atomicAdd(&(scoeff[0]),my_sum_1);
            atomicAdd(&(coeff[0]),my_sum_2);
            __syncthreads();
            if (tid==0)
                scoeff[0]=scoeff[0]/coeff[0];
            
        }

        """)
except:
    pass

try:

    ker_without_atomicAdd = SourceModule("""
        __global__ void beam_phase_sum(
            const double *ar1,
            const double *ar2,
            double *scoeff,
            double *coeff,
            int n_bins)
        {   
            int tid = threadIdx.x + blockDim.x * blockIdx.x;

            if (tid==0){
                scoeff[0]=0;
                coeff[0] =0;
            }
            double my_sum_1 = 0;
            double my_sum_2 = 0;
            if (tid==0){
                my_sum_1 += ar1[0]/2+ar1[n_bins-1]/2;
                my_sum_2 += ar2[0]/2+ar2[n_bins-1]/2;
            }
            for (int i = tid+1; i<n_bins-1; i+=gridDim.x*blockDim.x){
                my_sum_1 += ar1[i];
                my_sum_2 += ar2[i];
            }
            atomicAdd(&(scoeff[0]),my_sum_1);
            atomicAdd(&(coeff[0]),my_sum_2);
            __syncthreads();
            if (tid==0)
                scoeff[0]=scoeff[0]/coeff[0];
            
        }

        """)
except:
    pass

ker = SourceModule("""

    

    __global__ void halve_edges(double *my_array, int size){
        //__shared__ my_sum;
        int tid = threadIdx.x;
        if (tid==0){
            my_array[0] = my_array[0]/2.;
        }
        if (tid==32){
            my_array[size-1] = my_array[size-1]/2.;
        }
    }

    __global__ void simple_kick(
            const double  *beam_dt, 
            double        *beam_dE,
            const int n_rf, 
            const double  *voltage, 
            const double  *omega_RF, 
            const double  *phi_RF,
            const int n_macroparticles,
            const double acc_kick
            )
    {
        int tid = threadIdx.x + blockDim.x * blockIdx.x;
        double my_beam_dt;
        double sin_res;
        double dummy;
        for (int i=tid; i<n_macroparticles; i += blockDim.x*gridDim.x){
            my_beam_dt = beam_dt[i];
            for (int j=0; j<n_rf; j++){
                sincos(omega_RF[j]*my_beam_dt + phi_RF[j], &sin_res, &dummy);
                beam_dE[i] += voltage[j] * sin_res;
                }
            beam_dE[i] += acc_kick;
        }
    }
    

    __global__ void rf_volt_comp(  double *voltage,
                                    double *omega_rf,
                                    double *phi_rf,
                                    double *bin_centers,
                                    int n_rf,
                                    int n_bins,
                                    int f_rf,
                                    double *rf_voltage)
    {
        int tid = threadIdx.x + blockDim.x * blockIdx.x;
        extern __shared__ double s[];
        if (tid==0)
            for (int j=0; j<n_rf; j++){
                s[j] = voltage[j];
                s[j+n_rf] = omega_rf[j];
                s[j+2*n_rf] = phi_rf[j];
            }
        __syncthreads();
        for (int i = tid; i<n_bins; i+=blockDim.x*gridDim.x){
            for (int j=0; j<n_rf; j++)
                rf_voltage[i] = s[j]*sin(s[j+n_rf]*bin_centers[i] + s[j+2*n_rf]);
            
        }
    }

    __global__ void drift_simple(double *beam_dt,
                        const double *beam_dE,
                        const double T0, const double length_ratio,
                        const double eta0, const double beta,
                        const double energy,
                        const int n_macroparticles) 
    {
        double T = T0*length_ratio;
        double coeff = T*(eta0 / (beta*beta*energy));
        int tid = threadIdx.x + blockDim.x * blockIdx.x;
        for (int i = tid; i<n_macroparticles; i+=gridDim.x*blockDim.x)
            beam_dt[i] += coeff * beam_dE[i];
        
    }

    __global__ void drift_legacy_0(double *beam_dt,
                                const double *beam_dE,
                                const double T, 
                                const double eta0,
                                const int n_macroparticles)
    {
        int tid = threadIdx.x + blockDim.x * blockIdx.x;
        for (int i = tid; i < n_macroparticles; i+= blockDim.x*gridDim.x)
            beam_dt[i] += T * (1. / (1. - eta0 * beam_dE[i]) - 1.);
    }

    __global__ void drift_legacy_1(double *beam_dt,
                                const double *beam_dE,
                                const double T,
                                const double eta0, const double eta1,
                                const int n_macroparticles) 
    {
        int tid = threadIdx.x + blockDim.x * blockIdx.x;
        for (int i = tid; i < n_macroparticles; i+= blockDim.x*gridDim.x)
            beam_dt[i] += T * (1. / (1. - eta0 * beam_dE[i]
                                        - eta1 * beam_dE[i] * beam_dE[i]) - 1.);
    }

    __global__ void drift_legacy_2(double *beam_dt,
                                const double *beam_dE,
                                const double T,
                                const double eta0, const double eta1, const double eta2,
                                const int n_macroparticles) 
    {
        int tid = threadIdx.x + blockDim.x * blockIdx.x;
        double my_beam_dE;
        for (int i = tid; i < n_macroparticles; i+= blockDim.x*gridDim.x){
            my_beam_dE = beam_dE[i];
            beam_dt[i] += T * (1. / (1. - eta0 * my_beam_dE
                                        - eta1 * my_beam_dE * my_beam_dE
                                        - eta2 * my_beam_dE * my_beam_dE * my_beam_dE) - 1.);
        }
    }


    __global__ void drift_else(double *beam_dt,
                                const double *beam_dE,
                                const double invbetasq,
                                const double invenesq,
                                const double T,
                                const double alpha_zero,
                                const double alpha_one,
                                const double alpha_two,
                                const double energy,
                                const int n_macroparticles) 
    {
        double beam_delta;
        int tid = threadIdx.x + blockDim.x * blockIdx.x;
        for (int i = tid; i < n_macroparticles; i+= blockDim.x*gridDim.x){
            beam_delta = sqrt(1. + invbetasq *
                                    (beam_dE[i] * beam_dE[i] * invenesq + 2.*beam_dE[i] / energy)) - 1.;

            beam_dt[i] += T * (
                                (1. + alpha_zero * beam_delta +
                                alpha_one * (beam_delta * beam_delta) +
                                alpha_two * (beam_delta * beam_delta * beam_delta)) *
                                (1. + beam_dE[i] / energy) / (1. + beam_delta) - 1.);
        } 
    }   

    __global__ void histogram(double * input,
        int * output, const double cut_left,
        const double cut_right, const int n_slices,
        const int n_macroparticles)
    {
        int tid = threadIdx.x + blockDim.x * blockIdx.x;
        int target_bin;
        double const inv_bin_width = n_slices/(cut_right-cut_left);
        for (int i=tid; i<n_macroparticles; i=i+blockDim.x*gridDim.x){
            target_bin = floor((input[i] - cut_left) * inv_bin_width);
            if(target_bin<0 || target_bin>=n_slices)
                continue;
            atomicAdd(&(output[target_bin]),1);
        }
    }

    __global__ void hybrid_histogram(double * input,
                int * output, const double cut_left,
                const double cut_right, const unsigned int n_slices,
                const int n_macroparticles, const int capacity)
    {
        extern __shared__ int block_hist[];
        //reset shared memory
        for (int i=threadIdx.x; i<capacity; i+=blockDim.x)
            block_hist[i]=0;
        __syncthreads();
        int const tid = threadIdx.x + blockDim.x*blockIdx.x;
        int target_bin;
        double const inv_bin_width = n_slices/(cut_right-cut_left);

        const int low_tbin = (n_slices / 2) - (capacity/2);
        const int high_tbin = low_tbin + capacity;


        for (int i=tid; i<n_macroparticles; i+=blockDim.x*gridDim.x){
            target_bin = floor((input[i] - cut_left) * inv_bin_width);
            if (target_bin<0 || target_bin>=n_slices)
                continue;
            if (target_bin >= low_tbin && target_bin<high_tbin)
                atomicAdd(&(block_hist[target_bin-low_tbin]),1);
            else
                atomicAdd(&(output[target_bin]),1);

        }
        __syncthreads();
        for (int i=threadIdx.x; i<capacity; i+=blockDim.x)
                atomicAdd(&output[low_tbin+i],block_hist[i]);

    }

    __global__ void sm_histogram(double * input,
            int * output, const double cut_left,
            const double cut_right, const unsigned int n_slices,
            const int n_macroparticles)
    {
        extern __shared__ int block_hist[];
        for (int i=threadIdx.x; i<n_slices; i+=blockDim.x)
            block_hist[i]=0;
        __syncthreads();
        int const tid = threadIdx.x + blockDim.x*blockIdx.x;
        int target_bin;
        double const inv_bin_width = n_slices/(cut_right-cut_left);
        for (int i=tid; i<n_macroparticles; i+=blockDim.x*gridDim.x){
            target_bin = floor((input[i] - cut_left) * inv_bin_width);
            if (target_bin<0 || target_bin>=n_slices)
                continue;
            atomicAdd(&(block_hist[target_bin]),1);
        }
        __syncthreads();
        for (int i=threadIdx.x; i<n_slices; i+=blockDim.x)
                atomicAdd(&output[i],block_hist[i]);
    }

    __global__ void lik_only_gm_copy(
        double *beam_dt,
        double *beam_dE,
        const double *voltage_array,
        const double *bin_centers,
        const double charge,
        const int n_slices,
        const int n_macroparticles,
        const double acc_kick,
        double *glob_voltageKick,
        double *glob_factor
        )
    {
        int tid = threadIdx.x + blockDim.x * blockIdx.x;
        double const inv_bin_width = (n_slices-1)
            /(bin_centers[n_slices-1]-bin_centers[0]);


        for (int i=tid; i<n_slices-1; i+=gridDim.x*blockDim.x){
            glob_voltageKick[i] = charge * (voltage_array[i + 1] - voltage_array[i])
                * inv_bin_width;
            glob_factor[i] = (charge * voltage_array[i] - bin_centers[i] * glob_voltageKick[i])
                + acc_kick;
        }
    }

    __global__ void lik_only_gm_comp(
            double *beam_dt,
            double *beam_dE,
            const double *voltage_array,
            const double *bin_centers,
            const double charge,
            const int n_slices,
            const int n_macroparticles,
            const double acc_kick,
            double *glob_voltageKick,
            double *glob_factor
            )
    {
        int tid = threadIdx.x + blockDim.x * blockIdx.x;
        double const inv_bin_width = (n_slices-1)
            /(bin_centers[n_slices-1]-bin_centers[0]);
        int fbin;
        const double bin0 = bin_centers[0];
        for (int i=tid; i<n_macroparticles; i += blockDim.x*gridDim.x){
            fbin = floor((beam_dt[i] - bin0) * inv_bin_width);
            if ((fbin < n_slices - 1) && (fbin >= 0))
                beam_dE[i] += beam_dt[i] * glob_voltageKick[fbin] + glob_factor[fbin];
        }
    }

    __global__ void beam_phase_v2(
            const double *bin_centers,
            const int *profile,
            const double alpha,
            const double *omega_rf_ar,
            const double *phi_rf_ar,
            const int ind,
            const double bin_size,
            double *array1,
            double *array2,
            const int n_bins)
    {   
        double omega_rf = omega_rf_ar[ind];
        double phi_rf = phi_rf_ar[ind];
        int tid = threadIdx.x + blockDim.x * blockIdx.x;
        double a,b;
        double sin_res,cos_res;
        for (int i = tid; i<n_bins; i+=gridDim.x*blockDim.x){
            a = omega_rf * bin_centers[i] + phi_rf;
            sincos(a,&sin_res,&cos_res);
            b = exp(alpha * bin_centers[i]) * profile[i];
            array1[i] = b * sin_res;
            array2[i] = b * cos_res;
        }
    }

""")

synch_rad_ker = SourceModule("""
    #include <curand_kernel.h>
        extern "C" {
            __global__ void synchrotron_radiation(
                    double *  beam_dE,
                    const double U0,
                    const int n_macroparticles,
                    const double tau_z,
                    const int n_kicks)
            {
                
                int tid = threadIdx.x + blockDim.x * blockIdx.x;
                const double const_synch_rad = 2.0 / tau_z;
                
                for (int j=0; j<n_kicks; j++){
                    for (int i=tid; i<n_macroparticles; i+=blockDim.x*gridDim.x)
                        beam_dE[i] -= const_synch_rad * beam_dE[i] + U0;
                }
            }


            __global__ void synchrotron_radiation_full(
                    double *  beam_dE,
                    const double U0,
                    const int n_macroparticles,
                    const double sigma_dE,
                    const double tau_z,
                    const double energy,
                    const int n_kicks
                    )
            {   unsigned int seed = 0;
                int tid = threadIdx.x + blockDim.x * blockIdx.x;
                const double const_quantum_exc = 2.0 * sigma_dE / sqrt(tau_z) * energy;
                curandState_t state;
                curand_init(seed, tid, 0, &state);
                const double const_synch_rad = 2.0 / tau_z;
                for (int j=0; j<n_kicks; j++){
                    for (int i=tid; i<n_macroparticles; i+=blockDim.x*gridDim.x)
                        beam_dE[i] -= const_synch_rad * beam_dE[i] + U0 - const_quantum_exc*curand_normal_double(&state);       
                }
            }
        }
""", no_extern_c=True, )


#my_gpu = drv.Device(bm.gpuId())

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
halve_edges = ker.get_function("halve_edges")
beam_phase_v2 = ker.get_function("beam_phase_v2")
try:
    beam_phase_sum = ker_with_atomicAdd.get_function("beam_phase_sum")
except:
    beam_phase_sum = ker_without_atomicAdd.get_function("beam_phase_sum")

synch_rad = synch_rad_ker.get_function("synchrotron_radiation")
synch_rad_full = synch_rad_ker.get_function("synchrotron_radiation_full")

# beam phase kernels
bm_phase_exp_times_scalar = ElementwiseKernel(
    "double *a, double *b, double c, int *d",
    "a[i] = exp(c*b[i])*d[i]",
    "bm_phase_1")
bm_phase_mul_add = ElementwiseKernel(
    "double *a, double b, double *c, double d",
    "a[i] = b*c[i] + d",
    "bm_phase_2")

bm_sin_cos = ElementwiseKernel(
    "double *a, double *b, double *c",
    "sincos(a[i],&b[i], &c[i])",
    "bm_phase_3")

d_multiply = ElementwiseKernel(
    "double *a, double *b",
    "a[i] *= b[i]",
    "bm_phase_4")

d_multscalar = ElementwiseKernel(
    "double *a, double *b, double c",
    "a[i] = c*b[i]",
    "bm_phase_4")

cuda_sum = ReductionKernel(np.float64, neutral="0",
                           reduce_expr="a+b", map_expr="x[i]",
                           arguments="double *x")

cuda_sum_2 = ReductionKernel(np.float64, neutral="0",
                             reduce_expr="a+b", map_expr="x[i]*(0.5+0.5*(i>=1 && i<sz-1))",
                             arguments="double *x, int sz")


def gpu_rf_volt_comp(dev_voltage, dev_omega_rf, dev_phi_rf, dev_bin_centers, dev_rf_voltage, f_rf=0):
    rvc(dev_voltage, dev_omega_rf, dev_phi_rf, dev_bin_centers,
        np.int32(dev_voltage.size), np.int32(
            dev_bin_centers.size), np.int32(f_rf), dev_rf_voltage,
        block=block_size, grid=grid_size, shared=3*dev_voltage.size*64, time_kernel=True)


def gpu_kick(dev_voltage, dev_omega_rf, dev_phi_rf, charge, n_rf, acceleration_kick, beam):
    dev_voltage_kick = get_gpuarray((dev_voltage.size, np.float64, 0, 'vK'))

    #dev_voltage_kick  = np.float64(charge)*dev_voltage
    d_multscalar(dev_voltage_kick, dev_voltage, charge)

    kick_kernel(beam.dev_dt,
                beam.dev_dE,
                np.int32(n_rf),
                dev_voltage_kick,
                dev_omega_rf,
                dev_phi_rf,
                np.int32(beam.dev_dt.size),
                np.float64(acceleration_kick),
                block=block_size, grid=grid_size, time_kernel=True)
    beam.dt_obj.invalidate_cpu()


def gpu_drift(solver_utf8, t_rev, length_ratio, alpha_order, eta_0,
              eta_1, eta_2, alpha_0, alpha_1, alpha_2, beta, energy, beam):
    solver = solver_utf8.decode('utf-8')
    T = np.float64(t_rev)*np.float64(length_ratio)
    n_macroparticles = len(beam.dev_dt)
    if (solver == "simple"):
        ##### simple solver #####
        # coeff =  np.float64(eta_0) / (np.float64(beta)* np.float64(beta)* np.float64(energy))
        # beam.dev_dt += T*coeff*beam.dev_dE
        drift_simple(beam.dev_dt, beam.dev_dE,
                     (t_rev), np.float64(length_ratio),
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
                           np.float64(T), np.float64(eta0),
                           np.int32(n_macroparticles),
                           block=block_size, grid=grid_size, time_kernel=True)
        elif (alpha_order == 1):
            drift_legacy_1(beam.dev_dt, beam.dev_dE,
                           np.float64(T), np.float64(eta0),
                           np.float64(eta1), np.int32(n_macroparticles),
                           block=block_size, grid=grid_size, time_kernel=True)
        else:
            drift_legacy_2(beam.dev_dt, beam.dev_dE,
                           np.float64(T), np.float64(eta0),
                           np.float64(eta1), np.float64(eta2),
                           np.int32(n_macroparticles),
                           block=block_size, grid=grid_size, time_kernel=True)

    else:
        ##### other solver  #####
        invbetasq = 1. / (beta * beta)
        invenesq = 1. / (energy * energy)
        drift_else(beam.dev_dt, beam.dev_dE,
                   np.float64(invbetasq), np.float64(invenesq),
                   np.float64(T), np.float64(alpha_0), np.float64(alpha_1),
                   np.float64(alpha_2), np.float64(energy),
                   np.int32(n_macroparticles),
                   block=block_size, grid=grid_size, time_kernel=True)
    beam.dE_obj.invalidate_cpu()


def gpu_linear_interp_kick(dev_voltage,
                           dev_bin_centers, charge,
                           acceleration_kick, beam=None):
    macros = beam.dev_dt.size
    slices = dev_bin_centers.size

    dev_voltageKick = get_gpuarray((slices-1, np.float64, 0, 'vK'))
    dev_factor = get_gpuarray((slices-1, np.float64, 0, 'dF'))

    gm_linear_interp_kick_help(beam.dev_dt,
                               beam.dev_dE,
                               dev_voltage,
                               dev_bin_centers,
                               np.float64(charge),
                               np.int32(slices),
                               np.int32(macros),
                               np.float64(acceleration_kick),
                               dev_voltageKick,
                               dev_factor,
                               grid=grid_size, block=block_size,
                               time_kernel=True)
    gm_linear_interp_kick_comp(beam.dev_dt,
                               beam.dev_dE,
                               dev_voltage,
                               dev_bin_centers,
                               np.float64(charge),
                               np.int32(slices),
                               np.int32(macros),
                               np.float64(acceleration_kick),
                               dev_voltageKick,
                               dev_factor,
                               grid=grid_size, block=block_size,
                               time_kernel=True)
    beam.dE_obj.invalidate_cpu()


def gpu_slice(cut_left, cut_right, beam, profile):

    n_slices = profile.dev_n_macroparticles.size
    profile.dev_n_macroparticles.fill(0)
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
        sm_histogram(beam.dev_dt, profile.dev_n_macroparticles, np.float64(cut_left),
                     np.float64(cut_right), np.uint32(n_slices),
                     np.uint32(beam.dev_dt.size),
                     grid=grid_size, block=block_size, shared=4*n_slices, time_kernel=True)
    else:
        hybrid_histogram(beam.dev_dt, profile.dev_n_macroparticles, np.float64(cut_left),
                         np.float64(cut_right), np.uint32(n_slices),
                         np.uint32(beam.dev_dt.size), np.int32(
                             my_gpu.MAX_SHARED_MEMORY_PER_BLOCK/4),
                         grid=grid_size, block=block_size, shared=my_gpu.MAX_SHARED_MEMORY_PER_BLOCK, time_kernel=True)
    profile.n_macroparticles_obj.invalidate_cpu()
    return profile.dev_n_macroparticles


def gpu_synchrotron_radiation(dE, U0, n_kicks, tau_z):
    synch_rad(dE, np.float64(U0), np.int32(dE.size), np.float64(tau_z),
              np.int32(n_kicks), block=block_size, grid=(my_gpu.MULTIPROCESSOR_COUNT, 1, 1))


def gpu_synchrotron_radiation_full(dE, U0, n_kicks, tau_z, sigma_dE, energy):
    # print("Entering")
    synch_rad_full(dE, np.float64(U0), np.int32(dE.size), np.float64(sigma_dE), np.float64(energy),
                   np.int32(n_kicks), np.int32(1), block=block_size, grid=grid_size)


def gpu_beam_phase(bin_centers, profile, alpha, omega_rf, phi_rf, ind, bin_size):

    n_bins = bin_centers.size

    array1 = get_gpuarray((bin_centers.size, np.float64, 0, 'ar1'))
    array2 = get_gpuarray((bin_centers.size, np.float64, 0, 'ar2'))

    dev_scoeff = get_gpuarray((1, np.float64, 0, 'sc'))
    dev_coeff = get_gpuarray((1, np.float64, 0, 'co'))

    beam_phase_v2(bin_centers,
                  profile, np.float64(alpha), omega_rf, phi_rf, np.int32(
                      ind), np.float64(bin_size),
                  array1, array2, np.int32(n_bins),
                  block=block_size)  # , grid=grid_size)

    beam_phase_sum(array1, array2, dev_scoeff, dev_coeff, np.int32(
        n_bins), block=block_size, grid=(1, 1, 1), time_kernel=True)
    to_ret = dev_scoeff[0].get()

    return to_ret
