from skcuda import fft
from skcuda import integrate
import numpy as np
from copy import deepcopy

import pycuda.elementwise as elw
import pycuda.reduction as red
from ..gpu.cucache import get_gpuarray
from pycuda import gpuarray,driver as drv
from ..gpu import grid_size, block_size
from ..utils import bmath as bm
from pycuda.tools import dtype_to_ctype, VectorArg, ScalarArg
import os

get_precompiled = True
# create_cu_files = False
basedir = os.path.dirname(os.path.realpath(__file__))+"/cuda_kernels/"
## Elementwise

central_mod = bm.getMod()


### get precompiled Source Modules
if (get_precompiled):
    
    def custom_get_elwise_range_module(arguments, operation,
        name="kernel", keep=False, options=None,
        preamble="", loop_prep="", after_loop=""):
        #print(name, " range")
        return central_mod

    def custom_get_elwise_no_range_module(arguments, operation,
        name="kernel", keep=False, options=None,
        preamble="", loop_prep="", after_loop=""):
        #print(name, " no range")
        return central_mod

    def custom_get_elwise_kernel_and_types(arguments, operation,
        name="kernel", keep=False, options=None, use_range=False, **kwargs):
        if isinstance(arguments, str):
            from pycuda.tools import parse_c_arg
            arguments = [parse_c_arg(arg) for arg in arguments.split(",")]

        if use_range:
            arguments.extend([
                ScalarArg(np.intp, "start"),
                ScalarArg(np.intp, "stop"),
                ScalarArg(np.intp, "step"),
                ])
        else:
            arguments.append(ScalarArg(np.uintp, "n"))

        if use_range:
            module_builder = custom_get_elwise_range_module
        else:
            module_builder = custom_get_elwise_no_range_module

        mod = module_builder(arguments, operation, name,
                keep, options, **kwargs)

        func = mod.get_function(name+use_range*"_range")
        func.prepare("".join(arg.struct_char for arg in arguments))

        return mod, func, arguments

    elw.get_elwise_range_module = custom_get_elwise_range_module
    elw.get_elwise_module = custom_get_elwise_no_range_module
    elw.get_elwise_kernel_and_types = custom_get_elwise_kernel_and_types

    
    def get_reduction_module(out_type, block_size,
        neutral, reduce_expr, map_expr, arguments,
        name="reduce_kernel", keep=False, options=None, preamble=""):
        return central_mod

    red.get_reduction_module = get_reduction_module

ElementwiseKernel = elw.ElementwiseKernel
ReductionKernel = red.ReductionKernel

gpu_copy_i2d = ElementwiseKernel(
    f"{bm.precision.str} *x, int *y",
    f"x[i] = ({bm.precision.str}) y[i]*1.0",
    "gpu_copy_i2d")

gpu_copy_d2d = ElementwiseKernel(
    f"{bm.precision.str} *x,{bm.precision.str} *y",
    "x[i] = y[i]",
    "gpu_copy_d2d")

gpu_complex_copy = ElementwiseKernel(
    f"pycuda::complex<{bm.precision.str}> *x, pycuda::complex<{bm.precision.str}> *y",
    "x[i] = y[i]",
    "gpu_complex_copy",
    preamble="#include <pycuda-complex.hpp>")


# gpu_copy_i2d = central_mod.get_function('gpu_copy_i2d')
# gpu_copy_d2d = central_mod.get_function('gpu_copy_d2d')
# gpu_complex_copy = central_mod.get_function('gpu_complex_copy')


# gpu_beam

stdKernel = ReductionKernel(bm.precision.real_t, neutral="0",
                            reduce_expr="a+b", map_expr="(y[i]!=0)*(x[i]-m)*(x[i]-m)",
                            arguments=f"{bm.precision.str} *x, {bm.precision.str} *y, {bm.precision.str} m",
                            name = "stdKernel")

sum_non_zeros = ReductionKernel(bm.precision.real_t, neutral="0",
                                reduce_expr="a+b", map_expr="(x[i]!=0)",
                                arguments=f"{bm.precision.str} *x",
                                name="sum_non_zeros")

mean_non_zeros = ReductionKernel(bm.precision.real_t, neutral="0",
                                reduce_expr="a+b", map_expr="(id[i]!=0)*x[i]",
                                arguments=f"{bm.precision.str} *x, {bm.precision.str} *id",
                                name="mean_non_zeros")


# gpu_profile
cugradient = central_mod.get_function("cugradient")

custom_gpu_trapz = central_mod.get_function("gpu_trapz_custom")

gpu_diff = ElementwiseKernel(f"int *a, {bm.precision.str} *b, {bm.precision.str} c",
                             "b[i] = (a[i+1]-a[i])/c", "gpu_diff")


# impedances

set_zero_float = ElementwiseKernel(
    f"{bm.precision.str} *x",
    "x[i] = 0",
    "set_zero_float")

set_zero_double = ElementwiseKernel(
    f"double *x",
    "x[i] = 0",
    "set_zero_double")

set_zero_int = ElementwiseKernel(
    "int *x",
    "x[i] = 0",
    "set_zero_int")

set_zero_complex64 = ElementwiseKernel(
    f"pycuda::complex<{bm.precision.str}> *x",
    "x[i] = 0",
    "set_zero_complex",
    preamble="#include <pycuda-complex.hpp>")

set_zero_complex128 = ElementwiseKernel(
    f"pycuda::complex<double> *x",
    "x[i] = 0",
    "set_zero_complex128",
    preamble="#include <pycuda-complex.hpp>")


increase_by_value = ElementwiseKernel(
    f"{bm.precision.str} *x, {bm.precision.str} a",
    "x[i] += a",
    "increase_by_value")

add_array = ElementwiseKernel(
    f"{bm.precision.str} *x, {bm.precision.str} *y",
    "x[i] += y[i]",
    "add_array")

complex_mul = ElementwiseKernel(
    f"pycuda::complex<{bm.precision.str}> *x, pycuda::complex<{bm.precision.str}> *y, pycuda::complex<{bm.precision.str}> *z",
    "z[i] = x[i] * y[i]",
    "complex_mul",
    preamble="#include <pycuda-complex.hpp>")

gpu_mul = ElementwiseKernel(
    f"{bm.precision.str} *x, {bm.precision.str} *y, {bm.precision.str} a",
    "x[i] = a*y[i]",
    "gpu_mul")

# beam_feedback
gpu_copy_one = ElementwiseKernel(
    f"{bm.precision.str} *x, {bm.precision.str} *y, int ind",
    "x[i] = y[ind]",
    "gpu_copy_one")

triple_kernel = central_mod.get_function("gpu_beam_fb_track_other")

first_kernel_x = ElementwiseKernel(
    f"{bm.precision.str} *omega_rf, {bm.precision.str} *harmonic,  {bm.precision.str} domega_rf, int size, int counter",
    "omega_rf[i*size +counter] += domega_rf * harmonic[i*size + counter] / harmonic[counter]",
    "first_kernel_x")

second_kernel_x = ElementwiseKernel(
    f"{bm.precision.str} *dphi_rf, {bm.precision.str} *harmonic, {bm.precision.str} *omega_rf, {bm.precision.str} *omega_rf_d, int size, int counter, double pi",
    "dphi_rf[i] +=  2.0*pi*harmonic[size*i+counter]*(omega_rf[size*i+counter]-omega_rf_d[size*i+counter])/omega_rf_d[size*i+counter]",
    "second_kernel_x")

third_kernel_x = ElementwiseKernel(
    f"{bm.precision.str} *x, {bm.precision.str} *y, int size_0, int counter",
    "x[i*size_0 + counter] += y[i]",
    "third_kernel_x")

indexing_double = ElementwiseKernel(
    f"{bm.precision.str} *out, {bm.precision.str} *in, int *ind",
    "out[i] = in[ind[i]]",
    "indexing_double")

indexing_int = ElementwiseKernel(
    f"{bm.precision.str} *out, int *in, int *ind",
    "out[i] = in[ind[i]]",
    "indexing_int")

sincos_mul_add = ElementwiseKernel(
    f"{bm.precision.str} *ar, double a, double b, {bm.precision.str} *s, {bm.precision.str} *c",
    "sincos(a*ar[i]+b, &s[i], &c[i])",
    "sincos_mul_add")

sincos_mul_add_2 = ElementwiseKernel(
    f"{bm.precision.str} *ar, double a, double b, {bm.precision.str} *s, {bm.precision.str} *c",
    "s[i] = cos(a*ar[i]+b -3.141592653589793238462643383279502884197169399375105820974944592307816406286/2); c[i] = cos(a*ar[i]+b)",
    "sincos_mul_add_2")

gpu_trapz = ReductionKernel(bm.precision.real_t, neutral="0", reduce_expr="a+b",
                                   arguments=f"{bm.precision.str} *y, double x, int sz",
                                   map_expr="(i<sz-1) ? x*(y[i]+y[i+1])/2.0 : 0.0",
                                   name="gpu_trapz")

mul_d = ElementwiseKernel(
    f"{bm.precision.str} *a1, {bm.precision.str} *a2",
    "a1[i] *= a2[i]",
    "mul_d")

# tracker


add_kernel = ElementwiseKernel(
    f"{bm.precision.str} *a, {bm.precision.str} *b, {bm.precision.str} *c",
    "a[i]=b[i]+c[i]",
    "add_kernel")

first_kernel_tracker = ElementwiseKernel(
    f"{bm.precision.str} *phi_rf, double x, {bm.precision.str} *phi_noise, int len, int turn",
    "phi_rf[len*i + turn] += x * phi_noise[len*i + turn]",
    "first_kernel_tracker")

second_kernel_tracker = ElementwiseKernel(
    f"{bm.precision.str} *phi_rf, {bm.precision.str} *omega_rf, {bm.precision.str} *phi_mod0, {bm.precision.str} *phi_mod1, int size, int turn",
    "phi_rf[i*size+turn] += phi_mod0[i*size+turn]; omega_rf[i*size+turn] += phi_mod1[i*size+turn]",
    "second_kernel_tracker")

copy_column = ElementwiseKernel(
    f"{bm.precision.str} *x, {bm.precision.str} *y, int size, int column",
    "x[i] = y[i*size + column]",
    "copy_column")

rf_voltage_calculation_kernel = ElementwiseKernel(
    f"{bm.precision.str} *x, {bm.precision.str} *y, int size, int column",
    "x[i] = y[i*size + column]",
    "rf_voltage_calculation_kernel")

cavityFB_case = ElementwiseKernel(
    f"{bm.precision.str} *rf_voltage, {bm.precision.str} *voltage, {bm.precision.str} *omega_rf, {bm.precision.str} *phi_rf," +
    f"{bm.precision.str} *bin_centers, double V_corr, double phi_corr," +
    "int size, int column",
    "rf_voltage[i] = voltage[0] * V_corr * sin(omega_rf[0] * bin_centers[i]+phi_rf[0]+phi_corr)",
    "cavityFB_case")

gpu_rf_voltage_calc_mem_ops = central_mod.get_function("gpu_rf_voltage_calc_mem_ops")

cuinterp = central_mod.get_function("cuinterp")

## beam phase 


bm_phase_exp_times_scalar = ElementwiseKernel(
    f"{bm.precision.str} *a, {bm.precision.str} *b, double c, int *d",
    "a[i] = exp(c*b[i])*d[i]",
    "bm_phase_exp_times_scalar")

bm_phase_mul_add = ElementwiseKernel(
    f"{bm.precision.str} *a, double b, {bm.precision.str} *c, double d",
    "a[i] = b*c[i] + d",
    "bm_phase_mul_add")

bm_sin_cos = ElementwiseKernel(
    f"{bm.precision.str} *a, {bm.precision.str} *b, {bm.precision.str} *c",
    "sincos(a[i],&b[i], &c[i])",
    "bm_sin_cos")

d_multiply = ElementwiseKernel(
    f"{bm.precision.str} *a, {bm.precision.str} *b",
    "a[i] *= b[i]",
    "d_multiply")

d_multscalar = ElementwiseKernel(
    f"{bm.precision.str} *a, {bm.precision.str} *b, double c",
    "a[i] = c*b[i]",
    "d_multscalar")


plans_dict = {}
inverse_plans_dict = {}


## scale kernels
scale_int = ElementwiseKernel(
    "int a, int *b",
    "b[i] /= a ",
    "scale_kernel_int")

scale_double = ElementwiseKernel(
    f"double a, {bm.precision.str} *b",
    "b[i] /= a ",
    "scale_kernel_double")


def _get_scale_kernel(dtype):
    if (dtype==np.float64):
        return scale_double
    else:
        return scale_int
fft._get_scale_kernel = _get_scale_kernel

def find_plan(my_size):
    if (my_size not in plans_dict):
        plans_dict[my_size] = fft.Plan(my_size, np.float64, np.complex128)
    return plans_dict[my_size]


def inverse_find_plan(size):
    if (size not in inverse_plans_dict):
        inverse_plans_dict[size] = fft.Plan(
            size, in_dtype=np.complex128, out_dtype=np.float64)
    return inverse_plans_dict[size]


def gpu_rfft(dev_a, n=0, result=None, caller_id=None):
    if (n == 0) and (result == None):
        n = dev_a.size
    elif (n != 0) and (result == None):
        pass
    if (caller_id == None):
        result = gpuarray.empty(n//2 + 1, np.complex128)
    else:
        result = get_gpuarray((n//2 + 1, np.complex128, 0, 'rfft'),zero_fills=True)
    outSize = n // 2 + 1
    inSize = dev_a.size

    if (dev_a.dtype == np.int32):
        gpu_copy = gpu_copy_i2d
    else:
        gpu_copy = gpu_copy_d2d

    if (n == inSize):
        dev_in = get_gpuarray((n, np.float64, 0, 'rfft'))
        gpu_copy(dev_in, dev_a, slice=slice(0, n))
    else:
        dev_in = get_gpuarray((n, np.float64, 0, 'rfft'), zero_fills=True)
        if (n < inSize):
            gpu_copy(dev_in, dev_a, slice=slice(0, n))
        else:
            gpu_copy(dev_in, dev_a, slice=slice(0, inSize))
    plan = find_plan(dev_in.shape)
    fft.fft(dev_in, result, plan)
    return result


def gpu_irfft(dev_a, n=0, result=None, caller_id=None):
    if (n == 0) and (result == None):
        n = 2*(dev_a.size-1)
    elif (n != 0) and (result == None):
        pass

    if (caller_id == None):
        result = gpuarray.empty(n, dtype=np.float64)
    else:
        key = (n, np.float64, caller_id, 'irfft')
        result = get_gpuarray(key)

    outSize = n
    inSize = dev_a.size

    if (outSize == 0):
        outSize = 2*(inSize-1)
    n = outSize // 2 + 1

    if (n == inSize):
        dev_in = dev_a
    else:
        dev_in = get_gpuarray((n, np.complex128, 0, 'irfft'))
        if (n < inSize):
            gpu_complex_copy(dev_in, dev_a, slice=slice(0, n))
        else:
            gpu_complex_copy(dev_in, dev_a, slice=slice(0, n))

    inverse_plan = inverse_find_plan(outSize)
    fft.ifft(dev_in, result, inverse_plan, scale=True)
    return result


def gpu_rfftfreq(n, d=1.0, result=None):

    factor = 1/(d*n)
    result = factor*gpuarray.arange(0, n//2 + 1, dtype=bm.precision.real_t).get()
    return result


def gpu_convolve(signal, kernel, mode='full', result=None):
    if mode != 'full':
        # ConvolutionError
        raise RuntimeError('[convolve] Only full mode is supported')
    if result is None:
        result = np.empty(len(signal) + len(kernel) - 1, dtype=float)
    realSize = len(signal) + len(kernel) - 1
    complexSize = realSize // 2 + 1
    result1 = np.empty((complexSize), dtype=np.complex128)
    result2 = np.empty((complexSize), dtype=np.complex128)
    result1 = gpu_rfft(signal, result=result1, ret_gpu=True)
    result2 = gpu_rfft(kernel, result=result2, ret_gpu=True)
    result2 = result1*result2
    result = gpu_irfft(result2.get(), result=result).get()
    return result


def gpu_interp(dev_x, dev_xp, dev_yp, left=0.12345, right=0.12345, caller_id=None):
    if (caller_id == None):
        dev_res = get_gpuarray((dev_x.size, bm.precision.real_t, caller_id, 'interp'))
    else:
        dev_res = gpuarray.empty(dev_x.size, bm.precision.real_t)
    cuinterp(dev_x,  np.int32(dev_x.size),
             dev_xp, np.int32(dev_xp.size),
             dev_yp, dev_res,
             bm.precision.real_t(left), bm.precision.real_t(right),
             block=block_size, grid=grid_size)
    return dev_res




'''
if (create_cu_files):

    ## ElementWise
    f = open("no_range_elem_kernels.cu", "w")
    g = open("range_elem_kernels.cu", "w")
    def print_no_range_module(arguments, operation,
            name="kernel", keep=False, options=None,
            preamble="", loop_prep="", after_loop=""):
        from pycuda.compiler import SourceModule
        f.write("""
            #include <pycuda-complex.hpp>
            %(preamble)s
            extern "C"
            __global__ void %(name)s(%(arguments)s)
            {
            unsigned tid = threadIdx.x;
            unsigned total_threads = gridDim.x*blockDim.x;
            unsigned cta_start = blockDim.x*blockIdx.x;
            unsigned i;
            %(loop_prep)s;
            for (i = cta_start + tid; i < n; i += total_threads)
            {
                %(operation)s;
            }
            %(after_loop)s;
            }
            """ % {
                "arguments": arguments+",long n",
                "operation": operation,
                "name": name,
                "preamble": preamble,
                "loop_prep": loop_prep,
                "after_loop": after_loop,
                })
        
    def print_range_module(arguments, operation,
            name="kernel", keep=False, options=None,
            preamble="", loop_prep="", after_loop=""):
        from pycuda.compiler import SourceModule
        
        g.write("""
            #include <pycuda-complex.hpp>
            %(preamble)s
            extern "C"
            __global__ void %(name)s(%(arguments)s)
            {
                unsigned tid = threadIdx.x;
                unsigned total_threads = gridDim.x*blockDim.x;
                unsigned cta_start = blockDim.x*blockIdx.x;
                long i;
                %(loop_prep)s;
                if (step < 0)
                {
                for (i = start + (cta_start + tid)*step;
                    i > stop; i += total_threads*step)
                {
                    %(operation)s;
                }
                }
                else
                {
                for (i = start + (cta_start + tid)*step;
                    i < stop; i += total_threads*step)
                {
                    %(operation)s;
                }
                }
                %(after_loop)s;
            }
            """ % {
            "arguments": arguments +" ,long start, long stop, long step",
            "operation": operation,
            "name": name,
            "preamble": preamble,
            "loop_prep": loop_prep,
            "after_loop": after_loop,
            },
        )

    def elw__init__(self, arguments, operation,
                name="kernel", keep=False, options=None, **kwargs):
            self.gen_kwargs = kwargs.copy()
            self.gen_kwargs.update(dict(keep=keep, options=options, name=name,
                operation=operation, arguments=arguments))
            print_no_range_module(arguments, operation,
                name=name, keep=keep, options=options)
            print_range_module(arguments, operation,
                name=name, keep=False, options=None)

    ElementwiseKernel.__init__ = elw__init__

    ## Reduction
    
    def get_reduction_module(out_type, block_size,
        neutral, reduce_expr, map_expr, arguments,
        name="reduce_kernel", keep=False, options=None, preamble=""):


        src = """
            #include <pycuda-complex.hpp>
            #define BLOCK_SIZE %(block_size)d
            #define READ_AND_MAP(i) (%(map_expr)s)
            #define REDUCE(a, b) (%(reduce_expr)s)
            %(preamble)s
            typedef %(out_type)s out_type;
            extern "C"
            __global__
            void %(name)s(out_type *out, %(arguments)s,
            unsigned int seq_count, unsigned int n)
            {
            // Needs to be variable-size to prevent the braindead CUDA compiler from
            // running constructors on this array. Grrrr.
            extern __shared__ out_type sdata[];
            unsigned int tid = threadIdx.x;
            unsigned int i = blockIdx.x*BLOCK_SIZE*seq_count + tid;
            out_type acc = %(neutral)s;
            for (unsigned s = 0; s < seq_count; ++s)
            {
                if (i >= n)
                break;
                acc = REDUCE(acc, READ_AND_MAP(i));
                i += BLOCK_SIZE;
            }
            sdata[tid] = acc;
            __syncthreads();
            #if (BLOCK_SIZE >= 512)
                if (tid < 256) { sdata[tid] = REDUCE(sdata[tid], sdata[tid + 256]); }
                __syncthreads();
            #endif
            #if (BLOCK_SIZE >= 256)
                if (tid < 128) { sdata[tid] = REDUCE(sdata[tid], sdata[tid + 128]); }
                __syncthreads();
            #endif
            #if (BLOCK_SIZE >= 128)
                if (tid < 64) { sdata[tid] = REDUCE(sdata[tid], sdata[tid + 64]); }
                __syncthreads();
            #endif
            if (tid < 32)
            {
                // 'volatile' required according to Fermi compatibility guide 1.2.2
                volatile out_type *smem = sdata;
                if (BLOCK_SIZE >= 64) smem[tid] = REDUCE(smem[tid], smem[tid + 32]);
                if (BLOCK_SIZE >= 32) smem[tid] = REDUCE(smem[tid], smem[tid + 16]);
                if (BLOCK_SIZE >= 16) smem[tid] = REDUCE(smem[tid], smem[tid + 8]);
                if (BLOCK_SIZE >= 8)  smem[tid] = REDUCE(smem[tid], smem[tid + 4]);
                if (BLOCK_SIZE >= 4)  smem[tid] = REDUCE(smem[tid], smem[tid + 2]);
                if (BLOCK_SIZE >= 2)  smem[tid] = REDUCE(smem[tid], smem[tid + 1]);
            }
            if (tid == 0) out[blockIdx.x] = sdata[0];
            }
            """ % {
                "out_type": out_type,
                "arguments": arguments,
                "block_size": block_size,
                "neutral": neutral,
                "reduce_expr": reduce_expr,
                "map_expr": map_expr,
                "name": name,
                "preamble": preamble
                }
        e = open(name+".cu", "w")
        return SourceModule(src, options=options, keep=keep, no_extern_c=True)
    red.get_reduction_module = get_reduction_module
'''
