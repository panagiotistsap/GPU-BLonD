import numpy as np

import numpy as np
from pycuda.elementwise import ElementwiseKernel
from pycuda.reduction import ReductionKernel
from pycuda.compiler import SourceModule
import traceback
from ..utils.cucache import get_gpuarray
import pycuda.reduction as reduce
from pycuda import gpuarray, driver as drv, tools
import atexit      
from ..utils.bmath import gpu_num
drv.init()
#assert ( driver.Device.count() >= 1)
dev = drv.Device(gpu_num)

from skcuda import integrate

integrate.init()


from skcuda import fft

class myElementwiseKernel:
    def __init__(self, arguments, operation, name="new_func", preamble=None, options=None):
        mod = SourceModule("""
            #include <pycuda-complex.hpp>
            extern "C"
            __global__ void %(name)s(%(arguments)s, int start, int end, int step)
            {
            unsigned tid = threadIdx.x+blockDim.x*blockIdx.x;
            unsigned total_threads = gridDim.x*blockDim.x;
            unsigned i;
            for (i = tid*step+start; i < end; i += total_threads*step)
            {
                %(operation)s;
            }
            }
        """ % {
            "arguments": arguments,
            "operation": operation,
            "name": name,
            },
        no_extern_c=True)
        self.my_func = mod.get_function(name)

    def __call__(self, *args, **kwargs):
        #print(kwargs)
        if ('slice' not in kwargs):
            kwargs['slice'] = slice(0,args[0].size,1)
        if (kwargs['slice'].step==None):
            step=1
        else:
            step=kwargs['slice'].step
        #blocks = 1+( np.int32(kwargs['slice'].stop-np.int32(kwargs['slice'].start)/step
        self.my_func(*args, np.int32(kwargs['slice'].start), np.int32(kwargs['slice'].stop) ,np.int32(step), block=(1024,1,1), grid=(2*dev.MULTIPROCESSOR_COUNT,1,1),time_kernel=True)

#ElementwiseKernel = ElementwiseKernel
gpu_copy_i2d = ElementwiseKernel(
        "double *x, int *y",
        "x[i] = (double) y[i]*1.0",
        "copy_copy")

gpu_copy_d2d = ElementwiseKernel(
        "double *x,double *y",
        "x[i] = y[i]",
        "copy_copy")

gpu_complex_copy = ElementwiseKernel(
        "pycuda::complex<double> *x, pycuda::complex<double> *y",
        "x[i] = y[i]",
        "copy",
        preamble = "#include <pycuda-complex.hpp>")

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
        __global__ void gpu_trapz_custom(
                double *y,
                double x,
                int sz,
                double *res)
        {   
            int tid = threadIdx.x + blockDim.x * blockIdx.x;
            double my_sum = 0;
            for (int i = tid; i<sz-1; i+=gridDim.x*blockDim.x)
                my_sum += (y[i]+y[i+1])*x/2.0;

            atomicAdd(&(res[0]),my_sum);
        }
        """)

except:
    ## atomicAdd for doubles already specified
    pass
try:
    ker_without_atomicAdd = SourceModule("""
        
        __global__ void gpu_trapz_custom(
                double *y,
                double x,
                int sz,
                double *res)
        {   
            int tid = threadIdx.x + blockDim.x * blockIdx.x;
            double my_sum = 0;
            for (int i = tid; i<sz-1; i+=gridDim.x*blockDim.x)
                my_sum += (y[i]+y[i+1])*x/2.0;

            atomicAdd(&(res[0]),my_sum);
        }
        """)
except:
    ## atomicAdd already exists
    pass

ker = SourceModule("""

    
    

    __global__ void cuinterp(double *x,
            int x_size, 
            double *xp,
            int xp_size,
            double *yp,
            double *y,
            double left,
            double right)
    {
        if (left==0.12345)
            left = yp[0];
        if (right==0.12345)
            right = yp[xp_size-1];
        double curr;
        int lo;
        int mid;
        int hi;
        int tid = threadIdx.x + blockDim.x * blockIdx.x;
        for (int i=tid; i<x_size; i+= blockDim.x*gridDim.x){
            //need to find the right bin with binary search
            // looks like bisect_left
            curr = x[i];
            hi = xp_size;
            lo=0;
            while (lo<hi){
                mid = (lo+hi)/2;
                if (xp[mid] < curr)
                    lo = mid + 1;
                else
                    hi = mid;
            }
            if (lo == xp_size)
                y[i] = right;
            else if (xp[lo-1]==curr)
                y[i]=yp[i];
            else if (lo<=1)
                y[i]=left;
            else{
                y[i] = yp[lo - 1] +
                        (yp[lo] - yp[lo - 1]) * (x[i] - xp[lo - 1]) /
                        (xp[lo] - xp[lo - 1]);
                }
            
        }
    }
    
    __global__ void cugradient(
            double x, 
            int *y,
            double *g,
            int size)
    {
        int tid = threadIdx.x + blockDim.x * blockIdx.x;
        for (int i=tid+1; i<size-1; i+= blockDim.x*gridDim.x){

            g[i]=(y[i+1]-y[i-1])/(2*x);
            // g[i] = (hs*hs*fd + (hd*hd-hs*hs)*fx - hd*hd*fs)/
            //     (hs*hd*(hd+hs));
        }
        if (tid == 0)
            g[0] = (y[1]-y[0])/x;
        if (tid == 32)
            g[size-1] = (y[size-1]-y[size-2])/x;
    }

    
    __global__ void gpu_beam_fb_track_other(double *omega_rf,
                                            double *harmonic,
                                            double *dphi_rf,
                                            double *omega_rf_d,
                                            double *phi_rf,
                                            double pi,
                                            double domega_rf,
                                            int size,
                                            int counter,
                                            int n_rf)
    {
        double a,b,c;
        for (int i = threadIdx.x; i < n_rf; i +=blockDim.x){
            a = domega_rf * harmonic[i*size + counter] / harmonic[counter];
            b =  2.0*pi*harmonic[size*i+counter]*(a+omega_rf[i*size+counter]-omega_rf_d[size*i+counter])/omega_rf_d[size*i+counter];
            c = dphi_rf[i] +  b;
            omega_rf[i*size +counter] += a;
            dphi_rf[i] +=  b;
            phi_rf[size*i + counter] += c;
        }
    }

    __global__ void gpu_rf_voltage_calc_mem_ops(double *new_voltages,
                                            double *new_omega_rf,
                                            double *new_phi_rf,
                                            double *voltages,
                                            double *omega_rf,
                                            double *phi_rf,
                                            int start,
                                            int end,
                                            int step)
    {
        int idx =0;
        for (int i = threadIdx.x*step+start; i < end; i += blockDim.x*step){
            new_voltages[idx] = voltages[i];
            new_omega_rf[idx] = omega_rf[i];
            new_phi_rf[idx] = phi_rf[i];
            idx++;
        }
    }
    """)

## gpu_beam

stdKernel = ReductionKernel(np.float64, neutral="0",
        reduce_expr="a+b", map_expr="(y[i]!=0)*(x[i]-m)*(x[i]-m)",
        arguments="double *x, double *y, double m")
    
sum_non_zeros = ReductionKernel(np.float64, neutral="0",
        reduce_expr="a+b", map_expr="(x[i]!=0)",
        arguments="double *x")
    
mean_non_zeros = ReductionKernel(np.float64, neutral="0",
        reduce_expr="a+b", map_expr="(id[i]!=0)*x[i]",
        arguments="double *x, double *id")


## gpu_profile
cugradient = ker.get_function("cugradient")
try:
    custom_gpu_trapz = ker_with_atomicAdd.get_function("gpu_trapz_custom")
except:
    custom_gpu_trapz = ker_without_atomicAdd.get_function("gpu_trapz_custom")

gpu_diff = ElementwiseKernel("int *a, double *b, double c",
                                            "b[i] = (a[i+1]-a[i])/c","gpu_diff")
 
## impedances

set_zero = ElementwiseKernel(
        "double *x",
        "x[i] = 0",
        "zero")

increase_by_value = ElementwiseKernel(
        "double *x, double a",
        "x[i] += a",
        "increase")

add_array = ElementwiseKernel(
    "double *x, double *y",
    "x[i] += y[i]",
    "add_array")

complex_mul = ElementwiseKernel(
    "pycuda::complex<double> *x, pycuda::complex<double> *y, pycuda::complex<double> *z",
    "z[i] = x[i] * y[i]",
    "complex_mul",
    preamble="#include <pycuda-complex.hpp>")

gpu_mul  = ElementwiseKernel(
    "double *x, double *y, double a",
    "x[i] = a*y[i]",
    "mul_array")

## beam_feedback
gpu_copy_one = ElementwiseKernel(
    "double *x, double *y, int ind",
    "x[i] = y[ind]",
    "copy")

triple_kernel = ker.get_function("gpu_beam_fb_track_other")

first_kernel_x = ElementwiseKernel(
    "double *omega_rf, double *harmonic,  double domega_rf, int size, int counter",
    "omega_rf[i*size +counter] += domega_rf * harmonic[i*size + counter] / harmonic[counter]",
    "first_kernel")

second_kernel_x = ElementwiseKernel(
    "double *dphi_rf, double *harmonic, double *omega_rf, double *omega_rf_d, int size, int counter, double pi",
    "dphi_rf[i] +=  2.0*pi*harmonic[size*i+counter]*(omega_rf[size*i+counter]-omega_rf_d[size*i+counter])/omega_rf_d[size*i+counter]",
    "second_kernel_x")

third_kernel_x = ElementwiseKernel(
    "double *x, double *y, int size_0, int counter",
    "x[i*size_0 + counter] += y[i]",
    "increase_column")
    
indexing_double = ElementwiseKernel(
        "double *out, double *in, int *ind",
        "out[i] = in[ind[i]]",
        "indexing_double")

indexing_int = ElementwiseKernel(
        "double *out, int *in, int *ind",
        "out[i] = in[ind[i]]",
        "indexing_double")

sincos_mul_add = ElementwiseKernel(
        "double *ar, double a, double b, double *s, double *c",
        "sincos(a*ar[i]+b, &s[i], &c[i])",
        "sincos_mul_add")
sincos_mul_add_2 = ElementwiseKernel(
        "double *ar, double a, double b, double *s, double *c",
        "s[i] = cos(a*ar[i]+b -3.141592653589793238462643383279502884197169399375105820974944592307816406286/2); c[i] = cos(a*ar[i]+b)",
        "sincos_mul_add_2")

gpu_trapz = reduce.ReductionKernel(np.float64, neutral="0",reduce_expr="a+b",
        arguments = "double *y, double x, int sz",
        map_expr = "(i<sz-1) ? x*(y[i]+y[i+1])/2.0 : 0.0")

def gpu_trapz_2(ar1, dx, sz):
    res = gpuarray.zeros(1, np.float64)
    custom_gpu_trapz(ar1, np.float64(dx), np.int32(sz), res, block=(1024, 1,1), grid=(1,1,1))
    return res[0].get()
    #return gpu_trapz(ar1, np.float64(dx), sz).get()
    #return custom_trapz(ar1.get(), dx)

mul_d = ElementwiseKernel(
        "double *a1, double *a2",
        "a1[i] *= a2[i]",
        "mul_d")

## tracker 

add_kernel = ElementwiseKernel(
            "double *a, double *b, double *c",
            "a[i]=b[i]+c[i]",
            "add_kenerl")

first_kernel_tracker = ElementwiseKernel(
    "double *phi_rf, double x, double *phi_noise, int len, int turn",
    "phi_rf[len*i + turn] += x * phi_noise[len*i + turn]",
    "first_kernel_tracker")

second_kernel_tracker = ElementwiseKernel(
    "double *phi_rf, double *omega_rf, double *phi_mod0, double *phi_mod1, int size, int turn",
    "phi_rf[i*size+turn] += phi_mod0[i*size+turn]; omega_rf[i*size+turn] += phi_mod1[i*size+turn]",
    "second_kernel")

copy_column = ElementwiseKernel(
    "double *x, double *y, int size, int column",
    "x[i] = y[i*size + column]",
    "copu_column")

rf_voltage_calculation_kernel = ElementwiseKernel(
    "double *x, double *y, int size, int column",
    "x[i] = y[i*size + column]",
    "copu_column")

cavityFB_case = ElementwiseKernel(
    "double *rf_voltage, double *voltage, double *omega_rf, double *phi_rf,"+
    "double *bin_centers, double V_corr, double phi_corr,"+
    "int size, int column",
    "rf_voltage[i] = voltage[0] * V_corr * sin(omega_rf[0] * bin_centers[i]+phi_rf[0]+phi_corr)",
    "copu_column")
    
gpu_rf_voltage_calc_mem_ops = ker.get_function("gpu_rf_voltage_calc_mem_ops")

drv.init()
my_gpu = drv.Device(0)
cuinterp = ker.get_function("cuinterp")

plans_dict = {}
inverse_plans_dict = {}



def find_plan(my_size):
    if (my_size not in plans_dict):
        plans_dict[my_size] = fft.Plan(my_size, np.float64, np.complex128)
    return plans_dict[my_size]

def inverse_find_plan(size):
    if (size not in inverse_plans_dict):
        inverse_plans_dict[size] = fft.Plan(size, in_dtype=np.complex128, out_dtype=np.float64)
    return inverse_plans_dict[size]

def gpu_rfft(dev_a , n=0, result=None, caller_id = None):
    if (n == 0) and (result == None):
        n = dev_a.size
    elif (n != 0) and (result == None):
        pass    
    if (caller_id==None):
        result = gpuarray.zeros(n//2 + 1, np.complex128)
    else:
        result = get_gpuarray((n//2 + 1, np.complex128, 0, 'rfft'))
    result.fill(0)
    outSize = n // 2 + 1; 
    inSize = dev_a.size
    
    if (dev_a.dtype==np.int32):
        gpu_copy = gpu_copy_i2d
    else:
        gpu_copy = gpu_copy_d2d
    
    if (n == inSize):
        dev_in = get_gpuarray((n, np.float64, 0, 'rfft')) 
        gpu_copy(dev_in, dev_a, slice = slice(0,n))
    else:
        dev_in = get_gpuarray((n, np.float64, 0, 'rfft'))
        if (n < inSize):
            gpu_copy(dev_in, dev_a, slice = slice(0,n))
        else:
            dev_in.fill(0)
            gpu_copy(dev_in , dev_a, slice = slice(0,inSize))
    plan = find_plan(dev_in.shape)
    fft.fft(dev_in, result, plan)
    return result

def gpu_irfft(dev_a , n=0, result=None, caller_id=None):
    if (n == 0) and (result == None):
        n = 2*(dev_a.size-1)
    elif (n != 0) and (result == None):
        pass

    if (caller_id==None):
        result = gpuarray.zeros(n, dtype=np.float64)
    else:
        key = (n, np.float64, caller_id, 'irfft')
        result =  get_gpuarray(key)
    
    outSize = n
    inSize = dev_a.size
    
    if (outSize==0):
        outSize = 2*(inSize-1)
    n = outSize // 2 + 1

    if (n==inSize):
        dev_in = dev_a
    else:
        dev_in = get_gpuarray((n, np.complex128, 0, 'irfft'))
        if (n<inSize):
            gpu_complex_copy(dev_in, dev_a, slice = slice(0,n))
        else:
            gpu_complex_copy(dev_in, dev_a, slice = slice(0,n))
    
    inverse_plan = inverse_find_plan(outSize)
    fft.ifft(dev_in, result, inverse_plan, scale=True)
    return result

def gpu_rfftfreq(n, d=1.0, result=None):
    
    factor = 1/(d*n)
    result = factor*gpuarray.arange(0, n//2 + 1, dtype=np.float64).get()
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
    result1 = gpu_rfft(signal, result=result1,ret_gpu=True)
    result2 = gpu_rfft(kernel, result=result2,ret_gpu=True)
    result2 = result1*result2
    result = gpu_irfft(result2.get(), result=result).get()
    return result

def gpu_interp(dev_x, dev_xp, dev_yp, left=0.12345, right=0.12345, caller_id=None):
    if (caller_id==None):
        dev_res = get_gpuarray((dev_x.size, np.float64, caller_id, 'interp'))
    else:
        dev_res = gpuarray.zeros(dev_x.size, np.float64)
    cuinterp(   dev_x,  np.int32(dev_x.size), 
                dev_xp, np.int32(dev_xp.size),
                dev_yp, dev_res,
                np.float64(left), np.float64(right), 
                block = (1024,1,1), grid=(my_gpu.MULTIPROCESSOR_COUNT*2,1,1))
    return dev_res