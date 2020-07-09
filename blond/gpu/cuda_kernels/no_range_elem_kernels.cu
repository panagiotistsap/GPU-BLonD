
            #include <pycuda-complex.hpp>
            
            extern "C"
            __global__ void gpu_copy_i2d(double *x, int *y,long n)
            {
            unsigned tid = threadIdx.x;
            unsigned total_threads = gridDim.x*blockDim.x;
            unsigned cta_start = blockDim.x*blockIdx.x;
            unsigned i;
            ;
            for (i = cta_start + tid; i < n; i += total_threads)
            {
                x[i] = (double) y[i]*1.0;
            }
            ;
            }
            
            #include <pycuda-complex.hpp>
            
            extern "C"
            __global__ void gpu_copy_d2d(double *x,double *y,long n)
            {
            unsigned tid = threadIdx.x;
            unsigned total_threads = gridDim.x*blockDim.x;
            unsigned cta_start = blockDim.x*blockIdx.x;
            unsigned i;
            ;
            for (i = cta_start + tid; i < n; i += total_threads)
            {
                x[i] = y[i];
            }
            ;
            }
            
            #include <pycuda-complex.hpp>
            
            extern "C"
            __global__ void gpu_complex_copy(pycuda::complex<double> *x, pycuda::complex<double> *y,long n)
            {
            unsigned tid = threadIdx.x;
            unsigned total_threads = gridDim.x*blockDim.x;
            unsigned cta_start = blockDim.x*blockIdx.x;
            unsigned i;
            ;
            for (i = cta_start + tid; i < n; i += total_threads)
            {
                x[i] = y[i];
            }
            ;
            }
            
            #include <pycuda-complex.hpp>
            
            extern "C"
            __global__ void gpu_diff(int *a, double *b, double c,long n)
            {
            unsigned tid = threadIdx.x;
            unsigned total_threads = gridDim.x*blockDim.x;
            unsigned cta_start = blockDim.x*blockIdx.x;
            unsigned i;
            ;
            for (i = cta_start + tid; i < n; i += total_threads)
            {
                b[i] = (a[i+1]-a[i])/c;
            }
            ;
            }
            
            #include <pycuda-complex.hpp>
            
            extern "C"
            __global__ void set_zero_double(double *x,long n)
            {
            unsigned tid = threadIdx.x;
            unsigned total_threads = gridDim.x*blockDim.x;
            unsigned cta_start = blockDim.x*blockIdx.x;
            unsigned i;
            ;
            for (i = cta_start + tid; i < n; i += total_threads)
            {
                x[i] = 0;
            }
            ;
            }
            
            #include <pycuda-complex.hpp>
            
            extern "C"
            __global__ void set_zero_int(int *x,long n)
            {
            unsigned tid = threadIdx.x;
            unsigned total_threads = gridDim.x*blockDim.x;
            unsigned cta_start = blockDim.x*blockIdx.x;
            unsigned i;
            ;
            for (i = cta_start + tid; i < n; i += total_threads)
            {
                x[i] = 0;
            }
            ;
            }
            
            #include <pycuda-complex.hpp>
            
            extern "C"
            __global__ void set_zero_complex(pycuda::complex<double> *x,long n)
            {
            unsigned tid = threadIdx.x;
            unsigned total_threads = gridDim.x*blockDim.x;
            unsigned cta_start = blockDim.x*blockIdx.x;
            unsigned i;
            ;
            for (i = cta_start + tid; i < n; i += total_threads)
            {
                x[i] = 0;
            }
            ;
            }
            
            #include <pycuda-complex.hpp>
            
            extern "C"
            __global__ void increase_by_value(double *x, double a,long n)
            {
            unsigned tid = threadIdx.x;
            unsigned total_threads = gridDim.x*blockDim.x;
            unsigned cta_start = blockDim.x*blockIdx.x;
            unsigned i;
            ;
            for (i = cta_start + tid; i < n; i += total_threads)
            {
                x[i] += a;
            }
            ;
            }
            
            #include <pycuda-complex.hpp>
            
            extern "C"
            __global__ void add_array(double *x, double *y,long n)
            {
            unsigned tid = threadIdx.x;
            unsigned total_threads = gridDim.x*blockDim.x;
            unsigned cta_start = blockDim.x*blockIdx.x;
            unsigned i;
            ;
            for (i = cta_start + tid; i < n; i += total_threads)
            {
                x[i] += y[i];
            }
            ;
            }
            
            #include <pycuda-complex.hpp>
            
            extern "C"
            __global__ void complex_mul(pycuda::complex<double> *x, pycuda::complex<double> *y, pycuda::complex<double> *z,long n)
            {
            unsigned tid = threadIdx.x;
            unsigned total_threads = gridDim.x*blockDim.x;
            unsigned cta_start = blockDim.x*blockIdx.x;
            unsigned i;
            ;
            for (i = cta_start + tid; i < n; i += total_threads)
            {
                z[i] = x[i] * y[i];
            }
            ;
            }
            
            #include <pycuda-complex.hpp>
            
            extern "C"
            __global__ void gpu_mul(double *x, double *y, double a,long n)
            {
            unsigned tid = threadIdx.x;
            unsigned total_threads = gridDim.x*blockDim.x;
            unsigned cta_start = blockDim.x*blockIdx.x;
            unsigned i;
            ;
            for (i = cta_start + tid; i < n; i += total_threads)
            {
                x[i] = a*y[i];
            }
            ;
            }
            
            #include <pycuda-complex.hpp>
            
            extern "C"
            __global__ void gpu_copy_one(double *x, double *y, int ind,long n)
            {
            unsigned tid = threadIdx.x;
            unsigned total_threads = gridDim.x*blockDim.x;
            unsigned cta_start = blockDim.x*blockIdx.x;
            unsigned i;
            ;
            for (i = cta_start + tid; i < n; i += total_threads)
            {
                x[i] = y[ind];
            }
            ;
            }
            
            #include <pycuda-complex.hpp>
            
            extern "C"
            __global__ void first_kernel_x(double *omega_rf, double *harmonic,  double domega_rf, int size, int counter,long n)
            {
            unsigned tid = threadIdx.x;
            unsigned total_threads = gridDim.x*blockDim.x;
            unsigned cta_start = blockDim.x*blockIdx.x;
            unsigned i;
            ;
            for (i = cta_start + tid; i < n; i += total_threads)
            {
                omega_rf[i*size +counter] += domega_rf * harmonic[i*size + counter] / harmonic[counter];
            }
            ;
            }
            
            #include <pycuda-complex.hpp>
            
            extern "C"
            __global__ void second_kernel_x(double *dphi_rf, double *harmonic, double *omega_rf, double *omega_rf_d, int size, int counter, double pi,long n)
            {
            unsigned tid = threadIdx.x;
            unsigned total_threads = gridDim.x*blockDim.x;
            unsigned cta_start = blockDim.x*blockIdx.x;
            unsigned i;
            ;
            for (i = cta_start + tid; i < n; i += total_threads)
            {
                dphi_rf[i] +=  2.0*pi*harmonic[size*i+counter]*(omega_rf[size*i+counter]-omega_rf_d[size*i+counter])/omega_rf_d[size*i+counter];
            }
            ;
            }
            
            #include <pycuda-complex.hpp>
            
            extern "C"
            __global__ void third_kernel_x(double *x, double *y, int size_0, int counter,long n)
            {
            unsigned tid = threadIdx.x;
            unsigned total_threads = gridDim.x*blockDim.x;
            unsigned cta_start = blockDim.x*blockIdx.x;
            unsigned i;
            ;
            for (i = cta_start + tid; i < n; i += total_threads)
            {
                x[i*size_0 + counter] += y[i];
            }
            ;
            }
            
            #include <pycuda-complex.hpp>
            
            extern "C"
            __global__ void indexing_double(double *out, double *in, int *ind,long n)
            {
            unsigned tid = threadIdx.x;
            unsigned total_threads = gridDim.x*blockDim.x;
            unsigned cta_start = blockDim.x*blockIdx.x;
            unsigned i;
            ;
            for (i = cta_start + tid; i < n; i += total_threads)
            {
                out[i] = in[ind[i]];
            }
            ;
            }
            
            #include <pycuda-complex.hpp>
            
            extern "C"
            __global__ void indexing_int(double *out, int *in, int *ind,long n)
            {
            unsigned tid = threadIdx.x;
            unsigned total_threads = gridDim.x*blockDim.x;
            unsigned cta_start = blockDim.x*blockIdx.x;
            unsigned i;
            ;
            for (i = cta_start + tid; i < n; i += total_threads)
            {
                out[i] = in[ind[i]];
            }
            ;
            }
            
            #include <pycuda-complex.hpp>
            
            extern "C"
            __global__ void sincos_mul_add(double *ar, double a, double b, double *s, double *c,long n)
            {
            unsigned tid = threadIdx.x;
            unsigned total_threads = gridDim.x*blockDim.x;
            unsigned cta_start = blockDim.x*blockIdx.x;
            unsigned i;
            ;
            for (i = cta_start + tid; i < n; i += total_threads)
            {
                sincos(a*ar[i]+b, &s[i], &c[i]);
            }
            ;
            }
            
            #include <pycuda-complex.hpp>
            
            extern "C"
            __global__ void sincos_mul_add_2(double *ar, double a, double b, double *s, double *c,long n)
            {
            unsigned tid = threadIdx.x;
            unsigned total_threads = gridDim.x*blockDim.x;
            unsigned cta_start = blockDim.x*blockIdx.x;
            unsigned i;
            ;
            for (i = cta_start + tid; i < n; i += total_threads)
            {
                s[i] = cos(a*ar[i]+b -3.141592653589793238462643383279502884197169399375105820974944592307816406286/2); c[i] = cos(a*ar[i]+b);
            }
            ;
            }
            
            #include <pycuda-complex.hpp>
            
            extern "C"
            __global__ void mul_d(double *a1, double *a2,long n)
            {
            unsigned tid = threadIdx.x;
            unsigned total_threads = gridDim.x*blockDim.x;
            unsigned cta_start = blockDim.x*blockIdx.x;
            unsigned i;
            ;
            for (i = cta_start + tid; i < n; i += total_threads)
            {
                a1[i] *= a2[i];
            }
            ;
            }
            
            #include <pycuda-complex.hpp>
            
            extern "C"
            __global__ void add_kernel(double *a, double *b, double *c,long n)
            {
            unsigned tid = threadIdx.x;
            unsigned total_threads = gridDim.x*blockDim.x;
            unsigned cta_start = blockDim.x*blockIdx.x;
            unsigned i;
            ;
            for (i = cta_start + tid; i < n; i += total_threads)
            {
                a[i]=b[i]+c[i];
            }
            ;
            }
            
            #include <pycuda-complex.hpp>
            
            extern "C"
            __global__ void first_kernel_tracker(double *phi_rf, double x, double *phi_noise, int len, int turn,long n)
            {
            unsigned tid = threadIdx.x;
            unsigned total_threads = gridDim.x*blockDim.x;
            unsigned cta_start = blockDim.x*blockIdx.x;
            unsigned i;
            ;
            for (i = cta_start + tid; i < n; i += total_threads)
            {
                phi_rf[len*i + turn] += x * phi_noise[len*i + turn];
            }
            ;
            }
            
            #include <pycuda-complex.hpp>
            
            extern "C"
            __global__ void second_kernel_tracker(double *phi_rf, double *omega_rf, double *phi_mod0, double *phi_mod1, int size, int turn,long n)
            {
            unsigned tid = threadIdx.x;
            unsigned total_threads = gridDim.x*blockDim.x;
            unsigned cta_start = blockDim.x*blockIdx.x;
            unsigned i;
            ;
            for (i = cta_start + tid; i < n; i += total_threads)
            {
                phi_rf[i*size+turn] += phi_mod0[i*size+turn]; omega_rf[i*size+turn] += phi_mod1[i*size+turn];
            }
            ;
            }
            
            #include <pycuda-complex.hpp>
            
            extern "C"
            __global__ void copy_column(double *x, double *y, int size, int column,long n)
            {
            unsigned tid = threadIdx.x;
            unsigned total_threads = gridDim.x*blockDim.x;
            unsigned cta_start = blockDim.x*blockIdx.x;
            unsigned i;
            ;
            for (i = cta_start + tid; i < n; i += total_threads)
            {
                x[i] = y[i*size + column];
            }
            ;
            }
            
            #include <pycuda-complex.hpp>
            
            extern "C"
            __global__ void rf_voltage_calculation_kernel(double *x, double *y, int size, int column,long n)
            {
            unsigned tid = threadIdx.x;
            unsigned total_threads = gridDim.x*blockDim.x;
            unsigned cta_start = blockDim.x*blockIdx.x;
            unsigned i;
            ;
            for (i = cta_start + tid; i < n; i += total_threads)
            {
                x[i] = y[i*size + column];
            }
            ;
            }
            
            #include <pycuda-complex.hpp>
            
            extern "C"
            __global__ void cavityFB_case(double *rf_voltage, double *voltage, double *omega_rf, double *phi_rf,double *bin_centers, double V_corr, double phi_corr,int size, int column,long n)
            {
            unsigned tid = threadIdx.x;
            unsigned total_threads = gridDim.x*blockDim.x;
            unsigned cta_start = blockDim.x*blockIdx.x;
            unsigned i;
            ;
            for (i = cta_start + tid; i < n; i += total_threads)
            {
                rf_voltage[i] = voltage[0] * V_corr * sin(omega_rf[0] * bin_centers[i]+phi_rf[0]+phi_corr);
            }
            ;
            }
            
            #include <pycuda-complex.hpp>
            
            extern "C"
            __global__ void bm_phase_exp_times_scalar(double *a, double *b, double c, int *d,long n)
            {
            unsigned tid = threadIdx.x;
            unsigned total_threads = gridDim.x*blockDim.x;
            unsigned cta_start = blockDim.x*blockIdx.x;
            unsigned i;
            ;
            for (i = cta_start + tid; i < n; i += total_threads)
            {
                a[i] = exp(c*b[i])*d[i];
            }
            ;
            }
            
            #include <pycuda-complex.hpp>
            
            extern "C"
            __global__ void bm_phase_mul_add(double *a, double b, double *c, double d,long n)
            {
            unsigned tid = threadIdx.x;
            unsigned total_threads = gridDim.x*blockDim.x;
            unsigned cta_start = blockDim.x*blockIdx.x;
            unsigned i;
            ;
            for (i = cta_start + tid; i < n; i += total_threads)
            {
                a[i] = b*c[i] + d;
            }
            ;
            }
            
            #include <pycuda-complex.hpp>
            
            extern "C"
            __global__ void bm_sin_cos(double *a, double *b, double *c,long n)
            {
            unsigned tid = threadIdx.x;
            unsigned total_threads = gridDim.x*blockDim.x;
            unsigned cta_start = blockDim.x*blockIdx.x;
            unsigned i;
            ;
            for (i = cta_start + tid; i < n; i += total_threads)
            {
                sincos(a[i],&b[i], &c[i]);
            }
            ;
            }
            
            #include <pycuda-complex.hpp>
            
            extern "C"
            __global__ void d_multiply(double *a, double *b,long n)
            {
            unsigned tid = threadIdx.x;
            unsigned total_threads = gridDim.x*blockDim.x;
            unsigned cta_start = blockDim.x*blockIdx.x;
            unsigned i;
            ;
            for (i = cta_start + tid; i < n; i += total_threads)
            {
                a[i] *= b[i];
            }
            ;
            }
            
            #include <pycuda-complex.hpp>
            
            extern "C"
            __global__ void d_multscalar(double *a, double *b, double c,long n)
            {
            unsigned tid = threadIdx.x;
            unsigned total_threads = gridDim.x*blockDim.x;
            unsigned cta_start = blockDim.x*blockIdx.x;
            unsigned i;
            ;
            for (i = cta_start + tid; i < n; i += total_threads)
            {
                a[i] = c*b[i];
            }
            ;
            }
            
            #include <pycuda-complex.hpp>
            
            extern "C"
            __global__ void scale_kernel_int(int a, int *b,long n)
            {
            unsigned tid = threadIdx.x;
            unsigned total_threads = gridDim.x*blockDim.x;
            unsigned cta_start = blockDim.x*blockIdx.x;
            unsigned i;
            ;
            for (i = cta_start + tid; i < n; i += total_threads)
            {
                b[i] /= a ;
            }
            ;
            }
            
            #include <pycuda-complex.hpp>
            
            extern "C"
            __global__ void scale_kernel_double(double a, double *b,long n)
            {
            unsigned tid = threadIdx.x;
            unsigned total_threads = gridDim.x*blockDim.x;
            unsigned cta_start = blockDim.x*blockIdx.x;
            unsigned i;
            ;
            for (i = cta_start + tid; i < n; i += total_threads)
            {
                b[i] /= a ;
            }
            ;
            }
            