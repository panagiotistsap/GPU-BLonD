from pycuda.compiler import SourceModule
import pycuda.driver as drv 
import os
basedir = os.path.dirname(os.path.realpath(__file__))+"/cuda_kernels/"

get_precompiled = 1
create_files = 0
if (get_precompiled):
    try:
        beam_phase_sum_ker = drv.module_from_file(basedir+"beam_phase_sum_ker_aa.cubin")
        trapz_ker = drv.module_from_file(basedir+"trapz_ker_aa.cubin")
    except:
        beam_phase_sum_ker = drv.module_from_file(basedir+"beam_phase_sum_ker_na.cubin")
        trapz_ker = drv.module_from_file(basedir+"trapz_ker_na.cubin")
    physics_ker = drv.module_from_file(basedir+"physics_ker.cubin")
    synch_rad_ker = drv.module_from_file(basedir+"synch_rad_ker.cubin")
    butils_ker = drv.module_from_file(basedir+"butils_ker.cubin")
    beam_ker = drv.module_from_file(basedir+"beam_ker.cubin")
else:
    try:
        beam_phase_sum_ker = SourceModule("""
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
        trapz_ker = SourceModule("""
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
        beam_phase_sum_ker = SourceModule("""
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
        trapz_ker = SourceModule("""
                
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

    physics_ker = SourceModule("""

        

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

    beam_ker = SourceModule("""
        __global__ void gpu_losses_longitudinal_cut(
                        double *dt, 
                        double *dev_id, 
                        const int size,
                        const double min_dt,
                        const double max_dt)
        {
                int tid = threadIdx.x + blockDim.x*blockIdx.x;
                for (int i = tid; i<size; i += blockDim.x*gridDim.x)
                    if ((dt[i]-min_dt)*(max_dt-dt[i])<0)
                        dev_id[i]=0;
        }   

        __global__ void gpu_losses_energy_cut(
                        double *dE, 
                        double *dev_id, 
                        const int size,
                        const double min_dE,
                        const double max_dE)
        {
                int tid = threadIdx.x + blockDim.x*blockIdx.x;
                for (int i = tid; i<size; i += blockDim.x*gridDim.x)
                    if ((dE[i]-min_dE)*(max_dE-dE[i])<0)
                        dev_id[i]=0;
        } 

        __global__ void gpu_losses_below_energy(
                        double *dE, 
                        double *dev_id, 
                        const int size,
                        const double min_dE)
        {
                int tid = threadIdx.x + blockDim.x*blockIdx.x;
                for (int i = tid; i<size; i += blockDim.x*gridDim.x)
                    if (dE[i]-min_dE < 0)
                        dev_id[i]=0;
        } 

        """)
    
    butils_ker = SourceModule("""

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