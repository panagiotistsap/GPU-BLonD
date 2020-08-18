extern "C"
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

extern "C"
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

extern "C"
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

extern "C"
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

extern "C"
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


extern "C"
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

extern "C"
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

extern "C"
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


extern "C"
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

extern "C"
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


extern "C"
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


extern "C"
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


extern "C"
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


extern "C"
__global__ void lik_drift_only_gm_comp(
        double *beam_dt,
        double *beam_dE,
        const double *voltage_array,
        const double *bin_centers,
        const double charge,
        const int n_slices,
        const int n_macroparticles,
        const double acc_kick,
        double *glob_voltageKick,
        double *glob_factor,
        const double T0, const double length_ratio, 
        const double eta0, const double beta, const double energy
        )
{
    const double T = T0 * length_ratio * eta0 / (beta * beta * energy);

    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    double const inv_bin_width = (n_slices-1)
        /(bin_centers[n_slices-1]-bin_centers[0]);
    unsigned fbin;
    const double bin0 = bin_centers[0];
    for (int i=tid; i<n_macroparticles; i += blockDim.x*gridDim.x) {
        fbin = (unsigned) floor((beam_dt[i] - bin0) * inv_bin_width);
        if ((fbin < n_slices - 1))
            beam_dE[i] += beam_dt[i] * glob_voltageKick[fbin] + glob_factor[fbin];
        // beam_dt[i] += T * (1. / (1. - eta0 * beam_dE[i]) -1.);
        beam_dt[i] += T * beam_dE[i];
    }
}

extern "C"
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