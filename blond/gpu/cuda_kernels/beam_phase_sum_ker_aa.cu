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

extern "C"
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