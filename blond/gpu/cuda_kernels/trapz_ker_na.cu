extern "C"
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
