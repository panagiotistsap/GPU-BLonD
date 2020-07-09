extern "C"
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

extern "C"
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

extern "C"
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
        