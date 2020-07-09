#include <curand_kernel.h>

extern "C"   
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

extern "C"
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
