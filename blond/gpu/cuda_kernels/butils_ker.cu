extern "C"
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

extern "C"
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


extern "C"
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

extern "C"
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