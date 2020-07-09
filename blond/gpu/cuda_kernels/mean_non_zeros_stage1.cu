
            #include <pycuda-complex.hpp>
            #define BLOCK_SIZE 512
            #define READ_AND_MAP(i) ((id[i]!=0)*x[i])
            #define REDUCE(a, b) (a+b)
            
            typedef double out_type;
            extern "C"
            __global__
            void mean_non_zeros_stage1(out_type *out, double *x, double *id,
            unsigned int seq_count, unsigned int n)
            {
            // Needs to be variable-size to prevent the braindead CUDA compiler from
            // running constructors on this array. Grrrr.
            extern __shared__ out_type sdata[];
            unsigned int tid = threadIdx.x;
            unsigned int i = blockIdx.x*BLOCK_SIZE*seq_count + tid;
            out_type acc = 0;
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
            