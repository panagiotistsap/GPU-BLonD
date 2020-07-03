
from ..utils import bmath as bm

grid_size = (2*bm.gpuDev().MULTIPROCESSOR_COUNT, 1, 1)
block_size = (bm.gpuDev().MAX_THREADS_PER_BLOCK, 1, 1)
