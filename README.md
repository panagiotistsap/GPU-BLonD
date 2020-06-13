# CuBLonD - GPU Version of BLonD

## Requirments 
1. Cuda
2. PyCuda
3. Scikit-Cuda 
## Installation Guide

### Cuda
 
Download and install cuda from the following link https://developer.nvidia.com/cuda-downloads.

### PyCuda 

To install pycuda open a terminal and type 

### Scikit-Cuda 

To install scikit-cuda open a terminal and type 
`$ pip install scikit-cuda`

### Verification
To verify your installation, in a python terminal type the following:
```
import pycuda.autoinit
from pycuda import gpuarray
import numpy as np
a = gpuarray.to_gpu(np.zeros(1000,np.float64))
```

### GPU-BLonD

Finally after cloning the repo and adding it to your pythonpath, install it like the default BLonD with 

`$ python blond/compile.py install` and the flags of your choice

## How to use your GPU

To use the GPU version of BLonD you need to follow these 2 steps  
1. You need to do that import ```import blond.utils.bmath as bm```
2. Right before your main loop you need to add the following line of code:
```
bm.use_gpu()
``` 
and call the ```use_gpu()``` method of all your basic components. For example if you have a profile, a tracker and a TotalInducedVoltage object you need to write these lines before your main loop
```
bm.use_gpu()
my_tracker.use_gpu()
my_profile.use_gpu()
my_totalinducedvoltage().use_gpu()
```
### More information
- You do not need to call the use_gpu method for components you pass to your tracker as arguments.  
- You can enable an optimization with ```bm.enable_gpucache()```
- If you have multiple GPUs and you want to use a specific one you can choose which one by giving its id as an argument to the bm.use_gpu() like that ```bm.use_gpu(1)```. To view your GPUs you can type ```nvidia-smi``` in your terminal.

## For lxplus Users
You need to add these lines to your ~/.bashrc
```
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export PYTHONPATH=~/work/GPU-BLonD:$PYTHONPATH
```


