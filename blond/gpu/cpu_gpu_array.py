import numpy as np
from pycuda import gpuarray
from  ..utils import bmath as bm   

from pycuda import gpuarray
# from  ..utils import bmath as bm   

try:
    from pyprof import timing
except ImportError:
    from ..utils import profile_mock as timing

class my_gpuarray(gpuarray.GPUArray):
         
    def set_parent(self, parent):
        self.parent = parent

    def __setitem__(self, key, value):
        p = self.parent
        self.parent.gpu_validate()
        super().__setitem__(key, value)
        self.parent.cpu_valid = False
        return self
    
    def __getitem__(self, key):
        self.parent.gpu_validate()
        return super(my_gpuarray, self).__getitem__(key)


class my_cpuarray(np.ndarray):

    def __new__(cls, inputarr, dtype1=None, dtype2=None):
        if (inputarr is None):
            inputarr = np.array([],dtype=np.float64)

        obj = np.asarray(inputarr).view(cls)
        if (dtype1==None):
            obj.dtype1=inputarr.dtype
        else:
            obj.dtype1 = dtype1
        if (dtype2==None):
            obj.dtype2=inputarr.dtype
        else:
            obj.dtype2 = dtype2
        obj.__class__ = my_cpuarray
        obj.cpu_valid = True
        obj.gpu_valid = False
        obj.sp = inputarr.shape

        obj.dev_array = gpuarray.to_gpu(inputarr.flatten().astype(obj.dtype2))
        obj.dev_array.__class__ = my_gpuarray
        obj.dev_array.set_parent(obj)
        obj.gpu_valid = True

        return obj
  
    # @timing.timeit(key='serial:cpu_validate')
    def cpu_validate(self):
        
        if (not hasattr(self,"cpu_valid") or not self.cpu_valid):
            self.cpu_valid = True
            dummy = self.dev_array.get().reshape(self.sp).astype(self.dtype1)
            super().__setitem__(slice(None, None, None), dummy)
        self.cpu_valid = True
    
    # @timing.timeit(key='serial:gpu_validate')        
    def gpu_validate(self):
        if (not self.gpu_valid):
            self.dev_array.set(gpuarray.to_gpu(self.flatten().astype(self.dtype2)))

        self.gpu_valid = True
            
    def __setitem__(self, key, value):
    
        self.cpu_validate()
        self.gpu_valid = False
        super(my_cpuarray, self).__setitem__(key, value)

    def __getitem__(self, key):
        self.cpu_validate()
        if (len(self.shape)==1):
            return super(my_cpuarray, self).__getitem__(key)
        else:
            return np.array(super(my_cpuarray, self).__getitem__(key))

    
## example
class CGA():
    def __init__(self, inputarr, dtype1=None, dtype2=None):
        self.array_obj = my_cpuarray(inputarr, dtype1=dtype1, dtype2=dtype2)
        self._dev_array = self.array_obj.dev_array

    def invalidate_cpu(self):
        self.array_obj.cpu_valid = False
    
    def invalidate_gpu(self):
        self.array_obj.gpu_valid = False
        
    @property
    def my_array(self):
        self.array_obj.cpu_validate()
        return self.array_obj
    
    @my_array.setter
    def my_array(self, value):
        if (self.array_obj.size!=0 and value.dtype==self.array_obj.dtype1 and self.array_obj.shape == value.shape ):
           super(my_cpuarray, self.array_obj).__setitem__(slice(None, None, None), value)
        else:
            self.array_obj = my_cpuarray(value)
        
        self.array_obj.gpu_valid = False
        self.array_obj.cpu_valid = True

    @property
    def dev_my_array(self):
        self.array_obj.gpu_validate()
        return self.array_obj.dev_array
    
    @dev_my_array.setter
    def dev_my_array(self, value):
        if (self.array_obj.dev_array.size!=0 and value.dtype==self.array_obj.dtype2 and self.array_obj.dev_array.shape == value.shape):
            #super(my_gpuarray, self._dev_array).__setitem__(slice(None, None, None), value)
            self.array_obj.dev_array[:] = value
        else:
            self.array_obj = my_cpuarray(value.get())
            self.array_obj.dev_array = value
            self.array_obj.dev_array.__class__ = my_gpuarray
            self.array_obj.dev_array.set_parent(self.array_obj)
            self.array_obj.gpu_valid = True
            self.array_obj.cpu_valid = False

        self.array_obj.cpu_valid = False
        self.array_obj.gpu_valid = True
        

class ExampleClass():
    def __init__(self, bin_centers):
        self.bin_centers_obj = CGA(bin_centers)
    
    @property
    def bin_centers(self):
        return self.bin_centers_obj.my_array
    
    @bin_centers.setter
    def bin_centers(self, value):
        self.bin_centers_obj = value
    
    @property
    def dev_bin_centers(self):
        return self.bin_centers_obj.dev_my_array
    
    @dev_bin_centers.setter
    def dev_bin_centers(self,value):
        self.bin_centers_obj.dev_my_array = value





# C = ExampleClass(np.array([[1,2,3,4],[5,6,7,8]]).astype(np.float64))
# C.bin_centers[0] = 3
# print(C.dev_bin_centers)
# C.dev_bin_centers = gpuarray.zeros(8, np.float64)
# print(C.bin_centers)
# c = my_c.my_array
# #1,2,3,4
# print("chill")
# my_c.my_array[0]=15

# print("gpu array :", my_c.dev_my_array)
# print("cpu array :", my_c.my_array)

# my_c.dev_my_array = gpuarray.zeros(4, dtype=np.float64)
# # my_c.my_array[:] = np.array([1,2,3,4])
# print("gpu array :", my_c.dev_my_array)
# print("cpu array :", my_c.my_array)

# my_c.my_array[:] = np.array([4,5,6,7])

# print("gpu array :", my_c.dev_my_array)
# print("cpu array :", my_c.my_array)
# #print(d)