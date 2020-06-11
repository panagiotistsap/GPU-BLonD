import numpy as np
try:
    global gpuarray,drv
    import pycuda
    from pycuda import gpuarray, driver as drv
    import blond.utils.bmath as bm
    #bm.use_gpu()
    #drv.init()
    #my_gpu = drv.Device(bm.gpu_num)
except:
    pass


class my_gpuarray(pycuda.gpuarray.GPUArray):

         
    def set_parent(self, parent):
        self.parent = parent


    def __setitem__(self, key, value):
        p = self.parent
        if (not self.parent.gpu_valid):
            self.set(gpuarray.to_gpu(self.parent))
            self.parent.gpu_valid = True
        super().__setitem__(key, value)
        self.parent.cpu_valid = False
        return self
    
    def __getitem__(self, key):
        if (not self.parent.gpu_valid):
            self = gpuarray.to_gpu(self.parent)
            self.parent.gpu_valid = True
        return super(my_gpuarray, self).__getitem__(key)


class my_cpuarray(np.ndarray):

    def __new__(cls, inputarr):
        obj = np.asarray(inputarr).view(cls)
        obj.__class__ = my_cpuarray
        obj.cpu_valid = True
        obj.gpu_valid = False
        obj.x = inputarr.shape[0]
        try:
            obj.y = inputarr.shape[1]
        except:
            obj.y = 1
        try:
            obj.dev_array = gpuarray.to_gpu(inputarr)
            obj.dev_array.__class__ = my_gpuarray
            obj.dev_array.set_parent(obj)
            obj.gpu_valid = True
        except:
            obj.gpu_valid = False
        return obj



    def cpu_validate(self):
        
        if (not self.cpu_valid):
            self.cpu_valid = True
            super().__setitem__(slice(None, None, None), self.dev_array.get())
        self.cpu_valid = True
            
    def gpu_validate(self):
        if (not self.gpu_valid):
            self.dev_array.set(gpuarray.to_gpu(self))
        self.gpu_valid = True
            
    def __setitem__(self, key, value):
        self.cpu_validate()
        self.gpu_valid = False
        super(my_cpuarray, self).__setitem__(key, value)

    def __getitem__(self, key):
        self.cpu_validate()
        return super(my_cpuarray, self).__getitem__(key)

    
## example
class BigClass():
    def __init__(self):
        self._my_array = np.array([1,2,3,4]).astype(np.float64)
        self.array_obj = my_cpuarray(self._my_array)
        self._dev_array = self.array_obj.dev_array

    @property
    def my_array(self):
        self.array_obj.cpu_validate()
        return self.array_obj
    
    @my_array.setter
    def my_array(self, value):
        self.array_obj[:] = value

    @property
    def dev_my_array(self):
        self.array_obj.gpu_validate()
        return self.array_obj.dev_array
    
    @dev_my_array.setter
    def dev_my_array(self, value):
        self.array_obj.gpu_validate()
        self._dev_array[:] = value[:]
        self.array_obj.cpu_valid = False
        



# my_c = BigClass()
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