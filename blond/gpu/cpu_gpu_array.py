import numpy as np
try:
    global gpuarray,drv
    from pycuda import gpuarray, driver as drv
    import blond.utils.bmath as bm
    bm.use_gpu()
    drv.init()
    my_gpu = drv.Device(bm.gpu_num)
except:
    # missing pycuda
    pass

# class custom_gpuarray():
#     def __init__(self,cpu_array):
#         self.my_cpu_array = cpu_array
#         self.dtype = cpu_array.dtype
#         self.cpu_valid = True
#         self.gpu_valid = False
#         try:
#             self.my_gpu_array = gpuarray.to_gpu(self.my_cpu_array)
#             self.gpu_valid = True
#         except:
#             # missing pycuda
#             pass
    
#     def __setitem__(self, key, new_value):
#         if (isinstance(new_value,np.ndarray)):
#             self.my_cpu_array[key] = new_value
#             self.my_gpu_array[key] = gpuarray.to_gpu(new_value.astype(self.dtype))

#         elif (isinstance(new_value,gpuarray.GPUArray)):
#             self.my_gpu_array[key] = new_value
#             self.my_cpu_array[key] = new_value.get()

#     @property
#     def cpu_value(self):
#         return self.my_cpu_array[:]
      
   
#     @cpu_value.setter
#     def cpu_value(self,value):
#         self.my_cpu_array[:] = value
#         self.gpu_valid = False
    
#     @property
#     def gpu_value(self):
#         return self.my_gpu_array 


#     @gpu_value.setter
#     def gpu_value(self, value):
#         self.my_gpu_array = value

#     def validate_cpu(self):
#         if (not self.cpu_valid):
#             self.cpu_array = gpuarray.to_gpu(self.my_gpu_array)
        
class my_array(np.ndarray):

    def __init__(self, npar):
        self.ar = npar

    #@property
    #def __call__(self):
    #    return self.ar

    def __getitem__(self, key):
        #self.validate()
        return self.ar[key]

    def __setitem__(self,key,value):
        #self.invalidate_gpu()
        self[key] = value
    
    def validate(self):
        print("yo")
        #if (not self.parent.cpu_valid):
        #    self.ar = self.parent.gpu_obj.ar.get()

