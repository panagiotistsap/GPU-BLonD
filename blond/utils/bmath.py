'''
BLonD math and physics core functions

@author Stefan Hegglin, Konstantinos Iliakis
@date 20.10.2017
'''

import numpy as np
from ..utils import butils_wrap
from ..utils import bphysics_wrap
from numpy import fft
import os

precision = butils_wrap.precision

__exec_mode = 'single_node'
__gpu_dev = None

# dictionary storing the CPU versions of the desired functions #
_CPU_func_dict = {
    'rfft': fft.rfft,
    'irfft': fft.irfft,
    'rfftfreq': fft.rfftfreq,
    'irfft_packed': butils_wrap.irfft_packed,
    'sin': butils_wrap.sin,
    'cos': butils_wrap.cos,
    'exp': butils_wrap.exp,
    'mean': butils_wrap.mean,
    'std': butils_wrap.std,
    'where': butils_wrap.where,
    'interp': butils_wrap.interp,
    'interp_const_space': butils_wrap.interp_const_space,
    'cumtrapz': butils_wrap.cumtrapz,
    'trapz': butils_wrap.trapz,
    'linspace': butils_wrap.linspace,
    'argmin': butils_wrap.argmin,
    'argmax': butils_wrap.argmax,
    'convolve': butils_wrap.convolve,
    'arange': butils_wrap.arange,
    'sum': butils_wrap.sum,
    'sort': butils_wrap.sort,
    'add': butils_wrap.add,
    'mul': butils_wrap.mul,
    'beam_phase': bphysics_wrap.beam_phase,
    'fast_resonator': bphysics_wrap.fast_resonator,
    'kick': bphysics_wrap.kick,
    'rf_volt_comp': bphysics_wrap.rf_volt_comp,
    'drift': bphysics_wrap.drift,
    'linear_interp_kick': bphysics_wrap.linear_interp_kick,
    'LIKick_n_drift': bphysics_wrap.linear_interp_kick_n_drift,
    'synchrotron_radiation': bphysics_wrap.synchrotron_radiation,
    'synchrotron_radiation_full': bphysics_wrap.synchrotron_radiation_full,
    # 'linear_interp_time_translation': bphysics_wrap.linear_interp_time_translation,
    'slice': bphysics_wrap.slice,
    'slice_smooth': bphysics_wrap.slice_smooth,
    'music_track': bphysics_wrap.music_track,
    'music_track_multiturn': bphysics_wrap.music_track_multiturn,
    'diff': np.diff,
    'cumsum': np.cumsum,
    'cumprod': np.cumprod,
    'gradient': np.gradient,
    'sqrt': np.sqrt,
    'device': 'CPU'
}

_FFTW_func_dict = {
    'rfft': butils_wrap.rfft,
    'irfft': butils_wrap.irfft,
    'rfftfreq': butils_wrap.rfftfreq
}

_MPI_func_dict = {

}

def use_fftw():
    '''
    Replace the existing rfft and irfft implementations
    with the ones coming from butils_wrap.
    '''
    globals().update(_FFTW_func_dict)


# precision can be single or double
def use_precision(_precision='double'):
    global precision
    if _precision == 'single':
        print('WARNING: Only double precision supported')
        _precision = 'double'
    butils_wrap.precision = butils_wrap.Precision(_precision)
    precision = butils_wrap.precision


def use_mpi():
    '''
    Replace some bm functions with MPI implementations
    '''
    global __exec_mode
    globals().update(_MPI_func_dict)
    __exec_mode = 'multi_node'


def mpiMode():
    global __exec_mode
    return __exec_mode == 'multi_node'

def gpuMode():
    return globals()['device'] == 'GPU'


def enable_gpucache():
    from ..gpu import cucache as cc
    cc.enable_cache()


def disable_gpucache():
    from  ..gpu import cucache as cc
    cc.disable_cache()

def gpuId():
    return __gpu_dev.id

def gpuDev():
    return __gpu_dev.dev

def gpuCtx():
    return __gpu_dev.ctx

def getMod():
    return __gpu_dev.my_mod()
# def stop_gpu():
#     __gpu_ctx.pop()
#     update_active_dict(_CPU_func_dict)


class GPUDev:
    __instance = None
    def __init__(self, _gpu_num=0):
        if GPUDev.__instance != None:
            raise Exception("The GPUDev class is a singleton!")
        else:
            GPUDev.__instance = self
        from pycuda import driver as drv
        drv.init()
        self.id = _gpu_num
        self.dev = drv.Device(self.id)
        self.ctx = self.dev.make_context()
        this_dir = os.path.dirname(os.path.realpath(__file__)) + "/"

        self.mod = drv.module_from_file(os.path.join(this_dir, '../gpu/cuda_kernels/kernels.cubin'))



    def report_attributes(self):
        # Saves into a file all the device attributes 
        with open(f'{self.dev.name()}-attributes.txt', 'w') as f:
            for k, v in self.dev.get_attributes().items():
                f.write(f"{k}:{v}\n")

    def func(self, name):
        return self.mod.get_function(name)

    def __del__(self):
        self.ctx.pop()
        update_active_dict(_CPU_func_dict)

    def my_mod(self):
        return self.mod
    

def use_gpu(comps=[], gpu_id=0):
    from pycuda import driver as drv

    print("USING GPU")
    global __gpu_dev
    __gpu_dev = GPUDev(gpu_id)
    globals()['device'] = 'GPU'
    from ..gpu import gpu_physics_wrap
    from ..gpu import gpu_butils_wrap
    

    for obj in comps:
        if (hasattr(obj, "use_gpu")):
            print("using gpu")
            obj.use_gpu()
            
    _GPU_func_dict = {
        'rfft': gpu_butils_wrap.gpu_rfft,
        'irfft': gpu_butils_wrap.gpu_irfft,
        'rfftfreq': fft.rfftfreq,
        'irfft_packed': butils_wrap.irfft_packed,
        'sin': butils_wrap.sin,
        'cos': butils_wrap.cos,
        'exp': butils_wrap.exp,
        'mean': butils_wrap.mean,
        'std': butils_wrap.std,
        'where': butils_wrap.where,
        'interp': butils_wrap.interp,
        'interp_const_space': butils_wrap.interp_const_space,
        'cumtrapz': butils_wrap.cumtrapz,
        'trapz': butils_wrap.trapz,
        'linspace': butils_wrap.linspace,
        'argmin': butils_wrap.argmin,
        'argmax': butils_wrap.argmax,
        'convolve': gpu_butils_wrap.gpu_convolve,
        'arange': butils_wrap.arange,
        'sum': butils_wrap.sum,
        'sort': butils_wrap.sort,
        'add': butils_wrap.add,
        'mul': butils_wrap.mul,
        'beam_phase': gpu_physics_wrap.gpu_beam_phase,
        'fast_resonator': bphysics_wrap.fast_resonator,
        'kick': gpu_physics_wrap.gpu_kick,
        'rf_volt_comp': gpu_physics_wrap.gpu_rf_volt_comp,
        'drift' : gpu_physics_wrap.gpu_drift,
        'linear_interp_kick': gpu_physics_wrap.gpu_linear_interp_kick,
        'LIKick_n_drift': bphysics_wrap.linear_interp_kick_n_drift,
        'synchrotron_radiation': gpu_physics_wrap.gpu_synchrotron_radiation,
        'synchrotron_radiation_full': gpu_physics_wrap.gpu_synchrotron_radiation_full,
        # 'linear_interp_time_translation': bphysics_wrap.linear_interp_time_translation,
        'slice': gpu_physics_wrap.gpu_slice,
        'slice_smooth': bphysics_wrap.slice_smooth,
        'music_track': bphysics_wrap.music_track,
        'music_track_multiturn': bphysics_wrap.music_track_multiturn,
        'diff': np.diff,
        'cumsum': np.cumsum,
        'cumprod': np.cumprod,
        'gradient': np.gradient,
        'sqrt': np.sqrt,
        'device': 'GPU'
    }
    update_active_dict(_GPU_func_dict)
# print ('Available functions on GPU:\n' + str(_CPU_numpy_func_dict.keys()))
# print ('Available functions on CPU:\n' + str(_GPU_func_dict.keys()))


def update_active_dict(new_dict):
    '''
    Update the currently active dictionary. Removes the keys of the currently
    active dictionary from globals() and spills the keys
    from new_dict to globals()
    Args:
        new_dict A dictionary which contents will be spilled to globals()
    '''
    if not hasattr(update_active_dict, 'active_dict'):
        update_active_dict.active_dict = new_dict

    # delete all old implementations/references from globals()
    for key in update_active_dict.active_dict.keys():
        if key in globals():
            del globals()[key]
    # for key in globals().keys():
    #     if key in update_active_dict.active_dict.keys():
    #         del globals()[key]
    # add the new active dict to the globals()
    globals().update(new_dict)
    update_active_dict.active_dict = new_dict


################################################################################
update_active_dict(_CPU_func_dict)
################################################################################

