
# Copyright 2014-2017 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

'''
**Module to compute intensity effects**

:Authors: **Juan F. Esteban Mueller**, **Danilo Quartullo**,
          **Alexandre Lasheen**, **Markus Schwarz**
'''

from __future__ import division, print_function
from builtins import range, object
import numpy as np
from scipy.constants import e
from ..utils import bmath as bm
from types import MethodType
from ..gpu.cucache import get_gpuarray
from ..gpu.gpu_butils_wrap import gpu_copy_d2d, \
    increase_by_value, add_array, complex_mul, gpu_mul, gpu_interp, set_zero_real, d_multscalar

# import pycuda.reduction as reduce
import pycuda.cumath as cm
from pycuda import gpuarray
try:
    from pyprof import timing
except ImportError:
    from ..utils import profile_mock as timing
# , driver as drv, tools


# drv.init()
# dev = drv.Device(bm.gpuId())

def tiv_update_funcs(obj):
    if (bm.gpuMode()):
        obj.induced_voltage_sum = MethodType(gpu_induced_voltage_sum, obj)

from ..impedances.impedance import (TotalInducedVoltage, _InducedVoltage,
                            InducedVoltageFreq,InducedVoltageTime,
                            InductiveImpedance)
                            
class gpu_TotalInducedVoltage(TotalInducedVoltage):

    ## induced voltage

    @property
    def induced_voltage(self):
        return self.induced_voltage_obj.my_array

    @induced_voltage.setter
    def induced_voltage(self, value):
        self.induced_voltage_obj.my_array = value


    @property
    def dev_induced_voltage(self):
        return self.induced_voltage_obj.dev_my_array


    @dev_induced_voltage.setter
    def dev_induced_voltage(self, value):
        self.induced_voltage_obj.dev_my_array = value


    def induced_voltage_sum(self):
        """
        Method to sum all the induced voltages in one single array.
        """
        # For MPI, to avoid calulating beam spectrum multiple times
        beam_spectrum_dict = {}
        self.dev_induced_voltage = get_gpuarray(
            (self.profile.n_slices, bm.precision.real_t, id(self), 'iv'))
        set_zero_real(self.dev_induced_voltage)
        for induced_voltage_object in self.induced_voltage_list:
            induced_voltage_object.induced_voltage_generation(
                beam_spectrum_dict)
            if (not hasattr(induced_voltage_object, 'dev_induced_voltage')):
                induced_voltage_object.dev_induced_voltage = gpuarray.to_gpu(
                    induced_voltage_object.induced_voltage)
            add_array(self.dev_induced_voltage, induced_voltage_object.dev_induced_voltage,
                    slice=slice(0, self.profile.n_slices))



def iv_update_funcs(obj, is_ii=False):
    if (bm.gpuMode()):
        obj.shift_trev_freq = MethodType(gpu_shift_trev_freq, obj)
        obj.shift_trev_time = MethodType(gpu_shift_trev_time, obj)
        try:
            if (obj.mtw_mode == 'freq'):
                obj.shift_trev = MethodType(gpu_shift_trev_freq, obj)
            else:
                obj.shift_trev = MethodType(gpu_shift_trev_time, obj)
        except:
            pass

        if (is_ii):
            obj.induced_voltage_1turn = MethodType(
                ii_gpu_induced_voltage_1turn, obj)
        else:
            obj.induced_voltage_1turn = MethodType(
                gpu_induced_voltage_1turn, obj)
        obj.induced_voltage_mtw = MethodType(gpu_induced_voltage_mtw, obj)

        if (not obj.multi_turn_wake):
            if (is_ii):
                obj.induced_voltage_generation = MethodType(
                    ii_gpu_induced_voltage_1turn, obj)
            else:
                obj.induced_voltage_generation = MethodType(
                    gpu_induced_voltage_1turn, obj)
        else:
            obj.induced_voltage_generation = MethodType(
                gpu_induced_voltage_mtw, obj)

class gpu_InducedVoltage(_InducedVoltage):

    ## mtw_memory

    @property
    def mtw_memory(self):
        return self.mtw_memory_obj.my_array

    @mtw_memory.setter
    def mtw_memory(self, value):
        self.mtw_memory_obj.my_array = value


    @property
    def dev_mtw_memory(self):
        return self.mtw_memory_obj.dev_my_array


    @dev_mtw_memory.setter
    def dev_mtw_memory(self, value):
        self.mtw_memory_obj.dev_my_array = value

    ## total_impedance


    @property
    def total_impedance(self):
        return self.total_impedance_obj.my_array

    @total_impedance.setter
    def total_impedance(self, value):
        self.total_impedance_obj.my_array = value


    @property
    def dev_total_impedance(self):
        return self.total_impedance_obj.dev_my_array


    @dev_total_impedance.setter
    def dev_total_impedance(self, value):
        self.total_impedance_obj.dev_my_array = value



    def induced_voltage_1turn(self, beam_spectrum_dict={}):
        """
        Method to calculate the induced voltage at the current turn. DFTs are
        used for calculations in time and frequency domain (see classes below)
        """
        if self.n_fft not in beam_spectrum_dict:
            self.profile.beam_spectrum_generation(self.n_fft)
            beam_spectrum_dict[self.n_fft] = self.profile.dev_beam_spectrum
        self.profile.beam_spectrum_generation(self.n_fft)

        with timing.timed_region('serial:indVolt1Turn'):

            inp = get_gpuarray((self.profile.dev_beam_spectrum.size,
                                bm.precision.complex_t, id(self), 'inp'))
            complex_mul(self.dev_total_impedance, self.profile.dev_beam_spectrum, inp)
            # inp = self.dev_total_impedance * self.profile.dev_beam_spectrum
            my_res = bm.irfft(inp, caller_id=id(self))

            # dev_induced_voltage = - (self.beam.Particle.charge * e * self.beam.ratio *my_res )
            self.dev_induced_voltage = get_gpuarray(
                (self.n_induced_voltage, bm.precision.real_t, id(self), 'iv'))
            gpu_mul(self.dev_induced_voltage, my_res, bm.precision.real_t(-self.beam.Particle.charge *
                                                            e * self.beam.ratio), slice=slice(0, self.n_induced_voltage))


    def induced_voltage_mtw(self, beam_spectrum_dict={}):
        """
        Method to calculate the induced voltage taking into account the effect
        from previous passages (multi-turn wake)
        """
        # Shift of the memory wake field by the current revolution period
        self.shift_trev()

        # Induced voltage of the current turn calculation
        self.induced_voltage_1turn(beam_spectrum_dict)

        #print("mtw first point :", np.std(self.dev_induced_voltage.get()))
        # Setting to zero to the last part to remove the contribution from the
        # front wake
        # self.dev_induced_voltage[self.n_induced_voltage -
        #                     self.front_wake_buffer:] = 0

        set_zero_real(self.dev_induced_voltage, slice=slice(
            self.n_induced_voltage - self.front_wake_buffer, self.dev_induced_voltage.size))
        # Add the induced voltage of the current turn to the memory from
        # previous turns

        # self.mtw_memory[:self.n_induced_voltage] += self.induced_voltage
        add_array(self.dev_mtw_memory, self.dev_induced_voltage,
                slice=slice(0, self.n_induced_voltage))

        # self.induced_voltage = self.mtw_memory[:self.n_induced_voltage]
        self.dev_induced_voltage = get_gpuarray(
            (self.n_induced_voltage, bm.precision.real_t, id(self), 'mtw_iv'))
        gpu_copy_d2d(self.dev_induced_voltage, self.dev_mtw_memory,
                    slice=slice(0, self.n_induced_voltage))


    @timing.timeit(key='serial:shift_trev_freq')
    def shift_trev_freq(self):
        """
        Method to shift the induced voltage by a revolution period in the
        frequency domain
        """
        t_rev = self.RFParams.t_rev[self.RFParams.counter[0]]
        # Shift in frequency domain
        dev_induced_voltage_f = bm.rfft(self.dev_mtw_memory, self.n_mtw_fft)
        dev_induced_voltage_f *= cm.exp(self.dev_omegaj_mtw * t_rev)

        self.dev_mtw_memory = get_gpuarray(
            (self.n_mtw_memory, bm.precision.real_t, id(self), 'mtw_m'))
        dummy = bm.irfft(dev_induced_voltage_f, caller_id=self(id))
        gpu_copy_d2d(self.dev_mtw_memory, dummy, range=range(0, self.n_mtw_memory))
        set_zero_real(self.dev_mtw_memory, slice=slice(-int(self.buffer_size), None, None))


    @timing.timeit(key='serial:shift_trev_time')
    def shift_trev_time(self):
        """
        Method to shift the induced voltage by a revolution period in the
        frequency domain
        """
        t_rev = self.RFParams.t_rev[self.RFParams.counter[0]]
        inc_dev_time_mtw = get_gpuarray((self.dev_time_mtw.size, bm.precision.real_t, id(self), "time_mtw"))
        gpu_copy_d2d(inc_dev_time_mtw,self.dev_time_mtw)
        increase_by_value(inc_dev_time_mtw,t_rev)
        self.dev_mtw_memory = gpu_interp(inc_dev_time_mtw,
                                        self.dev_time_mtw, self.dev_mtw_memory,
                                        left=0, right=0, caller_id=id(self))


class gpu_InducedVoltageFreq(gpu_InducedVoltage, InducedVoltageFreq):
    pass

class gpu_InducedVoltageTime(gpu_InducedVoltage, InducedVoltageTime):


    def sum_wakes(self, time_array):
        self.total_wake = np.zeros(time_array.shape, dtype=bm.precision.real_t)
        for wake_object in self.wake_source_list:
            wake_object.wake_calc(time_array)
            self.total_wake += wake_object.wake

        # Pseudo-impedance used to calculate linear convolution in the
        # frequency domain (padding zeros)
        dev_total_wake = gpuarray.to_gpu(self.total_wake, dtype=bm.precision.real_t)
        self.dev_total_impedance = bm.rfft(
            self.total_wake, self.n_fft, caller_id=self(id))


class gpu_InductiveImpedance(gpu_InducedVoltage, InductiveImpedance):
    
    @timing.timeit(key='serial:InductiveImped')
    def induced_voltage_1turn(self, beam_spectrum_dict={}):
        """
        Method to calculate the induced voltage through the derivative of the
        profile. The impedance must be a constant Z/n.
        """
        
        index = self.RFParams.counter[0]

        sv = - (self.beam.Particle.charge * e / (2 * np.pi) *
                            self.beam.ratio * self.Z_over_n[index] *
                            self.RFParams.t_rev[index] / self.profile.bin_size)

        induced_voltage = self.profile.beam_profile_derivative(self.deriv_mode, caller_id=id(self))[1]
        d_multscalar(induced_voltage, induced_voltage, sv)

        self.dev_induced_voltage = get_gpuarray(
            (self.n_induced_voltage, bm.precision.real_t, id(self), "iv"))
        gpu_copy_d2d(self.dev_induced_voltage, induced_voltage,
                    slice=slice(0, self.n_induced_voltage))


def ii_update_funcs(obj):
    if (bm.gpuMode()):
        obj.induced_voltage_1turn = MethodType(
            ii_gpu_induced_voltage_1turn, obj)
