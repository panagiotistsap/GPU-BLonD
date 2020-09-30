# coding: utf-8
# Copyright 2017 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Module containing the fundamental beam class with methods to compute beam
statistics

:Authors: **Danilo Quartullo**, **Helga Timko**, **ALexandre Lasheen**

"""

from __future__ import division
import numpy as np
from ..utils import bmath as bm
from pycuda import gpuarray
# , driver as drv, tools
from types import MethodType
from ..gpu.gpu_butils_wrap import stdKernel, sum_non_zeros, mean_non_zeros
from ..gpu import grid_size, block_size

from ..beam.beam import Beam

gllc = bm.getMod().get_function("gpu_losses_longitudinal_cut")
glec = bm.getMod().get_function("gpu_losses_energy_cut")
glbe = bm.getMod().get_function("gpu_losses_energy_cut")

class gpu_Beam(Beam):

    ## dE property

    @property
    def dE(self):
        return self.dE_obj.my_array

    @dE.setter
    def dE(self, value):
        self.dE_obj.my_array = value


    @property
    def dev_dE(self):
        return self.dE_obj.dev_my_array


    @dev_dE.setter
    def dev_dE(self, value):
        self.dE_obj.dev_my_array = value
        
    ## dt property

    @property
    def dt(self):
        return self.dt_obj.my_array

    @dt.setter
    def dt(self, value):
        self.dt_obj.my_array = value


    @property
    def dev_dt(self):
        return self.dt_obj.dev_my_array


    @dev_dt.setter
    def dev_dt(self, value):
        self.dt_obj.dev_my_array = value

    ## id property
    
    @property
    def id(self):
        return self.id_obj.my_array

    @id.setter
    def id(self, value):
        self.id_obj.my_array = value


    @property
    def dev_id(self):
        return self.id_obj.dev_my_array


    @dev_id.setter
    def dev_id(self, value):
        self.id_obj.dev_my_array = value
        

    @property
    def n_macroparticles_lost(self):
        return self.n_macroparticles - int(gpuarray.sum(self.dev_id).get())


    def losses_longitudinal_cut(self, dt_min, dt_max):

        
        gllc(self.dev_dt, self.dev_id, np.int32(self.n_macroparticles), bm.precision.c_real_t(dt_min), bm.precision.c_real_t(dt_max),
            grid=grid_size, block=block_size)
        self.id_obj.invalidate_cpu()


    def losses_energy_cut(self, dE_min, dE_max):

        
        glec(self.dev_dE, self.dev_id, np.int32(self.n_macroparticles), bm.precision.c_real_t(dE_min), bm.precision.c_real_t(dE_max),
            grid=grid_size, block=block_size)
        self.id_obj.invalidate_cpu()


    def losses_below_energy(self, dE_min):

        
        glbe(self.dev_dE, self.dev_id, np.int32(self.n_macroparticles), bm.precision.c_real_t(dE_min),
            grid=grid_size, block=block_size)
        self.id_obj.invalidate_cpu()


    def statistics(self):
        ones_sum = sum_non_zeros(self.dev_id).get()
        # print(self.dev_id.dtype)
        self.ones_sum = ones_sum
        self.mean_dt = bm.precision.c_real_t(mean_non_zeros(
            self.dev_dt, self.dev_id).get()/ones_sum)
        self.mean_dE = bm.precision.c_real_t(mean_non_zeros(
            self.dev_dE, self.dev_id).get()/ones_sum)

        self.sigma_dt = bm.precision.c_real_t(
            np.sqrt(stdKernel(self.dev_dt, self.dev_id, self.mean_dt).get()/ones_sum))
        self.sigma_dE = bm.precision.c_real_t(
            np.sqrt(stdKernel(self.dev_dE, self.dev_id, self.mean_dE).get()/ones_sum))

        self.epsn_rms_l = np.pi*self.sigma_dE*self.sigma_dt  # in eVs


