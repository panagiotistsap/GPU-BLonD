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
from builtins import object
import numpy as np
from time import sleep
import traceback
import blond.utils.bmath as bm
from pycuda.compiler import SourceModule
import pycuda.reduction as reduce
from pycuda import gpuarray, driver as drv, tools
from blond.utils.bmath import gpu_num
from types import MethodType

drv.init()
dev = drv.Device(gpu_num)

def gpu_validate(self):
    if (not self.gpu_valid):
        self.dev_beam_dE = gpuarray.to_gpu(self._dE)
        self.dev_beam_dt = gpuarray.to_gpu(self._dt)
        self.dev_id    = gpuarray.to_gpu((self.id !=0).astype(np.float64))
        self.gpu_valid = True


@property
def dev_dE(self):
    self.gpu_validate()
    return self.dev_beam_dE


@dev_dE.setter
def dev_dE(self, value):
    self.gpu_validate()
    self.cpu_valid = False
    self.dev_beam_dE = value


@property
def dev_dt(self):
    self.gpu_validate()
    return self.dev_beam_dt


@dev_dt.setter
def dev_dt(self, value):
    self.gpu_validate()
    self.cpu_valid = False
    self.dev_beam_dt = value

#@property
# def gpu_n_macroparticles_lost(self):
#     return self.n_macroparticles - int(gpuarray.sum(self.dev_id).get())

#@property
# def gpu_n_macroparticles_alive(self):
#     return int(gpuarray.sum(self.dev_id).get())

################################
### G P U  F U N C T I O N S ###
################################


def funcs_update(obj):
    if (bm.get_exec_mode()=='GPU'):
        obj.losses_longitudinal_cut = MethodType(gpu_losses_longitudinal_cut,obj)
        obj.losses_energy_cut = MethodType(gpu_losses_energy_cut,obj)
        obj.losses_below_energy = MethodType(gpu_losses_below_energy,obj)
        obj.statistics = MethodType(gpu_statistics,obj)
        

def gpu_losses_longitudinal_cut(self, dt_min, dt_max):


    beam_ker = SourceModule("""
    __global__ void gpu_losses_longitudinal_cut(
                    double *dt, 
                    double *dev_id, 
                    const int size,
                    const double min_dt,
                    const double max_dt)
    {
            int tid = threadIdx.x + blockDim.x*blockIdx.x;
            for (int i = tid; i<size; i += blockDim.x*gridDim.x)
                if ((dt[i]-min_dt)*(max_dt-dt[i])<0)
                    dev_id[i]=0;
    }   

    """)
    gllc = beam_ker.get_function("gpu_losses_longitudinal_cut")
    gllc(self.dev_dt, self.dev_id, np.int32(self.n_macroparticles) , np.float64(dt_min), np.float64(dt_max),
        grid = (160, 1, 1), block =(1024, 1, 1))
    self.cpu_valid = False


def gpu_losses_energy_cut(self, dE_min, dE_max):

    beam_ker = SourceModule("""
    __global__ void gpu_losses_energy_cut(
                    double *dE, 
                    double *dev_id, 
                    const int size,
                    const double min_dE,
                    const double max_dE)
    {
            int tid = threadIdx.x + blockDim.x*blockIdx.x;
            for (int i = tid; i<size; i += blockDim.x*gridDim.x)
                if ((dE[i]-min_dE)*(max_dE-dE[i])<0)
                    dev_id[i]=0;
    }   

    """)
    glec = beam_ker.get_function("gpu_losses_energy_cut")
    glec(self.dev_dE, self.dev_id, np.int32(self.n_macroparticles) , np.float64(dE_min), np.float64(dE_max),
        grid = (160, 1, 1), block =(1024, 1, 1))
    self.cpu_valid = False


def gpu_losses_below_energy(self, dE_min):

    beam_ker = SourceModule("""
    __global__ void gpu_losses_below_energy(
                    double *dE, 
                    double *dev_id, 
                    const int size,
                    const double min_dE)
    {
            int tid = threadIdx.x + blockDim.x*blockIdx.x;
            for (int i = tid; i<size; i += blockDim.x*gridDim.x)
                if (dE[i]-min_dE < 0)
                    dev_id[i]=0;
    }   

    """)
    glbe = beam_ker.get_function("gpu_losses_energy_cut")
    glbe(self.dev_dE, self.dev_id, np.int32(self.n_macroparticles) , np.float64(dE_min),
        grid = (160, 1, 1), block =(1024, 1, 1))
    self.cpu_valid = False


def gpu_statistics(self):
    ones_sum = np.float64(gpuarray.sum(self.dev_id).get())
    self.mean_dt = np.float64(gpuarray.dot(self.dev_beam_dt, self.dev_id).get()/ones_sum)
    self.mean_dE = np.float64(gpuarray.dot(self.dev_dE, self.dev_id).get()/ones_sum)
    
    self.sigma_dt = np.float64(np.sqrt(self.stdKernel(self.dev_dt, self.dev_id, self.mean_dt).get()/ones_sum))
    self.sigma_dE = np.float64(np.sqrt(self.stdKernel(self.dev_dE, self.dev_id, self.mean_dE).get()/ones_sum))

    self.epsn_rms_l = np.pi*self.sigma_dE*self.sigma_dt  # in eVs





