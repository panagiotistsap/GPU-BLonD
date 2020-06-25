from __future__ import division, print_function
from builtins import range, object
import numpy as np
from ..utils import bmath as bm
from types import MethodType
from ..utils.cucache import get_gpuarray
from ..gpu.gpu_butils_wrap import gpu_copy_d2d,first_kernel_tracker,second_kernel_tracker,\
        copy_column, rf_voltage_calculation_kernel, cavityFB_case, add_kernel, gpu_rf_voltage_calc_mem_ops

from pycuda.compiler import SourceModule
from pycuda import gpuarray, driver as drv, tools

drv.init()
dev = drv.Device(bm.gpu_num)



arrays_dict = {}




def gpu_track(self):
    """Tracking method for the section. Applies first the kick, then the 
    drift. Calls also RF/beam feedbacks if applicable. Updates the counter
    of the corresponding RFStation class and the energy-related variables
    of the Beam class.
    """
    turn = self.counter[0]

    # Add phase noise directly to the cavity RF phase
    if self.phi_noise is not None:
        if self.noiseFB is not None:
            
            first_kernel_tracker(self.rf_station.dev_phi_rf, self.noiseFB.x, self.dev_phi_noise,
                self.rf_station.dev_phi_rf.shape[0], turn, slice = slice(0,self.rf_station.n_rf))
            # self.phi_rf[:, turn] += \
            #     self.noiseFB.x * self.phi_noise[:, turn]
        else:
            first_kernel_tracker(self.rf_station.dev_phi_rf, 1.0, self.rf_station.dev_phi_noise, 
                self.rf_station.dev_phi_rf.shape[0], turn, slice = slice(0,self.rf_station.n_rf))
            # self.phi_rf[:, turn] += \
            #     self.phi_noise[:, turn]

    # Add phase modulation directly to the cavity RF phase
    if self.phi_modulation is not None:
        second_kernel_tracker(self.dev_phi_rf, self.dev_phi_modulation[0], self.dev_phi_modulation[1], self.dev_phi_rf.shape[0], turn)
        # self.phi_rf[:, turn] += \
        #     self.phi_modulation[0][:, turn]
        # self.omega_rf[:, turn] += \
        #     self.phi_modulation[1][:, turn]

        # Determine phase loop correction on RF phase and frequency
        
    if self.beamFB is not None and turn >= self.beamFB.delay:
        self.beamFB.track()
            
    if self.periodicity:

        # Distinguish the particles inside the frame from the particles on
        # the right-hand side of the frame.
        self.indices_right_outside = \
            np.where(self.beam.dt > self.t_rev[turn + 1])[0]
        self.indices_inside_frame = \
            np.where(self.beam.dt < self.t_rev[turn + 1])[0]

        if len(self.indices_right_outside) > 0:
            # Change reference of all the particles on the right of the
            # current frame; these particles skip one kick and drift
            self.beam.dt[self.indices_right_outside] -= \
                self.t_rev[turn + 1]
            # Synchronize the bunch with the particles that are on the
            # RHS of the current frame applying kick and drift to the
            # bunch
            # After that all the particles are in the new updated frame
            self.insiders_dt = np.ascontiguousarray(
                self.beam.dt[self.indices_inside_frame])
            self.insiders_dE = np.ascontiguousarray(
                self.beam.dE[self.indices_inside_frame])
            self.kick(self.insiders_dt, self.insiders_dE, turn)
            self.drift(self.insiders_dt, self.insiders_dE, turn+1)
            self.beam.dt[self.indices_inside_frame] = self.insiders_dt
            self.beam.dE[self.indices_inside_frame] = self.insiders_dE
            # Check all the particles on the left of the just updated
            # frame and apply a second kick and drift to them with the
            # previous wave after having changed reference.
            self.indices_left_outside = np.where(self.beam.dt < 0)[0]

        else:
            self.kick(self.beam.dt, self.beam.dE, turn)
            self.drift(self.beam.dt, self.beam.dE, turn + 1)
            # Check all the particles on the left of the just updated
            # frame and apply a second kick and drift to them with the
            # previous wave after having changed reference.
            self.indices_left_outside = np.where(self.beam.dt < 0)[0]

        if len(self.indices_left_outside) > 0:
            left_outsiders_dt = np.ascontiguousarray(
                self.beam.dt[self.indices_left_outside])
            left_outsiders_dE = np.ascontiguousarray(
                self.beam.dE[self.indices_left_outside])
            left_outsiders_dt += self.t_rev[turn+1]
            self.kick(left_outsiders_dt, left_outsiders_dE, turn)
            self.drift(left_outsiders_dt, left_outsiders_dE, turn+1)
            self.beam.dt[self.indices_left_outside] = left_outsiders_dt
            self.beam.dE[self.indices_left_outside] = left_outsiders_dE

    else:
        if self.rf_params.empty is False:
            if self.interpolation:
                self.rf_voltage_calculation()
                self.dev_total_voltage = get_gpuarray((self.dev_rf_voltage.size, np.float64, id(self), "dtv"))
                if self.totalInducedVoltage is not None:
                    add_kernel(self.dev_total_voltage, self.dev_rf_voltage, self.totalInducedVoltage.dev_induced_voltage)
                else:
                    self.dev_total_voltage = self.dev_rf_voltage
                bm.linear_interp_kick(dev_voltage=self.dev_total_voltage,
                        dev_bin_centers= self.profile.dev_bin_centers,
                        charge=self.beam.Particle.charge,
                        acceleration_kick=self.acceleration_kick[turn],
                        beam=self.beam)
            else:
                self.kick(turn)
        self.drift(turn + 1)
        
    # Updating the beam synchronous momentum etc.
    self.beam.beta = self.rf_params.beta[turn+1]
    self.beam.gamma = self.rf_params.gamma[turn+1]
    self.beam.energy = self.rf_params.energy[turn+1]
    self.beam.momentum = self.rf_params.momentum[turn+1]

    # Increment by one the turn counter
    self.counter[0] += 1


def gpu_rf_voltage_calculation(self):
    """Function calculating the total, discretised RF voltage seen by the
    beam at a given turn. Requires a Profile object.
    """
   
    dev_voltages = get_gpuarray((self.rf_station.n_rf, np.float64, id(self), "v"))
    dev_omega_rf = get_gpuarray((self.rf_station.n_rf, np.float64, id(self), "omega"))
    dev_phi_rf   = get_gpuarray((self.rf_station.n_rf, np.float64, id(self), "phi"))
    n_turns  = self.rf_station.n_turns+1
    
    sz = self.n_rf
    my_end = self.rf_station.dev_voltage.size
    gpu_rf_voltage_calc_mem_ops(dev_voltages, dev_omega_rf,dev_phi_rf,
            self.rf_station.dev_voltage, self.rf_station.dev_omega_rf,
            self.rf_station.dev_phi_rf, np.int32(self.counter[0]),
            np.int32(my_end), np.int32(n_turns), block=(32,1,1), grid=(1,1,1))

    self.dev_rf_voltage = get_gpuarray((self.profile.dev_bin_centers.size, np.float64, id(self), "rf_v"))
   
    # TODO: test with multiple harmonics, think about 800 MHz OTFB

    if self.cavityFB:
        cavityFB_case(self.dev_rf_voltage, dev_voltages,dev_omega_rf,
                    dev_phi_rf, self.profile.dev_bin_centers,
                    self.cavityFB.V_corr,self.cavityFB.phi_corr)
        # self.rf_voltage = voltages[0] * self.cavityFB.V_corr * \
        #     bm.sin(omega_rf[0]*self.profile.bin_centers +
        #             phi_rf[0] + self.cavityFB.phi_corr)
        bm.rf_volt_comp(dev_voltages, dev_omega_rf, dev_phi_rf,
            self.profile.dev_bin_centers, self.dev_rf_voltage, f_rf=1)
    else:
        bm.rf_volt_comp(dev_voltages, dev_omega_rf, dev_phi_rf,
            self.profile.dev_bin_centers, self.dev_rf_voltage)
    #print("rf voltage mean, std", np.mean(self.dev_rf_voltage.get()), np.std(self.dev_rf_voltage.get()))


def gpu_kick(self, index):
    """Function updating the particle energy due to the RF kick in a given
    RF station. The kicks are summed over the different harmonic RF systems
    in the station. The cavity phase can be shifted by the user via 
    phi_offset. The main RF (harmonic[0]) has by definition phase = 0 at 
    time = 0 below transition. The phases of all other RF systems are 
    defined w.r.t.\ to the main RF. The increment in energy is given by the
    discrete equation of motion:

    .. math::
        \Delta E^{n+1} = \Delta E^n + \sum_{k=0}^{n_{\mathsf{rf}}-1}{e V_k^n \\sin{\\left(\omega_{\mathsf{rf,k}}^n \\Delta t^n + \phi_{\mathsf{rf,k}}^n \\right)}} - (E_s^{n+1} - E_s^n) 

    """

    
   
    dev_voltage  = get_gpuarray((self.rf_station.n_rf, np.float64, id(self), "v"))
    dev_omega_rf = get_gpuarray((self.rf_station.n_rf, np.float64, id(self), "omega"))
    dev_phi_rf  = get_gpuarray((self.rf_station.n_rf, np.float64, id(self), "phi"))
    
    my_end = self.rf_station.dev_voltage.size

    dev_voltage[:] = self.rf_station.dev_voltage[index:my_end:self.rf_station.n_turns+1]
    dev_omega_rf[:] = self.rf_station.dev_omega_rf[index:my_end:self.rf_station.n_turns+1]
    dev_phi_rf[:] = self.rf_station.dev_phi_rf[index:my_end:self.rf_station.n_turns+1]

    bm.kick(dev_voltage, dev_omega_rf, dev_phi_rf,
        self.charge, self.n_rf, self.acceleration_kick[index], self.beam)
    self.beam.dE.invalidate_cpu()
    

def gpu_drift(self, index):
        bm.drift(self.solver, self.t_rev[index],
                self.length_ratio, self.alpha_order, self.eta_0[index],
                self.eta_1[index], self.eta_2[index], self.alpha_0[index],
                self.alpha_1[index], self.alpha_2[index],
                self.rf_params.beta[index], self.rf_params.energy[index], self.beam)
        self.beam.dt_obj.invalidate_cpu()

def tracker_funcs_update(obj):
    if (bm.get_exec_mode()=='GPU'):
        obj.track = MethodType(gpu_track, obj)
        obj.rf_voltage_calculation = MethodType(gpu_rf_voltage_calculation,obj)
        obj.kick = MethodType(gpu_kick,obj)
        obj.drift = MethodType(gpu_drift,obj)