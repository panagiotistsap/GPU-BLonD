
# Copyright 2014-2017 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3), 
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities 
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

'''
Example input for simulation of acceleration
No intensity effects

:Authors: **Helga Timko**
'''
#  General Imports
from __future__ import division, print_function
from builtins import range
import numpy as np
import sys
import time

#  BLonD Imports
from blond.input_parameters.ring import Ring
from blond.input_parameters.rf_parameters import RFStation
from blond.trackers.tracker import RingAndRFTracker
from blond.beam.beam import Beam, Proton
from blond.beam.distributions import bigaussian
from blond.beam.profile import CutOptions, FitOptions, Profile
from blond.monitors.monitors import BunchMonitor
from blond.plots.plot import Plot
from blond.utils.bmath import use_gpu
import os
print(sys.argv)
if (len(sys.argv)<3):
    print("... <num_of_particles> <dt_plt> <gpu>")
    exit(0)

if (len(sys.argv)==4):
    use_gpu()
    print("Using Gpu")

this_directory = os.path.dirname(os.path.realpath(__file__)) + '/'

try:
    os.mkdir(this_directory + '../output_files')
except:
    pass
try:
    os.mkdir(this_directory + '../output_files/EX_01_fig')
except:
    pass
start = time.time()
# Simulation parameters -------------------------------------------------------
# Bunch parameters
N_b = 1e9           # Intensity
N_p = int(sys.argv[1])         # Macro-particles
tau_0 = 0.4e-9          # Initial bunch length, 4 sigma [s]

# Machine and RF parameters
C = 26658.883        # Machine circumference [m]
p_i = 450e9         # Synchronous momentum [eV/c]
p_f = 460.005e9      # Synchronous momentum, final
h = 35640            # Harmonic number
V = 6e6                # RF voltage [V]
dphi = 0             # Phase modulation/offset
gamma_t = 55.759505  # Transition gamma
alpha = 1./gamma_t/gamma_t        # First order mom. comp. factor

# Tracking details
N_t = 2000           # Number of turns to track
dt_plt = int(sys.argv[2])         # Time steps between plots


# Simulation setup ------------------------------------------------------------
print("Setting up the simulation...")
print("")


# Define general parameters
ring = Ring(C, alpha, np.linspace(p_i, p_f, N_t+1), Proton(), N_t)

# Define beam and distribution
beam = Beam(ring, N_p, N_b)


# Define RF station parameters and corresponding tracker
rf = RFStation(ring, [h], [V], [dphi])
long_tracker = RingAndRFTracker(rf, beam)


bigaussian(ring, rf, beam, tau_0/4, reinsertion = True, seed=1)


# Need slices for the Gaussian fit
# profile = Profile(beam, CutOptions(n_slices=N_p//1000),
#                  FitOptions(fit_option='gaussian'))         
                     
# Define what to save in file
# bunchmonitor = BunchMonitor(ring, rf, beam,
#                           this_directory + '../output_files/EX_01_output_data', Profile=profile)

# format_options = {'dirname': this_directory + '../output_files/EX_01_fig'}
# plots = Plot(ring, rf, beam, dt_plt, N_t, 0, 0.0001763*h,
#              -400e6, 400e6, xunit='rad', separatrix_plot=True, 
#              Profile=profile, h5file=this_directory + '../output_files/EX_01_output_data', 
#              format_options=format_options)
end = time.time()
print(end - start)
print(np.mean(beam.dE))
print(np.mean(beam.dt))
# Accelerator map
map_ = [long_tracker] #+ [profile] #+ [bunchmonitor] + [plots]
print("Map set")
print("")
for m in map_:
    m.track()
#enable()
start = time.time()
# Tracking --------------------------------------------------------------------
for i in range(2, N_t+1):
        
    # Plot has to be done before tracking (at least for cases with separatrix)
    # if (i % dt_plt) == 0:
    #     bunchmonitor.track()
    #     plots.track()
    #     print("Outputting at time step %d..." %i)
    #     print("   Beam momentum %.6e eV" %beam.momentum)
    #     print("   Beam gamma %3.3f" %beam.gamma)
    #     print("   Beam beta %3.3f" %beam.beta)
    #     print("   Beam energy %.6e eV" %beam.energy)
    #     print("   Four-times r.m.s. bunch length %.4e s" %(4.*beam.sigma_dt))
    #     print("   Gaussian bunch length %.4e s" %profile.bunchLength)
    #     print("")
        
    # Track
    for m in map_:
        m.track()
    # Define losses according to separatrix and/or longitudinal position
    # beam.losses_separatrix(ring, rf)
    # beam.losses_longitudinal_cut(0., 2.5e-9)
end = time.time()
print(end - start)
print("Done!")
#report()
print(np.mean(beam.dE))
print(np.mean(beam.dt))
