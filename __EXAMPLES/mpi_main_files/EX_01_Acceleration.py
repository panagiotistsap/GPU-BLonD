
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
# from __future__ import division, print_function
# from builtins import range
import numpy as np
import os
import time
try:
    from pyprof import timing
    from pyprof import mpiprof
except ImportError:
    from blond.utils import profile_mock as timing
    mpiprof = timing
#  BLonD Imports
from blond.monitors.monitors import SlicesMonitor
from blond.input_parameters.ring import Ring
from blond.input_parameters.rf_parameters import RFStation
from blond.trackers.tracker import RingAndRFTracker
from blond.beam.beam import Beam, Proton
from blond.beam.distributions import bigaussian
from blond.beam.profile import CutOptions, FitOptions, Profile
from blond.monitors.monitors import BunchMonitor
from blond.plots.plot import Plot
from blond.utils.input_parser import parse
from blond.utils import bmath as bm
from blond.utils.mpi_config import worker, mpiprint
bm.use_mpi()


this_directory = os.path.dirname(os.path.realpath(__file__)) + '/'


# Simulation parameters -------------------------------------------------------
# Bunch parameters
N_b = 1e9           # Intensity
n_particles = 1e6         # Macro-particles
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
n_turns = 1000000           # Number of turns to track
dt_plt = 1000         # Time steps between plots
n_slices = 100
n_bunches = 1
n_turns_reduce = 1
n_iterations = n_turns
seed = 0
worker.greet()
if worker.isMaster:
    worker.print_version()

# mpiprint('Done!')
# worker.finalize()
# exit()

args = parse()

n_iterations = n_iterations if args['turns'] == None else args['turns']
n_particles = n_particles if args['particles'] == None else args['particles']
n_bunches = n_bunches if args['bunches'] == None else args['bunches']
# n_turns_reduce = n_turns_reduce if args['reduce'] == None else args['reduce']
timing.mode = args['time']
os.environ['OMP_NUM_THREADS'] = str(args['omp'])
withtp = bool(args['withtp'])
seed = seed if args['seed'] == None else args['seed']
# approx = args['approx']
# precision = args['precision']
# bm.use_precision(precision)

worker.initLog(bool(args['log']), args['logdir'])
# worker.initTrace(bool(args['trace']), args['tracefile'])
worker.taskparallelism = withtp

mpiprint(args)


# Simulation setup ------------------------------------------------------------
mpiprint("Setting up the simulation...")


# Define general parameters
ring = Ring(C, alpha, np.linspace(p_i, p_f, n_turns+1), Proton(), n_turns)

# Define beam and distribution
beam = Beam(ring, n_particles, N_b)


# Define RF station parameters and corresponding tracker
rf = RFStation(ring, [h], [V], [dphi])


bigaussian(ring, rf, beam, tau_0/4, reinsertion=True, seed=seed)


# Need slices for the Gaussian fit
# TODO add the gaussian fit
profile = Profile(beam, CutOptions(n_slices=n_slices))
# FitOptions(fit_option='gaussian'))

long_tracker = RingAndRFTracker(rf, beam)

# beam.split_random()
beam.split()


if args['monitor'] > 0 and worker.isMaster:
    if args.get('monitorfile', None):
        filename = args['monitorfile']
    else:
        filename = 'monitorfiles/ex01-t{}-p{}-b{}-sl{}-approx{}-prec{}-r{}-m{}-se{}-w{}'.format(
            n_iterations, n_particles, n_bunches, n_slices, args['approx'], args['precision'],
            args['reduce'], args['monitor'], seed, worker.workers)
    slicesMonitor = SlicesMonitor(filename=filename,
                                  n_turns=np.ceil(
                                      n_iterations / args['monitor']),
                                  profile=profile,
                                  rf=rf,
                                  Nbunches=n_bunches)

# Accelerator map
# map_ = [long_tracker, profile]
mpiprint("Map set")

worker.initDLB(args['loadbalance'], args['loadbalancearg'], n_iterations)

worker.sync()
timing.reset()
start_t = time.time()


for turn in range(n_iterations):

    # Plot has to be done before tracking (at least for cases with separatrix)
    if (turn % dt_plt) == 0:
        mpiprint("Outputting at time step %d..." % turn)
        mpiprint("   Beam momentum %.6e eV" % beam.momentum)
        mpiprint("   Beam gamma %3.3f" % beam.gamma)
        mpiprint("   Beam beta %3.3f" % beam.beta)
        mpiprint("   Beam energy %.6e eV" % beam.energy)
        mpiprint("   Four-times r.m.s. bunch length %.4e s" %
                 (4.*beam.sigma_dt))
        mpiprint("   Gaussian bunch length %.4e s" % profile.bunchLength)
        mpiprint("")

    # Track
    long_tracker.track()

    profile.track()

    worker.DLB(turn, beam)

    if (args['monitor'] > 0) and (turn % args['monitor'] == 0):
        beam.statistics()
        beam.gather_statistics()
        if worker.isMaster:
            # profile.fwhm()
            slicesMonitor.track(turn)

        # mpiprint('dE mean: ', beam.mean_dE)
        # mpiprint('dE std: ', beam.sigma_dE)
        # mpiprint('dE min: ', beam.min_dE)
        # mpiprint('dE max: ', beam.max_dE)

        # mpiprint('dt mean: ', beam.mean_dt)
        # mpiprint('dt std: ', beam.sigma_dt)
        # mpiprint('dt min: ', beam.min_dt)
        # mpiprint('dt max: ', beam.max_dt)


beam.gather()

# mpiprint('real dE std: ', np.std(beam.dE))
# mpiprint('real dt std: ', np.std(beam.dt))

end_t = time.time()

timing.report(total_time=1e3*(end_t-start_t),
              out_dir=args['timedir'],
              out_file='worker-{}.csv'.format(worker.rank))
worker.finalize()


if args['monitor'] > 0:
    slicesMonitor.close()

mpiprint('dE mean: ', np.mean(beam.dE))
mpiprint('dE std: ', np.std(beam.dE))
mpiprint('profile mean: ', np.mean(profile.n_macroparticles))
mpiprint('profile std: ', np.std(profile.n_macroparticles))

mpiprint('Done!')
