import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gSpec
# from IPython import display
import time as tm
import scipy.constants as cont
import re
import subprocess
import sys
import matplotlib.animation as anim
import scipy.optimize as opt
import ctypes
import scipy.fftpack as fft
import yaml
import argparse
import os

try:
    from pyprof import timing
    from pyprof import mpiprof
except ImportError:
    from blond.utils import profile_mock as timing
    mpiprof = timing

# BLonD
import blond.input_parameters.ring as inputRing
import blond.input_parameters.ring_options as ringOpt
import blond.input_parameters.rf_parameters as inputRF
import blond.trackers.tracker as tracker
import blond.beam.beam as beam
import blond.beam.profile as prof
import blond.beam.coasting_beam as coastBeam
import blond.beam.distributions as distBeam
import blond.impedances.impedance_sources as impSource
import blond.impedances.impedance as imp
# import blond.utils.track_iteration as trackIt
import blond.llrf.offset_frequency as offFreq
import blond.llrf.rf_modulation as rfMod
import blond.llrf.beam_feedback as bfb
from blond.utils.mpi_config import worker, mpiprint
from blond.utils.input_parser import parse
from blond.monitors.monitors import SlicesMonitor

from blond.utils import bmath as bm

bm.use_mpi()
bm.use_fftw()

this_directory = os.path.dirname(os.path.realpath(__file__)) + '/'
inputDir = os.path.join(this_directory, '../input_files/PSB/')


n_particles = int(5E5)
n_slices = 2**7
n_bunches = 1
n_iterations = 10000

worker.greet()
if worker.isMaster:
    worker.print_version()
os.system("gcc --version")



args = parse()

n_iterations = n_iterations if args['turns'] == None else args['turns']
n_particles = n_particles if args['particles'] == None else args['particles']
n_bunches = n_bunches if args['bunches'] == None else args['bunches']
n_turns_reduce = n_turns_reduce if args['reduce'] == None else args['reduce']
seed = seed if args['seed'] == None else args['seed']
approx = args['approx']
timing.mode = args['time']
os.environ['OMP_NUM_THREADS'] = str(args['omp'])
withtp = bool(args['withtp'])
precision = args['precision']
# bm.use_precision(precision)


worker.initLog(bool(args['log']), args['logdir'])
# worker.initTrace(bool(args['trace']), args['tracefile'])
worker.taskparallelism = withtp

mpiprint(args)


#%%

initial_time = 490E-3
final_time = 805E-3
periodicity_tracking = False

targetIntensity = 350E10

#%%

#BLonD Simulation

radius = 25
gamma_transition = 4.07#psb.full_pars['gammaT']  # [1]
C = 2 * np.pi * radius  # [m]       
momentum_compaction = 1 / gamma_transition**2 # [1]
particle_type = 'proton'


momProg = np.load(os.path.join(inputDir, "newTrialMomentum.npy"))


#%%

ring_options = ringOpt.RingOptions("derivative", t_start = initial_time, 
                                   t_end = final_time)
ring = inputRing.Ring(C, momentum_compaction, (momProg[0], momProg[1]), 
                      beam.Proton(), RingOptions=ring_options)

n_turns = ring.n_turns
print("N Turns: " + str(n_turns))



#%%


voltage = np.load(os.path.join(inputDir, "doubleHFixedAreaVoltsLargeAcceptFullCycleH3Blow.npy"))
phase = np.load(os.path.join(inputDir, "doubleHFixedAreaPhaseLargeAcceptFullCycleH3Blow.npy"))
voltsH1 = [voltage[0], voltage[1]]
voltsH2 = [voltage[0], voltage[2]]
voltsH3 = [voltage[0], voltage[3]]
phaseH1 = [phase[0], phase[1]]
phaseH2 = [phase[0], phase[2]]
phaseH3 = [phase[0], phase[3]]

rf_params = inputRF.RFStation(ring, [1, 2, 3], (voltsH1, voltsH2, voltsH3), \
                              (phaseH1, phaseH2, phaseH3), 3)

#%%

noiseSpan = np.load(os.path.join(inputDir, "freqsH3Blow.npy"))
noise = ctypes.cdll.LoadLibrary(os.path.join(inputDir, './libnoise.so'))

nPoints = 10000
seed = 12345
nSamples = 200000

time = np.linspace(noiseSpan[0][0], noiseSpan[0][-1], nPoints)

fs0 = np.interp(time, noiseSpan[0], noiseSpan[1]*1.02)
fs1 = np.interp(time, noiseSpan[0], noiseSpan[2]*1.05)

np.random.seed(seed)
noiseProg = np.zeros(len(time))
rNums1 = np.random.rand(nSamples)
rNums2 = np.random.rand(nSamples)

noise.test_func(noiseProg.ctypes.data_as(ctypes.c_void_p),
                rNums1.ctypes.data_as(ctypes.c_void_p),
                rNums2.ctypes.data_as(ctypes.c_void_p),
                fs0.ctypes.data_as(ctypes.c_void_p),
                fs1.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_double(time[1]-time[0]),
                ctypes.c_uint(nSamples), ctypes.c_uint(len(time)))

noiseProg /= np.std(noiseProg)
noiseProg *= np.interp(time, [time[0], time[0]+0.010, time[-1]-0.010, 
                              time[-1]], [0, 1, 1, 0])

amplitude = np.linspace(0.005, 0.0075, len(noiseProg))
dphi = np.interp(ring.cycle_time + initial_time, time, noiseProg*amplitude)
domega = np.gradient(dphi)*ring.omega_rev/(2*np.pi)

for i, h in enumerate(rf_params.harmonic[:, 0]):
    rf_params.phi_rf[i] += h*dphi
    rf_params.omega_rf[i] += h*domega


#%%

my_beam = beam.Beam(ring, n_particles, 0)

cutOpts = prof.CutOptions(0, ring.t_rev[0], n_slices)
profile = prof.Profile(my_beam, cutOpts)

#%%

profile.track()
configuration = {'machine': 'PSB', 'PL_gain': 330, 'RL_gain': [1E7, 1E6],
                 'period': 10.0e-6}
phase_loop = bfb.BeamFeedback(ring, rf_params, profile,
                              configuration)

long_tracker = tracker.RingAndRFTracker(rf_params, my_beam,
                                        periodicity=periodicity_tracking,
                                        BeamFeedback = phase_loop)
full_ring = tracker.FullRingAndRF([long_tracker])

#%%


Z_Over_N_160 = np.zeros(n_turns)-603.18 #Ohm
steps_space = imp.InductiveImpedance(my_beam, profile, Z_Over_N_160,
                                     rf_params, deriv_mode='filter1d')

finemet1Cell = np.load(os.path.join(inputDir, "./FinemetImpedance1Cell.npy"))

finemetTable = impSource.InputTable(finemet1Cell[0], finemet1Cell[1]*36,
                                    finemet1Cell[2]*36)

notchHarmonics = np.arange(1, 9)
notchQs = np.array([200]*len(notchHarmonics))
notchRS = (1/63-1)*np.interp(notchHarmonics*ring.f_rev[0], finemet1Cell[0],
                             finemet1Cell[1]*36)

notchResons = impSource.Resonators(notchRS, notchHarmonics*ring.f_rev[0],
                                   notchQs)

profile.beam_spectrum_freq_generation(n_slices)

cutOpts.cut_right = ring.t_rev[0]
cutOpts.set_cuts()
profile.set_slices_parameters()
freqDomainInduced = imp.InducedVoltageFreq(my_beam, profile, [finemetTable,
                                                              notchResons])
totalInduced = imp.TotalInducedVoltage(my_beam, profile, [steps_space, \
                                                          freqDomainInduced])

#%%

sliceEdgeCount = np.arange(profile.n_slices+1)

def update_impedance(map_, turn):

    prof = map_[1]

    deltaRight = prof.cut_right - ring.t_rev[turn]

    prof.cut_right = ring.t_rev[turn]
    prof.edges -= (deltaRight/prof.n_slices)*sliceEdgeCount
    prof.bin_centers[:] = (prof.edges[:-1] + prof.edges[1:])/2
    prof.bin_size = (prof.cut_right - prof.cut_left) / prof.n_slices

    tInd = map_[2]
    indFreq = tInd.induced_voltage_list[1]

    notchResons.frequency_R = notchHarmonics*ring.f_rev[turn]
    notchResons.R_S = (1/63-1)*np.interp(notchHarmonics*ring.f_rev[turn],
                                         finemet1Cell[0],
                                         finemet1Cell[1]*36)

    prof.beam_spectrum_freq_generation(indFreq.n_fft)
    indFreq.freq = prof.beam_spectrum_freq
    indFreq.sum_impedances(indFreq.freq)

#%%

matchedBeam = distBeam.matched_from_distribution_function(my_beam,
                                                          full_ring,
                            distribution_type = 'parabolic_amplitude',
                            emittance = 1.8)

#%%

map_ = [full_ring, profile, totalInduced]

# tIt = trackIt.TrackIteration(map_)

# tIt.add_function(update_impedance, 1)

#%%

intensity_ramp = np.linspace(0, targetIntensity, 5000)
mpiprint('dE mean: ', np.mean(my_beam.dE))
mpiprint('dE std: ', np.std(my_beam.dE))
mpiprint('dt mean: ', np.mean(my_beam.dE))
mpiprint('dt std: ', np.std(my_beam.dE))

my_beam.split(random=False)

if args['monitor'] > 0 and worker.isMaster:
    if args.get('monitorfile', None):
        filename = args['monitorfile']
    else:
        filename = 'monitorfiles/psb-t{}-p{}-b{}-sl{}-approx{}-prec{}-r{}-m{}-se{}-w{}'.format(
            n_iterations, n_particles, n_bunches, n_slices, approx, args['precision'],
            n_turns_reduce, args['monitor'], seed, worker.workers)
    slicesMonitor = SlicesMonitor(filename=filename,
                                  n_turns=np.ceil(
                                      n_iterations / args['monitor']),
                                  profile=profile,
                                  rf=rf_params,
                                  Nbunches=n_bunches)


worker.sync()

timing.reset()
start_t = tm.time()

t0 = tm.process_time()
for turn in range(n_iterations):
    full_ring.track()
    profile.track()
    totalInduced.track()
    with timing.timed_region('serial:updateImp'):
        update_impedance(map_, turn)

    # for t in tIt:
    if turn < len(intensity_ramp):
        my_beam.intensity = intensity_ramp[turn]
        my_beam.ratio = my_beam.intensity/my_beam.n_macroparticles
    # else:
    #     break

    if (args['monitor'] > 0) and (turn % args['monitor'] == 0):
        my_beam.statistics()
        my_beam.gather_statistics()
        profile.fwhm()
    
        if worker.isMaster:
            slicesMonitor.track(turn)


    if turn%1000 == 0:
        print(f"Turn {turn} of {n_iterations}, time per 1k turns: {tm.process_time() - t0}")
        t0 = tm.process_time()

my_beam.gather()
end_t = tm.time()

timing.report(total_time=1e3*(end_t-start_t),
              out_dir=args['timedir'],
              out_file='worker-{}.csv'.format(worker.rank))

worker.finalize()

if args['monitor'] > 0:
    slicesMonitor.close()

mpiprint('dE mean: ', np.mean(my_beam.dE))
mpiprint('dE std: ', np.std(my_beam.dE))
mpiprint('dt mean: ', np.mean(my_beam.dt))
mpiprint('dt std: ', np.std(my_beam.dt))

# mpiprint('dt mean, 1st bunch: ', np.mean(beam.dt[:n_particles]))
# mpiprint('shift ', rf.phi_rf[0, turn]/rf.omega_rf[0, turn])

mpiprint('profile mean: ', np.mean(profile.n_macroparticles))
mpiprint('profile std: ', np.std(profile.n_macroparticles))

mpiprint('Done!')


# t0 = tm.process_time()
# for t in tIt:
#     if t%1000 == 0:
#         print(f"Turn {t} of {ring.n_turns}, time per 1k turns: {tm.process_time() - t0}")
#         t0 = tm.process_time()