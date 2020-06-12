import numpy as np
import time

# BLonD
import blond.input_parameters.ring as inputRing
import blond.input_parameters.ring_options as ringOpt
import blond.input_parameters.rf_parameters as inputRF
import blond.trackers.tracker as tracker
import blond.beam.beam as beam
import blond.beam.profile as prof
import blond.beam.distributions as distBeam
import blond.impedances.impedance_sources as impSource
import blond.impedances.impedance as imp
import blond.utils.bmath as bm
from pycuda import gpuarray
#%%


import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-g', default = False, action='store_true')
parser.add_argument('-u', default = False, action='store_true')
args = parser.parse_args()
print(args)


initial_time = 275E-3
final_time = 805E-3

#%%

#BLonD Simulation

radius = 25
gamma_transition = 4.07#psb.full_pars['gammaT']  # [1]
C = 2 * np.pi * radius  # [m]       
momentum_compaction = 1 / gamma_transition**2 # [1]
particle_type = 'proton'

momProg = np.load("../input_files/momentum.npy")


#%%

ring_options = ringOpt.RingOptions("derivative", t_start = 490E-3, 
                                   t_end = 700E-3)
ring = inputRing.Ring(C, momentum_compaction, (momProg[0], momProg[1]), 
                      beam.Proton(), RingOptions=ring_options)

n_turns = ring.n_turns
print("N Turns: " + str(n_turns))

n_macro = int(1E7)


#%%

rf_params = inputRF.RFStation(ring, 1, 18E3, np.pi, 1)

my_beam = beam.Beam(ring, n_macro, 3E11)

#%%

n_slices = 2**7
cutOpts = prof.CutOptions(0, ring.t_rev[0], n_slices)
profile = prof.Profile(my_beam, cutOpts)

long_tracker = tracker.RingAndRFTracker(rf_params, my_beam)
full_ring = tracker.FullRingAndRF([long_tracker])

#%%

FRange = np.array([0.9E6, 2E6])
reson1F = np.array([FRange, FRange*5])
reson2F = np.array([FRange, FRange*10])
reson3F = np.array([FRange, FRange*15])
reson4F = np.array([FRange, FRange*20])

reson1Q = 50
reson2Q = 60
reson3Q = 70
reson4Q = 80

reson1R = np.array([FRange, [500, 1000]])
reson2R = np.array([FRange, [1000, 800]])
reson3R = np.array([FRange, [100, 1000]])
reson4R = np.array([FRange, [5000, 1000]])

resonances = [reson1F, reson2F, reson3F, reson4F]
Qs = [reson1Q, reson2Q, reson3Q, reson4Q]
RShunts = [reson1R, reson2R, reson3R, reson4R]

initFs = [np.interp(ring.f_rev[0], f[0], f[1]) for f in resonances]
initRs = [np.interp(ring.f_rev[0], r[0], r[1]) for r in RShunts]

resonators = impSource.Resonators(initRs, initFs, Qs)

freqDomainInduced = imp.InducedVoltageFreq(my_beam, profile, [resonators])
totalInduced = imp.TotalInducedVoltage(my_beam, profile, [freqDomainInduced])

#%%

matchedBeam = distBeam.matched_from_distribution_function(my_beam,
                                                          full_ring,
                            distribution_type = 'parabolic_amplitude',
                            emittance = 1.4,
                            seed = 0,
                            TotalInducedVoltage = totalInduced,
                            n_iterations = 1)

#%%

map_ = [full_ring, profile, totalInduced]

sliceEdgeCount = np.arange(profile.n_slices+1)

def update_impedance(map_, turn):

    prof = map_[1]

    deltaRight = prof.cut_right - ring.t_rev[turn]

    prof.cut_right = ring.t_rev[turn]
    prof.edges -= (deltaRight/prof.n_slices)*sliceEdgeCount
    prof.bin_centers[:]  = (prof.edges[:-1] + prof.edges[1:])/2
    prof.bin_size = (prof.cut_right - prof.cut_left) / prof.n_slices

    tInd = map_[2]
    indFreq = tInd.induced_voltage_list[0]

    resonators.frequency_R = np.array([np.interp(ring.f_rev[turn], f[0], f[1])
                                       for f in resonances])
    resonators.R_S = np.array([np.interp(ring.f_rev[turn], r[0], r[1]) for r in
                               RShunts])

    prof.beam_spectrum_freq_generation(indFreq.n_fft)
    indFreq.freq = prof.beam_spectrum_freq
    indFreq.sum_impedances(indFreq.freq)

#%%
print("starting dE std",np.std(my_beam.dE))
import cuprof 
if (args.g):
    bm.use_gpu()
    bm.enable_gpucache()
    long_tracker.use_gpu()
    profile.use_gpu()
    totalInduced.use_gpu()
    tk = "gpu"
else:
    tk = "cpu"
    
cuprof.enable()
with cuprof.region_timer("main_loop",tk) as rtn:
    for i in range(1000):
        for m in map_:
            m.track()
        if (args.u):
            update_impedance(map_, i+1)
        else:
            pass
cuprof.report()
print(np.std(my_beam.dE))    
    