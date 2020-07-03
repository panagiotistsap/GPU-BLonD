# Simulation of LHC acceleration ramp to 6.5 TeV
# Noise injection through PL; both PL & SL closed
# Pre-distort noise spectrum to counteract PL action
# WITH intensity effects
#
# authors: H. Timko, K. Iliakis
#
import os
import time
import sys
import numpy as np
# try:
#     from pyprof import timing
#     from pyprof import mpiprof
# except ImportError:
#     from blond.utils import profile_mock as timing
#     mpiprof = timing

from blond.monitors.monitors import SlicesMonitor
from blond.toolbox.next_regular import next_regular
from blond.impedances.impedance import InducedVoltageFreq, TotalInducedVoltage
from blond.impedances.impedance_sources import InputTable
from blond.beam.profile import Profile, CutOptions
from blond.beam.distributions import bigaussian
from blond.beam.beam import Beam, Proton
from blond.llrf.rf_noise import FlatSpectrum, LHCNoiseFB
from blond.llrf.beam_feedback import BeamFeedback
from blond.trackers.tracker import RingAndRFTracker, FullRingAndRF
from blond.input_parameters.rf_parameters import RFStation
from blond.input_parameters.ring import Ring
from blond.utils import bmath as bm
from blond.utils import input_parser
args = input_parser.parse()

REAL_RAMP = True    # track full ramp
MONITORING = False   # turn off plots and monitors

if MONITORING:
    from blond.monitors.monitors import BunchMonitor
    from blond.plots.plot import Plot
    from blond.plots.plot_beams import plot_long_phase_space
    from blond.plots.plot_slices import plot_beam_profile


this_directory = os.path.dirname(os.path.realpath(__file__)) + '/'
inputDir = os.path.join(this_directory, '../input_files/LHC/')


# Simulation parameters --------------------------------------------------------
# Bunch parameters
N_b = 1.2e9          # Intensity
n_particles = 1000000         # Macro-particles
n_bunches = 1        # Number of bunches
freq_res = 2.09e5
# freq_res = 4.e5
# Machine and RF parameters
C = 26658.883        # Machine circumference [m]
h = 35640            # Harmonic number
dphi = 0.            # Phase modulation/offset
gamma_t = 55.759505  # Transition gamma
alpha = 1./gamma_t/gamma_t        # First order mom. comp. factor

# Tracking details
dt_plt = 10000      # Time steps between plots
dt_mon = 1           # Time steps between monitoring
dt_save = 1000000    # Time steps between saving coordinates
if REAL_RAMP:
    n_turns = 100000   # Number of turns to track; full ramp: 8700001
else:
    n_turns = 5000
bl_target = 1.25e-9  # 4 sigma r.m.s. target bunch length in [ns]

n_turns_reduce = 1
n_iterations = 2000
seed = 0





# Import pre-processed momentum and voltage for the acceleration ramp
if REAL_RAMP:
    ps = np.load(os.path.join(inputDir,'LHC_momentum_programme_6.5TeV.npz'))['arr_0']
    # ps = np.loadtxt(wrkDir+r'input/LHC_momentum_programme_6.5TeV.dat',
    # unpack=True)
    ps = np.ascontiguousarray(ps)
    ps = np.concatenate((ps, np.ones(436627)*6.5e12))
else:
    ps = 450.e9*np.ones(n_turns+1)
if REAL_RAMP:
    V = np.concatenate((np.linspace(6.e6, 12.e6, 13563374),
                        np.ones(436627)*12.e6))
else:
    V = 6.e6*np.ones(n_turns+1)

# Define general parameters
ring = Ring(C, alpha, ps[0:n_turns+1], Proton(), n_turns=n_turns)

# Define RF parameters (noise to be added for CC case)
rf = RFStation(ring, [h], [V[0:n_turns+1]], [0.])

# Generate RF phase noise
LHCnoise = FlatSpectrum(ring, rf, fmin_s0=0.8571, fmax_s0=1.001,
                        initial_amplitude=1.e-5,
                        predistortion='weightfunction')
LHCnoise.dphi = np.load(
    os.path.join(inputDir, 'LHCNoise_fmin0.8571_fmax1.001_ampl1e-5_weightfct_6.5TeV.npz'))['arr_0']
LHCnoise.dphi = np.ascontiguousarray(LHCnoise.dphi[0:n_turns+1])

# FULL BEAM
bunch = Beam(ring, n_particles, N_b)
beam = Beam(ring, n_particles*n_bunches, N_b)
bigaussian(ring, rf, bunch, 0.3e-9, reinsertion=True, seed=seed)
bunch_spacing_buckets = 10

for i in np.arange(n_bunches):
    beam.dt[i*n_particles:(i+1)*n_particles] = bunch.dt[0:n_particles] + i*rf.t_rf[0, 0]*10
    beam.dE[i*n_particles:(i+1)*n_particles] = bunch.dE[0:n_particles]

# Profile required for PL
cutRange = (n_bunches-1)*25.e-9+3.5e-9
n_slices = np.int(cutRange/0.025e-9 + 1)
n_slices = next_regular(n_slices)
profile = Profile(beam, CutOptions(n_slices=n_slices, cut_left=-0.5e-9,
                                   cut_right=(cutRange-0.5e-9)))

# Define emittance BUP feedback
noiseFB = LHCNoiseFB(rf, profile, bl_target)

# Define phase loop and frequency loop gain
PL_gain = 1./(5.*ring.t_rev[0])
SL_gain = PL_gain/10.

# Noise injected in the PL delayed by one turn and opposite sign
config = {'machine': 'LHC', 'PL_gain': PL_gain, 'SL_gain': SL_gain}
PL = BeamFeedback(ring, rf, profile, config, PhaseNoise=LHCnoise,
                  LHCNoiseFB=noiseFB)

# Injecting noise in the cavity, PL on

# Define machine impedance from http://impedance.web.cern.ch/impedance/
ZTot = np.loadtxt(os.path.join(inputDir, 'Zlong_Allthemachine_450GeV_B1_LHC_inj_450GeV_B1.dat'),
                  skiprows=1)
ZTable = InputTable(ZTot[:, 0], ZTot[:, 1], ZTot[:, 2])
indVoltage = InducedVoltageFreq(
    beam, profile, [ZTable], frequency_resolution=freq_res)
totVoltage = TotalInducedVoltage(beam, profile, [indVoltage])

# TODO add the noiseFB
tracker = RingAndRFTracker(rf, beam, BeamFeedback=PL, Profile=profile,
                           interpolation=True, TotalInducedVoltage=totVoltage)
# interpolation=True, TotalInducedVoltage=None)
# Fill beam distribution
fullring = FullRingAndRF([tracker])
# Juan's fit to LHC profiles: binomial w/ exponent 1.5
# matched_from_distribution_function(beam, fullring,
#    main_harmonic_option = 'lowest_freq',
#    distribution_exponent = 1.5, distribution_type='binomial',
#    bunch_length = 1.1e-9, bunch_length_fit = 'fwhm',
#    distribution_variable = 'Action')

# Initial losses, slicing, statistics
# beam.losses_separatrix(ring, rf)

## if you want to use the GPU uncomment the following lines
if args['gpu'] == 1:
    bm.use_gpu()
    tracker.use_gpu()
    totVoltage.use_gpu()
    beam.use_gpu()
    PL.use_gpu()
    bm.enable_gpucache()

for turn in range(n_iterations):
    # After the first 2/3 of the ramp, regulate down the bunch length
    if turn == 9042249:
        noiseFB.bl_targ = 1.1e-9
    profile.track()
    totVoltage.induced_voltage_sum()
    tracker.track()

print('dE mean: ', np.mean(beam.dE))
print('dE std: ', np.std(beam.dE))
print('profile mean: ', np.mean(profile.n_macroparticles))
print('profile std: ', np.std(profile.n_macroparticles))

print('Done!')
