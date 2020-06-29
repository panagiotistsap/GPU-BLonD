from pycuda import gpuarray
# , driver as drv, tools
from ..utils import bmath as bm
from types import MethodType

# drv.init()
# dev = drv.Device(bm.gpuId())


def update_synch_rad(obj):
    if (not obj.python):
        if obj.quantum_excitation:
            obj.track = MethodType(track_full_cuda, obj)
        else:
            obj.track = MethodType(track_SR_cuda, obj)


# Track particles with SR only (without quantum excitation)
# Cuda implementation
def track_SR_cuda(self):
    i_turn = self.rf_params.counter[0]
    # Recalculate SR parameters if energy changes
    if (i_turn != 0 and self.general_params.energy[0, i_turn] !=
            self.general_params.energy[0, i_turn-1]):
        self.calculate_SR_params()

    bm.synchrotron_radiation(self.beam.dev_dE, self.U0,
                             self.n_kicks, self.tau_z)

# Track particles with SR and quantum excitation.
# Cuda implementation


def track_full_cuda(self):
    i_turn = self.rf_params.counter[0]
    # Recalculate SR parameters if energy changes
    if (i_turn != 0 and self.general_params.energy[0, i_turn] !=
            self.general_params.energy[0, i_turn-1]):
        self.calculate_SR_params()

    bm.synchrotron_radiation_full(self.beam.dev_dE, self.U0, self.n_kicks,
                                  self.tau_z, self.sigma_dE,
                                  self.general_params.energy[0, i_turn])
