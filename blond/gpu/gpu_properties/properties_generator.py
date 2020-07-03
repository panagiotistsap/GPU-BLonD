props = """
@property
def {0}(self):
    return self.{0}_obj.my_array

@{0}.setter
def {0}(self, value):
    self.{0}_obj.my_array = value


@property
def dev_{0}(self):
    return self.{0}_obj.dev_my_array


@dev_{0}.setter
def dev_{0}(self, value):
    self.{0}_obj.dev_my_array = value
"""

## profile properties

exec(props.format("bin_centers"))
exec(props.format("n_macroparticles"))
exec(props.format("beam_spectrum"))
exec(props.format("beam_spectrum_freq"))

## impedance properties

exec(props.format("induced_voltage"))
exec(props.format("mtw_memory"))
exec(props.format("total_impedance"))

## beam properties

exec(props.format("dE"))
exec(props.format("dt"))
exec(props.format("id"))

## rf_station properties

exec(props.format("voltage"))
exec(props.format("phi_rf"))
exec(props.format("omega_rf"))
exec(props.format("omega_rf_d"))
exec(props.format("harmonic"))
exec(props.format("dphi_rf"))