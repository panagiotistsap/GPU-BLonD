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