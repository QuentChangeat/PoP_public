from taurex.binning import Binner
from taurex.log import Logger
import numpy as np

class LCBinner(Binner):
    def __init__(self, wavenumber):
        Logger.__init__(self, self.__class__.__name__)
        self.wavenumber = wavenumber

    def bindown(self, wngrid, spectrum, time=None, error=None):
        wngrid = np.array(wngrid).flatten()
        spectrum = np.array(spectrum).flatten()
        time = np.array(time).flatten()
        error = np.array(error).flatten()
        return wngrid, spectrum, error, time
        
    def bin_model(self, model_output):
        out = self.bindown(model_output[2], model_output[1], time = model_output[0])
        return out
        
    def generate_spectrum_output(self, model_output, output_size=None):
        output = {}
        return output