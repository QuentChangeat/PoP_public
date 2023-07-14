from taurex.log import Logger
from taurex.core import Fittable, fitparam
from pop.pipeline.units.baseunit import BaseUnit
import numpy as np

class HSTFlatten(BaseUnit):
    def __init__(self, name='flatten', fit = False):
        Logger.__init__(self,name)
        Fittable.__init__(self)
        super().__init__(name, fit=fit)

        #self.generate_fitting_params()
    
    def generate_fitting_params(self):
        pass
    
    def load_inputs(self, inputs=None):
        #print('inputs', inputs)
        self._timesF = inputs[0][0]
        self._input_dataF = inputs[1][0]
        self._errorsF = inputs[2][0]
        self._timesR = inputs[0][1]
        self._input_dataR = inputs[1][1]
        self._errorsR = inputs[2][1]
        self._wavelengths = inputs[3]

    def apply_step(self, inputs = None):
        self.load_inputs(inputs)

        times = np.concatenate([self._timesF, self._timesR])
        idxs = np.argsort(times)
        self._new_data = np.concatenate([self._input_dataF, self._input_dataR])[idxs]
        self._new_errors = np.concatenate([self._errorsF, self._errorsR])[idxs]
        self._new_times = times[idxs]

        self.outputs = [[self._new_times,], [self._new_data,], [self._new_errors,], [self._wavelengths,]]
        return self.outputs
        
    @classmethod
    def input_keywords(self):
        return ['HSTFlatten']