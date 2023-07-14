from taurex.log import Logger
from taurex.core import Fittable, fitparam
from pop.pipeline.units.baseunit import BaseUnit
import numpy as np

class IamUselessUnit(BaseUnit):
    def __init__(self, name='useless', fit = False):
        Logger.__init__(self,name)
        Fittable.__init__(self)
        super().__init__(name, fit=fit)

        #self.generate_fitting_params()
    
    def generate_fitting_params(self):
        pass
    
    def load_inputs(self, inputs=None):
        #print('inputs', inputs)
        self._times = inputs[0]
        self._input_data = inputs[1]
        self._errors = inputs[2]
        self._wavelengths = inputs[3]

    def apply_step(self, inputs = None):
        self.load_inputs(inputs)

        self.outputs = [self._times, self._input_data, self._errors, self._wavelengths]
        return self.outputs
        
    @classmethod
    def input_keywords(self):
        return ['UselessUnit']