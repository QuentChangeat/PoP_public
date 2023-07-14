from taurex.log import Logger
from taurex.core import Fittable, fitparam

class BaseUnit(Logger, Fittable):
    def __init__(self, name='base', fit=False):
        Logger.__init__(self, name)
        Fittable.__init__(self)
        
        self._name = name
        self._fit = fit
    
    def build(self, inputs, planet, star):
        self._planet = planet
        self._star = star
        out = self.apply_step(inputs)
        self.generate_fitting_params()
        return out
    
    def generate_fitting_params(self):
        pass
    
    def load_inputs(self, inputs=None):
        #print('inputs', inputs)
        self._times = inputs[0]
        self._input_data = inputs[1]
        self._errors = inputs[2]
        self._wavelengths = inputs[3]
        
    def apply_step(self,  inputs=None):
        raise NotImplementedError

    @classmethod
    def input_keywords(self):
        raise NotImplementedError