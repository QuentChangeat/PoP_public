from taurex.log import Logger
from taurex.core import Fittable, fitparam
from pop.pipeline.units.baseunit import BaseUnit
import numpy as np

class HSTOrbitFilter(BaseUnit):
    def __init__(self, name='orb_filt', remove_orbits=1, remove_first_exposures = 1, fit = False):
        Logger.__init__(self,name)
        Fittable.__init__(self)
        super().__init__(name, fit=fit)
        
        self._remove_orbits = remove_orbits
        if self._remove_orbits == 0 or self._remove_orbits is None:
            self._remove_orbits = None
        elif isinstance(self._remove_orbits, int):
            self._remove_orbits = np.arange(self._remove_orbits)
        else:
            self._remove_orbits = [ int(x) for x in self._remove_orbits ]
        self._remove_first_exposures = int(remove_first_exposures)

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

    def determine_orbits(self, times):
        orbits = np.where(abs(times - np.roll(times, 1)) > 20.0 / 60.0 / 24.0)[0]
        dumps = np.where(abs(times - np.roll(times, 1)) > 5.0 / 60.0 / 24.0)[0]
        return orbits, dumps

    def apply_step(self, inputs = None):
        self.load_inputs(inputs)

        idxsF = np.arange(len(self._timesF))
        if self._remove_orbits is not None:
            orbits, dumps = self.determine_orbits(self._timesF)
            #idxsF = idxsF[orbits[self._remove_orbits]:]
            idxsF = np.delete(idxsF,np.concatenate([np.arange(orbits[r], orbits[r+1]) for r in self._remove_orbits]))
        if self._remove_first_exposures is not None:
            orbits, dumps = self.determine_orbits(self._timesF[idxsF])
            idxsF = np.delete(idxsF,np.concatenate([orbits + i for i in range(self._remove_first_exposures)]))
        
        idxsR = np.arange(len(self._timesR))
        if len(idxsR) == 0:
            self._new_times = [self._timesF[idxsF], self._timesR]
            self._new_data = [self._input_dataF[idxsF],self._input_dataR]
            self._new_errors = [self._errorsF[idxsF], self._errorsR]
        else:
            if self._remove_orbits is not None:
                orbits, dumps = self.determine_orbits(self._timesR)
                #idxsR = idxsR[orbits[self._remove_orbits]:]
                idxsR = np.delete(idxsR,np.concatenate([np.arange(orbits[r], orbits[r+1]) for r in self._remove_orbits]))
            if self._remove_first_exposure is not None:
                orbits, dumps = self.determine_orbits(self._timesR[idxsR])
                idxsR = np.delete(idxsR,np.concatenate([orbits + i for i in range(self._remove_first_exposures)]))

            self._new_times = [self._timesF[idxsF], self._timesR[idxsR]]
            self._new_data = [self._input_dataF[idxsF],self._input_dataR[idxsR]]
            self._new_errors = [self._errorsF[idxsF], self._errorsR[idxsR]]

        self.outputs = [self._new_times, self._new_data, self._new_errors, self._wavelengths]
        return self.outputs
        
    @classmethod
    def input_keywords(self):
        return ['HSTOrbitFilter']