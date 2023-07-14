#from taurex.log import Logger
#from taurex.output.writeable import Writeable
#from taurex.core import Fittable
from taurex.data.spectrum import BaseSpectrum
import numpy as np
from pop.binning import LCBinner


class ObservationPipeline(BaseSpectrum):

    def __init__(self, units, order, planet=None, star=None):
        #Logger.__init__(self,'ObservationPipeline')
        #Fittable.__init__(self)
        super().__init__(self.__class__.__name__)
        
        self._order = order
        self._units = self.sort_units(units)
        
        self._planet = planet
        self._star = star

    def add_unit(self, unit, order = None):
        self._units.append(unit)
        if order is not None:
            self.define_order(order)
        else:
            self._order.append(unit._name)
        self._units = self.sort_units(self._units)
    
    def delete_unit(self, name=''):
        for i,u in enumerate(self._units):
            if name == u._name:
                self._units.pop(i)
                self._order.remove(name)
                self.info('Unit ' +str(u)+'with number '+str(i)+' was sucessfully removed')

    def define_order(self, order):
        self._order = order

    def sort_units(self, units):
        sorted_units = []
        unsorted_names = [u._name for u in units]
        for idx, n in enumerate(self._order):
            index = unsorted_names.index(n)
            sorted_units.append(units[index])
        return sorted_units
    
    def collect_fitting_parameters(self):
            self._fitting_parameters = {}
            self._fitting_parameters.update(self.fitting_parameters())
            for u in self._units:
                self._fitting_parameters.update(u.fitting_parameters())
            self.debug('Available Fitting params: %s',
                   list(self._fitting_parameters.keys()))
    
    def __getitem__(self, key):
        return self._fitting_parameters[key][2]()
    
    def __setitem__(self, key, value):
        self._fitting_parameters[key][3](value)
    
    def build(self, out=None):
        for idx, u in enumerate(self._units):
            out = u.build(out, self._planet, self._star)
        self.collect_fitting_parameters()
        self.out = out
    
    def run(self, out = None):
        for idx, u in enumerate(self._units):
            out = u.apply_step(out)
        self.out = out
        return out
    
    @property
    def spectrum(self):
        self.run()
        return np.array(self.out[1]).flatten()

    @property
    def shapedSpectrum(self):
        self.run()
        return np.array(self.out[1])

    @property
    def errorBar(self):
        self.run()
        return np.array(self.out[2]).flatten()
    
    @property
    def shapedErrorBar(self):
        self.run()
        return np.array(self.out[2])
    
    @property
    def timelist(self):
        self.run()
        return np.array(self.out[0])
    
    @property
    def wavelist(self):
        self.run()
        return np.array(self.out[3])
    
    @property
    def wavenumberGrid(self):
        return np.array(self.out[3])
    
    def create_binner(self):
        return LCBinner(self.wavenumberGrid)
    
    @property
    def fittingParameters(self):
        return self._fitting_parameters
    
    @property
    def derivedParameters(self):
        return self.derived_parameters()
    
    @classmethod
    def input_keywords(self):
        return ['obs_pipeline' ]