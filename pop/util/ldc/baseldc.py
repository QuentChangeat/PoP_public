from taurex.log import Logger
from taurex.data.fittable import fitparam, Fittable, derivedparam
from taurex.output.writeable import Writeable
from taurex.data.citation import Citable

class BaseStarLDC(Fittable, Logger, Writeable, Citable):
    """Holds information on the stellar Limb Darkening Coefficients

    Parameters
    -----------

    """

    def __init__(self):
        Logger.__init__(self, 'LDC')
        Fittable.__init__(self)
        
        self._ldc = None
        
    def get_LDC(self):
        return self._ldc
    
    #def initialize_star_params(self):
    #    raise NotImplementedError
        
    def computeLDC(self, wlgrid):
        raise NotImplementedError