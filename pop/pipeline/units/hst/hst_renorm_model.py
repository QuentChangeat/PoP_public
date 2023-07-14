from taurex.log import Logger
from taurex.core import Fittable, fitparam
from pop.pipeline.units.baseunit import BaseUnit
from taurex.mpi import barrier, get_rank, nprocs, broadcast
import numpy as np

class ModelRenormUnit(BaseUnit):
    def __init__(self, name='mrenorm', fit = False, white_res = None, res_prefix='_map'):
        Logger.__init__(self,name)
        Fittable.__init__(self)
        super().__init__(name, fit=fit)
        self._white_res = white_res
        self._res_prefix = res_prefix

        self.renorm_sp = 1.
        self.renorm_er = 1.
        if white_res is not None:
            self.load_white(white_res)
        #self.generate_fitting_params()

    def load_white(self, file):
        white_sp = None
        white_t = None
        if get_rank() == 0:
            self.info('Opening White Result file')

            import h5py 
            f = h5py.File(file)
            white_t = f['Outputs']['Results']['timelist'+self._res_prefix][()]
            white_sp = f['Outputs']['Results']['lightcurve'+self._res_prefix][()]
            white_t = list(white_t)
            white_sp = list(white_sp)

        self.info('Broadcasting White Result file')
        self.renorm_t = broadcast(white_t, 0)
        self.renorm_sp = broadcast(white_sp, 0)
        self.renorm_er = self.renorm_sp
        
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
        data = self._input_data
        err = self._errors
        if self._white_res is not None: 
            #both = set(list(self._times[0])).intersection(list(self.renorm_t[0]))
            #idx1 = [list(self._times[0]).index(x) for x in both]
            #idx2 = [list(self.renorm_t[0]).index(x) for x in both]
            common_elements, idx1, idx2 = np.intersect1d(self._times[0], self.renorm_t[0], return_indices=True)
            data = [self._input_data[i][idx1]/self.renorm_sp[0][idx2] for i in range(len(self._input_data))]
            #err = [self._errors[i]/self.renorm_er[0] for i in range(len(self._input_data))]
        self.outputs = [self._times, data, err, self._wavelengths]
        return self.outputs
        
    @classmethod
    def input_keywords(self):
        return ['ModelRenorm']