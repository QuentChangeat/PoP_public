from taurex.log import Logger
from taurex.core import Fittable, fitparam
from pop.pipeline.units.baseunit import BaseUnit
import numpy as np

class HSTObservationPhaseRampsBisUnit(BaseUnit):
    def __init__(self, name='norm', orbit_times = None, 
                    expfactorAF=1e-30, expfactorBF=1, expfactorCF = 1e-30, expfactorDF = 0.0, expfactorEF = 1., linfactorF=[1e-30], quadfactorF = [1e-30], normF=[1], 
                    expfactorAR=1e-30, expfactorBR=1, expfactorCR = 1e-30, expfactorDR = 0.0, expfactorER = 1., linfactorR=[1e-30], quadfactorR = [1e-30], normR=[1],
                    inflation_factor=1, segment_times = [1000.0,], visit_times = [1000.0],
                    fit = False):
        Logger.__init__(self,name)
        Fittable.__init__(self)
        super().__init__(name, fit=fit)
        
        self._orbit_times = orbit_times
        
        self._expfactorAF = expfactorAF
        self._expfactorBF = expfactorBF
        self._expfactorCF = expfactorCF
        self._expfactorDF = expfactorDF
        self._expfactorEF = expfactorEF
        self._linfactorF = linfactorF
        self._quadfactorF = quadfactorF
        self._expfactorAR = expfactorAR
        self._expfactorBR = expfactorBR
        self._expfactorCR = expfactorCR
        self._expfactorDR = expfactorDR
        self._expfactorER = expfactorER
        self._linfactorR = linfactorR
        self._quadfactorR = quadfactorR
        self._normF = normF
        self._normR = normR

        self._inf = inflation_factor

        self._segment_times = segment_times
        self._visit_times = visit_times
        
        #self.generate_fitting_params()
    def load_inputs(self, inputs=None):
        #print('inputs', inputs)
        self._timesF = inputs[0][0]
        self._input_dataF = inputs[1][0]
        self._errorsF = inputs[2][0]
        self._timesR = inputs[0][1]
        self._input_dataR = inputs[1][1]
        self._errorsR = inputs[2][1]
        self._wavelengths = inputs[3]

        if self._segment_times is None:
            self._segment_times = np.min(np.concatenate([self._timesR, self._timesF]))
        if self._visit_times is None:
            self._visit_times = np.min(np.concatenate([self._timesR, self._timesF]))

        self._segment_indexF = [(np.abs(self._timesF - stime)).argmin() for stime in self._segment_times]
        self._segment_indexF = np.append(self._segment_indexF, len(self._timesF))
        try:
            self._segment_indexR = [(np.abs(self._timesR - stime)).argmin() for stime in self._segment_times]
            self._segment_indexR = np.append(self._segment_indexR, len(self._timesR))
        except:
            self._segment_indexR = self._segment_indexF

        self._visit_indexF = [(np.abs(self._timesF - stime)).argmin() for stime in self._visit_times]
        self._visit_indexF = np.append(self._visit_indexF, len(self._timesF))
        try:
            self._visit_indexR = [(np.abs(self._timesR - stime)).argmin() for stime in self._visit_times]
            self._visit_indexR = np.append(self._visit_indexR, len(self._timesR))
        except:
            self._visit_indexR = self._visit_indexF

    def determine_orbits(self, times):
        orbits = np.where(abs(times - np.roll(times, 1)) > 20.0 / 60.0 / 24.0)[0]
        dumps = np.where(abs(times - np.roll(times, 1)) > 5.0 / 60.0 / 24.0)[0]
        return orbits, dumps
    
    def build(self, inputs, planet=None, star=None):
        
        self._planet = planet
        self._star = star
        
        self.load_inputs(inputs)
        
        orbits, dumps = self.determine_orbits(self._timesF)
        orbits = np.append(orbits, len(self._timesF))
        space = orbits[1:] - orbits[:-1]
        #self._t0F = np.array([[self._timesF[orbits[i]]]*space[i] for i in range(len(space))]).flatten()
        self._t0F = self.flatten([[self._timesF[orbits[i]]]*space[i] for i in range(len(space))])

        orbits, dumps = self.determine_orbits(self._timesR)
        orbits = np.append(orbits, len(self._timesR))
        space = orbits[1:] - orbits[:-1]
        #self._t0R = np.array([[self._timesR[orbits[i]]]*space[i] for i in range(len(space))]).flatten()
        self._t0R = self.flatten([[self._timesR[orbits[i]]]*space[i] for i in range(len(space))])

        space = self._segment_indexF[1:] - self._segment_indexF[:-1]
        self._tF_segment = self.flatten([[self._timesF[self._segment_indexF[i]]]*space[i] for i in range(len(space))])

        space = self._visit_indexF[1:] - self._visit_indexF[:-1]
        self._tF_visit = self.flatten([[self._timesF[self._visit_indexF[i]]]*space[i] for i in range(len(space))])

        if len(self._timesR) > 0 :
            #self._segment_indexR = np.append(self._segment_indexR, len(self._timesR))
            space = self._segment_indexR[1:] - self._segment_indexR[:-1]
            self._tR_segment = self.flatten([[self._timesR[self._segment_indexR[i]]]*space[i] for i in range(len(space))])

            space = self._visit_indexR[1:] - self._visit_indexR[:-1]
            self._tR_visit = self.flatten([[self._timesR[self._visit_indexR[i]]]*space[i] for i in range(len(space))])
        else:
            self._tR_segment = []
            self._tR_visit = []

        out = self.apply_step(inputs)
        self.generate_fitting_params()
        return out

    def flatten(self, l):
        return [item for sublist in l for item in sublist]
    
    def generate_fitting_params(self):
        
        for i, t in enumerate(self._visit_times):

            def read_expAF(self, i=i):
                return self._expfactorAF[i]
            def write_expAF(self, value, i=i):
                self._expfactorAF[i] = value    
            name_expAF='{}_expAF_vis{}'.format(self._name, i)
            self.add_fittable_param(name_expAF, name_expAF, read_expAF,
                                        write_expAF, 'linear', False, [-0.1, 0.1])
        
            def read_expBF(self, i=i):
                return self._expfactorBF[i]
            def write_expBF(self, value, i=i):
                self._expfactorBF[i] = value  
            name_expBF='{}_expBF_vis{}'.format(self._name, i)
            self.add_fittable_param(name_expBF, name_expBF, read_expBF,
                                        write_expBF, 'linear', False, [-0.1, 0.1])
            
            def read_expCF(self, i=i):
                return self._expfactorCF[i]
            def write_expCF(self, value, i=i):
                self._expfactorCF[i] = value  
            name_expCF='{}_expCF_vis{}'.format(self._name, i)
            self.add_fittable_param(name_expCF, name_expCF, read_expCF,
                                        write_expCF, 'linear', False, [-0.1, 0.1])
            def read_expDF(self, i=i):
                return self._expfactorDF[i]
            def write_expDF(self, value, i=i):
                self._expfactorDF[i] = value  
            name_expDF='{}_expDF_vis{}'.format(self._name, i)
            self.add_fittable_param(name_expDF, name_expDF, read_expDF,
                                        write_expDF, 'linear', False, [-0.1, 0.1])
            
            def read_expEF(self, i=i):
                return self._expfactorEF[i]
            def write_expEF(self, value, i=i):
                self._expfactorEF[i] = value  
            name_expEF='{}_expEF_vis{}'.format(self._name, i)
            self.add_fittable_param(name_expEF, name_expEF, read_expEF,
                                        write_expEF, 'linear', False, [-0.1, 0.1])

            def read_expAR(self, i=i):
                return self._expfactorAR[i]
            def write_expAR(self, value, i=i):
                self._expfactorAR[i] = value    
            name_expAR='{}_expAR_vis{}'.format(self._name, i)
            self.add_fittable_param(name_expAR, name_expAR, read_expAR,
                                        write_expAR, 'linear', False, [-0.1, 0.1])
        
            def read_expBR(self, i=i):
                return self._expfactorBR[i]
            def write_expBR(self, value, i=i):
                self._expfactorBR[i] = value    
            name_expBR='{}_expBR_vis{}'.format(self._name, i)
            self.add_fittable_param(name_expBR, name_expBR, read_expBR,
                                        write_expBR, 'linear', False, [-0.1, 0.1])
            
            def read_expCR(self, i=i):
                return self._expfactorCR[i]
            def write_expCR(self, value, i=i):
                self._expfactorCR[i] = value  
            name_expCR='{}_expCR_vis{}'.format(self._name, i)
            self.add_fittable_param(name_expCR, name_expCR, read_expCR,
                                        write_expCR, 'linear', False, [-0.1, 0.1])
            def read_expDR(self, i=i):
                return self._expfactorDR[i]
            def write_expDR(self, value, i=i):
                self._expfactorDR[i] = value  
            name_expDR='{}_expDR_vis{}'.format(self._name, i)
            self.add_fittable_param(name_expDR, name_expDR, read_expDR,
                                        write_expDR, 'linear', False, [-0.1, 0.1])
            
            def read_expER(self, i=i):
                return self._expfactorER[i]
            def write_expER(self, value, i=i):
                self._expfactorER[i] = value  
            name_expER='{}_expER_vis{}'.format(self._name, i)
            self.add_fittable_param(name_expER, name_expER, read_expER,
                                        write_expER, 'linear', False, [-0.1, 0.1])

        for i, t in enumerate(self._segment_times):

            def read_linF(self, i=i):
                return self._linfactorF[i]
            def write_linF(self, value, i=i):
                self._linfactorF[i] = value    
            name_linF='{}_linF_seg{}'.format(self._name, i)
            self.add_fittable_param(name_linF, name_linF, read_linF,
                                        write_linF, 'linear', False, [-0.1, 0.1])

            def read_quadF(self, i=i):
                return self._quadfactorF[i]
            def write_quadF(self, value, i=i):
                self._quadfactorF[i] = value    
            name_quadF='{}_quadF_seg{}'.format(self._name, i)
            self.add_fittable_param(name_quadF, name_quadF, read_quadF,
                                        write_quadF, 'linear', False, [-0.1, 0.1])

            def read_nF(self, i=i):
                return self._normF[i]
            def write_nF(self, value, i=i):
                self._normF[i] = value    
            name_nF='{}_normF_seg{}'.format(self._name, i)
            self.add_fittable_param(name_nF, name_nF, read_nF,
                                        write_nF, 'log', False, [1e2, 1e10])
            
            def read_linR(self, i=i):
                return self._linfactorR[i]
            def write_linR(self, value, i=i):
                self._linfactorR[i] = value    
            name_linR='{}_linR_seg{}'.format(self._name, i)
            self.add_fittable_param(name_linR, name_linR, read_linR,
                                        write_linR, 'linear', False, [-0.1, 0.1])


            def read_quadR(self, i=i):
                return self._quadfactorR[i]
            def write_quadR(self, value, i=i):
                self._quadfactorR[i] = value    
            name_quadR='{}_quadR_seg{}'.format(self._name, i)
            self.add_fittable_param(name_quadR, name_quadR, read_quadR,
                                        write_quadR, 'linear', False, [-0.1, 0.1])

            def read_nR(self, i=i):
                return self._normR[i]
            def write_nR(self, value, i=i):
                self._normR[i] = value    
            name_nR='{}_normR_seg{}'.format(self._name, i)
            self.add_fittable_param(name_nR, name_nR, read_nR,
                                        write_nR, 'log', False, [1e2, 1e10])

        def read_inf(self):
            return self._inf
        def write_inf(self, value):
            self._inf = value    
        name_inf='{}_inf'.format(self._name)
        self.add_fittable_param(name_inf, name_inf, read_inf,
                                    write_inf, 'linear', False, [0.8, 1.2])

    def apply_step(self, inputs = None):
        self.load_inputs(inputs)
        self._mid_time = self._planet._mid_time
        #self.t0 = np.min(np.concatenate([self._timesR, self._timesF]))

        space = self._segment_indexF[1:] - self._segment_indexF[:-1]
        #expfactorAF = self.flatten([[self._expfactorAF[i]]*space[i] for i in range(len(space))])
        #expfactorBF = self.flatten([[self._expfactorBF[i]]*space[i] for i in range(len(space))])
        #exprampF = (1+expfactorAF*np.exp(-(self._timesF-self._tF_segment)/expfactorBF))

        linfactorF = self.flatten([[self._linfactorF[i]]*space[i] for i in range(len(space))])
        quadfactorF = self.flatten([[self._quadfactorF[i]]*space[i] for i in range(len(space))])
        normF = self.flatten([[self._normF[i]]*space[i] for i in range(len(space))])
        linrampF = (1 - linfactorF*(self._timesF-self._tF_segment) - quadfactorF*(self._timesF-self._tF_segment)**2)*normF

        space = self._visit_indexF[1:] - self._visit_indexF[:-1]
        expfactorAF = self.flatten([[self._expfactorAF[i]]*space[i] for i in range(len(space))])
        expfactorBF = self.flatten([[self._expfactorBF[i]]*space[i] for i in range(len(space))])
        expfactorCF = self.flatten([[self._expfactorCF[i]]*space[i] for i in range(len(space))])
        expfactorDF = self.flatten([[self._expfactorDF[i]]*space[i] for i in range(len(space))])
        exprampF = (1+expfactorAF*np.exp(-(self._timesF-self._tF_visit)/expfactorBF))
        exprampF2 = (1+expfactorCF*np.exp(-(self._timesF-self._t0F)/(expfactorDF * exprampF)))

        new_dataF = self._input_dataF/linrampF/exprampF/exprampF2 
        new_errorsF = self._errorsF/linrampF/exprampF/exprampF2*self._inf

        try:
            space = self._segment_indexR[1:] - self._segment_indexR[:-1]
            #expfactorAR = self.flatten([[self._expfactorAR[i]]*space[i] for i in range(len(space))])
            #expfactorBR = self.flatten([[self._expfactorBR[i]]*space[i] for i in range(len(space))])
            #exprampR = (1+expfactorAR*np.exp(-(self._timesR-self._tR_segment)/expfactorBR))

            linfactorR = self.flatten([[self._linfactorR[i]]*space[i] for i in range(len(space))])
            quadfactorR = self.flatten([[self._quadfactorR[i]]*space[i] for i in range(len(space))])
            normR = self.flatten([[self._normR[i]]*space[i] for i in range(len(space))])
            linrampR = (1 - linfactorR*(self._timesR-self._tR_segment) - quadfactorR*(self._timesR-self._tR_segment)**2)*normR

            space = self._visit_indexR[1:] - self._visit_indexR[:-1]
            expfactorAR = self.flatten([[self._expfactorAR[i]]*space[i] for i in range(len(space))])
            expfactorBR = self.flatten([[self._expfactorBR[i]]*space[i] for i in range(len(space))])
            expfactorCR = self.flatten([[self._expfactorCR[i]]*space[i] for i in range(len(space))])
            expfactorDR = self.flatten([[self._expfactorDR[i]]*space[i] for i in range(len(space))])
            exprampR = (1+expfactorAR*np.exp(-(self._timesR-self._tR_visit)/expfactorBR))
            exprampR2 = (1+expfactorCR*np.exp(-(self._timesR-self._t0R)/(expfactorDR * exprampR)))

            new_dataR = self._input_dataR/linrampR/exprampR/exprampR2 
            new_errorsR = self._errorsR/linrampR/exprampR/exprampR2*self._inf
        except:
            new_dataR = []
            new_errorsR = []

        self.outputs = [[self._timesF, self._timesR], [new_dataF, new_dataR], [new_errorsF, new_errorsR], self._wavelengths]
        return self.outputs
        
    @classmethod
    def input_keywords(self):
        return ['HSTPhaseRampsWit']