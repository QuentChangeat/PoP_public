from taurex.model.model import ForwardModel
from pop.util.pylightcurve_utils import planet_orbit
from pop.util.pylightcurve_utils import eclipse, transit, eclipse_mid_time
from taurex.mpi import barrier, get_rank, nprocs, broadcast
import numpy as np
from taurex.util.util import create_grid_res

class PhaseOrbitalModel(ForwardModel):
    def __init__(self,
                 planet=None,
                 star=None,
                 observation=None,
                 wngrid =None,
                 modelpipe = None,
                 precision = 3,
                 ldc_mode = 'exotethys',
                 ldc_method = 'claret',
                 stellar_grid = 'Atlas_2000',
                 passbands = 'uniform_phoenix_2012_13',
                 manual_ldc = [ 0.37127814,  0.28283567, -0.15814152,  0.0016998 ]):
        super().__init__('orbitalModel')
        self._planet = planet
        self._star = star
        self._modelpipe = modelpipe
        self._observation = observation
        self._precision = precision
        self._userwngrid = wngrid
        self._ldc_mode = ldc_mode
        self._ldc_method = ldc_method
        self._stellar_grid = stellar_grid
        self._passbands = passbands
        
        self._default_rp2_bounds = [-0.003, 0.003]
        self._default_fp_bounds = [1e-4, 1e-2]
        self._default_c_ppm_bounds = [0,1e-3]
        self._default_c_rad_bounds = [-np.pi/2, np.pi/2]
        self.ldc = None
        self._default_ldc = manual_ldc
    
    def load_ldc(self):
        if self._ldc_mode == 'exotethys':
            ldc = None 
            if get_rank() == 0:
                self.info('Computing LDC')
                from pop.util.ldc import ExotethysLDC
                ### With this, no interpolation will be done!
                exogrid = self.wlgrid
                exowidth = self.wl_edges

                if self._observation is None:
                    gridtemp = create_grid_res(100, 0.5, 2)
                    exogrid = gridtemp[::-1,0]
                self.exoLDC = ExotethysLDC(stellar_models_grid=self._stellar_grid, limb_darkening_laws='claret4', passbands=self._passbands)

                passbandtest, bandmin, bandmax = self.exoLDC.check_passband_limits(self.wl_edges, self._stellar_grid)
                off = 1e-8 ### This is just to make sure edge bins are still bins
                bandmin = bandmin*1e-4+off
                bandmax = bandmax*1e-4-off
                if passbandtest is False:
                    self.warning('Careful, the requested wavelengths are larger than the stellar grids in '+ self._stellar_grid+' with wave range: '+str(bandmin)+' - '+str(bandmax))
                    exogrid2 = [np.maximum(np.minimum(e, bandmax-off), bandmin+off) for e in exogrid]
                    exowidth2 = [[np.maximum(np.minimum(e[0], bandmax-2*off), bandmin),np.maximum(np.minimum(e[1], bandmax), bandmin+2*off)] for e in exowidth]
                    exogrid = exogrid2
                    exowidth = exowidth2

                self.exoLDC.set_star_params(star_effective_temperature=self._star.temperature, 
                    star_log_gravity=self._star._logg, star_metallicity=self._star._metallicity)
                self.exoLDC.computeLDC(wlgrid_edges=exowidth, wlgrid=exogrid)
                self._ldc = self.exoLDC.get_LDC()
                ldc0 = np.interp(self.wlgrid, exogrid, [self._ldc[i][0] for i in range(len(self._ldc))])
                ldc1 = np.interp(self.wlgrid, exogrid, [self._ldc[i][1] for i in range(len(self._ldc))])
                ldc2 = np.interp(self.wlgrid, exogrid, [self._ldc[i][2] for i in range(len(self._ldc))])
                ldc3 = np.interp(self.wlgrid, exogrid, [self._ldc[i][3] for i in range(len(self._ldc))])
                ldc = [[ldc0[i], ldc1[i], ldc2[i], ldc3[i]] for i in range(len(ldc0))]
            self.info('Sending LDC to all processes')
            ldc = broadcast(ldc, 0)
            self.info('Received LDC from all processes')
            self.ldc = ldc
        elif self._ldc_mode == 'from_star':
            self.ldc = self._star.ldc
        else:
            self.ldc = [self._default_ldc]*len(self.wngrid)
        
        self.info('I am using ldc: '+str(self._ldc_method))
        self.info(self.ldc)

    def initialize_model(self):
        self.period = self._planet.orbitalPeriod
        if self._planet.inclination is None:
            self.inclination = np.arccos(self._star.radius * self._planet.impactParameter / self._planet.get_planet_semimajoraxis(unit='m'))*180/np.pi
        else:
            self.inclination = self._planet.inclination
        self.sma_over_rs = self._planet.get_planet_semimajoraxis(unit='m')/self._star.radius
        #eccentricity = planet.eccentricity
        self.eccentricity = self._planet._eccentricity
        self.periastron = self._planet._pericentre_long
        self.mid_time = self._planet._mid_time
        self.eclipse_mid_time = eclipse_mid_time(self.period, self.sma_over_rs, self.eccentricity, self.inclination, self.periastron, self.mid_time)
        if self.ldc is None:
            self.load_ldc()
    
    def generate_auto_rp2_params(self, bounds = None):
        self.rpbounds = bounds or self._default_rp2_bounds
        for idx, val in enumerate(self.list_rp_rs2):
            point_num = idx+1
            param_name = 'rp_rs2_w{}'.format(point_num)
            param_latex = '$rp_rs2_w{}$'.format(point_num)

            def read_point(self, idx=idx):
                return self.list_rp_rs2[idx]

            def write_point(self, value, idx=idx):
                self.list_rp_rs2[idx] = value

            fget_point = read_point
            fset_point = write_point
            self.debug('FGet_location %s', fget_point)
            default_fit = False
            self.add_fittable_param(param_name, param_latex, fget_point,
                                    fset_point, 'linear', default_fit, self.rpbounds)
            
    def generate_auto_fp_params(self, bounds = None):
        self.fpbounds = bounds or self._default_fp_bounds
        for idx, val in enumerate(self.list_fp_fs):
            point_num = idx+1
            param_name = 'fp_fs_w{}'.format(point_num)
            param_latex = '$fp_fs_w{}$'.format(point_num)

            def read_point(self, idx=idx):
                return self.list_fp_fs[idx]

            def write_point(self, value, idx=idx):
                self.list_fp_fs[idx] = value

            fget_point = read_point
            fset_point = write_point
            self.debug('FGet_location %s', fget_point)
            default_fit = False
            self.add_fittable_param(param_name, param_latex, fget_point,
                                    fset_point, 'log', default_fit, self.fpbounds)

    def generate_auto_coeffs(self, bounds=None):
        self.cppmbounds = bounds or self._default_c_ppm_bounds
        self.cradbounds = self._default_c_rad_bounds
        for idx, val in enumerate(self.list_c0):

            def read_point_c0(self, idx=idx):
                return self.list_c0[idx]
            def write_point_c0(self, value, idx=idx):
                self.list_c0[idx] = value
            self.add_fittable_param('c0_w{}'.format(idx+1), '$c0_w{}$'.format(idx+1), read_point_c0,
                                    write_point_c0, 'linear', False, self.cppmbounds)
            
            def read_point_c1(self, idx=idx):
                return self.list_c1[idx]
            def write_point_c1(self, value, idx=idx):
                self.list_c1[idx] = value
            self.add_fittable_param('c1_w{}'.format(idx+1), '$c1_w{}$'.format(idx+1), read_point_c1,
                                    write_point_c1, 'linear', False, self.cppmbounds)

            def read_point_c3(self, idx=idx):
                return self.list_c3[idx]
            def write_point_c3(self, value, idx=idx):
                self.list_c3[idx] = value
            self.add_fittable_param('c3_w{}'.format(idx+1), '$c3_w{}$'.format(idx+1), read_point_c3,
                                    write_point_c3, 'linear', False, self.cppmbounds)

            def read_point_c2(self, idx=idx):
                return self.list_c2[idx]
            def write_point_c2(self, value, idx=idx):
                self.list_c2[idx] = value
            self.add_fittable_param('c2_w{}'.format(idx+1), '$c2_w{}$'.format(idx+1), read_point_c2,
                                    write_point_c2, 'linear', False, self.cradbounds)
            
            def read_point_c4(self, idx=idx):
                return self.list_c4[idx]
            def write_point_c4(self, value, idx=idx):
                self.list_c4[idx] = value
            self.add_fittable_param('c4_w{}'.format(idx+1), '$c4_w{}$'.format(idx+1), read_point_c4,
                                    write_point_c4, 'linear', False, self.cradbounds)

            def read_point_e0(self, idx=idx):
                return self.list_e0[idx]
            def write_point_e0(self, value, idx=idx):
                self.list_e0[idx] = value
            self.add_fittable_param('e0_w{}'.format(idx+1), '$e0_w{}$'.format(idx+1), read_point_e0,
                                    write_point_e0, 'linear', False, [0,1])                        
            
    def collect_fitting_parameters(self):
        #self._fitting_parameters = {}
        #self._fitting_parameters.update(self.fitting_parameters())
        self._fitting_parameters.update(self._planet.fitting_parameters())
        if self._star is not None:
            self._fitting_parameters.update(self._star.fitting_parameters())
            
        if self._modelpipe is not None:
            self._fitting_parameters.update(self._modelpipe.fittingParameters)
        self.debug('Available Fitting params: %s',
                   list(self._fitting_parameters.keys()))
    
    def build(self):
        self.timelist = self._observation.timelist
        if self._userwngrid is None:
            self.wl_edges = self._observation.wavelist
            self.wlgrid = np.array([(e[1]+e[0])/2 for e in self.wl_edges])
            self.wngrid = 10000/self.wlgrid
        else:
            self.wngrid = self._userwngrid
        
        self.list_rp_rs2 = [1.0*(self._planet.get_planet_radius(unit='m')/self._star.radius)**2]*len(self.wngrid)
        self.list_fp_fs = [1.0*self._planet.get_planet_radius(unit='m')/self._star.radius]*len(self.wngrid)
        
        self.list_c0 = [1e-30]*len(self.wngrid)
        self.list_c1 = [1e-30]*len(self.wngrid)
        self.list_c2 = [np.pi]*len(self.wngrid)
        self.list_c3 = [1e-30]*len(self.wngrid)
        self.list_c4 = [np.pi]*len(self.wngrid)
        self.list_e0 = [0.0]*len(self.wngrid)
        
        self.generate_auto_rp2_params()
        self.generate_auto_fp_params()
        self.generate_auto_coeffs()
        
        self.initialize_model()
        
        t,s,w = self.model(run_pipe=False)
        inputs = [t,s,None,w]
        if self._modelpipe:
            out2 = self._modelpipe.build(out=inputs)
        
        self.collect_fitting_parameters()
        
        
    def defaultBinner(self):
        raise NotImplementedError
    
    def model(self, wngrid=None, cutoff_grid=True, run_pipe = True):
        
        if self.wngrid is None:
            self.wngrid = wngrid 
        self.initialize_model()
    
        self.lc_flux = []

        np.seterr(divide='ignore', invalid='ignore')
        
        for idx, wl in enumerate(self.wngrid):
            #rp_over_rs = self._planet.get_planet_radius(unit='m')/self._star.radius+self.list_rp_rs[idx]
            rp_over_rs = np.sqrt(self.list_rp_rs2[idx])
            fp_over_fs = self.list_fp_fs[idx]
            c0 = self.list_c0[idx]
            c1 = self.list_c1[idx]
            c2 = self.list_c2[idx]
            c3 = self.list_c3[idx]
            c4 = self.list_c4[idx]
            e0 = self.list_e0[idx]
        
            time = np.array(self.timelist[0])
            phi = 2*np.pi*(time - self.mid_time)/self.period

            eclipse_light_curve = eclipse(fp_over_fs, rp_over_rs, self.period, self.sma_over_rs, self.eccentricity,
                                          self.inclination, self.periastron, self.eclipse_mid_time, 
                                          time, precision=self._precision)

            transit_light_curve = transit(self.ldc[idx], rp_over_rs, self.period, self.sma_over_rs, 
                                          self.eccentricity, self.inclination, self.periastron, self.mid_time, 
                                          time, method=self._ldc_method, precision=self._precision)

            cosine_variation = c0 + (c1/2)* (1 - np.cos( phi - c2 )) + (c3/2)*(1 - np.cos( 2*phi - c4 ))

            ## Ellipsoidal var
            cosine_variation = cosine_variation * (1+ (e0/2)*(1-np.cos(2*phi)) )

            # to be changed to add PC model
            #f = 0
            #lc_flux = transit_light_curve + f * (eclipse_light_curve - 1 + cosine_variation) / (cosine_variation)
            
            lc_flux = transit_light_curve + cosine_variation*(eclipse_light_curve-np.min(eclipse_light_curve))/(np.max(eclipse_light_curve)-np.min(eclipse_light_curve))
        
            self.lc_flux.append(lc_flux)
            
        if run_pipe and self._modelpipe:
            in_model = [self.timelist, self.lc_flux, None, self.wngrid]
            self.timelist, self.lc_flux,_, self.wngrid = self._modelpipe.run(in_model)
        
        return self.timelist, self.lc_flux, self.wngrid
    
    @classmethod
    def input_keywords(self):
        return ['pylightcurve_phase', 'plc_phase' ]