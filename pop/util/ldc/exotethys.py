
from re import A, M
from .baseldc import BaseStarLDC
import numpy as np
from taurex.mpi import barrier, get_rank, nprocs


class ExotethysLDC(BaseStarLDC):
    """Computes the Limb Darkening Coefficients from Exothetis
        https://github.com/ucl-exoplanets/ExoTETHyS
    Parameters
    -----------

    """
    
    def __init__(self, calculation_type='individual', stellar_models_grid='Phoenix_2012_13',
                 limb_darkening_laws='claret4', passbands='uniform_phoenix_2012_13'):
        super().__init__()
        
        self._calculation_type = calculation_type
        self._stellar_models_grid = stellar_models_grid
        self._limb_darkening_laws = limb_darkening_laws
        self.set_passband(passbands=passbands)
    
    def set_star_params(self, star_effective_temperature=6065.0, star_log_gravity=4.36, star_metallicity=0.0):
        self._star_effective_temperature = star_effective_temperature
        self._star_log_gravity = star_log_gravity
        self._star_metallicity = star_metallicity
    
    def set_passband(self, passbands='HST_WFC3_G141', wavelength_bins_path='aux_files/wavelength_bins_files/',
                     wavelength_bins_files='WFC3_G141_bins_Tsiaras2018_low.txt'):
        self._passbands = passbands
        self._wavelength_bins_path = wavelength_bins_path
        self._wavelength_bins_files = wavelength_bins_files

    def check_passband_limits(self, pb_waves, stellar_models_grid):
        """
        TAKEN AS IS FROM EXOTETHYS
        This function checks that the wavelengths read from a passband file are within the limits for the chosen stellar_models_grid and returns a boolean value.
        
        :param quantity array pb_waves: 1D array of wavelengths read from the passband file
        :params str stellar_models_grid: the name of the chosen stellar database
        :return: True if the wavelengths are within the limits, False otherwise
        :rtype: bool
        """
        check = True
        minimum_wavelength = 0.0
        maximum_wavelength = 0.0
        if stellar_models_grid == 'Phoenix_2018':
            minimum_wavelength = 500.0 
            maximum_wavelength = 25999.0 
            if np.min(pb_waves)<minimum_wavelength or np.max(pb_waves)>maximum_wavelength:
                check = False
        elif stellar_models_grid == 'Phoenix_2012_13':
            minimum_wavelength = 2500.0 
            maximum_wavelength = 99995.0 
            if np.min(pb_waves)<minimum_wavelength or np.max(pb_waves)>maximum_wavelength:
                check = False
        elif stellar_models_grid == 'Phoenix_drift_2012':
            minimum_wavelength = 10.0 
            maximum_wavelength = 9000000.0 
            if np.min(pb_waves)<minimum_wavelength or np.max(pb_waves)>maximum_wavelength:
                check = False
        elif stellar_models_grid == 'Atlas_2000':
            minimum_wavelength = 90.9 
            maximum_wavelength = 1600000.0 
            if np.min(pb_waves)<minimum_wavelength or np.max(pb_waves)>maximum_wavelength:
                check = False
        elif stellar_models_grid == 'Stagger_2018':
            minimum_wavelength = 1010.0 
            maximum_wavelength = 199960.16 
            if np.min(pb_waves)<minimum_wavelength or np.max(pb_waves)>maximum_wavelength:
                check = False
        elif stellar_models_grid == 'Stagger_2015':
            minimum_wavelength = 2000.172119140625 
            maximum_wavelength = 10000.0791015625 
            if np.min(pb_waves)<minimum_wavelength or np.max(pb_waves)>maximum_wavelength:
                check = False
        return check, minimum_wavelength, maximum_wavelength

    def auto_passband(self, wlgrid_edges=None, wlgrid=None):
        from pathlib import Path
        directory = 'exothetis_temp/'
        Path(directory).mkdir(parents=True, exist_ok=True)
        self._wavelength_bins_path = directory
        self._wavelength_bins_files = 'taurex_auto_bins.txt'
        if wlgrid_edges is not None:
            np.savetxt(self._wavelength_bins_path+self._wavelength_bins_files, np.array(wlgrid_edges)*1e4)
        elif wlgrid is not None:
            width = [(wlgrid[i+1]- wlgrid[i]) for i in range(len(wlgrid)-1)]
            width.append(wlgrid[-1]-wlgrid[-2])       
            bands = [[wlgrid[i]-width[i]/2, wlgrid[i]+width[i]/2] for i in range(len(wlgrid))]
            np.savetxt(self._wavelength_bins_path+self._wavelength_bins_files, np.array(bands)*1e4)
        else:
            print('The wlgrid was not provided for auto LDC')
            raise NotImplementedError
    
    def create_dico(self):
        dico = dict({'calculation_type': [self._calculation_type],
                    'stellar_models_grid': [self._stellar_models_grid],
                    'limb_darkening_laws': [self._limb_darkening_laws],
                    'passbands': [self._passbands],
                    'wavelength_bins_path': [self._wavelength_bins_path],
                    'wavelength_bins_files': [self._wavelength_bins_files],
                    'star_effective_temperature': [self._star_effective_temperature],
                    'star_log_gravity': [self._star_log_gravity],
                    'star_metallicity': [self._star_metallicity],
                    'target_names': ['mytarget'],
                    'output_path': ['exothetis_temp/']})
        return dico
    
    def clean_res(self, res):
        thekeys = dict((key,value) for key, value in res[0]['target']['mytarget']['passbands'].items() if self._passbands+'_' in key)
        final = [thekeys[i][self._limb_darkening_laws]['coefficients'] for i in thekeys.keys()] 
        
        return final
    
    def computeLDC(self, wlgrid_edges=None, wlgrid=None):
        """Is called to compute the LDC initially
            wlgrid_passband has the form [[wl_min0, wl_max0], [wl_min1, wl_max1], ...]
        """
        from exotethys import sail
        self.auto_passband(wlgrid_edges=wlgrid_edges, wlgrid=wlgrid)
        self._dico = self.create_dico()
        [check, self._checked_dict] = sail.check_configuration(self._dico)
        res = sail.process_configuration(self._checked_dict)
        self._ldc = self.clean_res(res)
        
    def computeLDCfromFile(self, passbands='HST_WFC3_G141', wavelength_bins_path='aux_files/wavelength_bins_files/',
                     wavelength_bins_files='WFC3_G141_bins_Tsiaras2018_low.txt'):
        from exotethys import sail
        self.set_passband(passbands=passbands, wavelength_bins_path=wavelength_bins_path,
                     wavelength_bins_files=wavelength_bins_files)
        self._dico = self.create_dico()
        [check, self._checked_dict] = sail.check_configuration(self._dico)
        res = sail.process_configuration(self._checked_dict)
        self._ldc = self.clean_res(res)