from taurex.log import Logger
from taurex.core import Fittable, fitparam
from pop.pipeline.units.baseunit import BaseUnit
from taurex.mpi import barrier, get_rank, nprocs, broadcast
import numpy as np


class IraclisLightLoader(BaseUnit):
    def __init__(self, name='iraclis_loader', file_path=None, fit = False, updown_correction = True):
        Logger.__init__(self,name)
        Fittable.__init__(self)
        super().__init__(name, fit=fit)
        
        self._path = file_path
        self.updown_correction = updown_correction
        self.data = []
        self.load_inputs()
    
    def load_inputs(self, inputs=None):
        if inputs is not None:
            pass
        else:
            self.load_files(self._path)
    
    def determine_orbits(self):
        orbits = np.where(abs(self._full_time - np.roll(self._full_time, 1)) > 20.0 / 60.0 / 24.0)[0]
        dumps = np.where(abs(self._full_time - np.roll(self._full_time, 1)) > 5.0 / 60.0 / 24.0)[0]
        return orbits, dumps

    def apply_updown_correction(self, idxs):
        zerofr = self._star_ypos[idxs]
        sigmfr = self._scan_length[idxs]
        begnfr = zerofr
        fitfr = np.poly1d(np.polyfit(begnfr, sigmfr, 1))
        
        self._lc_white[idxs] = self._lc_white[idxs] * fitfr(begnfr[0]) / fitfr(begnfr)
        self._er_white[idxs] = self._er_white[idxs] * fitfr(begnfr[0]) / fitfr(begnfr)
        for b in range(len(self._raw_lc_bins)):
            self._raw_lc_bins[b][idxs] = self._raw_lc_bins[b][idxs] * fitfr(begnfr[0]) / fitfr(begnfr)
            self._raw_er_bins[b][idxs] = self._raw_er_bins[b][idxs] * fitfr(begnfr[0]) / fitfr(begnfr)

    def load_files(self, phase_path):
        import glob
        import pathlib
        import re
        import os
        import pickle as pkl

        direct_array = None
        full_time = None
        star_ypos = None
        scan_length = None
        lc_white = None
        er_white = None
        raw_lc_bins = None
        raw_er_bins = None
        wl_edges = None
        wl = None

        if get_rank() == 0:
            self.info('Opening Iraclis reduction file')
            d = open(self._path,'rb')
            data = pkl.load(d, encoding="latin1")
            
            direct_array = data['spectrum_direction_array']
            full_time = data['bjd_tdb_array']
            star_ypos = data['y_star_array']
            scan_length = data['scan_length_array']

            lc_white = data['white']['flux']
            er_white = data['white']['error']
            
            raw_lc_bins = []
            raw_er_bins = []
            wl = []
            wl_edges = []

            for lc in data.items():
                if 'bin_' in lc[0]:
                    raw_lc_bins.append(data[lc[0]]['flux'])
                    raw_er_bins.append(data[lc[0]]['error'])
                    minwl = data[lc[0]]['lower_wavelength']
                    maxwl = data[lc[0]]['upper_wavelength']
                    wl_edges.append([minwl*1e-4, maxwl*1e-4])
                    wl.append((maxwl+minwl)/2*1e-4)

            raw_lc_bins = np.array(raw_lc_bins)
            raw_er_bins = np.array(raw_er_bins)
            wl_edges = np.array(wl_edges)
            wl = np.array(wl)
            
            direct_array = list(direct_array)
            full_time = list(full_time)
            star_ypos = list(star_ypos)
            scan_length = list(scan_length)
            lc_white = list(lc_white)
            er_white = list(er_white)
            raw_lc_bins = list(raw_lc_bins)
            raw_er_bins = list(raw_er_bins)
            wl_edges = list(wl_edges)
            wl = list(wl)
            #print(type(direct_array))
        self.info('Broadcasting Inputs...')
        #print(list(direct_array))
        

        self._direct_array = broadcast(direct_array, 0)
        self._full_time = broadcast(full_time, 0)
        self._star_ypos = broadcast(star_ypos, 0)
        self._scan_length = broadcast(scan_length, 0)
        self._lc_white = broadcast(lc_white, 0)
        self._er_white = broadcast(er_white, 0)
        self._raw_lc_bins = broadcast(raw_lc_bins, 0)
        self._raw_er_bins = broadcast(raw_er_bins, 0)
        self._wl_edges = broadcast(wl_edges, 0)
        self._wl = broadcast(wl, 0)
        self.info('Inputs received by all cores.')

        self._direct_array = np.array(self._direct_array)
        self._full_time = np.array(self._full_time)
        self._star_ypos = np.array(self._star_ypos)
        self._scan_length = np.array(self._scan_length)
        self._lc_white = np.array(self._lc_white)
        self._er_white = np.array(self._er_white)
        self._raw_lc_bins = np.array(self._raw_lc_bins)
        self._raw_er_bins = np.array(self._raw_er_bins)
        self._wl_edges = np.array(self._wl_edges)
        self._wl = np.array(self._wl)

        #self._wl = np.array(self._wl)
        #self._wl_edges = np.array(self._wl_edges)

        self._orbits, self._dumps = self.determine_orbits()

        self._forward_idxs = np.where(self._direct_array == 1)[0]
        self._reverse_idxs = np.where(self._direct_array == -1)[0]
        
        if self.updown_correction:
            self.apply_updown_correction(self._forward_idxs)
            if len(np.where(self._reverse_idxs == -1)[0]) > 0:
                self.apply_updown_correction(self._reverse_idxs)

        #if self._direction == 'forward':
        #    self.idxs_direction = self._forward_idxs
        #elif self._direction == 'reverse':
        #    self.idxs_direction = self._reverse_idxs
        #else:
        #    self.idxs_direction = np.arange(1, len(self._direct_array)-1, 1, dtype=int)

        # set default to white
        self.set_spectrum(idx='white')

    def set_spectrum(self, idx = 'white', div_white = False):
        if idx == 'white':
            self._times = [self._full_time[self._forward_idxs], self._full_time[self._reverse_idxs]]
            self._spectrum = [self._lc_white[self._forward_idxs],self._lc_white[self._reverse_idxs]]
            self._error = [self._er_white[self._forward_idxs], self._er_white[self._reverse_idxs]]
            self._wlgrid = np.average(self._wl)
            self._wlgrid_edge = [np.min(self._wl_edges), np.max(self._wl_edges)]
            self._wngrid = 10000/self._wlgrid 
        else:
            self._times = [self._full_time[self._forward_idxs], self._full_time[self._reverse_idxs]]
            if div_white:
                self._spectrum = [self._raw_lc_bins[idx][self._forward_idxs]/self._lc_white[self._forward_idxs],self._raw_lc_bins[idx][self._reverse_idxs]/self._lc_white[self._reverse_idxs]]
                self._error = [self._raw_er_bins[idx][self._forward_idxs]/self._lc_white[self._forward_idxs], self._raw_er_bins[idx][self._reverse_idxs]/self._lc_white[self._reverse_idxs]]
            else:
                self._spectrum = [self._raw_lc_bins[idx][self._forward_idxs],self._raw_lc_bins[idx][self._reverse_idxs]]
                self._error = [self._raw_er_bins[idx][self._forward_idxs], self._raw_er_bins[idx][self._reverse_idxs]]
            self._wlgrid = self._wl[idx]
            self._wlgrid_edge = self._wl_edges[idx]
            self._wngrid = 10000/self._wlgrid 
        
    def apply_step(self, inputs = None):
        self._time = self._times
        self._input_data = self._spectrum
        self.outputs = [self._times, self._spectrum, self._error, self._wlgrid_edge]
        return self.outputs
    
    @classmethod
    def input_keywords(self):
        return ['lc_iraclis_loader']