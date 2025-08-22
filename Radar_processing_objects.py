import matplotlib.pyplot as plt
import numpy
import xarray
import pandas as pd
import matplotlib.lines as mlines

import pyart
import netCDF4
import h5py as h5
from glob import glob
import os
import numpy as np
import cartopy.crs as ccrs
import warnings 
import time 
import cartopy.feature as cfeature
import cartopy.io.img_tiles as cimgt
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
#import rasterio
import matplotlib
import wradlib
warnings.filterwarnings("ignore", category=DeprecationWarning)

from copy import deepcopy

from Aggregated_h5_loading import read_nimrod_aggregated_odim_h5
from wradlib import trafo

class NightSweeper():

    def __init__(self, files: dict, times: dict[list], pulses: list, apply_filter: bool = True):

        self.times = times
        
        if type(pulses) != list:
            pulses = [pulses]
        
        self.pulses = pulses
        print(self.pulses)
        
        self.evening_radars = {pulse : {time : None for time in times['evening']} for pulse in pulses}
        self.night_radars = {pulse : {time : None for time in times['night']} for pulse in pulses}

        print('Loading evening files...')
        
        for pulse in pulses:
            for time in times['evening']:
                print(pulse, time)       
                self.evening_radars[pulse][time] = self._load_file(files['evening'], pulse, time)
                                
        print('Loading night files...')
        for pulse in pulses:
            for time in times['night']:
                print(pulse, time)
                self.night_radars[pulse][time] = self._load_file(files['night'], pulse, time)

        self.radar_dict = self._combine_radars_from_two_files()      

        self.processed_radar_dict = {
            pulse: {
                time : None for time in self.full_time_list
            } for pulse in pulses
        }

        for pulse in self.pulses:
            for time in self.full_time_list:
                self.processed_radar_dict[pulse][time] = self._set_up_radar_filtered(self.radar_dict[pulse][time])

        self.elevation_list = [0.5,1,2,3,4]

    def _load_file(self, path, pulse, time):
        return read_nimrod_aggregated_odim_h5(path, pulse, time)
        

    def _combine_radars_from_two_files(self):

        self.full_time_list = self.times['evening'] + self.times['night']
        radar_dict = {
            pulse: {
                time : None for time in self.full_time_list
            } for pulse in self.pulses
        }

        for pulse in self.pulses:
            for time in self.times['evening']:
                radar_dict[pulse][time] = self.evening_radars[pulse][time]
            for time in self.times['night']:
                radar_dict[pulse][time] = self.night_radars[pulse][time]

        return radar_dict
        
    
    def _set_up_radar_filtered(self,radar_input):

        radar = deepcopy(radar_input)
        
        ZDR_lin= trafo.idecibel(radar.fields['differential_reflectivity']['data'].data)
        DR_num=ZDR_lin+1-2*ZDR_lin**(1/2)*radar.fields['cross_correlation_ratio']['data'].data
        DR_denom=ZDR_lin+1+2*ZDR_lin**(1/2)*radar.fields['cross_correlation_ratio']['data'].data
    
        DR=trafo.decibel(DR_num/DR_denom)
        
        alts=[]
        for e, e_val in enumerate(radar.fixed_angle['data']):
            lat, lon, alt= radar.get_gate_lat_lon_alt(e)
            if e==0:
                alts=alt
            else:
                alts=np.append(alts,alt,axis=0)
        

        
        excluded = np.where(
                        np.logical_or(
                            np.logical_or(
                                np.logical_or(
                                    np.logical_or(DR < -12, 
                                                  radar.fields['normalized_coherent_power']['data'].data < 0.5),
                                    radar.fields['reflectivity']['data'].data > 35),
                                radar.fields['differential_reflectivity']['data'].data<0),
                            radar.fields['normalized_coherent_power']['data'].data > 0.8)
        )
                        

        radar.fields['reflectivity']['data'].data[excluded]=np.nan
        radar.fields['differential_reflectivity']['data'].data[excluded]=np.nan
        radar.fields['cross_correlation_ratio']['data'].data[excluded]=np.nan
        radar.fields['differential_phase']['data'].data[excluded]=np.nan
        radar.fields['velocity']['data'].data[excluded]=np.nan
        radar.fields['normalized_coherent_power']['data'].data[excluded]=np.nan

        return radar

    @staticmethod
    def make_square(ax):
        ax.set_aspect('equal', 'box')
        return ax
    
    def plot_ppis(self, 
                  R, 
                  ele,
                  pulse,
                  filtered: bool = True, 
                  time_subset: list = None , 
                  save_fig: str = None,
                  plot_ncp: bool = False,
                  plot_velocity: bool = False,
                  plot_kwargs: dict = None):

        if time_subset:
            for time in time_subset:
                if not time in self.times:
                    raise ValueError('Times in time_subset must be in original time list') 

        if filtered:
            radar_dict = self.processed_radar_dict
        else: 
            radar_dict = self.radar_dict 
        
        nrows = len(self.times['evening']) + len(self.times['night'])
        if plot_ncp or plot_velocity:
            ncols =5
        else:
            ncols=4
        
        fig = plt.figure(figsize = (15,20))



        # For now just manually select variables
        
        counter = 0
        for time, radar in radar_dict[pulse].items():
            display = pyart.graph.RadarDisplay(radar)

            ax1 = fig.add_subplot(nrows,ncols,1+counter)
            display.plot('reflectivity',ele,ax=ax1, vmin=-32, vmax=45., title='Horizontal Reflectivity', colorbar_label=radar.fields['reflectivity']['units'], axislabels=('', ''))
            display.set_limits((-R, R), (-R, R), ax=ax1)
            self.make_square(ax1)
            ax1.set_ylabel(str(time), fontsize=14, fontweight='bold')
    
            ax2 = fig.add_subplot(nrows,ncols,2+counter)
            display.plot('differential_reflectivity', ele,ax=ax2, vmin=-2, vmax=10., title='Differential Reflectivity', colorbar_label=radar.fields['differential_reflectivity']['units'],
                         axislabels=('', ''), cmap = 'pyart_RefDiff')
            display.set_limits((-R, R), (-R, R), ax=ax2)
            self.make_square(ax2)
        
            ax3 = fig.add_subplot(nrows,ncols,3+counter)
            display.plot('cross_correlation_ratio', ele,ax=ax3, vmin=0.0, vmax=1.0, title='Cross Correlation Ratio', colorbar_label=radar.fields['cross_correlation_ratio']['units'],
                         axislabels=('', ''), cmap = 'pyart_RefDiff')
            display.set_limits((-R, R), (-R, R), ax=ax3)
            self.make_square(ax3)
            
            ax4 = fig.add_subplot(nrows,ncols,4+counter)
            display.plot('differential_phase', ele,ax=ax4, vmin=-5, vmax=120., title='Differential Phase', colorbar_label=radar.fields['differential_phase']['units'],axislabels=('', ''),
                         cmap = 'pyart_RefDiff') #cmap = 'pyart_Wild25')
            display.set_limits((-R, R), (-R, R), ax=ax4)
            self.make_square(ax4)

            if plot_ncp:
                ax5 = fig.add_subplot(nrows,ncols,5+counter)
                display.plot('normalized_coherent_power', ele, ax=ax5, title='Normalized Coherent Power', colorbar_label=radar.fields['normalized_coherent_power']['units'],axislabels=('', ''),
                            cmap = 'pyart_Carbone17')
                display.set_limits((-R, R), (-R, R), ax=ax5)

            if plot_velocity:

                ax5 = fig.add_subplot(nrows,ncols,5+counter)
                display.plot('velocity',ele,ax=ax5, vmin=-2, vmax=2., title='Doppler Velocity', colorbar_label=radar.fields['velocity']['units'],
                             axislabels=('', ''), cmap = 'pyart_BuDRd18')
                display.set_limits((-R, R), (-R, R), ax=ax5)
                
            
            if plot_ncp or plot_velocity:
                counter +=5
            else:
                counter +=4

            for rr in [5, 10, 15, 20]:   # pick ranges you want
                for ax in [ax1, ax2, ax3, ax4]:
                    display.plot_range_ring(rr, ax=ax, ls = '--', lw = 0.5)

        fig = plt.gcf()
        y = 1 - 1/nrows   # normalized figure coordinate where row 1 ends
        line = mlines.Line2D([0.05, 0.95], [y, y], color='k', linestyle='--', transform=fig.transFigure)
        fig.add_artist(line)  

        fig.text(0.5, y + 0.01, "Sunset", ha='left', va='bottom',
        fontsize=12, fontweight='bold')
        
        plt.tight_layout()

        if save_fig:
            plt.savefig(save_fig)


    def _get_altitude(self, ele_index, R):
        ele_deg = self.elevation_list[ele_index]
        return np.deg2rad(ele_deg)*R

    def get_data_in_range(self, R: int, gates: list[int], ele_index:int, pulse: str, time: str, field_name: str, filtered: bool, smoothing_window: int = 20):

        if filtered:
            radar = deepcopy(self.processed_radar_dict[pulse][time])
        else:
            radar = deepcopy(self.radar_dict[pulse][time])
        
        # Get the start and end ray indices for the elevation sweep
        start_ray = radar.sweep_start_ray_index['data'][ele_index]
        end_ray = radar.sweep_end_ray_index['data'][ele_index] + 1
        
        # Pick a specific range gate index (e.g., ~20 km away)
        gate_index = np.argmin(np.abs(radar.range['data'] - R))
        
        # Extract azimuths for the sweep
        azimuths = radar.azimuth['data'][start_ray:end_ray]
        
        
        values_array = np.zeros((len(gates),len(azimuths),))
        values_array[:]=np.nan
        range_labels = []
        masks = np.zeros((len(gates), len(azimuths)))
        
        plt.figure(figsize=(8,5))
        for index, gate in enumerate(gates):
            adjusted_gate_index = gate_index + gate
            field_values = radar.fields[field_name]['data'][start_ray:end_ray, adjusted_gate_index]
            values_array[index, :] = field_values[:]
            masks[index, :] = field_values.mask
            
            range_label = radar.range['data'][adjusted_gate_index]/1000
            range_labels.append(range_label)
            #plt.plot(azimuths, np.ma.array(values_array[index, :],mask=masks[index, :]), marker='o', linestyle = '', label = f'{range_label:.1f} km')
        
        masked_values = np.ma.array(values_array,mask=masks)
        means_array = np.nanmean(masked_values, axis=0)
        smoothed_means= pd.Series(means_array).rolling(window=smoothing_window, center=True, min_periods=1).mean().to_numpy()
        
        max_alt = self._get_altitude(ele_index, radar.range['data'][gate_index + gates[-1]])
        min_alt = self._get_altitude(ele_index, radar.range['data'][gate_index + gates[0]])
        
        return smoothed_means, masked_values, range_labels, (min_alt, max_alt), azimuths
    
    def plot_range_single_field_cartesian(self, R: int, gates: list[int], ele_index:int, pulse: str, time: str, field_name: str, filtered: bool, smoothing_window: int = 20, save_fig: str = None):

        dummy_radar = self.radar_dict[pulse][time]
        
        smoothed_means, values_array, range_labels, alts, azimuths = self.get_data_in_range(R, gates, ele_index, pulse, time, field_name, filtered, smoothing_window)
        
        #fig, ax = plt.subplots(1,1,figsize=(8,5),subplot_kw={'projection':''})
        plt.figure(figsize=(8,5))
        
        for index, gate in enumerate(gates):
            range_label = range_labels[index]

            #ignoring potential azimuths issue -> ask neely if azimuths alwasy 0-360
            plt.plot(azimuths, values_array[index, :], marker='o', linestyle = '', label = f'{range_label:.1f} km')
                
        #ax.plot(np.deg2rad(np.arange(0,360)),smoothed_means,label = f'mean {field_name}', color = 'cyan', linewidth = 1)
        plt.plot(azimuths,smoothed_means,label = f'mean {field_name}', color = 'cyan', linewidth = 1)
        #ax.set_theta_zero_location("N")
        
        plt.legend()
        
        plt.xlabel("Azimuth (degrees)")
        plt.ylabel(f"{field_name} ({dummy_radar.fields[field_name]['units']})")
        plt.title(f"{field_name} vs Azimuth at {R/1000} km, elev.: {self.elevation_list[ele_index]}, alt.: ~{alts[0]:.0f}-{alts[1]:.0f}m")
        plt.grid(True)
        plt.show()

        if save_fig:
            plt.savefig(save_fig)


    def plot_range_single_field_polar(self, R: int, gates: list[int], ele_index:int, pulse: str, time: str, field_name: str, filtered: bool, smoothing_window: int = 20, save_fig: str = None):

        dummy_radar = self.radar_dict[pulse][time]
        
        smoothed_means, values_array, range_labels, alts, azimuths = self.get_data_in_range(R, gates, ele_index, pulse, time, field_name, filtered, smoothing_window)
        
        fig, ax = plt.subplots(1,1,figsize=(12,8),subplot_kw={'projection':'polar'})
        
        for index, gate in enumerate(gates):
            range_label = range_labels[index]

            plt.plot(np.deg2rad(azimuths), values_array[index, :], marker='o', linestyle = '', label = f'{range_label:.1f} km', alpha = 0.6)
                
        ax.plot(np.deg2rad(azimuths),smoothed_means,label = f'mean {field_name}', color = 'cyan', linewidth = 5)
        ax.set_theta_zero_location("N")
        
        plt.legend(loc='lower left')
    
        plt.xlabel("Azimuth (degrees)")
        plt.ylabel(f"{field_name} ({dummy_radar.fields[field_name]['units']})")
        plt.title(f"{field_name} vs Azimuth at {R/1000} km, elev.: {self.elevation_list[ele_index]}, alt.: ~{alts[0]:.0f}-{alts[1]:.0f}m")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        if save_fig:
            plt.savefig(save_fig)
                        
