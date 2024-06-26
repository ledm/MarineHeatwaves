import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, LinearSegmentedColormap


from glob import glob
from netCDF4 import Dataset,num2date
import cftime
from bisect import bisect
from scipy import interpolate

import os
import csv
import math

import numpy as np
import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
#import cartopy.io.shapereader as shpreader
#from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
#import cmocean
from shelve import open as shopen
from matplotlib.ticker import MaxNLocator
from matplotlib import cm
#import nctoolkit
import pickle

from earthsystemmusic2.music_utils import folder
from add_distortion import add_distortion_to_fig

#from .earthsystemmusic2 import music_utils as mutils 
#from earthsystemmusic2.climate_music_maker import climate_music_maker

daily_count = 'daily17'
global_plot_every_days=2
video_format = False # '720p' # '4K' #'720p' # 'UHD'
do_distortion = False
reduced_years = np.arange(1975, 2072)
no_new_plots=False


central_longitude = -14.36816 #W #-160.+3.5
central_latitude = -7.94097 # S
mpa_radius = 2.88

ortho_pro=ccrs.Orthographic(-15, -15)
pc_proj=cartopy.crs.PlateCarree()

#TO DO:
# v16 notes:
 # noise did not work.
 # maybe jsut do a blend between the filtered one? without noise?
# 2068.
# legend font is jumpy. Can that be continuous?
# can we just run a short few years at a time?
# Not sure the jump to anomaly globe is needed at 1986. Maybe save it for 2020?


# v 14:
# done start is too quick. remove 1977. frame?
# done legend  background black?
# done legend HW pulse too quick, no fade
# done Faster movement in 2052-2054.
# done More movement in 2060-2065
# time series extends backwards too soon. Maybe later? start in 2066, 

# Turn less at the end.
# colorbar slides too early
# show both globes at once and make them dance?
# Rotate globes 360 and back? pulsate with the beats?
# circle around each other?
# emphasise heatwaves with rapid shaking?

# Last few years: 
# expand TS axes to cover more of figure. - done
# Move zoomed out globe to centre right.

# Mpa circle:
# Only have a circle when note changes. Load velocity from midi maker or duplicate calculation method.
# Circle radius is a function of time since the note changed and is not linked to number of days or frame rate. 
# Colour is continuous scale linked to real data? 
# Line width scales with axes size.

# Anomaly plot goes behind other one.
# Colour bar need to move closer to timing of switch.


climate_stripes_colours = [
    '#08306b', '#08519c', '#2171b5', '#4292c6',
    '#6baed6', '#9ecae1', '#c6dbef', '#deebf7',
    '#fee0d2', '#fcbba1', '#fc9272', '#fb6a4a',
    '#ef3b2c', '#cb181d', '#a50f15', '#67000d',
   ]
climate_stripes = ListedColormap(climate_stripes_colours)
anom_bins=20
smooth_stripes = LinearSegmentedColormap.from_list('smooth_stripes', climate_stripes_colours,N=anom_bins)

# temperature_anom_cm = cm.hot_r
# temperature_anom_cm = cm.PuRd
temperature_anom_cm = cm.winter_r

temperature_anom_norm = matplotlib.colors.Normalize(vmin=1, vmax=3.5)
temperature_bins = 5 

vmins = {
    'thetao_con': 22.5,
    'so_abs': 34.95,
    'O3_TA': 2000., 
    'O3_pH':7.8,
    'N3_n': 0., 
    'O3_c': 2000.,   
    'uo': -0.4,
    'vo': -1.,
    'Ptot_c_result': 0.,
    'Ztot_c_result': 0.,
    'Ptot_Chl_result': 0.,
    'Ptot_NPP_result': 0.,
    }
vmaxs = {
    'thetao_con': 32.,
    'so_abs': 36.55,
    'O3_TA': 2500., 
    'O3_pH': 8.3,
    'N3_n': 2., 
    'O3_c': 2400.,   
    'uo': 0.4,
    'vo': 1.,    
    'Ptot_c_result': 42.,
    'Ztot_c_result': 30.,
    'Ptot_Chl_result': 0.5,
    'Ptot_NPP_result': 45.,
    }

anom_mins = {
    'thetao_con': -1.5, 
    # 'thetao_con':-15,
    # 'so_abs': 34.95,
    # 'O3_TA': 2000., 
    'O3_pH':-0.32,
    # 'N3_n': 0., 
    # 'O3_c': 2000.,   
    # 'uo': -0.4,
    # 'vo': -1.,
    'Ptot_c_result': -25,
    'Ztot_c_result': -12,
    # 'Ptot_Chl_result': 0.,
    # 'Ptot_NPP_result': 0.,                
    }
anom_maxs = {
    'thetao_con': 4.0, 
    # 'thetao_con':-15,
    # 'so_abs': 34.95,
    # 'O3_TA': 2000., 
    'O3_pH': 0.05,
    # 'N3_n': 0., 
    # 'O3_c': 2000.,   
    # 'uo': -0.4,
    # 'vo': -1.,
    'Ptot_c_result': 10.,
    'Ztot_c_result': 7.5,
    # 'Ptot_Chl_result': 0.,
    # 'Ptot_NPP_result': 0.,                
    }
cbar_vmin = {
    'thetao_con': 5.0,
    'thetao_con_anomaly': -5.,
    }
cbar_vmax = {
    'thetao_con': 31.0,
    'thetao_con_anomaly': 5.,
    }

cm_bins = {
    'thetao_con': 26,
    'thetao_con_anomaly': anom_bins,
}

cmaps = {
    'thetao_con': 'plasma', #'viridis',
    'thetao_con_anomaly': smooth_stripes, #climate_stripes, #'seismic', #'viridis',

#     'so_abs': 'viridis'
#    'O3_TA': 2500., 
#    'N3_n': 20., 
#    'O3_c': 2500.,   
#    'uo': 1.,
#    'vo': 1.,     
    }
land_color = '#D3D3D3' #'#F5F5F5' #'#DCDCDC' # '#E0E0E0' ##F8F8F8'

def mn_str(month):
    mn = '%02d' %  month
    return mn

def makeLonSafe(lon):
    """
    Makes sure that the value is between -180 and 180.
    """
    while True:
        if -180 < lon <= 180: return lon
        if lon <= -180: lon += 360.
        if lon > 180: lon -= 360.


def makeLatSafe(lat):
    """
    Makes sure that the value is between -90 and 90.
    """
    if -90. <= lat <= 90.: return lat
    if lat is np.ma.masked: return lat
    print("makeLatSafe:\tERROR:\tYou can\'t have a latitude > 90 or <-90", lat)
    assert False


def makeLonSafeArr(lon):
    """
    Makes sure that the entire array is between -180 and 180.
    """

    if lon.ndim == 3:
        for (
                l,
                ll,
                lll,
        ), lo in np.ndenumerate(lon):
            lon[l, ll, lll] = makeLonSafe(lo)
        return lon
    if lon.ndim == 2:
        for l, lon1 in enumerate(lon):
            for ll, lon2 in enumerate(lon1):
                lon[l, ll] = makeLonSafe(lon2)
        return lon
    if lon.ndim == 1:
        for l, lon1 in enumerate(lon):
            lon[l] = makeLonSafe(lon1)
        return lon
    assert False


def closest_index(arr, value):
    """
    Lat:
    Value
    """
    index = np.argmin(abs(arr - value))
    print("closest_index found: ", index, (arr[index], 'for', value))
    return index


def getOrcaIndexCC(
    lat,
    lon,
    latcc,
    loncc,
    debug=True,
):
    """
    Takes a lat and long coordinate, an returns the position of the closest coordinate in the grid.
    """
    km = 10.E20
    la_ind, lo_ind = -1, -1
    lat = makeLatSafe(lat)
    lon = makeLonSafe(lon)

    c = (latcc - lat)**2 + (loncc - lon)**2

    (la_ind, lo_ind) = np.unravel_index(c.argmin(), c.shape)

    if debug:
        print('location ', [la_ind, lo_ind], '(', latcc[la_ind, lo_ind],
              loncc[la_ind, lo_ind], ') is closest to:', [lat, lon])
    return la_ind, lo_ind



def get_paths():
    """
    Get paths for the models.
    """
    paths = {}
    paths['CNRM_hist'] = '/data/proteus3/scratch/gig/MissionAtlantic/CNRM_hist/OUTPUT/CNRM_hist/'
    paths['CNRM_ssp126'] = '/data/proteus3/scratch/gig/MissionAtlantic/CNRM_ssp126/OUTPUT/CNRM_ssp126/'
    paths['CNRM_ssp370'] = '/data/proteus3/scratch/gig/MissionAtlantic/CNRM_ssp370/OUTPUT/CNRM_ssp370/'
    paths['GFDL_hist'] = '/data/proteus3/scratch/gig/MissionAtlantic/GFDL_hist/OUTPUT/GFDL_hist/'
    paths['GFDL_ssp126'] = '/data/proteus4/scratch/ledm/MissionAtlantic/GFDL_ssp126/OUTPUT/GFDL_ssp126/'
    paths['GFDL_ssp370'] = '/data/proteus4/scratch/ledm/MissionAtlantic/GFDL_ssp370/OUTPUT/GFDL_ssp370/'
    paths['output'] = '/data/proteus3/scratch/ledm/MissionAtlantic/post_proc/'
    return paths



def find_corners(nc):
    """
    location  [640, 1080] ( -10.933042 -17.25 ) is closest to: [-10.820978677133848, -17.248164721459744]
    location  [664, 1103] ( -4.9936657 -11.5 ) is closest to: [-5.060978677133847, -11.488164721459743]
    Lower left, lat lon: (640, 1080)
    Upper right, lat lon: (664, 1103)
    """
    if isinstance(nc, 'str'):
        nc = Dataset(nc, 'r')
    lats = nc.variables['nav_lat'][:]
    lons = nc.variables['nav_lon'][:]

    AI_lat_ll, AI_lon_ll = getOrcaIndexCC(
        central_latitude-mpa_radius,
        central_longitude-mpa_radius,
        lats,
        lons)
    AI_lat_ur, AI_lon_ur = getOrcaIndexCC(
        central_latitude+mpa_radius,
        central_longitude+mpa_radius,
        lats,
        lons)

    print('Lower left, lat lon:', (AI_lat_ll, AI_lon_ll))
    print('Upper right, lat lon:', (AI_lat_ur, AI_lon_ur))

    new_lats = lats[640:664, 1080:1103]
    new_lons = lons[640:664, 1080:1103]

    print('lat:', new_lats.min(), new_lats.mean(), new_lats.max())
    print('lon:', new_lons.min(), new_lons.mean(), new_lons.max())


def calc_mean_aimpa(nc, model='CNRM', field = 'thetao_con'):
    """
    Calculate the mean temperature in the AIMPA over the monthly data
    """
    if model == 'CNRM':
        try:
            temp = nc.variables[field][:, 0, 640:664, 1080:1103]
        except:
            return {}, {}
        area = nc.variables['area'][640:664, 1080:1103]
    else:
        assert 0
    times = nc.variables['time_centered']
    dates = num2date(times[:], times.units, calendar=times.calendar)

    data_dict = {}
    date_dict = {}

    for t, dt in enumerate(dates):
        meantemp = np.average(temp[t], weights=area)
        date_key = (dt.year, dt.month, dt.day)
        print(t, dt, date_key, meantemp)
        data_dict[date_key] = meantemp
        date_dict[date_key] = dt
    return  data_dict, date_dict


def save_shelve(shelvefn, finished_files, datas_dict, dates_dict):
    """
    Saves a shelve
    """ 
    print('saving shelve:', shelvefn, len(datas_dict.keys()))
    sh = shopen(shelvefn)
    sh['files'] = finished_files
    sh['datas_dict'] = datas_dict
    sh['dates_dict'] = dates_dict
    sh.close()


def load_shelve(shelvefn):
    """
    Load a shelve
    """
    if len(glob(shelvefn+'*')):
          print('loading:', shelvefn)
          sh = shopen(shelvefn)
          a = sh['files']
          b = sh['datas_dict']
          c = sh['dates_dict']
          sh.close()
    else:
          print('unable to load shelve:', shelvefn)
          a = []
          b = {}
          c = {}
    return a, b, c


def decimal_year(dt, year,month,day):
    """
    Takes y,m,d and returns decimal year.
    """
    t0 = cftime.DatetimeGregorian(year, 1, 1, 12., 0, 0, 0)
    td = dt - t0
    t_end = cftime.DatetimeGregorian(year, 12, 31, 12., 0, 0, 0)

    days_in_year = t_end - t0
    days_in_year = days_in_year.days +1
    dec_t = year + (float(td.days)/days_in_year)
    #print('decimal_year:', dt, td.days, 'of', days_in_year, dec_t)
    return dec_t


def calculate_clim(datas_dict, dates_dict, clim_range=[1976,1985]):
    """
    Calculate the climatology data.
    """
    #create dicts:
    clim_datas = {}
    clim_doy = {}
    clim_month = {m:[] for m in np.arange(1, 13, 1)}

    # Fill dictionarys:
    for time_key in sorted(datas_dict.keys()):
        (year, month, day) = time_key
        if year < np.min(clim_range): 
            continue
        if year > np.max(clim_range):
            continue
        dat = datas_dict[time_key]
        dt = dates_dict[time_key]
        clim_key = (month, day) 
        doy = dt.dayofyr

        if clim_key in clim_datas.keys():
            clim_datas[clim_key].append(dat)
        else:
            clim_datas[clim_key] = [dat, ]

        if doy in clim_doy.keys():
            clim_doy[doy].append(dat)
        else:
            clim_doy[doy] = [dat, ]

        clim_month[month].append(dat)

    # calculate clims
    for key, values in clim_datas.items():
        clim_datas[key] = np.mean(values)

    for key, values in clim_doy.items():
        clim_doy[key] = np.mean(values)

    for key, values in clim_month.items():
        clim_month[key] = np.mean(values)

    return clim_datas, clim_doy, clim_month


def plot_single_year_ts(datas_dict, dates_dict, plot_year=1976, field=None):
    """
    Save single plot.
    """
    x = []
    y = []
    for time_key in sorted(datas_dict.keys()):
        (year, month, day) = time_key
        if plot_year != year: 
            continue

        dat = datas_dict[time_key]
        dt = dates_dict[time_key]

        dcy = decimal_year(dt, year,month,day)
        x.append(dcy)
        y.append(dat)

    if not len(x):
#        print('no time in ', year)
        return


    fig = pyplot.figure()
    pyplot.plot(x, y)
    pyplot.title(plot_year)
    fn = folder('images/single_year/'+field)+str(plot_year)+'.png'
    print('Saving', fn)
    pyplot.savefig(fn)
    pyplot.close()

def smooth_axes(x, y, interp='nearest'):
    """
    Takes low time resolution data and smooths it out. 
    """
    func1 = interpolate.interp1d(x, y, kind=interp)
    x = np.array(x)
    new_x = np.arange(x.min(), x.max(), 1000.)
    return new_x, func1(new_x)


def fill_between(
        times,
        y_values,
        clim_y,
        plot_type='anom',
        field='',
        window=2.5,
        alpha=1.,
        fig=None,
        ax=None, 
        ):
    """
    Do the fill between stuff?
    """
    pyplot.sca(ax)

    anom = np.array(y_values) - np.array(clim_y)
    zeros = np.array([0. for i in times])

    if plot_type=='anom':
        y_mins_down = anom
        y_maxs_down = zeros
        y_mins_up = zeros
        y_maxs_up = anom
        black_line = zeros
        purple_line = anom  
        #if anom_mins.get(field, False):
        ax.set_ylim([anom_mins[field], anom_maxs[field]])        

    elif plot_type =='bar_anom':
        black_line = zeros
        purple_line = anom

        norm_down = matplotlib.colors.Normalize(vmin=anom_mins[field], vmax=0.)
        norm_up   = matplotlib.colors.Normalize(vmin=0., vmax=anom_maxs[field])

        bar_colours = []
        for value in anom:
            if value< 0:
                bar_colours.append(cm.cividis(norm_down(value)))
            if value > 0:
                bar_colours.append(cm.inferno_r(norm_up(value)))

        #if anom_mins.get(field, False):
        ax.set_ylim([anom_mins[field], anom_maxs[field]])     
           
    elif plot_type=='bar_ts':

        black_line = clim_y
        purple_line = y_values

        norm_down = matplotlib.colors.Normalize(vmin=anom_mins[field], vmax=0.)
        norm_up   = matplotlib.colors.Normalize(vmin=0., vmax=anom_maxs[field])

        bar_colours = []
        for value in anom:
            if value < 0:
                bar_colours.append(cm.cividis(norm_down(value)))
            if value > 0:
                bar_colours.append(cm.autumn_r(norm_up(value)))
        top_line, bottom_line = [], []

        for clim, value in zip(clim_y, y_values):
            if clim >= value:
                top_line.append(clim - value)
                bottom_line.append(value)
            else:
                top_line.append(value - clim)
                bottom_line.append(clim)              

        #if vmins.get(field, False):
        ax.set_ylim([vmins[field], vmaxs[field]])

    elif plot_type=='just_ts':
        black_line = clim_y
        purple_line = y_values
    elif plot_type == 'just_anom':
        black_line = zeros
        purple_line = anom
    else:
        y_mins_down = y_values
        y_maxs_down = clim_y
        y_mins_up = clim_y
        y_maxs_up = y_values  
        black_line = clim_y
        purple_line = y_values  
        #if vmins.get(field, False):
        ax.set_ylim([vmins[field], vmaxs[field]])

    ax.plot(times, purple_line, 'white', lw=1.7, zorder=10, alpha=alpha)

    ax.set_xlim([times[-1] - window, times[-1]])

    if plot_type=='just_ts':    
        ax.plot(times, black_line, 'w', ls=':', lw=1., zorder=5, alpha=alpha)
        ax.set_ylim([vmins[field], vmaxs[field]])                    
        return fig, ax  
          
    if plot_type=='just_anom':    
        ax.plot(times, black_line, 'w', ls=':', lw=1., zorder=5, alpha=alpha)
        ax.set_ylim([anom_mins[field], anom_maxs[field]])     
        return fig, ax         
  
    if plot_type == 'bar_anom':
        ax.bar(
                times,
                anom,
                color= bar_colours, 
                width=1/365.25,
                alpha=alpha
           )
        #if anom_mins.get(field, False):
        ax.set_ylim([anom_mins[field], anom_maxs[field]])          
        return fig, ax
               
    if plot_type == 'bar_ts':
        ax.bar(
                times,
                top_line,
                bottom=bottom_line, 
                color=bar_colours, #'red', #np.ma.masked_where(downwhere,down_colours).compressed(),
                width=1/365.25,
                alpha=alpha,
           )  
        #if vmins.get(field, False):
        ax.set_ylim([vmins[field], vmaxs[field]])                    
        return fig, ax
    assert 0 

    # number_of_colours = 5.
    # colours =['purple', 'red', 'orange', 'yellow', 'white']        
    # for i in np.arange(1, number_of_colours+1):

    #     down_cut_1 = anom_mins[field]*i/number_of_colours
    #     down_cut_m1 = anom_mins[field]*(i-1)/number_of_colours

    #     up_cut_1 = anom_maxs[field]*i/number_of_colours
    #     up_cut_m1 = anom_maxs[field]*(i-1)/number_of_colours

    #     downwhere = np.ma.masked_outside(anom, down_cut_1, down_cut_m1).mask
    #     upwhere = np.ma.masked_outside(anom, up_cut_m1, up_cut_1).mask

    #     down_colour = cm.winter((number_of_colours-float(i))/number_of_colours) #, bytes=True)
    #     up_colour = cm.hot(float(i)/number_of_colours)
    #     down_colour = up_colour = colours[int(i-1)]

    #     print(field, i, 'DOWN:', ('between', down_cut_1, down_cut_m1), ': of one:', #norm_down(down_cut_1), 
    #           (up_cut_m1, up_cut_1), ':', #norm_up(up_cut_1), 
    #           [np.mean(clim_y), np.mean(y_values), np.mean(anom)], down_colour, len(times) - np.sum(downwhere)
    #           )
    #     print(field, i, 'UP:', ('between:', up_cut_m1, up_cut_1), ': of one:', #norm_up(up_cut_1), 
    #           [np.mean(clim_y), np.mean(y_values), np.mean(anom)], up_colour, len(times) - np.sum(upwhere)
    #           )

    #     if plot_type == 'bar_anom':
    #          if len(times) - np.sum(downwhere):
    #             pyplot.bar(
    #                 times,
    #                 np.ma.masked_where(downwhere, y_maxs_down),
    #                 bottom =  np.ma.masked_where(downwhere, y_mins_down),
    #                 color=down_colour,
    #                 width=1/365.25,
    #             )
    #          if len(times) - np.sum(upwhere):
    #             pyplot.bar(
    #                 times,
    #                 np.ma.masked_where(upwhere, y_maxs_up),
    #                 bottom =  np.ma.masked_where(downwhere, y_mins_up),
    #                 color=up_colour,
    #                 width=1/365.25,
    #             )
    #     else:
    #         if len(times) - np.sum(downwhere):
    #             pyplot.fill_between(
    #                 times,
    #                 y_mins_down,
    #                 y_maxs_down,
    #                 color=down_colour,
    #                 where=downwhere,
    #                 #facecolor=down_colour,
    #                 interpolate=True,
    #                 )
    #         if len(times) - np.sum(upwhere):   
    #             pyplot.fill_between(
    #                 times,
    #                 y_mins_up,
    #                 y_maxs_up,
    #                 color=up_colour,
    #                 where = upwhere,
    #                 #facecolor=up_colour, 
    #                 interpolate=True,
    #                 )
    # return fig, ax

    # else:
    #     if np.sum(downwhere):
    #         norm = matplotlib.colors.Normalize(vmin=vmins[field], vmax=vmaxs[field])
    #         rgba_colors = [cm.cool(norm(y_val), 10 ) for y_val in y]# if downwhere]

    #         pyplot.fill_between(x,
    #                 y,
    #                 clim_y,
    #                 color=rgba_colors,
    #                 where = downwhere,
    #                 #facecolor='dodgerblue',
    #                 )
    #     if np.sum(upwhere):
    #         norm = matplotlib.colors.Normalize(vmin=vmins[field], vmax=vmaxs[field])
    #         rgba_colors = [cm.hot(norm(y_val), 10 ) for y_val in y ]# if upwhere]            
    #         pyplot.fill_between(x,
    #                 clim_y,
    #                 y,
    #                 color=rgba_colors,
    #                 where = upwhere,
    #                 #facecolor='#e2062c', # candy apple red
    #                 )
    #     pyplot.plot(x, y, 'purple', lw=0.7)
    #     pyplot.plot(clim_x, clim_y, 'k', lw=0.7)
    #     #pyplot.plot(x, zeros, 'w', lw=0.5)
    #     #pyplot.plot(x, zeros+1, 'w', lw=0.5)
    #     #pyplot.plot(x, zeros-1, 'w', lw=0.5)

    #     pyplot.xlim([x[-1] - window, x[-1]])
    #     #pyplot.ylim([-1.5, 4.0])

    #     if vmins.get(field, False):
    #         pyplot.ylim([vmins[field], vmaxs[field]])

def highlight_endpoint(fig, ax, decimal_t, times, data, colours=()):
    """
    Emphasise the current musical note.
    """
    pyplot.sca(ax)
    loc = np.argmin(np.abs(np.array(times) - decimal_t))
    time = times[loc]
    dat = data[loc]
    col = np.array(colours[loc][:-1]) # strip alpha value
    ax.scatter(time, dat, s=150, marker='o', c=col/5, alpha=0.5, zorder=55 , edgecolor=(0., 0., 0., 0.,))

    #print('highlight_endpoint', time, dat, col)
    for i in np.arange(1, 11, 1):
        ax.scatter(time-(float(i)/20.), dat, s=150/float(i), marker='o', c=col/(5+float(i)), alpha=0.4/float(i), zorder=55 ,  edgecolor=(0., 0., 0., 0.,))
        return fig, ax



def plot_musical_notes(
        datas_dict, 
        dates_dict, 
        time_key=(2000, 1, 1), 
        decimal_t=0,
        recalc_fn = '',
        field='',
        window=2.5, # years
        active_time_range = [],
        #linecolor='red',
        plot_type='bar_ts',
        monthly=False,
        fig=None,
        ax=None
    ):

    pyplot.sca(ax)

    times = []
    qdata = []
    alphas = []
    sizes = []
    #markers = []
    colours = []   

    with open(recalc_fn) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            time = float(row['dtimes'])
            if time-0.25 > decimal_t:
                continue
            if time < decimal_t - window:
                continue
            if int(time) not in active_time_range:
                continue
            alpha = np.clip((time- decimal_t)*1.5 + 1.7, 0., 1.)
            if alpha ==0:
                continue
            
            times.append(time)
            qdata.append(float(row[' recalc']))
            alphas.append(alpha)
            if np.abs(time-decimal_t) < 0.15:
                sizes.append(10)
            else:
                sizes.append(np.clip((time- decimal_t)*1.7 + 1.7, 0.5, 3.))

            # ['thetao_con', 'O3_pH', 'Ptot_c_result', 'Ztot_c_result']    
            if field == 'thetao_con':
                colours.append((alpha, 0., 0., alpha)) # red to black
            if field == 'O3_pH':
                colours.append((0., 0., alpha,  alpha)) # Blue to black    
            if field == 'Ptot_c_result':
                colours.append((0.,  alpha, 0.,  alpha)) # Green to black                            
            if field == 'Ztot_c_result':
                colours.append((alpha/2, 0.,  alpha, alpha)) # purple to black   

    if (monthly and len(times) > 2) or (not monthly and 2< len(times)<100):

        new_times = np.arange(np.min(times), np.max(times), np.abs((np.max(times)- np.min(times))/200.))
        new_colours = []
        new_data = interpolate.interp1d(times, qdata, kind='nearest')(new_times)  
        new_sizes = interpolate.interp1d(times, sizes, kind='linear')(new_times) 
        new_alphas = interpolate.interp1d(times, alphas, kind='linear')(new_times) 
        if field == 'thetao_con':
            new_colours = [(a, 0., 0., a) for a in new_alphas] # red to black
        if field == 'O3_pH':
            new_colours = [(0, 0., a, a) for a in new_alphas] # Blue to black  
        if field == 'Ptot_c_result':
            new_colours = [(0, 0.8*a, 0., a) for a in new_alphas] # Green to black 
        if field == 'Ztot_c_result':
            new_colours = [(a/2., 0., a, a) for a in new_alphas]

        ax.scatter(new_times, new_data, s=new_sizes, marker='s', c=new_colours, zorder=50,)

        #fig, ax = highlight_endpoint(fig, ax, decimal_t, new_times, new_data, colours=new_colours)

        #col = new_colours[-1][:-1]

        #print(, new_sizes[-1], col)
        #assert 0        
        #ax.scatter(new_times[-1], new_data[-1], s=new_sizes[-1]*200, marker='o', c=col, alpha=0.5, zorder=55,)
        #ax.scatter(new_times[-1], new_data[-1], s=new_sizes[-1]*400, marker='o', c=col, alpha=0.25, zorder=55,)
       
    else:
        ax.scatter(times, qdata, s=sizes, marker='s', c=colours, zorder=50,)

       #fig, ax = highlight_endpoint(fig, ax, decimal_t, times, qdata, colours=colours)

        # col = colours[-1][:-1]
        # #print(times[-1], qdata[-1], sizes[-1], col)
        # #assert 0
        # ax.scatter(times[-1], qdata[-1], s=sizes[-1]*200, marker='o', c=col, alpha=0.5, zorder=55,)
        # ax.scatter(times[-1], qdata[-1], s=sizes[-1]*400, marker='o', c=col, alpha=0.25, zorder=55,)

    if plot_type == ['bar_anom', 'just_anom']:
        ax.set_ylim([anom_mins[field], anom_maxs[field]])     
        pyplot.ylim([anom_mins[field], anom_maxs[field]])
    if plot_type in ['bar_ts', 'just_ts']:
        ax.set_ylim([vmins[field], vmaxs[field]])
        pyplot.ylim([vmins[field], vmaxs[field]])
    return fig, ax


def plot_past_year_just_anom_ts(
        datas_dict,
        dates_dict, 
        target_time_key=(2000,1,1), 
        window=2.5, # years
        clim_range=[1976,1985],
        alpha=1.,
        field='thetao_con',
        active_time_range = [],
        clim_stuff=(),
        plot_type='anom',
        fig=None,
        ax=None):
    """
    Create single plot.
    """
    times = []
    y_values = []
    clim_times = []
    clim_y= []   
    target_string = '-'.join([str(t) for t in target_time_key])
    target_dt = dates_dict[target_time_key]
    target_decimal = decimal_year(target_dt, target_time_key[0], target_time_key[1], target_time_key[2])
    (clim_datas, clim_doy, clim_month) = clim_stuff

    for time_key in sorted(datas_dict.keys()):
        (year, month, day) = time_key
        dt = dates_dict[time_key]
        dcy = decimal_year(dt, time_key[0], time_key[1], time_key[2])
        if dcy > target_decimal:
            continue
        if dcy < target_decimal - window:
            continue
        if year not in active_time_range:
            continue
        dat = datas_dict[time_key]
        times.append(dcy)
        y_values.append(dat)
        clim_times.append(dcy)
        clim_y.append(clim_datas[(month, day)])

    if not len(times):
        print('No data!', field)
        print(active_time_range, ': active_time_range')
        assert 0
   
    if fig is None:
        fig = pyplot.figure()
        returnfig=False
    else:
        pyplot.sca(ax)
        returnfig=True
   
    fig, ax = fill_between(
        times,
        y_values,
        clim_y,
        plot_type=plot_type,
        field=field,
        window=window,
        alpha=alpha,
        fig=fig,
        ax=ax, 
        )
    
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    axis_color='white'

    fig, ax = set_axes_alpha(fig, ax, alpha=alpha, axes_color=axis_color)

    if returnfig:
        return fig, ax

    fn = folder('images/Just_anom_single_year/')+str(target_string)+'.png'
    print('Saving', fn)
    pyplot.savefig(fn)
    pyplot.close()



def plot_single_year_just_anom_ts(datas_dict, dates_dict, plot_year=1976, clim_range=[1976,1985],
                                  field=None,
        fig=None, ax=None):
    """
    Save single plot.
    """
    x = []
    y = []
    clim_y = []
    clim_datas, clim_doy, clim_month = calculate_clim(datas_dict, dates_dict, clim_range=clim_range)

    for time_key in sorted(datas_dict.keys()):
        (year, month, day) = time_key
        if plot_year != year:
            continue

        dat = datas_dict[time_key]
        dt = dates_dict[time_key]

        dcy = decimal_year(dt, year,month,day)
        x.append(dcy)
        y.append(dat)
        clim_y.append(clim_datas[( month, day)])

    if not len(x):
        return

    #lim_y = [clim_doy[i+1] for i, time in enumerate(x)]

    anom = np.array(y) - np.array(clim_y)
    zeros = np.array([0. for i in anom])

    if fig is None:
        fig = pyplot.figure()
        returnfig=False
    else:
        pyplot.sca(ax)
        returnfig=True 

    downwhere = np.ma.masked_less(anom, 0.).mask
    if np.sum(downwhere):
        pyplot.fill_between(x,
                zeros,
                anom,
                where = downwhere,
                facecolor='dodgerblue',
                )
    upwhere = np.ma.masked_greater(anom, 0.).mask
    if np.sum(upwhere):
        pyplot.fill_between(x,
                anom,
                zeros,
                where = upwhere,
                facecolor='#e2062c',
                )
    pyplot.plot(x, np.array(y) - np.array(clim_y))
    pyplot.title(plot_year)
    pyplot.plot(x, zeros, 'w', lw=0.5)
    pyplot.plot(x, zeros+1, 'w', lw=0.5)
    pyplot.plot(x, zeros-1, 'w', lw=0.5)

    pyplot.xlim([x[-1]-window, x[-1]])
    if anom_mins.get(field, False):
        pyplot.ylim([anom_mins[field], anom_maxs[field]])

    if returnfig:
        return fig, ax

    fn = folder('images/Just_anom_single_year/'+field)+str(plot_year)+'.png'
    print('Saving', fn)
    pyplot.savefig(fn)
    pyplot.close()


def set_axes_alpha(fig, ax, alpha=1.,axes_color='white'):
    """
    Tool that sets an entire ax to a transparent value. 
    Does not change the plotted values, which need to be done elsewhere.
    Unclear whether it works on cartopy.
    """
    white_alpha = (1,1,1,alpha)
    black_alpha = (0,0,0,alpha)

    #ax.set_facecolor((1., 1., 1., 0.35)) # transparent black
    ax.set_facecolor((0., 0., 0., 0.1)) # transparent black

    if axes_color == 'white':
        color_alpha = white_alpha
    if axes_color == 'black':
        color_alpha = (0,0,0,alpha)

    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_alpha(alpha)
        axis.set_tick_params(color=color_alpha, labelcolor=color_alpha)
    for spine_loc in ['top', 'bottom', 'left', 'right']:
        ax.spines[spine_loc].set_color(color_alpha)

    return fig, ax


def add_cbar(fig, ax=None, field='thetao_con', globe_type='ts', cbar_ax_loc = [0.925, 0.2, 0.02, 0.6]):
    """
    Add a color bar on the side.
    """
    if ax == None:
        ax = fig.add_axes([1.85, 1.2, 0.015, 0.6]) # (left, bottom, width, height)

    pyplot.sca(ax)
    x = np.array([[0., 1.], [0., 1.]])

    if globe_type=='ts':
        cmap = cm.get_cmap(cmaps.get(field, 'viridis'), cm_bins[field])    # 11 discrete colors
        vmin=cbar_vmin.get(field, 0.)
        vmax=cbar_vmax.get(field, 1.)
        label = 'Sea Surface Temperature, '+r'$\degree$'+'C'
                    
    elif globe_type=='anomaly':
        anom_field = field+'_anomaly'
        cmap = cm.get_cmap(cmaps.get(anom_field, 'viridis'), cm_bins[anom_field])    # 11 discrete colors        
        vmin=cbar_vmin.get(anom_field, 0.)
        vmax=cbar_vmax.get(anom_field, 1.)
        label = 'Sea Surface Temperature Anomaly, '+r'$\degree$'+'C'
                    
    img = pyplot.pcolormesh(x,x,x,
                            cmap=cmap,
                            vmin=vmin,
                            vmax=vmax)
    img.set_visible(False)
    ax.set_visible(False)
    
    #ax_cbar = fig.add_axes([0.85, 0.2, 0.05, 0.6]) # (left, bottom, width, height)
    axc = fig.add_axes(cbar_ax_loc, zorder=1000) # (left, bottom, width, height)
    # axc.set_facecolor((0., 0., 0., 0.35)) # transparent black

    cbar = pyplot.colorbar(cax=axc, orientation="vertical", extend='both',  )

    # font size scales with height of cbar    
    font_size = np.clip(6/.4*cbar_ax_loc[3] +5., 8., 14.)
    cbar.set_label(label=label, color='white', size=font_size )#, weight='bold')
    cbar.ax.tick_params(color='white', labelcolor='white')

    return ax


def add_mpa(
        ax, 
        decimal_t,
        linewidth=2.1,   
        max_heatwave=0.75, # decimal years
        heatwaves={}, # data from recalc (ie the notes.)
        axes_size=1., 
        draw_study_region=False, 
        discrete_colours=False):
    """
    Add the MPA circle and study region square.
    """
    # whole MPA
    proj = ccrs.PlateCarree()
    
    ax.add_patch(mpatches.Circle(xy=[central_longitude, central_latitude, ], linewidth=1.5,
            radius=mpa_radius, ec='white', fc=(0., 0., 0., 0.), transform=proj, zorder=31))

    if max_heatwave>2.: 
        print('max_heatwave is in units of decimal years. Typically use a between 0.5 adn 1.5.')
        assert 0

    reduced_heatwaves = {}
    times = [time for time in sorted(heatwaves.keys())]
    for t, time in enumerate(times):
        if time > decimal_t: 
            # no heatwaves in the future
            continue
        if time < decimal_t - max_heatwave:
            # no heatwaves in the distant past
            continue
        hwl_value = heatwaves[time]

        if hwl_value <= 1.: 
            #no heatwave for 1 degree
            continue
        if t > 0:
            hwl_value_m1 = heatwaves[times[t-1]]
            if hwl_value_m1 == hwl_value:
                # Skip doubles - only show waves when note changes.
                continue
        reduced_heatwaves[time] = hwl_value

    ring_speed = 10./1. # 5 degrees per year
    alpha_speed = -1.5/1. # loose x alphas per year 
    thickness_speed = -2.5/1. # looose 2 thicknesses per year

    min_alpha = 0.05
    min_lw = 0.25 
    lw_factor = 0.6
    min_rad = 0.

    for time, hwl_value in reduced_heatwaves.items():
        if hwl_value <= 1.: 
            #no heatwave for 1 degree
            continue

        if discrete_colours:
            hwl_value = int(hwl_value)
        
        tdiff = np.abs(decimal_t - time)

        circle_alpha = np.max([1  + alpha_speed * tdiff, min_alpha])

        rgba_color = temperature_anom_cm(temperature_anom_norm(hwl_value), 5 )
        mpa_circle_colour = (rgba_color[0], rgba_color[1], rgba_color[2], circle_alpha)

        lw = np.max([linewidth +  thickness_speed * tdiff, min_lw]) 
        lw = lw_factor * lw * axes_size * np.max([hwl_value**1.25, 1.])
        rad = mpa_radius + 0.01 + np.max([tdiff * ring_speed, min_rad])**0.95

        ax.add_patch(
            mpatches.Circle(xy=[central_longitude, central_latitude, ], 
            linewidth=lw,
            radius=rad,
            ec=mpa_circle_colour, 
            fc=(0., 0., 0., 0.), 
            transform=proj, 
            zorder=30))        

    if not draw_study_region:
        return ax
    ax.plot(
        mpa_lon_corners,
        mpa_lat_corners,
        color=study_region_colour,
        linewidth=linewidth,
        transform=proj,
        zorder=30,
        )
    return ax



def plot_globe(ax, nc=None, t=None, quick=True, field = 'thetao_con',  globe_type='ts', clim_dat=None):
    """
    Add the globe to the axes.
    """
    pyplot.sca(ax)
    binning=1
    lats = nc.variables['nav_lat'][::binning]
    lons = nc.variables['nav_lon'][::binning]
    if quick:
        ax.add_feature(cfeature.OCEAN, zorder=0)
        ax.add_feature(cfeature.LAND, zorder=0, edgecolor='black')
    elif globe_type=='ts':

        data = nc.variables[field][t, 0, ::binning, ::binning]
        cmap = cm.get_cmap(cmaps.get(field, 'viridis'),  cm_bins[field])    # 11 discrete colors

        pyplot.pcolormesh(
                    lons,
                    lats,
                    data,
                    #transform=proj,
                    transform=ccrs.PlateCarree(),
                    cmap=cmap,
                    vmin=cbar_vmin.get(field, 0.),
                    vmax=cbar_vmax.get(field, 1.),
                    zorder=8, 
                    )        
        ax.coastlines()
        ax.add_feature(cfeature.LAND, edgecolor='black', facecolor=land_color, linewidth=0.5, zorder=8)                    
    elif globe_type=='anomaly':

        data = nc.variables[field][t, 0, ::binning, ::binning]
        #clim = clim_nc.variables[field][t, ::binning, ::binning]
        anom_field = field+'_anomaly'
        cmap = cm.get_cmap(cmaps.get(anom_field, 'viridis'), cm_bins[anom_field])    # 11 discrete colors        
        pyplot.pcolormesh(
                    lons,
                    lats,
                    data - clim_dat,
                    #transform=proj,
                    transform=ccrs.PlateCarree(),
                    cmap=cmap,
                    vmin=cbar_vmin.get(anom_field, 0.),
                    vmax=cbar_vmax.get(anom_field, 1.),
                    zorder=6, 
                    )        
        ax.coastlines()
        ax.add_feature(cfeature.LAND, edgecolor='black', facecolor=land_color, linewidth=0.5, zorder=7)

    ax.set_global()
    xlocs = [-150., -120., -90., -60., -30., 0., 30., 60., 90., 120., 150., 180., ]
    ylocs = [-60, -30, 0., 30.,  60.]
    #ax.set_xticks(, crs=ccrs.PlateCarree())
    #ax.set_xticklabels([]) #120., 140., 160., 180., -160., -140., -120.], color='red', weight='bold')
    #ax.set_yticks([-60, -30, 0., 30.,  60.], crs=ccrs.PlateCarree())    
    #ax.set_yticklabels([])
    ax.gridlines(xlocs=xlocs, ylocs=ylocs, draw_labels=False)
    #ax.gridlines()
    return ax


def calc_midoint(decimal_time, path, interp=None, window=0.25):
    """
    Calculates a panning value, assuming even distribution between two points.
    """
    times = np.array(sorted(path.keys()))
    coords = np.array([path[t] for t in times])
    if interp == None:
        if decimal_time <= times.min():
            return path[times.min()]
        if decimal_time >= times.max():
            return path[times.max()]
        index = bisect(times, decimal_time)

        t0 = times[index -1]
        t1 = times[index]
        y0 = path[t0]
        y1 = path[t1]
        slope = (y1-y0)/(t1-t0)
        intersect = y1 - slope*t1
        return slope*decimal_time + intersect
    if interp == 'smooth':
        x = np.linspace(decimal_time-window, decimal_time+window, 100,)
        func1 = interpolate.interp1d(times, coords, kind='linear')
        y = func1(x)
        return np.mean(y)

    func1 = interpolate.interp1d(times, coords, kind=interp)
    return func1(decimal_time)


def calc_heat_wave(datas_dict, 
                   dates_dict, 
                   date_key=(),
                   clim_range=(),
                   clim_stuff=(),
                   threshold = 2.5,
                   max_heatwave=30,
                   ):
    """
    Calculate length of heatwave
    returns a list of anomalies for each day before this.
    """
    target_dt = dates_dict[date_key]
    target_decimal = decimal_year(target_dt, date_key[0], date_key[1], date_key[2])

    dates_list = []
    for dkey, date in dates_dict.items():
        dcy = decimal_year(date, dkey[0], dkey[1], dkey[2])
        if dcy> target_decimal: continue
        if np.abs(dcy-target_decimal) > max_heatwave/365.25:
            continue
        dates_list.append(dkey)
    
    dates_list.sort(reverse=True)
    mhw = {}
    clim = clim_stuff[0]
    for dkey in dates_list:
        diff = datas_dict[dkey] - clim[(dkey[1], dkey[2])]
        mhw[dkey] = diff
    return mhw


def calc_heat_wave_csv(
        datas_dict, 
        dates_dict, 
        clim_range=(),
        clim_stuff=(),
    ):
    """
    Calculate time steps with of heatwaves
    and save the CSV.
    """
    heatwaves_csv = folder('csv/heatweaves/')+'CNRM_SSP370_heatwaves.cvs'
    if os.path.exists(heatwaves_csv):
        print('Heatwaves csv', heatwaves_csv)
        return 

    dates_list = []
    dcys = {}
    for dkey, date in dates_dict.items():
        dcy = decimal_year(date, dkey[0], dkey[1], dkey[2])
        dates_list.append(dkey)
        dcys[dkey] = dcy
    
    dates_list.sort(reverse=True)
    #mhw = []
    clim = clim_stuff[0]
    txt = '# times,temperature_anom\n'
    for dkey in sorted(dates_list):
        diff = datas_dict[dkey] - clim[(dkey[1], dkey[2])]
        txt = ''.join([txt, str(dcys[dkey]), ',', str(int(diff)),'\n'])
    file = open(heatwaves_csv, 'w')
    file.write(txt)
    file.close()
    #return mhw

    # (clim_datas, clim_doy, clim_month) = clim_stuff

    # for time_key in sorted(datas_dict.keys()):
    #     (year, month, day) = time_key
    #     dt = dates_dict[time_key]
    #     dcy = decimal_year(dt, time_key[0], time_key[1], time_key[2])
    #     if dcy > target_decimal: continue
    #     if dcy < target_decimal - window: continue

    #     dat = datas_dict[time_key]
    #     x.append(dcy)
    #     y.append(dat)
    #     clim_x.append(dcy)
    #     clim_y.append(clim_datas[(month, day)])

def add_blank_legend_entry(fig, ax):
    pyplot.plot([], [], marker='s', color=(0, 0, 0, 0), label='                ')

def plot_legend(
    active_time_ranges={},
    time_key = (),
    decimal_time = 0.,
    clim_range=(),
    fig=None,
    ax=None,):
    """
    Add legend to plot.
    """
    year = time_key[0]
    pyplot.plot([], [], lw=2, c='white', label='Model Mean')
    #pyplot.plot([], [], lw=1.5, ls=':', c='white', label=''.join([str(clim_range[0]), '-', str(clim_range[1]), ' Mean'] ))
    #add_blank_legend_entry(fig, ax)
    pyplot.plot([], [], lw=1.5, ls=':', c='white', label='Climatology')
    pyplot.plot([], [], lw=1.5, ls=':', c=(0,0,0,0), label=''.join(['    (', str(clim_range[0]), '-', str(clim_range[1]), ' Mean)',] ))

    #add_blank_legend_entry(fig, ax)    
    # add_blank_legend_entry(fig, ax)
    pyplot.scatter([], [], marker='o', color='white', facecolor=(0, 0, 0, 0), s = 100, label = 'Ascension Island MPA') #

    if year >= np.min(active_time_ranges['thetao_con']):
        pyplot.scatter([], [], marker='s', color=(1., 0.,  0., 1.), label = 'Piano') # red
    if year >= np.min(active_time_ranges['O3_pH']):
        pyplot.scatter([], [], marker='s', color=(0., 0.,  1., 1.), label = 'Synth Top') # blue
    else:
        add_blank_legend_entry(fig, ax)
    if year >= np.min(active_time_ranges['Ptot_c_result']):
        pyplot.scatter([], [], marker='s', color=(0., 0.8, 0., 1.), label = 'Synth Bass') # green
    else:
        add_blank_legend_entry(fig, ax)
    if year >= np.min(active_time_ranges['Ztot_c_result']):
        pyplot.scatter([], [], marker='s', color=(0.5, 0., 1., 1.), label = 'Synth Mid') # purple
    else:
        add_blank_legend_entry(fig, ax)
    #
    first_deg_waves = {
        1: 1997.049,
        2: 2011.920,
        3: 2052.185,
    }
    for deg_thresh in [1, 2, 3]:
        if decimal_time >= first_deg_waves[deg_thresh]:
            rgba_color = temperature_anom_cm(temperature_anom_norm(deg_thresh), 5 )#, bytes=True) 
            cycles_per_year= 1
            time_in_rads = math.radians(((decimal_time - year - first_deg_waves[deg_thresh])%(1./cycles_per_year))*90.)
            size = (math.sin(time_in_rads)+1) * 100.
            alpha = math.cos(time_in_rads)
            rgba_color = (rgba_color[0], rgba_color[1], rgba_color[2], alpha)
            #size = (decimal_time - year- (first_1_deg_wave - int(first_1_deg_wave)) +2. )* 20 # pulse from 10-15.
            pyplot.scatter([], [], 
                    marker='o', 
                    color=rgba_color, 
                    facecolor=(0, 0, 0, 0),
                    s = size,
                    #alpha = alpha,
                    label = str(int(deg_thresh))+' '+r'$\degree$'+ 'C heatwave')
        else:
            add_blank_legend_entry(fig, ax)

    fig, ax = set_axes_alpha(fig, ax, alpha=1.) #, axes_color=axis_color)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_axis_off()


    leg = ax.legend(
        framealpha=0.15, 
        ncol=3, 
        loc = 'lower left', 
        mode = "expand",
        labelcolor='white',
        facecolor='black',
    ) #handles=handles)
    frame = leg.get_frame()
    frame.set_edgecolor('black')

    return fig, ax


def get_image_path(date_key, dt):
    """
    Get a predeinfed path name
    """
    year=str(date_key[0])
    fn = folder('images/'+daily_count+'_'+video_format+'/'+year)+'daily'
    date_string = '-'.join([mn_str(tt) for tt in date_key])
    fn = ''.join([fn, '_', date_string])+ '.png'
    return fn


global_heatwaves = {}
def load_heatwaves():
    global global_heatwaves
    if len(global_heatwaves.keys()):
        return global_heatwaves
    
    recalc_fn = 'output/MHW/recalc/MarineHeatWaves_f_cnrm_temp_anom.csv'
    heatwaves={}
    with open(recalc_fn) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            time = float(row['dtimes'])
            data = float(row[' recalc'])
            heatwaves[time] = data
    global_heatwaves = heatwaves
    return heatwaves

pan = {}
pan_years = {}
def make_global_panning_points():
    """
    Make the global panning point here, so it's not created every time step.
    """
    global pan
    global pan_years

    # panning points:
    both_sizes = 0.35
    bottom = 0.25
    mid = 0.4
    top = 0.63
    left = 0.37
    centre = 0.5
    right = 0.624 
    margin = 0.025
    pan={# name:           [  X,     Y,    L,    B,    W,    H ]
        'tiny_dot':        [ -5.,   -85.,  0.6,  0.4,  0.05,  0.05, ], 

        'tiny_dot_BR':     [ -25.,   15.,  right-margin,  bottom +margin,  both_sizes*1.,  both_sizes*1., ], 
        'tiny_dot_BC':     [ -30.,   -5.,  centre,  bottom,  both_sizes,  both_sizes, ], 
        'tiny_dot_MR':     [  5.,   15.,  right,  mid,  both_sizes,  both_sizes, ], 
        'tiny_dot_BL':     [ -25.,  -45.,  left+margin,  bottom+margin,  both_sizes*1.,  both_sizes*1., ], 
        'tiny_dot_ML':     [ -5.,  -45.,  left,  mid,  both_sizes,  both_sizes, ], 
        'tiny_dot_TL':     [ 15., -45.,  left+margin,  top-margin,  both_sizes*1.,  both_sizes*1., ], 
        'tiny_dot_TC':     [ 20.,  -5.,   centre,  top,  both_sizes,  both_sizes, ], 
        'tiny_dot_TR':     [ 15.,  15.,  right-margin,  top-margin,  both_sizes*1.,  both_sizes*1. , ], 
    
        'tiny_dot_r':      [  35.,   65.,  0.75,  0.65,  0.05,  0.05, ],
        'tiny_dot_2':      [  35.,   0.,  0.75,  0.65,  0.05,  0.05, ], 

        'dot_r':           [ -10.,  -15.,  0.75,  0.65,  0.005,  0.005, ], 


        'vvfar_out':       [ -24.,  -20.,  0.4,  0.3,  0.6,  0.6 ],    
        'vvfar_out_w':     [ -24.,  -70.,  0.4,  0.3,  0.6,  0.6 ],         
        'vvfar_out_b':     [ -24.,  -20.,  0.35,  0.2,  0.7,  0.7 ],         

        'vfar_out':        [ -28.,  -28.,  0.3,  0.2,  0.7,  0.7 ],        
        'far_out':         [ -25.,  -25.,  0.3,  0.1,  0.8,  0.8 ],
        'far_out_w':       [  65.,    5.,  0.3,  0.1,  0.8,  0.8 ],
        'big':             [ -20.,  -20,   0.1,  -0.1, 1.2,  1.2 ],
        'big_b':           [ -21.,  -19,   0.05, -0.15, 1.3,  1.3 ],

        'big_left':        [ -28.,  -22,   0.1,  -0.3, 1.2,  1.2 ],
        'big_left_b':      [ -24.,  -18,   0.1,  -0.3, 1.3,  1.3 ],
        'big_right':       [ -5.,  -22,   0.1,  -0.1, 1.2,  1.2 ],

        'big_up':          [ -5., -27.,   0.1,  0., 1.0,  1.0 ],
        'big_down':        [ -9.,  25,    0.125,  -0.15, 1.5,  1.5 ],

        'vbig':            [ -10.,  -10,   0.0,  -0.3, 1.6,  1.6 ],
        'vbig_2':          [ 0.5, 2,   0.0,  -0.2, 1.5,  1.5 ],
        'vbig_b':          [ -7.,  7,  -0.1,  -0.4, 1.8,  1.8 ],

        'vbig_low':        [ -30,   -7.,   -0.15, -0.7, 1.6,  1.6 ],
        'vbig_low_b':      [ -28,   -11.,   -0.20, -0.65, 1.68, 1.68 ],

        'vbig_low1':       [ -20,   -17.,   -0.35, -0.7, 1.96,  1.96 ],
        'vbig_low1_b':     [ -23,   -5.,   -0.2, -0.7, 1.7,  1.7 ],
        'vbig_low1_c':     [ -18,   -25.,   -0.35, -0.7, 1.8,  1.8 ],


        'vbig_low2':       [ -30,   -27.,  -0.25, -0.6, 1.6,  1.6 ],
        'vbig_low2_swoop': [ -25,   -22.,  0.1, -0.3, 1.4,  1.4 ],       
        'vbig_low2_b':     [ -35,   -23.,  -0.125, -0.35, 1.4,  1.4 ],
#        'big_left':        [ -40.,  -22,   0.1,  -0.1, 1.2,  1.2 ],

        'vbig_ai':         [ central_latitude+2,   central_longitude-4,   -0.15, -0.3, 1.6,  1.6 ],
        'vvbig_ai':        [ central_latitude-1,   central_longitude+2,   -0.45, -0.45, 1.9,  1.9 ],
        'vvvbig_ai':       [ central_latitude-2,   central_longitude+2,  -0.3, -0.6, 2.2,  2.2 ],
        'vvvbig_ai_centre': [ central_latitude-6,   central_longitude-4,  -0.5, -0.6, 2.2,  2.2 ],

        'vvvbig_ai_R':     [ central_latitude-2,   central_longitude-5,  -0.5, -0.6, 2.1,  2.1 ],
        'vvvbig_ai_R_out':     [ central_latitude-2,   central_longitude-5,  -0.65, -0.75, 2.4,  2.4 ],
            
        'vvvbig_ai_Up':    [ central_latitude-5,   central_longitude+10,  -0.2, -0.6, 2.1,  2.1 ],
        # off screen
        'os_right':        [ -24.,  -20.,  1.00,  1.,  0.6,  0.6 ],         
        'os_below':        [ -24.,  -20.,  0.,  -0.6,  0.6,  0.6 ],         
        'os_above':        [  45.,  20.,   0.3,  1.,  0.45,  0.45 ], 
        'os_above_vbig':   [ 10.,  10,   0.0,  1.0, 1.6,  1.6 ],
    }
    pan_years = {
        1970.: 'far_out_w',
        1975.7: 'far_out_w',
        #1976.5: 'far_out',
        1978.:  'big', 
        1980.:  'vbig',
        1981.: 'vbig_2',
        1982.: 'vbig',
        1983.: 'vbig_2',
        1984:   'vbig_low',
        1985.:  'vbig_low_b',        
        1986.:  'vbig_low',
        1987.:  'vbig_low_b',   
        #1988.:  'vbig_low',
        1989.:  'vbig_low_b',   
        1990.:  'vbig_low1',
        1991.:  'vbig_low1_c',
        1992.:  'vbig_low1_b',        
        1993.:  'vbig_low1',
        1994.:  'vbig_low1_b',   
        1995.:  'vbig_low1',
#        1996.:  'vbig_low1_b',   
#        1997.:  'vbig_low1',
        1998.:  'vbig_low1_b',                           
        2000.:  'vbig_low2',
        #2000.5: 'vbig_low2',
        2002:   'vbig_low2_b',
        2003.:  'vbig_low2',
        2004.:  'big_left',
        2005.:   'big',
        2006:   'big_left',
        2007:    'big',
        2008.:  'big_right',
        2009.:  'big_left_b',
        2010.:   'big',        
        2011.: 'vbig',
        2012.:  'big_left',
        2012.5:  'vbig',
        2013.:  'big_left',
        2013.5:  'vbig',
        2014.:  'big_right',
        2014.5:  'big_down',
        2015.:  'big_right',
        2015.5:  'big_left',
        2016.:  'big_up',
        2016.5:  'big_down',        
        2017.:  'big_right',
        2017.5:  'big_up',
        2018.:  'big_left',
        2018.5:  'big_down',  
        #2012.5:  'big',
        2019.: 'vbig_ai',
        2019.75: 'vvvbig_ai_centre',
        #2000.25: 'far_out',
        2024.: 'vvfar_out', 
        2028.: 'vvfar_out_b',
        2029: 'far_out',
        2030.: 'vvfar_out_b',
        2031: 'vvfar_out',
        2032.: 'far_out',
        2040.: 'vbig_low',
        2042.: 'vbig_low1',
        2044.:  'vbig_low_b',  
        2045.: 'vbig_low2',
        2046.: 'vbig_low2_b',
        2047.: 'vbig_low2',
        2047.5: 'vbig_low2_swoop',
        2048.: 'tiny_dot', #,
        # These years have both globes.
        # 2049.: 'vbig_low1',
        # 2053.: 'big',
        # 2054.: 'big_b',
        # 2055.: 'vbig_low2_b',
        # 2056: 'tiny_dot',
        #2058.: 'tiny_dot', 
        #2058.5: 'big_b',
        2059.: 'big', 
        2059.5: 'big_b',        
        2060.: 'big_left',
        2060.5: 'vbig_b',
        2061.: 'vbig',
        2061.5: 'vbig_b',
        2062.: 'big_left',
        2062.5: 'vbig_ai',
        2063.: 'big_left',
        2063.5: 'vbig_ai',        
        2064: 'big_down',
        2064.5: 'vbig_ai',
        2065: 'vbig_low1',
        2065.5: 'vvbig_ai',
        2066: 'vbig',
        2066.5: 'vvvbig_ai_R_out',
        2067.: 'vvvbig_ai_R',
        2067.5: 'vvvbig_ai_R_out',
        2068.: 'vvvbig_ai_R',
        2068.5: 'vvvbig_ai_R_out',
        2069.0: 'vvvbig_ai_Up',
        2069.5: 'vvvbig_ai_R_out',
        2070.5: 'tiny_dot_2',
        2070.975: 'dot_r',
        }
make_global_panning_points()
    

def daily_plot(
        nc,
        t, 
        date_key, 
        datas_dict, 
        dates_dict, 
        clim_stuff=(), 
        clim_range=[1976,1985], 
        model='CNRM', 
        field='thetao_con', 
        clim_dat=None,
        plot_every_days=9,
        ):
    """
    The main daily plot.
    returns image path
    """
    dt = dates_dict[field][date_key]
    fn =  get_image_path(date_key, dt)
    if os.path.exists(fn):
        return fn
    if no_new_plots: 
        return None

    date_string = '-'.join([mn_str(tt) for tt in date_key])
    (year, month, day) = date_key
    dct = decimal_year(dt, year,month,day)
   
   
    # Create main figure.
    if video_format == '720p':
        image_size = [1280, 720]
        dpi = 100. 

    if video_format == 'HD':
        image_size = [1920., 1080.]
        dpi = 150.

    if video_format == 'QHD':
        image_size = [2560., 1440.]
        dpi = 200.

    if video_format == '4K':
        image_size = [3840., 2160.]
        dpi = 300.

    fig = pyplot.figure(facecolor='black')
    fig.set_size_inches(image_size[0]/dpi,image_size[1]/dpi)
    
    pan_years_ts = pan_years.copy()
    # 1984 transision

    pan_years_ts[1987.5] = 'vbig'
    pan_years_ts[1988] = 'os_right'
    pan_years_ts[1989.0] = 'os_right'
  
    pan_years_anom = pan_years.copy()
    pan_years_anom[1987.0] = 'os_below'
    pan_years_anom[1987.5] = 'os_below'
    pan_years_anom[1988] = 'vbig'

    # 1997 transition
    pan_years_ts[1995] = 'os_right'
    pan_years_ts[1996] = 'os_right'
    pan_years_ts[1997] = 'vbig_low1'
    pan_years_anom[1996] = 'vbig_low1_b'
    pan_years_anom[1997] = 'os_below'
    pan_years_anom[1998] = 'os_below'

    #2020 transition is sudden.

    # 2036 transition to ts.
    pan_years_ts[2035.] = 'os_above_vbig'
    pan_years_ts[2036.] = 'os_above_vbig'
    pan_years_ts[2036.5] = 'vbig'

    pan_years_anom[2035] = 'vbig'
    pan_years_anom[2036.5] = 'os_below'
    pan_years_anom[2037.5] = 'os_below'

    #2048-2056 - both spinning:
    pan_years_ts[2048  ] = 'tiny_dot_r'
    pan_years_ts[2048.5  ] = 'tiny_dot_TC'
    pan_years_ts[2049  ] = 'tiny_dot_TL'
    pan_years_ts[2049.5  ] = 'tiny_dot_TC'
    pan_years_ts[2050  ] = 'tiny_dot_TR'
    pan_years_ts[2051  ] = 'tiny_dot_BR'    
    pan_years_ts[2051.5  ] = 'tiny_dot_BC'    
    pan_years_ts[2052  ] = 'tiny_dot_BL'
    pan_years_ts[2052.5  ] = 'tiny_dot_BC'      
    pan_years_ts[2053  ] = 'tiny_dot_BR'
    pan_years_ts[2054  ] = 'tiny_dot_TR'
    pan_years_ts[2054.5  ] = 'tiny_dot_TC'
    pan_years_ts[2055  ] = 'tiny_dot_TL'
    pan_years_ts[2055.25  ] = 'tiny_dot_TC'
    pan_years_ts[2055.5  ] = 'tiny_dot_TL'
    pan_years_ts[2055.75  ] = 'tiny_dot_TC'
    pan_years_ts[2056  ] = 'tiny_dot_TR'
    pan_years_ts[2056.25  ] = 'tiny_dot_MR'
    pan_years_ts[2056.5  ] = 'tiny_dot_BR'
    pan_years_ts[2056.75  ] = 'tiny_dot_MR'
    pan_years_ts[2057  ] = 'tiny_dot_TR'    
    pan_years_ts[2057.25  ] = 'tiny_dot_MR'    
    pan_years_ts[2058] = 'tiny_dot'    
  
    pan_years_anom[2048  ] = 'tiny_dot'
    pan_years_anom[2048.5  ] = 'tiny_dot_BC'
    pan_years_anom[2049  ] = 'tiny_dot_BR'
    pan_years_anom[2049.5  ] = 'tiny_dot_BC'
    pan_years_anom[2050  ] = 'tiny_dot_BL'
    pan_years_anom[2051  ] = 'tiny_dot_TL'  
    pan_years_anom[2051.5  ] = 'tiny_dot_TC'    
    pan_years_anom[2052  ] = 'tiny_dot_TR' 
    pan_years_anom[2052.5  ] = 'tiny_dot_TC'    
    pan_years_anom[2053  ] = 'tiny_dot_TL'
    pan_years_anom[2054  ] = 'tiny_dot_BL'
    pan_years_anom[2054.5 ] = 'tiny_dot_BC'
    pan_years_anom[2055  ] = 'tiny_dot_BR' 
    pan_years_anom[2055.25 ] = 'tiny_dot_BC'
    pan_years_anom[2055.5  ] = 'tiny_dot_BR' 
    pan_years_anom[2055.75 ] = 'tiny_dot_BC'
    pan_years_anom[2056  ] = 'tiny_dot_BL'    
    pan_years_anom[2056.25 ] = 'tiny_dot_ML'    
    pan_years_anom[2056.5 ] = 'tiny_dot_TL'    
    pan_years_anom[2056.75 ] = 'tiny_dot_ML'    
    pan_years_anom[2057  ] = 'tiny_dot_BL' 
    pan_years_anom[2057.25 ] = 'tiny_dot_ML'    
    pan_years_anom[2058] = 'big_b'   

    # #2048-2056 - both spinning:
    # pan_years_ts[2048  ] = 'tiny_dot_r'
    # pan_years_ts[2048.5] = 'tiny_dot_TL'
    # pan_years_ts[2049  ] = 'tiny_dot_TC'
    # pan_years_ts[2049.5] = 'tiny_dot_TR'
    # pan_years_ts[2050  ] = 'tiny_dot_MR'
    # pan_years_ts[2050.5] = 'tiny_dot_BR'    
    # pan_years_ts[2051  ] = 'tiny_dot_BC'    
    # pan_years_ts[2051.5] = 'tiny_dot_BL'    
    # pan_years_ts[2052  ] = 'tiny_dot_ML'    
    # pan_years_ts[2052.5] = 'tiny_dot_TL'    
    # pan_years_ts[2053  ] = 'tiny_dot_TC'
    # pan_years_ts[2053.5] = 'tiny_dot_TR'
    # pan_years_ts[2054  ] = 'tiny_dot_MR'
    # pan_years_ts[2054.5] = 'tiny_dot_BR'    
    # pan_years_ts[2055  ] = 'tiny_dot_BC'    
    # pan_years_ts[2055.5] = 'tiny_dot_BL'    
    # pan_years_ts[2056  ] = 'tiny_dot_ML'    
    # pan_years_ts[2056.5] = 'tiny_dot_TL'    
    # pan_years_ts[2057  ] = 'tiny_dot_ML'    
    # pan_years_ts[2057.5] = 'tiny_dot_BL'

    # pan_years_ts[2058] = 'tiny_dot'    
  
    # pan_years_anom[2048  ] = 'tiny_dot'
    # pan_years_anom[2048.5] = 'tiny_dot_BR'
    # pan_years_anom[2049  ] = 'tiny_dot_BC'
    # pan_years_anom[2049.5] = 'tiny_dot_BL'
    # pan_years_anom[2050  ] = 'tiny_dot_ML'
    # pan_years_anom[2050.5] = 'tiny_dot_TL'    
    # pan_years_anom[2051  ] = 'tiny_dot_TC'    
    # pan_years_anom[2051.5] = 'tiny_dot_TR'    
    # pan_years_anom[2052  ] = 'tiny_dot_MR'    
    # pan_years_anom[2052.5] = 'tiny_dot_BR'    
    # pan_years_anom[2053  ] = 'tiny_dot_BC'
    # pan_years_anom[2053.5] = 'tiny_dot_BL'
    # pan_years_anom[2054  ] = 'tiny_dot_ML'
    # pan_years_anom[2054.5] = 'tiny_dot_TL'    
    # pan_years_anom[2055  ] = 'tiny_dot_TC'    
    # pan_years_anom[2055.5] = 'tiny_dot_TR'    
    # pan_years_anom[2056  ] = 'tiny_dot_MR'    
    # pan_years_anom[2056.5] = 'tiny_dot_BR' 
    # pan_years_anom[2057  ] = 'tiny_dot_MR'    
    # pan_years_anom[2057.5] = 'tiny_dot_TR'    

    # pan_years_anom[2058] = 'big_b'   

    # #2048-2056 - both spinning:
    # pan_years_ts[2048] = 'tiny_dot_r'
    # pan_years_ts[2049] = 'tiny_dot_TL'
    # pan_years_ts[2050] = 'tiny_dot_TC'
    # pan_years_ts[2051] = 'tiny_dot_TR'
    # pan_years_ts[2052] = 'tiny_dot_MR'
    # pan_years_ts[2053] = 'tiny_dot_BR'    
    # pan_years_ts[2054] = 'tiny_dot_BC'    
    # pan_years_ts[2055] = 'tiny_dot_BL'    
    # pan_years_ts[2056] = 'tiny_dot_ML'    
    # pan_years_ts[2057] = 'tiny_dot_TL'    
    # pan_years_ts[2058] = 'tiny_dot'    
  
    # pan_years_anom[2048] = 'tiny_dot'
    # pan_years_anom[2049] = 'tiny_dot_BR'
    # pan_years_anom[2050] = 'tiny_dot_BC'
    # pan_years_anom[2051] = 'tiny_dot_BL'
    # pan_years_anom[2052] = 'tiny_dot_ML'
    # pan_years_anom[2053] = 'tiny_dot_TL'    
    # pan_years_anom[2054] = 'tiny_dot_TC'    
    # pan_years_anom[2055] = 'tiny_dot_TR'    
    # pan_years_anom[2056] = 'tiny_dot_MR'    
    # pan_years_anom[2057] = 'tiny_dot_BR'    
    # pan_years_anom[2058] = 'big_b'    

    # 2020 is sudden
    globe_type_years={}
    globe_type_years.update({t:'ts' for t in np.arange(1970, 1988)})
    globe_type_years[1987] = 'both'
    globe_type_years.update({t:'anomaly' for t in np.arange(1988, 1996)}) 
    globe_type_years[1996] = 'both'
    globe_type_years.update({t:'ts' for t in np.arange(1997, 2020)})

    ## 2048 us a sudden change via the tiny_dot:
    globe_type_years.update({t:'anomaly' for t in np.arange(2020, 2036)})
    globe_type_years[2036] = 'both'


    globe_type_years.update({t:'ts' for t in np.arange(2037, 2048)})
    globe_type_years.update({t:'both' for t in np.arange(2048, 2058)})
    globe_type_years.update({t:'anomaly' for t in np.arange(2058, 2075)})

    max_heatwaves_panning = {
        1976: 0.5,
        ##2020: 0.,
        #2020: 180., 
        2076: 0.85, 
    }
    max_heatwave = calc_midoint(dct, max_heatwaves_panning)

    # heatwaves = calc_heat_wave(datas_dict['thetao_con'],
    #                           dates_dict['thetao_con'], 
    #                           date_key=date_key,
    #                           clim_range=clim_range,
    #                           clim_stuff=clim_stuff['thetao_con'],
    #                           max_heatwave=hw_len)
    heatwaves = load_heatwaves()

    cbar_ax_locs = { #(left, bottom, width, height)
        'main':  [0.9, 0.2, 0.02, 0.6],
        'above': [0.95, 1.0, 0.015, 0.45],
        'below': [0.95, -0.6, 0.015, 0.45],
        'tophalf' : [0.9, 0.55, 0.02, 0.3],
        'bottomhalf' : [0.9, 0.15, 0.02, 0.3],
    
    }
    cbar_ax_loc_pann_ts = {}
    cbar_ax_loc_pann_anom = {}
    globe_years = sorted([yr for yr in globe_type_years.keys()])
    for i, yr in enumerate(globe_years):
        plot_type2 = globe_type_years[yr]
        
        #plot_type2_p1 = globe_type_years[globe_years[i+1]]
        #plot_type2_m1 = globe_type_years[globe_years[i-1]]

        # situations: 
        if plot_type2 == 'both':
            cbar_ax_loc_pann_ts[yr] = cbar_ax_locs['tophalf']
            cbar_ax_loc_pann_anom[yr] = cbar_ax_locs['bottomhalf']

        if plot_type2 == 'ts':
            cbar_ax_loc_pann_ts[yr] = cbar_ax_locs['main']
            cbar_ax_loc_pann_anom[yr] = cbar_ax_locs['below']
            #cbar_ax_loc_pann_anom[yr-1] = cbar_ax_locs['below']
               
        if plot_type2 == 'anomaly':
            cbar_ax_loc_pann_ts[yr] = cbar_ax_locs['above']
            #cbar_ax_loc_pann_ts[yr-0.5] = cbar_ax_locs['above']
            cbar_ax_loc_pann_anom[yr] = cbar_ax_locs['main']

    # transistions
    cbar_ax_loc_pann_ts[1986] = cbar_ax_locs['main']
    cbar_ax_loc_pann_anom[1986] = cbar_ax_locs['below']
    cbar_ax_loc_pann_ts[1987] = cbar_ax_locs['main']
    cbar_ax_loc_pann_anom[1987] = cbar_ax_locs['below'] 
    cbar_ax_loc_pann_ts[1987.5] = cbar_ax_locs['main']
    cbar_ax_loc_pann_anom[1987.5] = cbar_ax_locs['below']     
    cbar_ax_loc_pann_ts[1987.75] = cbar_ax_locs['tophalf']
    cbar_ax_loc_pann_anom[1987.75] = cbar_ax_locs['bottomhalf']         
    cbar_ax_loc_pann_ts[1988] = cbar_ax_locs['above']
    cbar_ax_loc_pann_anom[1988] = cbar_ax_locs['main']

    cbar_ax_loc_pann_ts[1996] = cbar_ax_locs['above']
    cbar_ax_loc_pann_anom[1996] = cbar_ax_locs['main']
    cbar_ax_loc_pann_ts[1996.5] = cbar_ax_locs['tophalf']
    cbar_ax_loc_pann_anom[1996.5] = cbar_ax_locs['bottomhalf']    
    cbar_ax_loc_pann_ts[1997] = cbar_ax_locs['main']
    cbar_ax_loc_pann_anom[1997] = cbar_ax_locs['below'] 

    # abrupt change:
    cbar_ax_loc_pann_ts[2019.85] = cbar_ax_locs['main']
    cbar_ax_loc_pann_ts[2020] = cbar_ax_locs['main']
    cbar_ax_loc_pann_anom[2020] = cbar_ax_locs['main']
    cbar_ax_loc_pann_anom[2020.1] = cbar_ax_locs['main']


    cbar_ax_loc_pann_ts[2036] = cbar_ax_locs['above']
    cbar_ax_loc_pann_anom[2036] = cbar_ax_locs['main']
    cbar_ax_loc_pann_ts[2036.5] = cbar_ax_locs['tophalf']
    cbar_ax_loc_pann_anom[2036.5] = cbar_ax_locs['bottomhalf']    
    cbar_ax_loc_pann_ts[2037] = cbar_ax_locs['main']
    cbar_ax_loc_pann_anom[2037] = cbar_ax_locs['below'] 


    cbar_ax_loc_pann_ts[2047] = cbar_ax_locs['main']    
    cbar_ax_loc_pann_anom[2047] = cbar_ax_locs['below']
    cbar_ax_loc_pann_ts[2047.5] = cbar_ax_locs['main']
    cbar_ax_loc_pann_anom[2047.5] = cbar_ax_locs['below']    
    cbar_ax_loc_pann_ts[2048] = cbar_ax_locs['tophalf']
    cbar_ax_loc_pann_anom[2048] = cbar_ax_locs['bottomhalf']   

    globe_type_yr = globe_type_years[year] #.get(year, 'ts')
    if globe_type_yr in ['both', 'ts']:
        ortho_path_x = {yr: pan[value][0] for yr, value in pan_years_ts.items()}
        ortho_path_y = {yr: pan[value][1] for yr, value in pan_years_ts.items()}
        axes_path_L = {yr: pan[value][2] for yr, value in pan_years_ts.items()}
        axes_path_B = {yr: pan[value][3] for yr, value in pan_years_ts.items()}
        axes_path_W = {yr: pan[value][4] for yr, value in pan_years_ts.items()}
        axes_path_H = {yr: pan[value][5] for yr, value in pan_years_ts.items()}

        ortho_x = calc_midoint(dct, ortho_path_x)
        ortho_y = calc_midoint(dct, ortho_path_y)
        axes_L = calc_midoint(dct, axes_path_L)
        axes_B = calc_midoint(dct, axes_path_B)
        axes_W = calc_midoint(dct, axes_path_W)
        axes_H = calc_midoint(dct, axes_path_H)

        cbar_path_l = {yr: value[0] for yr, value in cbar_ax_loc_pann_ts.items()}
        cbar_path_b = {yr: value[1] for yr, value in cbar_ax_loc_pann_ts.items()}
        cbar_path_w = {yr: value[2] for yr, value in cbar_ax_loc_pann_ts.items()}
        cbar_path_h = {yr: value[3] for yr, value in cbar_ax_loc_pann_ts.items()}

        cbar_ax_loc = [
            calc_midoint(dct, cbar_path_l),
            calc_midoint(dct, cbar_path_b),
            calc_midoint(dct, cbar_path_w),
            calc_midoint(dct, cbar_path_h),
        ]

        ortho_pro=ccrs.Orthographic(ortho_y, ortho_x)

        ax2 = fig.add_axes([axes_L, axes_B, axes_W, axes_H], projection=ortho_pro) # (left, bottom, width, height)
        globe_type='ts'
        ax2 = plot_globe(ax2, nc=nc, t=t, quick=False, globe_type=globe_type, clim_dat=clim_dat)

        #if datas_dict['thetao_con'][date_key]
        ax2 = add_mpa(ax2, dct, heatwaves=heatwaves, max_heatwave=max_heatwave, axes_size=axes_W)

        ax2 = add_cbar(fig, field=field,  globe_type=globe_type, cbar_ax_loc = cbar_ax_loc) #ax_cbar)

    if globe_type_yr in ['both', 'anomaly']:
        ortho_path_x = {yr: pan[value][0] for yr, value in pan_years_anom.items()}
        ortho_path_y = {yr: pan[value][1] for yr, value in pan_years_anom.items()}
        axes_path_L = {yr: pan[value][2] for yr, value in pan_years_anom.items()}
        axes_path_B = {yr: pan[value][3] for yr, value in pan_years_anom.items()}
        axes_path_W = {yr: pan[value][4] for yr, value in pan_years_anom.items()}
        axes_path_H = {yr: pan[value][5] for yr, value in pan_years_anom.items()}

        ortho_x = calc_midoint(dct, ortho_path_x)
        ortho_y = calc_midoint(dct, ortho_path_y)
        axes_L = calc_midoint(dct, axes_path_L)
        axes_B = calc_midoint(dct, axes_path_B)
        axes_W = calc_midoint(dct, axes_path_W)
        axes_H = calc_midoint(dct, axes_path_H)

        cbar_path_l = {yr: value[0] for yr, value in cbar_ax_loc_pann_anom.items()}
        cbar_path_b = {yr: value[1] for yr, value in cbar_ax_loc_pann_anom.items()}
        cbar_path_w = {yr: value[2] for yr, value in cbar_ax_loc_pann_anom.items()}
        cbar_path_h = {yr: value[3] for yr, value in cbar_ax_loc_pann_anom.items()}

        cbar_ax_loc = [
            calc_midoint(dct, cbar_path_l),
            calc_midoint(dct, cbar_path_b),
            calc_midoint(dct, cbar_path_w),
            calc_midoint(dct, cbar_path_h),
        ]

        ortho_pro=ccrs.Orthographic(ortho_y, ortho_x)
        ax2b = fig.add_axes([axes_L, axes_B, axes_W, axes_H], projection=ortho_pro) # (left, bottom, width, height)
        globe_type='anomaly'
        ax2b = plot_globe(ax2b, nc=nc, t=t, quick=False, globe_type=globe_type, clim_dat=clim_dat)
        ax2b = add_mpa(ax2b, dct, heatwaves=heatwaves, max_heatwave=max_heatwave, axes_size=axes_W)

        ax2b = add_cbar(fig, field=field,  globe_type=globe_type, cbar_ax_loc = cbar_ax_loc) #ax_cbar)


    # time series axes
    # Music is:
    # piano: temperature anomaly:
    # synth bass: Ptot_c
    # tops: pH
    # high Synth: Ztot
    axes_alphas = {
        'thetao_con': {1970: 0., 1976.:0., 1976.15:1, 2070.5:1., 2071: 0.},
        'O3_pH': {1970: 0., 1976.5:1, 1992:1,  2020:1, 2021:0., 2051:0, 2052:1, 2070.5:1., 2071: 0.},
        'Ptot_c_result': {1970: 0., 1976.5:1, 1984:1, 2020:1, 2021:0, 2044:1, 2070.5:1., 2071: 0.},
        'Ztot_c_result': {1970: 0., 1999:1, 2000:1, 2020:1, 2021:0, 2060:1, 2070.5:1., 2071: 0.},
    }

    # when to show the time series.
    active_time_ranges={
        'thetao_con': list(np.arange(1976, 2076)),
        'O3_pH': list(np.arange(1992, 2020)),
        'Ptot_c_result': list(np.arange(1984, 2020)),
        'Ztot_c_result': list(np.arange(2000, 2020)),
        }
    
    active_time_ranges['O3_pH' ].extend(list(np.arange(2052, 2076)))
    active_time_ranges['Ptot_c_result' ].extend(list(np.arange(2044, 2076)))
    active_time_ranges['Ztot_c_result' ].extend(list(np.arange(2060, 2076)))

    # Wheere the time series axes are:
    left = 0.08
    widths = {1970: 0.3, 2067: 0.35, 2068: 0.4, 2071: 0.55}
    width = calc_midoint(dct, widths)

    plot_heights =       {1: 0.6, 2: 0.3,  3: 0.2,  4: 0.17}
    plot_B_corner = { # the index is the number of plots on screen at once.
        'thetao_con'   : {1: 0.2, 2: 0.52, 3: 0.65, 4: 0.715},
        'O3_pH'        : {1: 0,   2: 0.,   3: 0.4,  4: 0.515},
        'Ptot_c_result': {1: 0,   2: 0.18, 3: 0.15, 4: 0.315},
        'Ztot_c_result': {1: 0,   2: 0,    3: 0,    4: 0.115},
        }
    
    number_of_panes = {
            1970: 1,
            1976: 1,
            1983: 1,
            1984: 2,
            1991: 2,
            1992: 3, 
            1999: 3,
            2000: 4,
            2020: 4,
            2022: 1,
            2043: 1,
            2044: 2,           
            2051: 2,
            2052: 3,
            2059: 3,
            2060: 4,
            2075: 4,
    }

    ax_panning_B = {
        'thetao_con':    {yr: plot_B_corner['thetao_con'   ][i] for yr, i in number_of_panes.items()},
        'O3_pH':         {yr: plot_B_corner['O3_pH'        ][i] for yr, i in number_of_panes.items()},
        'Ptot_c_result': {yr: plot_B_corner['Ptot_c_result'][i] for yr, i in number_of_panes.items()},
        'Ztot_c_result': {yr: plot_B_corner['Ztot_c_result'][i] for yr, i in number_of_panes.items()},
    }

    window_panning = {
            1970: 2.1,
            1976: 2.1,
            1977: 2.5,
            1978: 2.7,
            1979: 3.1,
            2000: 4.5,
            2004: 3.2,
            2010: 3.,
            2016: 2.1,
            2020: 2.1,
            2021: 5.,
            2030: 7.,
            2040: 10.,
            2043: 6.,
            2044: 5.,           
            2051: 4.,
            2059: 2.5,
            2060: 2.3,
            2064.: 5.,
#           2065: 2.1,
#           2069: 20,
            2069.5: 96.,
            2071: 98.,
            2075: 95,
    }

    ax_panning_H = {yr: plot_heights[i] for yr, i in number_of_panes.items()}
    axes_path_H = {yr: pan[value][5] for yr, value in pan_years.items()} 
    window =  calc_midoint(dct, window_panning)
    y_height = calc_midoint(dct, ax_panning_H)


    add_legend = {int(yr):True for yr in np.arange(1970, 2068)}
    add_legend[2069] = False
    add_legend[2070] = False
    add_legend[2071] = False
   

    if add_legend.get(year, False):
        ax_leg = fig.add_axes([0.45, 0.1, 0.4, 0.2]) # (left, bottom, width, height)
        fig, ax_leg = plot_legend(
                active_time_ranges,
                time_key=date_key,
                decimal_time=dct,
                clim_range=clim_range,
                fig=fig, 
                ax=ax_leg)

    if year in active_time_ranges['thetao_con']:
        alpha_ax3 = calc_midoint(dct, axes_alphas['thetao_con'])
        ax3_B = calc_midoint(dct, ax_panning_B['thetao_con'])
        ax3 = fig.add_axes([left, ax3_B, width, y_height]) # (left, bottom, width, height) 

        fig, ax3 = plot_past_year_just_anom_ts(
            datas_dict['thetao_con'],
            dates_dict['thetao_con'], 
            target_time_key=date_key, 
            window=window, 
            clim_range=clim_range,
            clim_stuff=clim_stuff['thetao_con'],
            active_time_range=active_time_ranges['thetao_con'],
            plot_type='just_anom',
            field='thetao_con',
            fig=fig, 
            ax=ax3, 
            alpha=alpha_ax3
        )
        fig, ax3 = plot_musical_notes(
            datas_dict,
            dates_dict, 
            time_key=date_key, 
            decimal_t=dct,
            recalc_fn = 'output/MHW/recalc/MarineHeatWaves_f_cnrm_temp_anom.csv', 
            field='thetao_con',
            active_time_range=active_time_ranges['thetao_con'],
            window=window, # years
            plot_type='just_anom',
            monthly=False,
            fig=fig, 
            ax=ax3,
        )        
        # ax3.text(0.02, 0.05, ''.join(['Temperature Anomaly ', , str(clim_range[0]), '-', str(clim_range[1]), ]), 
        #          color='white', alpha=alpha_ax3, transform=ax3.transAxes)
        ax3.text(0.02, 0.05, 'Temperature Anomaly ',
                 color='white', alpha=alpha_ax3, transform=ax3.transAxes)        
        ax3.set_ylabel(r'$\Delta \degree$'+'C', color='white', alpha=alpha_ax3,)
    

    # pH
    if year in active_time_ranges['O3_pH']:
        ax4_B = calc_midoint(dct, ax_panning_B['O3_pH'])
        ax4 = fig.add_axes([left, ax4_B, width, y_height]) # (left, bottom, width, height) 
        alpha_ax4 = calc_midoint(dct, axes_alphas['O3_pH'])

        active_time_range = active_time_ranges['O3_pH']
        if year >=2062.5: 
            active_time_range = active_time_ranges['thetao_con']

        fig, ax4 = plot_past_year_just_anom_ts(
            datas_dict['O3_pH'],
            dates_dict['O3_pH'],
            target_time_key=date_key, 
            window=window, 
            clim_range=clim_range,
            clim_stuff=clim_stuff['O3_pH'],
            active_time_range=active_time_range,
            plot_type='just_ts',
            field='O3_pH',
            fig=fig, 
            ax=ax4,
            alpha=alpha_ax4)
        ax4.text(0.02, 0.05, 'pH', color='white', alpha=alpha_ax4, transform=ax4.transAxes)
        fig, ax4 = plot_musical_notes(datas_dict, dates_dict, 
            time_key=date_key, 
            decimal_t=dct,
            recalc_fn = 'output/MHW/recalc/MarineHeatWaves_f_pH_result.csv',
            active_time_range=active_time_range,
            field='O3_pH',
            plot_type='just_ts',
            monthly=True,
            window=window, # years
            fig=fig, ax=ax4
        )
        #ax4.set_ylabel('pH', color='white', alpha=alpha_ax4,)


    # Ptot_c_result
    if year in active_time_ranges['Ptot_c_result']:
        ax5_B = calc_midoint(dct, ax_panning_B['Ptot_c_result'])
        ax5 = fig.add_axes([left, ax5_B, width, y_height]) # (left, bottom, width, height) 
        alpha_ax5 = calc_midoint(dct, axes_alphas['Ptot_c_result'])
        active_time_range = active_time_ranges['Ptot_c_result']
        if year >=2062.5: 
            active_time_range = active_time_ranges['thetao_con']


        fig, ax5 = plot_past_year_just_anom_ts(
            datas_dict['Ptot_c_result'],
            dates_dict['Ptot_c_result'], 
            target_time_key=date_key, 
            window=window, 
            clim_range=clim_range,
            clim_stuff=clim_stuff['Ptot_c_result'],
            field='Ptot_c_result',
            active_time_range=active_time_range,
            plot_type='just_ts',
            fig=fig, 
            ax=ax5, 
            alpha=alpha_ax5)
        ax5.text(0.02, 0.87, 'Phytoplankton', color='white',alpha=alpha_ax5, transform=ax5.transAxes)
        fig, ax5 = plot_musical_notes(datas_dict, dates_dict, 
            time_key=date_key, 
            decimal_t=dct,
            recalc_fn = 'output/MHW/recalc/MarineHeatWaves_f_Ptot_c_result.csv',
            active_time_range=active_time_range,
            field='Ptot_c_result',
            plot_type='just_ts',
            monthly=True,
            window=window, # years
            fig=fig, ax=ax5
        )
        ax5.set_ylabel('mg Cm'+r'$^{3}$', color='white', alpha=alpha_ax5,)

    # Ztot_c_result
    if year in active_time_ranges['Ztot_c_result']:
        ax6_B = calc_midoint(dct, ax_panning_B['Ztot_c_result'])

        ax6 = fig.add_axes([left, ax6_B, width, y_height]) # (left, bottom, width, height) 
        alpha_ax6 = calc_midoint(dct, axes_alphas['Ztot_c_result'])
        active_time_range = active_time_ranges['Ztot_c_result']
        if year >=2062.5:    
            active_time_range = active_time_ranges['thetao_con']

        fig, ax6 = plot_past_year_just_anom_ts(
            datas_dict['Ztot_c_result'],
            dates_dict['Ztot_c_result'], 
            target_time_key=date_key,
            window=window,
            clim_range=clim_range,
            clim_stuff=clim_stuff['Ztot_c_result'],
            field='Ztot_c_result',
            active_time_range=active_time_range,
            plot_type='just_ts',
            fig=fig, 
            ax=ax6, 
            alpha=alpha_ax6)
        ax6.text(0.02, 0.87, 'Zooplankton', color='white', alpha=alpha_ax6, transform=ax6.transAxes)
        fig, ax6 = plot_musical_notes(
            datas_dict, dates_dict, 
            time_key=date_key, 
            decimal_t=dct,
            recalc_fn = 'output/MHW/recalc/MarineHeatWaves_f_Ztot_c_result.csv',
            field='Ztot_c_result',
            active_time_range=active_time_range,
            plot_type='just_ts',
            window=window, # years
            monthly=True,
            fig=fig, ax=ax6
        )
        ax6.set_ylabel('mg Cm'+r'$^{3}$', color='white', alpha=alpha_ax6,)
    
    # Adjust x ticks:
    if year in active_time_ranges['Ptot_c_result']:
        ax3.xaxis.set_ticklabels([]) # thetao axis
    if year in active_time_ranges['O3_pH']:
        ax4.xaxis.set_ticklabels([]) # O3_pH
        # Never has ticks!
    if year in active_time_ranges['Ztot_c_result']:
        ax5.xaxis.set_ticklabels([]) # Phyto   
    # # Zoo always has x ticks.    

    #ax7 = fig.add_axes([0.1, 0.05, 0.3, 0.]) # (left, bottom, width, height) 
    #pyplot.plot()

    font = {#'family': 'monospace',
            'color':  'white',
            'weight': 'bold',
            'size': 14,
            }
    scenario = 'Historical'
    if year>=2015:
        scenario = 'SSP3-7.0'    
    fig.text(left, 0.932, 'Marine Heatwaves' , fontdict=font)
    #fig.text(left, 0.895, ' '.join([date_string, scenario]), fontdict=font)
    fig.text(left, 0.90, date_string, fontdict={'color':  'white', 'size': 12,})
    #fig.text(left, ax3_B+ y_height + 0.0075, date_string, fontdict=font)



    print('saving:', date_string, fn)  
    pyplot.savefig(fn, dpi=dpi)
    pyplot.close()

    return fn
    # end of daily_plot


def calc_and_add_distortion(fn, decimal_t):
    """
    Adds distortion to the images.
    More heatwave, more distortion.
    """
    output_file = fn.replace('.png', '_distorted.png')

    if os.path.exists(output_file):
        return output_file
    
    heatwaves = load_heatwaves()

    reduced_heatwaves = {}
    times = [time for time in sorted(heatwaves.keys())]
    for t, time in enumerate(times):
        if time > decimal_t: 
            # no heatwaves in the future
            continue

        hwl_value = heatwaves[time]

        if t > 0:
            hwl_value_m1 = heatwaves[times[t-1]]
            if hwl_value_m1 == hwl_value:
                # Skip doubles - only show waves when note changes.
                continue
        reduced_heatwaves[time] = hwl_value

    temperature_anom = heatwaves[decimal_t] # the current piano note
    reduced_heatwaves_times = np.array(sorted(reduced_heatwaves.keys()))

    if temperature_anom < 1.: 
        # no distortion below 1 degree
        return fn
    
    if reduced_heatwaves.get(decimal_t, False):
        # Current time step is a new note.
        time_diff = 0.
    else:
        # Calculate last first step.
        time_init =  reduced_heatwaves_times[reduced_heatwaves_times < decimal_t].max()
        time_diff = np.abs(decimal_t - time_init)
       
    print("calc_and_add_distortion", temperature_anom, time_diff, fn, '->', output_file)
    output_file = add_distortion_to_fig(temperature_anom, time_diff, input_file=fn, output_file=output_file)
    return output_file


def get_aligned_clim(dt, year, month, day, clim_datas, clim_doy, clim_month):
    """
    Get the rught climatological value for a day.
    """
    if len(clim_datas.keys()) < 25:
        # monthly data:
        return clim_month[month]
    if (month, day) in clim_datas.keys():
        return clim_datas[(month, day)]
    assert 0


def plot_single_year_anomaly_ts(datas_dict, dates_dict, plot_year=1976, clim_range=[1976,1985], field=None, fig=None, ax=None):
    """
    Save single plot.
    """
    x = []
    y = []
    clim_y = []
    clim_datas, clim_doy, clim_month = calculate_clim(datas_dict, dates_dict, clim_range=clim_range)

    for time_key in sorted(datas_dict.keys()):
        (year, month, day) = time_key
        if plot_year != year:
            continue

        dat = datas_dict[time_key]
        dt = dates_dict[time_key]

        dcy = decimal_year(dt, year,month,day)
        x.append(dcy)
        y.append(dat)
        clim = get_aligned_clim(dt, year,month,day, clim_datas,clim_doy, clim_month)

        clim_y.append(clim)
    if not len(x):
        print('no time in ', field)
        return

    returnfig = True
    if fig is None:
        fig = pyplot.figure()
        returnfig= False
    pyplot.plot(x, y, c='green', zorder=10, label=str(plot_year))
    pyplot.plot(x, clim_y, c='k', label = 'Climatology') 
    pyplot.ylim([vmins.get(field, 0.), vmaxs.get(field, 1.)])

    down_where = np.ma.masked_less(y, clim_y).mask
    up_where = np.ma.masked_greater(y, clim_y).mask

    if np.sum(down_where):

        pyplot.fill_between(
            x,
            clim_y,
            y,
            where = down_where, # * np.ma.masked_less_equal(y, clim_y-1).mask,
            facecolor='dodgerblue'
            )
    if np.sum(up_where):
        pyplot.fill_between(
            x,
            y,
            clim_y,
            where = up_where, # * np.ma.masked_less_equal(y, clim_y-1).mask,
            facecolor='#e2062c', # 'red'
            )

    pyplot.title(field + ' '+str(plot_year))
    pyplot.legend()
    pyplot.xlim([x[0], x[-1]])
    pyplot.ylim([vmins.get(field, 0.), vmaxs.get(field, 1.)])

    fn = folder('images/anom_year/'+field)+field+'_anom_'+str(plot_year)+'.png'
    print('Saving', fn)
    pyplot.savefig(fn)
    pyplot.close()




def get_clim_dat(fn, date_key, field, clim_range, model='CNRM'):
    """
    returns or calculates a netcdf of the climatalogical surface field.
    """
    clim_range_str = '-'.join([str(int(c)) for c in clim_range])
    month_str = mn_str(date_key[1])
    day_str = mn_str(date_key[2])

    output_path = folder(['pickles', model, field, clim_range_str, '/'])
    output_path = ''.join([output_path, model, '-',field, '-', clim_range_str, '_', month_str, '-', day_str, '.pkl'])

    if os.path.exists(output_path):
        with open(output_path, 'rb') as pfile:
            data = pickle.load(pfile)
        return data

    path_list =  get_file_list(model, field, ssp=None, remove_2015=True)
    path_restricted = []

    # curate a short list:
    clim_years = np.arange(clim_range[0], clim_range[1]+1)
    for fn in sorted(path_list):
        #print(fn.split('/'))
        yr = int(fn.split('/')[-3])
        mn = fn.split('/')[-2]
        #print(yr, mn, month_str)       
        if yr not in clim_years:
            continue
        #print('found year, ', yr, mn)
        if mn == month_str:
            pass
        else:
            continue
        #print('found one:', fn, yr, mn)
        path_restricted.append(fn)

    print(field, model, path_restricted, clim_years)  
    if len(path_restricted) != len(clim_years):
        assert 0

    print('calculating clim:', date_key[1], date_key[2], path_restricted[0])
    nc = Dataset(path_restricted[0], 'r')
    data = nc.variables[field][date_key[2]-1, 0, :, :]
    nc.close()

    count = 1
    leap_day =  (date_key[1] == 2 and date_key[2] == 29)
    for fn in path_restricted[1:]:
        print('calculating clim:', date_key[1], date_key[2], fn)
        nc = Dataset(fn, 'r')
        if leap_day: # Leap day
            try:
                data += nc.variables[field][date_key[2]-1, 0, :, :]
            except: pass
        else:
            data += nc.variables[field][date_key[2]-1, 0, :, :]
        nc.close()        
        count+=1
    data = data/count
    with open(output_path, 'wb') as pfile:
        pickle.dump(data, pfile)
    return data

    # ds = nctoolkit.open_data(path_restricted)
    # #ds.subset(years = range(clim_range[0], clim_range[1]+1))
    # ds.subset(month=date_key[1], day=date_key[2])
    # ds.subset(variables = field)
    # ds.merge("time")
    # ds.top()
    # ds.tmean("day")
    # nctoolkit.options(cores = 6)

    # ds.to_nc(output_path)
    # return output_path


def iterate_daily_plots(
        fn, 
        datas_dict, 
        dates_dict, 
        clim_stuff=(), 
        clim_range=[1976,1985], 
        field='thetao_con', 
        plot_every_days=9,
        ):

    nc = Dataset(fn, 'r')

    times = nc.variables['time_centered']
    dates = num2date(times[:], times.units, calendar=times.calendar)

    images = []
    for t, dt in enumerate(dates[:]):
        date_key = (dt.year, dt.month, dt.day)
        if date_key[1] == 2 and date_key[2] == 29:
            # skip problematic leap days
            continue

        dcy = decimal_year(dt, dt.year, dt.month, dt.day)
        plot_every_days_ = plot_every_days

        if dcy >= 2069.95:
            plot_every_days_ = int(plot_every_days/3.) # 3x slower final year.
        if dcy > 2070. + 11./12. or plot_every_days_ == 0:
            plot_every_days_ = 1

        time_delta = dt-cftime.DatetimeGregorian(1976, 1, 1, 12., 0, 0, 0)
        if time_delta.days % plot_every_days_: 
            continue
        
        # loading climatology map:
        clim_dat = get_clim_dat(fn, date_key, field, clim_range)
        #clim_dat = Dataset(clim_dat_fn, 'r')       
        #clim_dat.close()

        # Make Daily_plot.
        img_fn = daily_plot(nc, t, date_key, datas_dict, dates_dict, clim_stuff=clim_stuff, clim_range=clim_range, clim_dat=clim_dat) # no_new_plots=no_new_plots)

        if img_fn in [False, None]:
            continue
        if not do_distortion:
            images.append(img_fn)
        else:
            distorted_fn = calc_and_add_distortion(img_fn, dcy)
            #if year >= 2070:
            #    add_distortion_to_fig(month+2, 0.,  input_file=distorted_fn, output_file = distorted_fn) 
            images.append(distorted_fn)
    nc.close()
    return images
        #return


def export_csv(datas_dict, dates_dict, model, field, overwrite=True):
    fn = folder('csv/')+model+'_'+field+'.csv'
    print('exporting csv', fn)
    if os.path.exists(fn) and not overwrite:
        return 
    txt = '# year, '+field +'\n'

    for tk in sorted(datas_dict.keys()):
        (year, month, day) = tk
        dat = datas_dict[tk]
        dt = dates_dict[tk]
        dcy = decimal_year(dt, year,month,day)

        txt =  ''.join([txt, 
            str(dcy), ',',
            str(dat), '\n'])
    csv = open(fn, 'w')
    csv.write(txt)
    csv.close()
        

def export_csv_clim(datas_dict, dates_dict, model, field,repeats=4,clim_range=[1976,1985], overwrite=True):
    """
    Export the climatology data as a csv.
    repeats is there so that it can be done to music.
    """
    fn = folder('csv/')+model+'_'+field+'_clim'+'_'+str(repeats)+'.csv'
    print('exporting csv', fn)
    if os.path.exists(fn) and not overwrite:
        return 
    txt = '# year, '+field +'_clim\n'
    clim_datas, clim_doy, clim_month = calculate_clim(datas_dict, dates_dict, clim_range=clim_range)
    for repeat in range(repeats):
        for doy in sorted(clim_doy.keys()):
            doyd = float(repeat) +doy/366.
            txt =  ''.join([txt, 
                str(doyd), ',',
                str(clim_doy[doy]), '\n'])
        
    csv = open(fn, 'w')
    csv.write(txt)
    csv.close()

def export_csv_anomaly(datas_dict, dates_dict, model, field,clim_range=[1976,1985], overwrite=True):
    """
    Export the anomaly data as a csv.
    """
    fn = folder('csv/')+model+'_'+field+'_anomaly.csv'
    print('exporting csv', fn)
    if os.path.exists(fn) and not overwrite:
        return 
    txt = '# year, '+field +'_anom\n'
    clim_datas, clim_doy, clim_month = calculate_clim(datas_dict, dates_dict, clim_range=clim_range)

    for tk in sorted(datas_dict.keys()):
        (year, month, day) = tk
        dat = datas_dict[tk]
        dt = dates_dict[tk]
        dcy = decimal_year(dt, year,month,day)
        dat = dat - clim_datas[(month, day)]
        txt =  ''.join([txt, 
            str(dcy), ',',
            str(dat), '\n'])
                
    csv = open(fn, 'w')
    csv.write(txt)
    csv.close()


def get_file_list(model, field, ssp='ssp370', remove_2015=True):
    """
    Get a list of netcdfile file paths.
    """
    if field in ['thetao_con', 'so_abs']:
        suffix  = '*_1d_*_grid_T.nc'

    elif field in ['vo',]:
        suffix  = '*_1d_*_grid_V.nc'

    elif field in ['uo',]:
        suffix  = '*_1d_*_grid_U.nc'   

    elif field in ['O3_c', 'N3_n', 'O3_TA']: 
          suffix  = '*_1d_*_ptrc_T.nc'             

    elif field in ['O3_pH', ]: # monthly:
         suffix  = '*_1m_*_ptrc_T*.nc'

    elif field in ['Ptot_c_result', 'Ztot_c_result', 'Ptot_Chl_result', 'Ptot_NPP_result']:
         suffix  = '*_1m_*_diag_T*.nc'       

    else:
       print('field not recognised:', field)
       assert 0

    path_hist = get_paths()[model+'_hist']
    files = glob(path_hist+'*/*/'+ suffix)

    if ssp:
        path_ssp = get_paths()[model+'_'+ssp]
        files.extend(glob(path_ssp+'*/*/'+ suffix))
    
    if remove_2015:
        remove_2015 = model+'_hist/2015/'
        files = [fn for fn in files if fn.find(remove_2015)==-1]
    #CNRM_ssp370/2015
    return files


def main(field = 'thetao_con', no_new_data=True):
    """
    Maing algo for caluculating stuff.
    """
    model = 'CNRM'

    files = get_file_list(model, field, ssp='ssp370')

    if not len(files): 
        print('No files found.', field, files)
        assert 0

    failures=[]
    #shelvefn = 'shelves/CNRM_hist_temp.shelve'
    shelvefn = folder('shelves/') + model + '_' + field + '.shelve'
    finished_files, datas_dict, dates_dict = load_shelve(shelvefn)

    clim_range=[1976,1985]

    for fn in sorted(files):
        if no_new_data:
            continue
        if fn in finished_files:
            continue
        print('loading:', fn)
        try:
            nc = Dataset(fn, 'r')
        except: 
            print('unable to load:', fn)
            failures.append(fn)
            continue
        #find_corners(nc)
        data_dict, date_dict = calc_mean_aimpa(nc, model=model, field=field)
        if 0 in [len(data_dict), len(date_dict)]:
            print('unable to load data from ', fn)
            failures.append(fn)
            continue
        datas_dict.update(data_dict)
        dates_dict.update(date_dict)        
        finished_files.append(fn)
        save_shelve(shelvefn, finished_files, datas_dict, dates_dict)
        nc.close()
    print('finished_files', finished_files)
    print('datas_dict', datas_dict)

    do_export_csv = True
    if do_export_csv:
        export_csv(datas_dict, dates_dict, model, field)
        export_csv_anomaly(datas_dict, dates_dict, model, field,clim_range=clim_range)
        export_csv_clim(datas_dict, dates_dict, model, field,clim_range=clim_range)

    #return
    for year in range(1976, 2071):
        #continue 
        print(year)
        #plot_single_year_ts(datas_dict, dates_dict, field=field, plot_year=year,)
        plot_single_year_just_anom_ts(datas_dict, dates_dict, field=field, plot_year=year, clim_range=clim_range)
        #plot_single_year_anomaly_ts(datas_dict, dates_dict, field=field, plot_year=year, clim_range=clim_range)
   
    print('Failures:', failures)
    return


def reduce_years(temp_files, reduced_years):
    """
    Reduce the number of input files.
    """
    out_files = []
    for fn in temp_files:
        yr = fn.split('/')[-3]
        if int(yr) not in reduced_years:
            continue
        print('Keeping', fn)
        out_files.append(fn)
    return out_files


def thumbnail(field = 'thetao_con',video_format = '4K',  globe_type='ts'):
    """
    Make a thumbnail figure
    """
    if video_format == '4K':
        image_size = [3840., 2160.]
        dpi = 300.



    files = [
        "/data/proteus3/scratch/gig/MissionAtlantic/CNRM_ssp370/OUTPUT/CNRM_ssp370/2060/12/CNRM_ssp370_1d_20601201_20601231_grid_T.nc",
        "/data/proteus3/scratch/gig/MissionAtlantic/CNRM_hist/OUTPUT/CNRM_hist/1999/06/CNRM_hist_1d_19990601_19990630_grid_T.nc",
        "/data/proteus3/scratch/gig/MissionAtlantic/CNRM_ssp370/OUTPUT/CNRM_ssp370/2024/04/CNRM_ssp370_1d_20240401_20240430_grid_T.nc",
        ]
    date_keys = [
        (2060, 12, 1), 
        (1999, 6, 1), 
        (2024, 4, 1), 
    ]

    for fn, date_key in zip(files, date_keys):
        fig = pyplot.figure(facecolor='black')
        fig.set_size_inches(image_size[0]/dpi,image_size[1]/dpi)

        clim_range=[1976,1985] # shouldn't matter?


        nc = Dataset(fn, 'r')
        date_str = ''.join([str(x) for x in date_key])
        t = 0 
        clim_dat = get_clim_dat(fn, date_key, field, clim_range)

        ortho_pro=ccrs.Orthographic(-15., 7.5)
        proj = ccrs.PlateCarree()

        ax2 = fig.add_axes([0.3, 0.05, 0.75, 0.9], projection=ortho_pro) # (left, bottom, width, height)

        ax2 = plot_globe(ax2, nc=nc, t=t, quick=False, globe_type=globe_type, clim_dat=clim_dat)
        
        ax2.add_patch(mpatches.Circle(xy=[central_longitude, central_latitude, ], linewidth=1.5,
                radius=mpa_radius, ec='white', fc=(0., 0., 0., 0.), transform=proj, zorder=31))
        
        
        fn = folder('images/thumbnail/')+'thumbnail_'+globe_type+'_'+video_format+'_'+date_str+'.png'
        print('Saving', fn)
        pyplot.savefig(fn)
        pyplot.close()




def make_daily_plots(
        model='CNRM',
        clim_range=[1976,1985],
        plot_every_days=global_plot_every_days, 
        #no_new_plots = False,
):
    """
    Main tool for making daily plots.
    """

    temp_files = get_file_list(model, 'thetao_con', ssp='ssp370')
   
    finished_files = {}
    datas_dict = {}
    dates_dict = {}
    clim_stuff = {}
   
    for field in ['thetao_con', 'O3_pH', 'Ptot_c_result', 'Ztot_c_result']:
        shelvefn = folder('shelves/') + model + '_' + field + '.shelve'
        finished_files[field], datas_dict[field], dates_dict[field] = load_shelve(shelvefn) 

    # Extend monthly datasets to closest value.    
    method = 'linear'
    for field in ['O3_pH', 'Ptot_c_result', 'Ztot_c_result']:
        if method == 'nearest':
            for time_key in sorted(datas_dict['thetao_con'].keys()):
                dates_dict[field][time_key] = dates_dict['thetao_con'][time_key]
                new_val = datas_dict[field].get((time_key[0], time_key[1], 16), 
                                datas_dict[field].get((time_key[0], time_key[1], 15), ))
                datas_dict[field][time_key] = new_val

        if method == 'linear':
            new_decimal_times = {}
            for dkey,date in dates_dict[field].items():
                new_decimal_times[decimal_year(date, dkey[0], dkey[1], dkey[2])] = dkey
            
            dec_times = sorted(new_decimal_times.keys())
            y = [datas_dict[field][new_decimal_times[dc]] for dc in dec_times]

            func1 = interpolate.interp1d(dec_times, y, kind='linear') #,bounds_error=False, fill_value=0.)
            for dkey, date in dates_dict['thetao_con'].items():
                thetao_decimal_times = decimal_year(date, dkey[0], dkey[1], dkey[2])
                if thetao_decimal_times < dec_times[0]:
                    thetao_decimal_times = dec_times[0]
                elif thetao_decimal_times > dec_times[-1]:
                    thetao_decimal_times = dec_times[-1]  
                new_val = func1(thetao_decimal_times)

                datas_dict[field][dkey] = new_val
                dates_dict[field][dkey] = date

                #print( dkey, date, new_val)

    # Calculate clims (just the one time)
    for field in ['thetao_con', 'O3_pH', 'Ptot_c_result', 'Ztot_c_result']:
        clim_datas, clim_doy, clim_month = calculate_clim(datas_dict[field], dates_dict[field], clim_range=clim_range)
        clim_stuff[field] = (clim_datas, clim_doy, clim_month)


    calc_heat_wave_csv(
        datas_dict['thetao_con'],
        dates_dict['thetao_con'], 
        clim_range=clim_range,
        clim_stuff=clim_stuff['thetao_con']
    )

    if len(reduced_years):
        temp_files = reduce_years(temp_files, reduced_years)
        years_str = str(reduced_years[0])+'-'+str(reduced_years[-1])
    else:
        years_str='Full'

    images = []
    for fn in sorted(temp_files): #reverse=True)[:]:
        #nc = Dataset(fn, 'r')
        imgs = iterate_daily_plots(fn, datas_dict, dates_dict, clim_stuff=clim_stuff, clim_range=clim_range, field='thetao_con', plot_every_days=plot_every_days ) #, no_new_plots=no_new_plots)
        images.extend(imgs)
        #nc.close()

    # Make video links:  
    if no_new_plots:
        video_frames = folder('video/frames/'+daily_count+'/'+years_str+'/NNP_'+str(plot_every_days))
    else:
        video_frames = folder('video/frames/'+daily_count+'/'+years_str+'/'+str(plot_every_days))

    video_path = folder('video/mp4/')+daily_count+'_'+years_str+'_'+str(plot_every_days)+'.mp4'
    print('creating links in', video_frames)
    last_frame = 0

    for plot_id, img_fn in enumerate(sorted(images)):
        outpath = ''.join([video_frames, 'img'+str(plot_id).zfill(6), '.png'])
        #add_distortion(img_fn, outpath)
        if not os.path.exists(outpath):
            os.symlink(os.path.abspath(img_fn), os.path.abspath(outpath))
        last_frame=plot_id

    img_fn = sorted(images)[-1]
    for i in np.arange(1, 60):
        outpath = ''.join([video_frames, 'img'+str(last_frame+i).zfill(6), '.png'])
        if not os.path.exists(outpath):
            os.symlink(os.path.abspath(img_fn), os.path.abspath(outpath))

   
    #beats_per_second = 120./60.
    #beats_per_year = 8.
    seconds_per_year = 2.
    days_per_year = 365.25         
    #frames_per_year= days_per_year/plot_every_days
    frames_per_second = days_per_year/(plot_every_days*seconds_per_year) 
    #beats_per_second * frames_per_year / beats_per_year
    framerate = str(frames_per_second)

    if video_format == '720p':
        image_size = [1280, 720]

    if video_format == 'HD':
        image_size = [1920., 1080.]

    if video_format == 'QHD':
        image_size = [2560., 1440.]

    if video_format == '4K':
        image_size = [3840., 2160.]

    image_size_str = 'x'.join([str(int(image_size[0])), str(int(image_size[1]))])
    video_command = "ffmpeg -nostdin -y -framerate "+framerate+"  -s "+image_size_str+" -i "+video_frames+"img%06d.png -pix_fmt yuv420p -c:v libx264  -preset ultrafast "+video_path
    print('Make video with:\n', video_command)


    return


def  run_all():
    failures = {}

    global video_format
    video_format = '4K' #'720p' # 'UHD'
    make_daily_plots(plot_every_days=global_plot_every_days) #, no_new_plots=True)

    #thumbnail(globe_type='ts')
    #thumbnail(globe_type='anomaly')    

    return    
    plot_every_days = 16 # 8*8  # 54*3  # 135

    make_daily_plots(plot_every_days=plot_every_days)
    #main(field='O3_pH') 
    return
    for field in ['Ptot_c_result', 'Ztot_c_result', 'Ptot_Chl_result', 'Ptot_NPP_result', 'O3_pH']:
         failures[field] = main(field=field)
    for field in ['O3_c', 'uo', 'vo', 'N3_n', 'thetao_con', 'so_abs', 'O3_TA']:
         failures[field] = main(field = field)
    print('Failures:', failures)

#   main(field='O3_c')
#   main(field='uo')
#   main(field='vo')
#   main(field='N3_n') 
#   main(field='thetao_con')
#   main(field='so_abs')


if __name__ == "__main__":
    run_all()


