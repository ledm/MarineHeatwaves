import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot
import matplotlib.patches as mpatches

from glob import glob
from netCDF4 import Dataset,num2date
import cftime
from bisect import bisect
from scipy import interpolate

import os

import numpy as np
import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cmocean
from shelve import open as shopen
from matplotlib.ticker import MaxNLocator
from matplotlib import cm

from earthsystemmusic2.music_utils import folder
#from .earthsystemmusic2 import music_utils as mutils 
from earthsystemmusic2.climate_music_maker import climate_music_maker

central_longitude = -14.368164721459744 #W #-160.+3.5
central_latitude = -7.940978677133847 # S
mpa_radius = 2.88

ortho_pro=ccrs.Orthographic(-15, -15)
pc_proj=cartopy.crs.PlateCarree()
#mpa_lon_corners = [central_longitude-mpa_radius, central_longitude+mpa_radius, central_longitude+mpa_radius, central_longitude-mpa_radius, central_longitude-mpa_radius ]
#mpa_lat_corners = [central_latitude-mpa_radius, central_latitude-mpa_radius, central_latitude+mpa_radius, central_latitude+mpa_radius, central_latitude-mpa_radius]

#TO DO:
#Add date
#make cbar text white
#figure out how to display time axes.
#add section turning on and off various axes with music.
#Make MPA red when over threshold.

#cm_bins = 19
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
    'thetao_con': 10.0,
    }
cbar_vmax = {
    'thetao_con': 31.0,
    }
cm_bins = 21

cmaps = {
    'thetao_con': 'plasma', #'viridis',
#     'so_abs': 'viridis'
#    'O3_TA': 2500., 
#    'N3_n': 20., 
#    'O3_c': 2500.,   
#    'uo': 1.,
#    'vo': 1.,     
    }
land_color = '#D3D3D3' #'#F5F5F5' #'#DCDCDC' # '#E0E0E0' ##F8F8F8'

def mnStr(month):
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
        fig, ax, 
        ):
    """
    Do the fill between stuff?
    """
    

def plot_past_year_just_anom_ts(datas_dict, dates_dict, 
                                target_time_key=(2000,1,1), 
                                window=2.5, # years
                                clim_range=[1976,1985],
                                alpha=1.,
                                field='thetao_con',
                                clim_stuff=(),
                                plot_type='anom',
                                fig=None, ax=None):
    """
    Create single plot.
    """
    x = []
    y = []
    clim_x = []
    clim_y= []   
    target_string = '-'.join([str(t) for t in target_time_key])
    target_dt = dates_dict[target_time_key]
    target_decimal = decimal_year(target_dt, target_time_key[0], target_time_key[1], target_time_key[2])
    #clim_datas, clim_doy, clim_month = calculate_clim(datas_dict, dates_dict, clim_range=clim_range)
    (clim_datas, clim_doy, clim_month) = clim_stuff

    for time_key in sorted(datas_dict.keys()):
        (year, month, day) = time_key
        dt = dates_dict[time_key]
        dcy = decimal_year(dt, time_key[0], time_key[1], time_key[2])
        if dcy > target_decimal: continue
        if dcy < target_decimal - window: continue

        dat = datas_dict[time_key]
        x.append(dcy)
        y.append(dat)
        clim_x.append(dcy)
        clim_y.append(clim_datas[(month, day)])

    if not len(x):
        return
   
    # smooth out monthly data.
    if 2 < len(x) < 36:
        newx, newy = smooth_axes(x, y)
        newclim_x, newclim_y = smooth_axes(clim_x, clim_y)
        x = newx
        y = newy
        clim_x = newclim_x
        clim_y = newclim_y
        print(x, y)

    if fig is None:
        fig = pyplot.figure()
        returnfig=False
    else:
        pyplot.sca(ax)
        returnfig=True

    anom = np.array(y) - np.array(clim_y)
    zeros = np.array([0. for i in anom])

    downwhere = np.ma.masked_less(anom, 0.).mask
    upwhere = np.ma.masked_greater(anom, 0.).mask

    if plot_type=='anom':
     
        if np.sum(downwhere):
            norm = matplotlib.colors.Normalize(vmin=anom_mins[field], vmax=0.)
            rgba_colors = [cm.cool(norm(y_val), 10 ) for y_val in anom]# if downwhere]
            
            pyplot.fill_between(x,
                    zeros,
                    anom,
                    color=rgba_colors,
                    where = downwhere,
                    #facecolor='dodgerblue',
                    )
        if np.sum(upwhere):
            norm = matplotlib.colors.Normalize(vmin=0., vmax=anom_maxs[field])
            rgba_colors = [cm.cool(norm(y_val), 10 ) for y_val in anom]# if upwhere]            
            pyplot.fill_between(x,
                    anom,
                    zeros,
                    where = upwhere,
                    color=rgba_colors,
                    #facecolor='#e2062c', # candy apple red
                    )
        pyplot.plot(clim_x, np.array(y) - np.array(clim_y), 'purple', lw=0.7)
        pyplot.plot(x, zeros, 'w', lw=0.5)
        #pyplot.plot(x, zeros+1, 'w', lw=0.5)
        #pyplot.plot(x, zeros-1, 'w', lw=0.5)

        pyplot.xlim([x[-1] - window, x[-1]])
        #pyplot.ylim([-1.5, 4.0])

        if anom_mins.get(field, False):
            pyplot.ylim([anom_mins[field], anom_maxs[field]])
    else:
        if np.sum(downwhere):
            norm = matplotlib.colors.Normalize(vmin=vmins[field], vmax=vmaxs[field])
            rgba_colors = [cm.cool(norm(y_val), 10 ) for y_val in y]# if downwhere]

            pyplot.fill_between(x,
                    y,
                    clim_y,
                    color=rgba_colors,
                    where = downwhere,
                    #facecolor='dodgerblue',
                    )
        if np.sum(upwhere):
            norm = matplotlib.colors.Normalize(vmin=vmins[field], vmax=vmaxs[field])
            rgba_colors = [cm.hot(norm(y_val), 10 ) for y_val in y ]# if upwhere]            
            pyplot.fill_between(x,
                    clim_y,
                    y,
                    color=rgba_colors,
                    where = upwhere,
                    #facecolor='#e2062c', # candy apple red
                    )
        pyplot.plot(x, y, 'purple', lw=0.7)
        pyplot.plot(clim_x, clim_y, 'k', lw=0.7)
        #pyplot.plot(x, zeros, 'w', lw=0.5)
        #pyplot.plot(x, zeros+1, 'w', lw=0.5)
        #pyplot.plot(x, zeros-1, 'w', lw=0.5)

        pyplot.xlim([x[-1] - window, x[-1]])
        #pyplot.ylim([-1.5, 4.0])

        if vmins.get(field, False):
            pyplot.ylim([vmins[field], vmaxs[field]])

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    axis_color='white'
    #ax.spines['bottom'].set_color(axis_color)
    #ax.spines['top'].set_color(axis_color)
    #ax.spines['right'].set_color(axis_color)
    #ax.spines['left'].set_color(axis_color)
    #ax.tick_params(axis='both', colors=axis_color)
    #ax.yaxis.label.set_color(axis_color)
    #ax.xaxis.label.set_color(axis_color)
    ax.title.set_color(axis_color)

    fig, ax = set_axes_alpha(fig, ax, alpha=alpha,axes_color=axis_color)

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

    pyplot.xlim([x[0], x[-1]])
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


def add_cbar(fig, ax=None, field='thetao_con'):
    # Add the cbar at the bottom.
    if ax == None:
        ax = fig.add_axes([0.85, 0.2, 0.015, 0.6]) # (left, bottom, width, height)

    pyplot.sca(ax)
    x = np.array([[0., 1.], [0., 1.]])

    cmap = cm.get_cmap(cmaps.get(field, 'viridis'), cm_bins)    # 11 discrete colors

    img = pyplot.pcolormesh(x,x,x,
                            cmap=cmap,
                            vmin=cbar_vmin.get(field, 0.),
                            vmax=cbar_vmax.get(field, 1.))
    img.set_visible(False)
    ax.set_visible(False)
    
    #ax_cbar = fig.add_axes([0.85, 0.2, 0.05, 0.6]) # (left, bottom, width, height)
    axc = fig.add_axes([0.925, 0.2, 0.02, 0.6]) # (left, bottom, width, height)
    cbar = pyplot.colorbar(cax=axc, orientation="vertical", extend='both', )
    cbar.set_label(label='SST, '+r'$\degree$'+'C', color='white', size=14 )#, weight='bold')
    cbar.ax.tick_params(color='white', labelcolor='white')

    return ax


def add_mpa(ax, linewidth=2.1, draw_study_region=False, heatwaves=0):
    """
    Add the MPA circle and study region square.
    """
    # whole MPA
    proj = ccrs.PlateCarree()
    # proj = ccrs.Geodetic()
    
    ax.add_patch(mpatches.Circle(xy=[central_longitude, central_latitude, ], linewidth=1.5,
            radius=mpa_radius, ec='k', fc=(0., 0., 0., 0.), transform=proj, zorder=31))    

    norm = matplotlib.colors.Normalize(vmin=1, vmax=3.5)

    for hwl, hwl_value in enumerate(heatwaves):

        if hwl_value <= 1.: 
            #no heatwave for 1 degree
            continue

        circle_alpha = np.max([1 - 0.03*hwl, 0.05])
        rgba_color = cm.hot_r(norm(hwl_value), 5 )#, bytes=True) 
        mpa_circle_colour = (rgba_color[0], rgba_color[1], rgba_color[2], circle_alpha)

        lw = np.max([2. - 0.05*hwl, 0.1])
        rad = mpa_radius + 0.1 + (0.25 * hwl**0.85)


        ax.add_patch(mpatches.Circle(xy=[central_longitude, central_latitude, ], 
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



def plot_globe(ax, nc=None, t=None, quick=True, field = 'thetao_con'):
    pyplot.sca(ax)
    if quick:
        ax.add_feature(cfeature.OCEAN, zorder=0)
        ax.add_feature(cfeature.LAND, zorder=0, edgecolor='black')
    else:
        #c = Dataset(bathy_fn, 'r')
        binning=1
        lats = nc.variables['nav_lat'][::binning]
        lons = nc.variables['nav_lon'][::binning]

        data = nc.variables[field][t, 0, ::binning, ::binning]

#        data = np.ma.masked_where(data.mask + data < -1.80, data)

        cmap = cm.get_cmap(cmaps.get(field, 'viridis'), cm_bins)    # 11 discrete colors

        pyplot.pcolormesh(
                    lons,
                    lats,
                    data,
                    #transform=proj,
                    transform=ccrs.PlateCarree(),
                    cmap=cmap,
                    vmin=cbar_vmin.get(field, 0.),
                    vmax=cbar_vmax.get(field, 1.),
                    )
        ax.coastlines()
        ax.add_feature(cfeature.LAND, edgecolor='black', facecolor=land_color, linewidth=0.5, zorder=9)

    ax.set_global()
    ax.gridlines()
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
    #today_clim = datas_dict[date_key] - clim_stuff[0][(date_key[1], date_key[2])]
    #if today_clim < threshold:
    #   # no heatwave
    #    return 0

    target_dt = dates_dict[date_key]
    target_decimal = decimal_year(target_dt, date_key[0], date_key[1], date_key[2])

    dates_list = []
    for dkey, date in dates_dict.items():
        dcy = decimal_year(date, dkey[0], dkey[1], dkey[2])
        if dcy> target_decimal: continue
        if np.abs(dcy-target_decimal) > max_heatwave/365.25:
            continue
        dates_list.append(dkey)

    #if len(dates_list) < 2: 
    #    return 0
    
    dates_list.sort(reverse=True)
    mhw = []
    clim = clim_stuff[0]
    for dkey in dates_list:
        diff = datas_dict[dkey] - clim[(dkey[1], dkey[2])]
        mhw.append(diff)
        #if diff < threshold:
        #    mhw.append(False)
        #else:
        #    mhw.append(True)
        # mhw+=1
    return mhw



    (clim_datas, clim_doy, clim_month) = clim_stuff

    for time_key in sorted(datas_dict.keys()):
        (year, month, day) = time_key
        dt = dates_dict[time_key]
        dcy = decimal_year(dt, time_key[0], time_key[1], time_key[2])
        if dcy > target_decimal: continue
        if dcy < target_decimal - window: continue

        dat = datas_dict[time_key]
        x.append(dcy)
        y.append(dat)
        clim_x.append(dcy)
        clim_y.append(clim_datas[(month, day)])



def daily_plot(nc, t, date_key, datas_dict, dates_dict, clim_stuff=(), clim_range=[1976,1985], model='CNRM', field='thetao_con'):
    """
    The main daily plot.
    """

    fn = folder('images/daily4/')+'daily'

    date_string = '-'.join([mnStr(tt) for tt in date_key])
    #print(dates_dict.keys())
    dt = dates_dict[field][date_key]
    (year, month, day) = date_key
    dct = decimal_year(dt, year,month,day)

    fn = ''.join([fn, '_', date_string])+ '.png'
    if os.path.exists(fn):
        return

    fig = pyplot.figure(facecolor='black')
    dpi = 100. 
    video_format = '720p' # 'HD' #self.globals.get('image_res','HD')
    if video_format == '720p':
        image_size = [1280, 720]

    if video_format == 'HD':
        image_size = [1920., 1280.]
        dpi = 250.
    if video_format == '4K':
        image_size = [3840., 2160.]
        dpi = 390.

    # panning points:
    pan={# name:           [  X,     Y,    L,    B,    W,    H ]
        'vvfar_out':       [ -24.,  -20.,  0.3,  0.3,  0.6,  0.6 ],         
        'vfar_out':        [ -28.,  -28.,  0.3,  0.2,  0.7,  0.7 ],        
        'far_out':         [ -25.,  -25.,  0.3,  0.1,  0.8,  0.8 ],
        'big':             [ -20.,  -20,   0.1,  -0.1, 1.2,  1.2 ],
        'vbig':            [ -10.,  -10,   0.0,  -0.3, 1.6,  1.6 ],
        'vbig_low':        [ -30,   -7.,   -0.15, -0.7, 1.6,  1.6 ],
        'vbig_low1':       [ -20,   -17.,   -0.15, -0.7, 1.96,  1.96 ],
        'vbig_low2':       [ -30,   -27.,   -0.15, -0.7, 1.6,  1.6 ],
        'vbig_ai':         [ central_latitude,   central_longitude,   -0.15, -0.3, 1.6,  1.6 ],
        'vvbig_ai':        [ central_latitude-1,   central_longitude+2,   -0.35, -0.45, 1.9,  1.9 ],
        'vvvbig_ai':       [ central_latitude-2,   central_longitude+5,  -0.5, -0.5, 2.2,  2.2 ],

    }
    pan_years = {
        1970.: 'far_out',
        1976.: 'far_out',
        1978: 'big', 
        1980: 'vbig',
        1985: 'vbig_low',
        1990: 'vbig_low1',
        2000: 'vbig_low2',
        2001: 'far_out',
        2005: 'big',
        2011: 'vbig',
        2017: 'vbig_ai',
        2021.: 'far_out',
        2024: 'vvfar_out',
        2032: 'far_out',
        2040: 'vbig_low',
        2046: 'vbig_low1',
        2050: 'vbig_low2',
        2058: 'big', 
        2066: 'vbig', 
        2068: 'vbig_ai',
        2069: 'vvbig_ai',
        2070: 'vvvbig_ai',
        2071.: 'vvfar_out',
        }

    ortho_path_x = {yr: pan[value][0] for yr, value in pan_years.items()}
    ortho_path_y = {yr: pan[value][1] for yr, value in pan_years.items()}
    axes_path_L = {yr: pan[value][2] for yr, value in pan_years.items()}
    axes_path_B = {yr: pan[value][3] for yr, value in pan_years.items()}
    axes_path_W = {yr: pan[value][4] for yr, value in pan_years.items()}
    axes_path_H = {yr: pan[value][5] for yr, value in pan_years.items()}

    ortho_x = calc_midoint(dct, ortho_path_x)
    ortho_y = calc_midoint(dct, ortho_path_y)
    axes_L = calc_midoint(dct, axes_path_L)
    axes_B = calc_midoint(dct, axes_path_B)
    axes_W = calc_midoint(dct, axes_path_W)
    axes_H = calc_midoint(dct, axes_path_H)

    ortho_pro=ccrs.Orthographic(ortho_y, ortho_x)

    heatwaves = calc_heat_wave(datas_dict['thetao_con'],
                              dates_dict['thetao_con'], 
                              date_key=date_key,
                              clim_range=clim_range,
                              clim_stuff=clim_stuff['thetao_con'])

    fig.set_size_inches(image_size[0]/dpi,image_size[1]/dpi)
    ax2 = fig.add_axes([axes_L, axes_B, axes_W, axes_H], projection=ortho_pro) # (left, bottom, width, height)
    quick=False
    ax2 = plot_globe(ax2, nc=nc, t=t, quick=quick)

    #if datas_dict['thetao_con'][date_key]
    ax2 = add_mpa(ax2, heatwaves=heatwaves)

    if not quick:
        add_cbar(fig, field=field) #ax_cbar)

    # time series axes
    # Music is:
    # piano: temperature anomaly:
    # synth bass: Ptot_c
    # tops: pH
    # high Synth: Ztot
    
    # Temperature
    axes_alphas = {
        'thetao_con': {1970: 0., 1976.:0., 1976.5:1, 2075:1., 2076: 0.},
        'O3_pH': {1970: 0., 1976.:0., 1976.5:1, 2075:1.,  2076: 0.},
        'Ptot_c_result': {1970: 0., 1976.:0., 1976.5:1, 2075:1., 2076: 0.},
        'Ztot_c_result': {1970: 0., 1976.:0., 1976.5:1, 2075:1.,  2076: 0.},
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

    left = 0.08
    width=0.3
    y_off = 0.005
    y_height = 0.165
    # temperature
    if year in active_time_ranges['thetao_con']:
        ax3 = fig.add_axes([left, 0.7+y_off, width, y_height]) # (left, bottom, width, height) 
        ax3.set_facecolor((1., 1., 1., 0.)) # transparent
        alpha_ax3 = calc_midoint(dct, axes_alphas['thetao_con'])
        plot_past_year_just_anom_ts(datas_dict['thetao_con'], dates_dict['thetao_con'], 
                                    target_time_key=date_key, window=2.5, clim_range=clim_range,
                                    clim_stuff=clim_stuff['thetao_con'],
                                    plot_type='anom',
                                    field='thetao_con',
                                    fig=fig, ax=ax3, alpha=alpha_ax3)
        ax3.text(0.02, 0.05, 'Temperature Anomaly', color='white', transform=ax3.transAxes)

    # pH
    if year in active_time_ranges['O3_pH']:
        ax4 = fig.add_axes([left, 0.5+y_off, width, y_height]) # (left, bottom, width, height) 
        ax4.set_facecolor((1., 1., 1., 0.)) # transparent
        alpha_ax4 = calc_midoint(dct, axes_alphas['O3_pH'])
        plot_past_year_just_anom_ts(datas_dict['O3_pH'], dates_dict['O3_pH'],
                                    target_time_key=date_key, window=2.5, clim_range=clim_range,
                                    clim_stuff=clim_stuff['O3_pH'],
                                    plot_type='ts',
                                    field='O3_pH',
                                    fig=fig, ax=ax4, alpha=alpha_ax4)
        ax4.text(0.02, 0.05, 'pH', color='white', transform=ax4.transAxes)

    # Ptot_c_result
    if year in active_time_ranges['Ptot_c_result']:
        ax5 = fig.add_axes([left, 0.3+y_off, width, y_height]) # (left, bottom, width, height) 
        ax5.set_facecolor((1., 1., 1., 0.)) # transparent
        alpha_ax5 = calc_midoint(dct, axes_alphas['Ptot_c_result'])
        plot_past_year_just_anom_ts(datas_dict['Ptot_c_result'], dates_dict['Ptot_c_result'], 
                                    target_time_key=date_key, window=2.5, clim_range=clim_range,
                                    clim_stuff=clim_stuff['Ptot_c_result'],
                                    field='Ptot_c_result',
                                    plot_type='ts',
                                    fig=fig, ax=ax5, alpha=alpha_ax5)
        ax5.text(0.02, 0.05, 'Phytoplankton', color='white', transform=ax5.transAxes)

    # Ztot_c_result
    if year in active_time_ranges['Ztot_c_result']:
        ax6 = fig.add_axes([left, 0.1+y_off, width, y_height]) # (left, bottom, width, height) 
        ax6.set_facecolor((1., 1., 1., 0.)) # transparent
        alpha_ax6 = calc_midoint(dct, axes_alphas['Ztot_c_result'])
        plot_past_year_just_anom_ts(datas_dict['Ztot_c_result'], dates_dict['Ztot_c_result'], 
                                    target_time_key=date_key, window=2.5, clim_range=clim_range,
                                    clim_stuff=clim_stuff['Ztot_c_result'],
                                    field='Ztot_c_result',
                                    plot_type='ts',
                                    fig=fig, ax=ax6, alpha=alpha_ax6)
        ax6.text(0.02, 0.05, 'Zooplankton', color='white', transform=ax6.transAxes)

    #ax7 = fig.add_axes([0.1, 0.05, 0.3, 0.]) # (left, bottom, width, height) 
    #pyplot.plot()

    font = {'family': 'monospace',
            'color':  'white',
            #'weight': 'bold',
            'size': 14,
            }
    fig.text(left, 0.9, date_string, fontdict=font)

    print('saving:', date_string, fn)  
    pyplot.savefig(fn, dpi=dpi)
    pyplot.close()
    #assert 0
    # end of daily_plot


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


def iterate_daily_plots(nc, datas_dict, dates_dict, clim_stuff=(), clim_range=[1976,1985], field='thetao_con'):

    times = nc.variables['time_centered']
    dates = num2date(times[:], times.units, calendar=times.calendar)

    for t, dt in enumerate(dates[:]):
        date_key = (dt.year, dt.month, dt.day)
        daily_plot(nc, t, date_key, datas_dict, dates_dict, clim_stuff=clim_stuff, clim_range=clim_range)
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


def get_file_list(model, field, ssp='ssp370'):
    """
    Get a list of netcdfile file paths.
    """
    path_hist = get_paths()[model+'_hist']
    path_ssp = get_paths()[model+'_'+ssp]

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

    files = glob(path_hist+'*/*/'+ suffix)
    files.extend(glob(path_ssp+'*/*/'+ suffix))
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


def make_daily_plots(model='CNRM', clim_range=[1976,1985]):
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

    for fn in sorted(temp_files)[-15:]:
        nc = Dataset(fn, 'r')
        iterate_daily_plots(nc, datas_dict, dates_dict, clim_stuff=clim_stuff, clim_range=clim_range, field='thetao_con')
        nc.close()

    return


def  run_all():
    failures = {}
    make_daily_plots()
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

