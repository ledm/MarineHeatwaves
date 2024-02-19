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

cbar_vmin = {
    'thetao_con': 22.5,
    }
cbar_vmax = {
    'thetao_con': 32.0,
    }

cmaps = {
    'thetao_con': 'viridis',
#     'so_abs': 'viridis'
#    'O3_TA': 2500., 
#    'N3_n': 20., 
#    'O3_c': 2500.,   
#    'uo': 1.,
#    'vo': 1.,     
    }
land_color = '#F5F5F5' #'#DCDCDC' # '#E0E0E0' ##F8F8F8'


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


def plot_past_year_just_anom_ts(datas_dict, dates_dict, 
                                target_time_key=(2000,1,1), 
                                window=2.5, # years
                                clim_range=[1976,1985],
                                alpha=1.,
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
    clim_datas, clim_doy, clim_month = calculate_clim(datas_dict, dates_dict, clim_range=clim_range)

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
        print()
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
                facecolor='#e2062c', # candy apple red
                )
    pyplot.plot(clim_x, np.array(y) - np.array(clim_y))
    #pyplot.title(target_string)
    pyplot.plot(x, zeros, 'w', lw=0.5)
    pyplot.plot(x, zeros+1, 'w', lw=0.5)
    pyplot.plot(x, zeros-1, 'w', lw=0.5)

    pyplot.xlim([x[-1] - window, x[-1]])
    pyplot.ylim([-1.5, 4.0])

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
    pyplot.ylim([-1.5, 4.0])
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
    img = pyplot.pcolormesh(x,x,x,
                            cmap=cmaps.get(field, 'viridis'),
                            vmin=cbar_vmin.get(field, 0.),
                            vmax=cbar_vmax.get(field, 1.))
    img.set_visible(False)
    ax.set_visible(False)
    
    #ax_cbar = fig.add_axes([0.85, 0.2, 0.05, 0.6]) # (left, bottom, width, height)
    axc = fig.add_axes([0.925, 0.2, 0.02, 0.6]) # (left, bottom, width, height)

    #cax = pyplot.axes([0.05, 0.06, 0.035, 0.9, ]) # Left, bottom, width, heigh (figure )
    pyplot.colorbar(cax=axc, orientation="vertical", extend='both', label='SST, '+r'$\degree$'+'C')
    return ax


def add_mpa(ax, linewidth=1.7, draw_study_region=False):
    """
    Add the MPA circle and study region square.
    """
    # whole MPA
    proj = ccrs.PlateCarree()
    # proj = ccrs.Geodetic()
    mpa_circle_colour='k'
    alpha=0.
    ax.add_patch(mpatches.Circle(xy=[central_longitude, central_latitude, ], linewidth=linewidth,
        radius=mpa_radius, ec=mpa_circle_colour, fc=(0., 0., 0., alpha), transform=proj, zorder=30))
    #AI
    #ax.add_patch(mpatches.Circle(xy=[central_longitude, central_latitude, ],
    #    radius=mpa_radius/50., ec='black', fc=(1.,1.,1.,0.), transform=proj, zorder=30))

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

        pyplot.pcolormesh(
                    lons,
                    lats,
                    data,
                    #transform=proj,
                    transform=ccrs.PlateCarree(),
                    cmap=cmaps.get(field, 'viridis'),
                    vmin=cbar_vmin.get(field, 0.),
                    vmax=cbar_vmax.get(field, 1.),
                    )
        ax.coastlines()
        ax.add_feature(cfeature.LAND, edgecolor='black', facecolor=land_color, linewidth=0.5, zorder=9)

    ax.set_global()
    ax.gridlines()
    add_mpa(ax)

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



def daily_plot(nc, t, date_key, datas_dict, dates_dict, clim_range=[1976,1985], model='CNRM', field='thetao_con'):
    """
    The main daily plot.
    """

    fn = folder('images/daily/')+'daily'

    date_string = '-'.join([str(tt) for tt in date_key])
    dt = dates_dict[date_key]
    (year, month, day) = date_key
    dct = decimal_year(dt, year,month,day)


    fn = ''.join([fn, '_', date_string])+ '.png'

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

    fig.set_size_inches(image_size[0]/dpi,image_size[1]/dpi)

    # big globe
    """
    # Panning route:
    ortho_path_x = {
            1976.: -25.,
            1978: -12,
            1980: 1,
            1985: -15,
            1995: -7.,
            2005: -25.
            }
    ortho_path_y = {
            1976.: -25.,
            1978: -12, 
            1980: -10,
            1982: -17,
            1985: -5,
            1995: -13,
            2005: -25.,
            }

    axes_path_L= { 
            1976: 0.4,
            1978.: 0.3, 
            1980.: 0.15,
            1982: 0.25,
            1986: 0.2,
            1995: 0.1,
            2005:0.05,
            }
    axes_path_B= { 
            1976.: 0.05, 
            1980.: -0.1,
            1985: -0.2,
            2005: -0.1
            }
    axes_path_W= {
            1976.: 0.7, 
            1980.: 0.9,
            1985: 1.,
            2005: 1.1,
            }
    axes_path_H= {
            1976.: 1., 
            1980.: 1.2, 
            1985: 1.4,
            2005: 1.1,
            }
    """
    # panning points:
    pan={# name:           [  X,     Y,    L,    B,    W,    H ]
        'far_out':         [ -25.,  -25.,  0.3,  0.1,  0.8,  0.8 ],
        'big':             [ -20.,  -20,   0.1,  -0.1, 1.2,  1.2 ],
        'vbig':            [ -10.,  -10,   0.0,  -0.3, 1.6,  1.6 ],
        'vbig_low':        [ -30,   -7.,   -0.15, -0.7, 1.6,  1.6 ],
        'vbig_low1':        [ -20,   -17.,   -0.15, -0.7, 1.96,  1.96 ],
        'vbig_low2':        [ -30,   -27.,   -0.15, -0.7, 1.6,  1.6 ],
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
        2005: 'big'
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

    ax2 = fig.add_axes([axes_L, axes_B, axes_W, axes_H], projection=ortho_pro) # (left, bottom, width, height)
    quick=False
    ax2 = plot_globe(ax2, nc=nc, t=t, quick=quick)
    if not quick:
        add_cbar(fig, field=field) #ax_cbar)

    # time series axes
    
    ax3 = fig.add_axes([0.1, 0.7, 0.3, 0.2]) # (left, bottom, width, height) 
    ax3.set_facecolor((1., 1., 1., 0.)) # transparent

    alpha_ax3 = calc_midoint(dct, {1970: 0., 1976.:0., 1976.5:1, 2075:1.})

    plot_past_year_just_anom_ts(datas_dict, dates_dict, target_time_key=date_key, window=2.5, clim_range=clim_range,
                fig=fig, ax=ax3,alpha=alpha_ax3)

    print('saving:', date_string, fn)  
    pyplot.savefig(fn,dpi=dpi)
    pyplot.close()


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

    """
    if np.sum(down_where):
        pyplot.fill_between(x,
            y,
            yb,
            where = down_where * np.ma.masked_less_equal(y, clim_y-1).mask,
            facecolor=c,
            )

        for a, c in zip([1., 2., 3.], ['#b8dbff' ,'#b8dbff', '#b8dbff']): # https://html-color.codes/color-names/dodger-blue
            
            topclim = y_clim - a + 1
            botclim = y_clim - a
            datline = y
            
            toplinne = np.ma.masked_where(topclim< datline , topclim) 

            ya = [np.max([cl+a-step, yy]) for cl, yy in zip(y_clim, y)]

            yb = [np.max([cl-a+step, yy]) for cl, yy in zip(y_clim, y) if down_where]

            pyplot.fill_between(x, 
                y, 
                yb,
                where = down_where * np.ma.masked_less_equal(y, clim_y-1).mask,
                facecolor=c,
                )
        if np.sum(up_where):
            pyplot.fill_between(x, 
                y-a, 
                clim_y, 
                where = up_where, 
                facecolor='red',
                alpha=1-a/3.)
    """
    pyplot.title(field + ' '+str(plot_year))
    pyplot.legend()
    pyplot.xlim([x[0], x[-1]])
    pyplot.ylim([vmins.get(field, 0.), vmaxs.get(field, 1.)])

    fn = folder('images/anom_year/'+field)+field+'_anom_'+str(plot_year)+'.png'
    print('Saving', fn)
    pyplot.savefig(fn)
    pyplot.close()






def iterate_daily_plots(nc, datas_dict, dates_dict, clim_range=[1976,1985], field='thetao_con'):

    times = nc.variables['time_centered']
    dates = num2date(times[:], times.units, calendar=times.calendar)

    for t, dt in enumerate(dates):

        date_key = (dt.year, dt.month, dt.day)
        daily_plot(nc, t, date_key, datas_dict, dates_dict, clim_range=clim_range)
        return


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

    #find_corners(files[0])
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

    return
    for year in range(1976, 2071):
        #continue 
        print(year)
        plot_single_year_ts(datas_dict, dates_dict, field=field, plot_year=year,)
        plot_single_year_just_anom_ts(datas_dict, dates_dict, field=field, plot_year=year, clim_range=clim_range)
        plot_single_year_anomaly_ts(datas_dict, dates_dict, field=field, plot_year=year, clim_range=clim_range)
   
    print('Failures:', failures)

    temp_files = get_file_list(model, 'thetao_con', ssp='ssp370')
    shelvefn = folder('shelves/') + model + '_' + 'thetao_con' + '.shelve'
    temp_finished_files, temp_datas_dict, temp_dates_dict = load_shelve(shelvefn)    
    for fn in sorted(temp_files)[:12]:
        nc = Dataset(fn, 'r')
        iterate_daily_plots(nc, temp_datas_dict, temp_dates_dict, clim_range=clim_range, field='thetao_con')
        nc.close()

        #return
    return failures


def  run_all():
    failures = {}
    #main(field='O3_pH') 
    #return
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

