

# =============================================================================
# region import packages


# basic library
from satpy.scene import Scene
from datetime import datetime
import numpy as np
import xarray as xr
import os
import glob
import pickle
import gc

import sys  # print(sys.path)
sys.path.append(
    '/Users/gao/OneDrive - whu.edu.cn/ETH/Courses/4. Semester/DEoAI')
sys.path.append('/project/pr94/qgao/DEoAI')
sys.path.append('/scratch/snx3000/qgao')


# plot
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mpcolors
from matplotlib.legend import Legend
from matplotlib.patches import Rectangle
import matplotlib.ticker as mticker
from matplotlib import font_manager as fm
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import BoundaryNorm
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib.animation as animation

fontprop_tnr = fm.FontProperties(
    fname='/project/pr94/qgao/DEoAI/data_source/TimesNewRoman.ttf')
# mpl.rcParams['font.family'] = fontprop_tnr.get_name()
mpl.rcParams['figure.dpi'] = 600
mpl.rc('font', family='Times New Roman', size=10)
mpl.rcParams['backend'] = 'Qt4Agg'  #
# mpl.get_backend()

plt.rcParams.update({"mathtext.fontset": "stix"})
plt.rcParams["font.serif"] = ["Times New Roman"]

import cartopy as ctp
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from DEoAI_analysis.module.mapplot import (
    ticks_labels,
    scale_bar,
    framework_plot,
)


# data analysis
import pandas as pd
import xesmf as xe
import metpy.calc as mpcalc
from metpy.interpolate import cross_section
from metpy.units import units
from haversine import haversine
from scipy import interpolate
import dask
dask.config.set({"array.slicing.split_large_chunks": False})
import rasterio as rio

from DEoAI_analysis.module.namelist import (
    month,
    seasons,
    months,
    years,
    years_months,
    timing,
    quantiles,
    folder_1km,
    g,
    m,
    r0,
    cp,
    r,
    r_v,
    p0sl,
    t0sl,
    extent1km,
    extent3d_m,
    extent3d_g,
    extent3d_t,
    extentm,
    extentc,
    extent12km,
    extent1km_lb,
    ticklabel1km,
    ticklabelm,
    ticklabelc,
    ticklabel12km,
    ticklabel1km_lb,
    transform,
    coastline,
    borders,
)

from DEoAI_analysis.module.statistics_calculate import(
    get_statistics
)

from DEoAI_analysis.module.spatial_analysis import(
    rotate_wind
)

# endregion

# region mask outside region

daily_pre_1km_sim = xr.open_dataset(
    "scratch/precipitation/daily_pre_1km_sim.nc")
lon1 = daily_pre_1km_sim.lon.values
lat1 = daily_pre_1km_sim.lat.values
analysis_region = np.zeros_like(lon1)
analysis_region[80:920, 80:920] = 1
cs = plt.contour(lon1, lat1, analysis_region, levels=np.array([0.5]))
poly_path = cs.collections[0].get_paths()[0]

mask_lon = np.arange(-23.5, -11.2, 0.02)
mask_lat = np.arange(24.1, 34.9, 0.02)
mask_lon2, mask_lat2 = np.meshgrid(mask_lon, mask_lat)
coors = np.hstack((mask_lon2.reshape(-1, 1), mask_lat2.reshape(-1, 1)))
mask = poly_path.contains_points(coors).reshape(
    mask_lon2.shape[0], mask_lon2.shape[1])
masked = np.ones_like(mask_lon2)
masked[mask] = np.nan

ascat_bd = True
if ascat_bd:
    file = 'scratch/ascat_hires_winds0/ascat_20100805_103000_metopa_19687_srv_o_063_ovw.nc'
    ncfile = xr.open_dataset(file)
    lon = ncfile.lon.values
    lat = ncfile.lat.values
    
    middle_i = int(lon.shape[1]/2)
    lon_m = lon.copy()
    lon_m[lon_m >= 180] = lon_m[lon_m >= 180] - 360
    lon_s2 = lon_m[:, middle_i:]
    lat_s2 = lat[:, middle_i:]
    # fil_rvor2 = median_filter(rvor2, 3, )
    coors_s2 = np.hstack((lon_s2.reshape(-1, 1), lat_s2.reshape(-1, 1)))
    mask_s2 = poly_path.contains_points(coors_s2).reshape(
        lon_s2.shape[0], lon_s2.shape[1])
    masked_s2 = np.zeros_like(lon_s2)
    masked_s2[mask_s2] = 1
    masked_s2[:, 0] = 0
    masked_s2[:, -1] = 0

# endregion
# =============================================================================


# =============================================================================
# region plot modis data

# Red: Band 1: 620 - 670 nm
# Green: Band 4: 545 - 565 nm
# Blue: Band 3: 459 - 479 nm

import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import numpy as np
from pyhdf.SD import SD, SDC
from pprint import pprint

def si2reflectance(scaled_integers, scales, offsets, gamma = 2.2):
    
    si = np.ma.array(scaled_integers)
    si[si > 32767] = np.ma.masked
    reflectance = scales * (si - offsets)
    
    # Apply range limits for each channel. RGB values must be between 0 and 1
    reflectance = np.clip(reflectance, 0, 1)
    
    # Apply a gamma correction to the image to correct ABI detector brightness
    reflectance = np.power(reflectance, 1/gamma)
    
    return reflectance


def modis_l1b_1km(filename):
    hdf = SD(filename, SDC.READ)
    # hdf.datasets()
    
    EV_250_Aggr1km_RefSB = hdf.select('EV_250_Aggr1km_RefSB')
    EV_500_Aggr1km_RefSB = hdf.select('EV_500_Aggr1km_RefSB')
    # pprint(EV_250_Aggr1km_RefSB.attributes())
    # pprint(EV_500_Aggr1km_RefSB.attributes())
    
    red_reflectance = si2reflectance(
        EV_250_Aggr1km_RefSB[0, :, :],
        scales=EV_250_Aggr1km_RefSB.attributes()['reflectance_scales'][0],
        offsets=EV_250_Aggr1km_RefSB.attributes()['reflectance_offsets'][0],
    )
    
    green_reflectance = si2reflectance(
        EV_500_Aggr1km_RefSB[1, :, :],
        scales=EV_500_Aggr1km_RefSB.attributes()['reflectance_scales'][1],
        offsets=EV_500_Aggr1km_RefSB.attributes()['reflectance_offsets'][1],
    )
    
    blue_reflectance = si2reflectance(
        EV_500_Aggr1km_RefSB[0, :, :],
        scales=EV_500_Aggr1km_RefSB.attributes()['reflectance_scales'][0],
        offsets=EV_500_Aggr1km_RefSB.attributes()['reflectance_offsets'][0],
    )
    
    rgb = np.dstack([red_reflectance, green_reflectance, blue_reflectance])
    color_tuples = np.array(
        [red_reflectance.flatten(), green_reflectance.flatten(),
         blue_reflectance.flatten()]).transpose()
    
    scn = Scene(filenames={'modis_l1b': [filename]})
    scn.load(["longitude", "latitude"])
    lon = scn["longitude"].values
    lat = scn["latitude"].values
    
    return {'rgb' : rgb,
            'color_tuples' : color_tuples,
            'red_reflectance' : red_reflectance,
            'green_reflectance' : green_reflectance,
            'blue_reflectance' : blue_reflectance,
            'lon' : lon,
            'lat' : lat,
            }


# outmask = False
# filename = 'scratch/modis/20100801_0809MOD_MYD_021KM/MOD021KM.A2010217.1145.061.2017255034847.hdf'
# outputfile = 'figures/05_clouds/5.1.0 MOD021KM_2010217_1145.png'
# timepoint = '2010-08-05 11:45 UTC'
# filename = 'data_source/modis/MOD021KM.A2010226.1140.061.2017255192030.hdf'
# outputfile = 'figures/05_clouds/5.1.1 MOD021KM_2010226_1140.png'
# timepoint = '2010-08-14 11:40 UTC'

outmask = True
# filename = 'scratch/modis/20100801_0809MOD_MYD_021KM/MOD021KM.A2010217.1145.061.2017255034847.hdf'
# outputfile = 'figures/05_clouds/5.3.0 MOD021KM_2010217_1145_withmask.png'
# timepoint = '2010-08-05 11:45 UTC'
filename = 'data_source/modis/MOD021KM.A2010226.1140.061.2017255192030.hdf'
outputfile = 'figures/05_clouds/5.3.1 MOD021KM_2010226_1140_withmask.png'
timepoint = '2010-08-14 11:40 UTC'

modis_l1b = modis_l1b_1km(filename)

fig, ax = framework_plot("1km_lb", figsize=np.array([8.8, 8.8]) / 2.54,)
ma_lon = modis_l1b['lon']; ma_lat = modis_l1b['lat']
ma_lon[np.isnan(ma_lon)] = 999; ma_lat[np.isnan(ma_lat)] = 999

im = ax.pcolormesh(modis_l1b['lon'], modis_l1b['lat'],
                   modis_l1b['red_reflectance'],
                   color=modis_l1b['color_tuples'],
                   transform=transform,
                   shading='auto', rasterized=True)

filename1 = 'data_source/modis/MOD021KM.A2010226.1135.061.2017255191955.hdf'
modis_l1b1 = modis_l1b_1km(filename1)
im1 = ax.pcolormesh(modis_l1b1['lon'], modis_l1b1['lat'],
                    modis_l1b1['red_reflectance'],
                    color=modis_l1b1['color_tuples'],
                    transform=transform,
                    shading='auto', rasterized=True)

if outmask:
    ax.contourf(
        mask_lon2, mask_lat2, masked,
        colors='white', levels=np.array([0.5, 1.5]))

ax.text(-23, 34, timepoint)
ax.set_xlabel('True Color Image from MODIS Terra product \n MOD021KM (Band 1, 4, 3)')
fig.subplots_adjust(left=0.12, right=0.99, bottom=0.15, top=0.99)
fig.savefig(outputfile, dpi=600)


'''
# https://unidata.github.io/python-training/gallery/mapping_goes16_truecolor/

# im.set_array(None)

dataset = rio.open(
    '/project/pr94/qgao/DEoAI/scratch/modis/20100801_09MOD021KM/MOD021KM.A2010215.1200.061.2017255144812.gscs_000501505242.hdf')
for name in dataset.subdatasets:
    print(name)
dataset.meta

'''

# endregion
# =============================================================================


# =============================================================================
# region clouds plot from EUMETSAT-MSG meteosat 9

# 12->0, 27->15, 42->30, 57->45
# outmask = False
# inputfile = sorted(glob.glob(
#     "data_source/eumetsat/1448098/*201008051057*201008*.nat"))
# outputfile = 'figures/05_clouds/5.5.0 EUMETSAT_MSG_meteosat9_rgb_20100805_1045.png'
# inputfile = sorted(glob.glob(
#     "data_source/eumetsat/1448098/*201008052212*201008*.nat"))
# outputfile = 'figures/05_clouds/5.5.1 EUMETSAT_MSG_meteosat9_rgb_20100805_2200.png'

outmask = True
inputfile = sorted(glob.glob(
    "data_source/eumetsat/1448098/*201008051057*201008*.nat"))
outputfile = 'figures/05_clouds/5.4.0 EUMETSAT_MSG_meteosat9_rgb_20100805_1045_withmask.png'
# inputfile = sorted(glob.glob(
#     "data_source/eumetsat/1448098/*201008052212*201008*.nat"))
# outputfile = 'figures/05_clouds/5.4.1 EUMETSAT_MSG_meteosat9_rgb_20100805_2200_withmask.png'


global_scene = Scene(reader="seviri_l1b_native", filenames=[inputfile[0]])
# global_scene.available_dataset_names()

gamma = 2.2
if (global_scene.start_time.hour > 7) & (global_scene.start_time.hour < 20):
    composite = 'natural_color'
    global_scene.load([composite], upper_right_corner='NE')
    lon, lat = global_scene[composite].attrs['area'].get_lonlats()
    color_tuples = np.array(
        [global_scene[composite][0, :, :].values.flatten(),
         global_scene[composite][1, :, :].values.flatten(),
         global_scene[composite][2, :, :].values.flatten()]).transpose()
    color_tuples = np.power(color_tuples / 100, 1/gamma)
    color_tuples = np.clip(color_tuples, 0, 1)
else:
    # eumetsat suggested rgb colors
    global_scene.load(['IR_039', 'IR_108', 'IR_120'],
                      upper_right_corner='NE')
    lon, lat = global_scene['IR_108'].attrs['area'].get_lonlats()
    r_beam = global_scene['IR_120'].values - global_scene['IR_108'].values
    g_beam = global_scene['IR_108'].values - global_scene['IR_039'].values
    b_beam = global_scene['IR_108'].values
    r_beam_c = np.clip((r_beam + 4)/6, 0, 1)
    g_beam_c = np.clip(g_beam/10, 0, 1)
    b_beam_c = np.clip((b_beam - 243)/(293 - 243), 0, 1)
    color_tuples = np.array(
        [r_beam_c.flatten(), g_beam_c.flatten(), b_beam_c.flatten()]
    ).transpose()

fig, ax = framework_plot(
    "1km_lb", figsize=np.array([8.8, 8.8]) / 2.54, border_color = 'white', grid_color = 'white',)
im = ax.pcolormesh(
    lon, lat, np.zeros_like(lon), color=color_tuples,
    transform=transform, rasterized=True, shading='auto')

if outmask:
    ax.contourf(
        mask_lon2, mask_lat2, masked,
        colors='white', levels=np.array([0.5, 1.5]))
    ax.text(-23, 34, str(global_scene.start_time)[0:16] + ' UTC',)
    scale_bar(
        ax, bars=2, length=200, location=(0.02, 0.015), barheight=20,
        linewidth=0.15, col='black', middle_label=False)
else:
    ax.text(-23, 34, str(global_scene.start_time)[0:16] + ' UTC',
            backgroundcolor='white')

if ascat_bd:
    ax.contour(
        lon_s2, lat_s2, masked_s2, colors='m', levels=np.array([0.5]),
        linewidths=0.5, linestyles='solid')

ax.set_xlabel(
    'Satellite images from EUMETSAT MSG \n Rectified (level 1.5) Meteosat-9 SEVIRI image data')
fig.subplots_adjust(left=0.12, right=0.99, bottom=0.15, top=0.99)
fig.savefig(outputfile, dpi=600)


'''
http://www.eumetrain.org/RGBguide/recipes/RGB_recipes.pdf
http://www.eumetrain.org/RGBguide/rgbs.html?page=1&sat=-1&rgb=-1&colour=-1&phenom=-1&recent=false

filenames = sorted(glob.glob(
    "/project/pr94/qgao/DEoAI/data_source/eumetsat/1447992-1of1/MSG2-SEVI-MSG15-0100-NA-20100803121241.951000000Z-20100803121254-1447992.nat"))
filenames = sorted(glob.glob(
    "data_source/eumetsat/1447992-1of1/*.nat"))

from satpy import available_readers
available_readers()
global_scene.available_composite_names()
vis006_lon, vis006_lat = global_scene["VIS006"].attrs['area'].get_lonlats()
crs = global_scene['VIS006'].attrs['area'].to_cartopy_crs()
# crs.bounds
# global_scene.load(['overview'])
# global_scene.show('overview')

# local_scene = global_scene.resample()
# local_scene.load(['hrv_clouds'])
# local_scene.show('hrv_clouds')
# global_scene["VIS006"].values.shape
# global_scene["VIS008"].values.shape
# global_scene["IR_016"].values.shape
# global_scene[composite]

global_scene.load(['natural_color_with_night_ir'], upper_right_corner='NE')
global_scene['natural_color_with_night_ir'].prerequisites


# infrared channel 10.8
gamma3 = 5
global_scene.load(['IR_108'], upper_right_corner='NE')
ir108 = global_scene['IR_108'].values
# Normalize: cleanIR = (cleanIR-minimumValue)/(maximumValue-minimumValue)
ir108 = (ir108-90)/(313-90)
ir108 = np.clip(ir108, 0, 1)
# Invert colors so that cold clouds are white
ir108 = 1 - ir108
# Lessen the brightness of the coldest clouds so they don't appear so bright
# when we overlay it on the true color image.
ir108 = ir108 / 1
color_tuples3 = np.array(
    [ir108.flatten(), ir108.flatten(), ir108.flatten()]).transpose()
color_tuples3 = np.power(color_tuples3, 1/gamma3)


# infrared channel 3.9
# https://unidata.github.io/python-training/gallery/mapping_goes16_truecolor/
gamma4 = 5
global_scene.load(['IR_039'], upper_right_corner='NE')
ir039 = global_scene['IR_039'].values
# Normalize: cleanIR = (cleanIR-minimumValue)/(maximumValue-minimumValue)
ir039 = (ir039-90)/(313-90)
ir039 = np.clip(ir039, 0, 1)
# Invert colors so that cold clouds are white
ir039 = 1 - ir039
# Lessen the brightness of the coldest clouds so they don't appear so bright
# when we overlay it on the true color image.
ir039 = ir039 / 1
color_tuples4 = np.array(
    [ir039.flatten(), ir039.flatten(), ir039.flatten()]).transpose()
color_tuples4 = np.power(color_tuples4, 1/gamma4)

'''
# endregion
# =============================================================================


# =============================================================================
# region clouds animation from EUMETSAT-MSG meteosat 9

filenames = sorted(glob.glob(
    "/project/pr94/qgao/DEoAI/data_source/eumetsat/1448098/*.nat"))

istart = 272
ifinal = 800  # 280  #

fig, ax = framework_plot(
    "1km_lb", figsize=np.array([8.8, 8.8]) / 2.54, border_color='white', grid_color='white',)
ims = []

# outmask = False
# file_input = np.arange(istart, ifinal, 4)
# outputfile = 'figures/05_clouds/5.2.4 EUMETSAT_MSG_meteosat9_rgb2010080320_0907_combined.mp4'
# file_input = np.arange(istart, ifinal)
# outputfile = 'figures/05_clouds/5.2.5 EUMETSAT_MSG_meteosat9_rgb2010080320_0907_combined_all.mp4'

outmask = True
file_input = np.arange(istart, ifinal, 4)
outputfile = 'figures/05_clouds/5.2.6 EUMETSAT_MSG_meteosat9_rgb2010080320_0907_combined_withmask.mp4'

for i in file_input:  # np.arange(istart, ifinal):
    # i=320
    global_scene = Scene(reader="seviri_l1b_native", filenames=[filenames[i]])
    # visible colors
    if (global_scene.start_time.hour > 7) & (global_scene.start_time.hour < 20):
        composite = 'natural_color'
        global_scene.load([composite], upper_right_corner='NE')
        lon, lat = global_scene[composite].attrs['area'].get_lonlats()
        color_tuples = np.array(
            [global_scene[composite][0, :, :].values.flatten(),
             global_scene[composite][1, :, :].values.flatten(),
             global_scene[composite][2, :, :].values.flatten()]).transpose()
        color_tuples = np.power(color_tuples / 100, 1/2.2)
        color_tuples = np.clip(color_tuples, 0, 1)
    else :
        # eumetsat suggested rgb colors
        global_scene.load(['IR_039', 'IR_108', 'IR_120'],
                          upper_right_corner='NE')
        lon, lat = global_scene['IR_108'].attrs['area'].get_lonlats()
        r_beam = global_scene['IR_120'].values - global_scene['IR_108'].values
        g_beam = global_scene['IR_108'].values - global_scene['IR_039'].values
        b_beam = global_scene['IR_108'].values
        r_beam_c = np.clip((r_beam + 4)/6, 0, 1)
        g_beam_c = np.clip(g_beam/10, 0, 1)
        b_beam_c = np.clip((b_beam - 243)/(293 - 243), 0, 1)
        color_tuples = np.array(
            [r_beam_c.flatten(), g_beam_c.flatten(), b_beam_c.flatten()]
            ).transpose()
    
    im = ax.pcolormesh(lon, lat,
                       np.zeros_like(lon),
                       color=color_tuples,
                       transform=transform, rasterized=True, zorder=-2,
                       shading='auto'
                       )
    
    if outmask:
        im_time = ax.text(
            -23, 34, str(global_scene.start_time)[0:16] + ' UTC',)
    else:
        im_time = ax.text(
            -23, 34, str(global_scene.start_time)[0:16] + ' UTC',
            backgroundcolor='white')
    
    ims.append([im, im_time])
    print(str(i) + '/' + str(len(filenames) - 1))

if outmask:
    ax.contourf(
        mask_lon2, mask_lat2, masked,
        colors='white', levels=np.array([0.5, 1.5]))
    scale_bar(
        ax, bars=2, length=200, location=(0.02, 0.015), barheight=20,
        linewidth=0.15, col='black', middle_label=False)

ax.set_xlabel(
    'Satellite images from EUMETSAT MSG \n Rectified (level 1.5) Meteosat-9 SEVIRI image data')
ax.set_rasterization_zorder(-1)
fig.subplots_adjust(left=0.12, right=0.99, bottom=0.15, top=0.99)
ani = animation.ArtistAnimation(fig, ims, interval=150, blit=True)
ani.save(
    outputfile,
    progress_callback=lambda i, n: print(f'Saving frame {i} of {n}'),
    dpi = 600)

# endregion
# =============================================================================


