

# =============================================================================
# region import packages


# basic library
import datetime
import numpy as np
import xarray as xr
import os
import glob
import pickle
import gc

import sys

from xarray.backends.api import open_dataset  # print(sys.path)
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
# =============================================================================


# =============================================================================
# region check era5 vertical temperature profiles

era5_example = xr.open_dataset(
    'data_source/era5/adaptor.mars.internal-1603200664.7498538-23233-12-0f7403c4-7ee1-44c2-8d5f-ab5a6d67e189.nc')

height = mpcalc.geopotential_to_height(
    era5_example.z.squeeze().values * units('meter ** 2 / second ** 2'))

era5_example.level


# plot T and theta

fig, ax = plt.subplots(1, 1, figsize=np.array([8.8, 8]) / 2.54)

plt_T, = ax.plot(
    era5_example.t[0, -11:, np.where(era5_example.latitude == 33.0)[0][0],
                   np.where(era5_example.longitude == -16.75)[0][0]],
    height[0, -11:, np.where(era5_example.latitude == 33.0)[0][0],
           np.where(era5_example.longitude == -16.75)[0][0]],
    linewidth=0.25, color='blue'
)

ax.set_xticks(np.arange(285, 316, 5))
ax.set_yticks(np.arange(0, 3001, 500))
ax.set_xticklabels(np.arange(285, 316, 5), size=8)
ax.set_yticklabels(np.arange(0, 3.1, 0.5), size=8)
ax.set_xlabel("Temperature in ERA5 [K]", size=10)
ax.set_ylabel("Height [km]", size=10)

fig.subplots_adjust(left=0.14, right=0.95, bottom=0.2, top=0.99)
fig.savefig(
    'figures/06_inversion_layer/6.0.1 inversion layer in 201008 in ERA5.png', dpi=1200)
plt.close('all')

# endregion
# =============================================================================


# =============================================================================
# region download era5 data using cdsapi
import cdsapi

c = cdsapi.Client()

# 100m wind: ERA5 hourly data on single levels from 1979 to present
# https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels?tab=form

c.retrieve(
    'reanalysis-era5-single-levels',
    {
        'product_type': 'reanalysis',
        'variable': '100m_v_component_of_wind',
        'year': [
            '2006', '2007', '2008',
            '2009', '2010', '2011',
            '2012', '2013', '2014',
            '2015',
        ],
        'month': [
            '01', '02', '03',
            '04', '05', '06',
            '07', '08', '09',
            '10', '11', '12',
        ],
        'day': [
            '01', '02', '03',
            '04', '05', '06',
            '07', '08', '09',
            '10', '11', '12',
            '13', '14', '15',
            '16', '17', '18',
            '19', '20', '21',
            '22', '23', '24',
            '25', '26', '27',
            '28', '29', '30',
            '31',
        ],
        'time': [
            '00:00', '01:00', '02:00',
            '03:00', '04:00', '05:00',
            '06:00', '07:00', '08:00',
            '09:00', '10:00', '11:00',
            '12:00', '13:00', '14:00',
            '15:00', '16:00', '17:00',
            '18:00', '19:00', '20:00',
            '21:00', '22:00', '23:00',
        ],
        'area': [
            34.88, -23.42, 24.17,
            -11.28,
        ],
        'format': 'netcdf',
    },
    'scratch/obs/era5/hourly_100m_wind_v.nc')

c.retrieve(
    'reanalysis-era5-single-levels',
    {
        'product_type': 'reanalysis',
        'variable': '100m_u_component_of_wind',
        'year': [
            '2006', '2007', '2008',
            '2009', '2010', '2011',
            '2012', '2013', '2014',
            '2015',
        ],
        'month': [
            '01', '02', '03',
            '04', '05', '06',
            '07', '08', '09',
            '10', '11', '12',
        ],
        'day': [
            '01', '02', '03',
            '04', '05', '06',
            '07', '08', '09',
            '10', '11', '12',
            '13', '14', '15',
            '16', '17', '18',
            '19', '20', '21',
            '22', '23', '24',
            '25', '26', '27',
            '28', '29', '30',
            '31',
        ],
        'time': [
            '00:00', '01:00', '02:00',
            '03:00', '04:00', '05:00',
            '06:00', '07:00', '08:00',
            '09:00', '10:00', '11:00',
            '12:00', '13:00', '14:00',
            '15:00', '16:00', '17:00',
            '18:00', '19:00', '20:00',
            '21:00', '22:00', '23:00',
        ],
        'area': [
            34.88, -23.42, 24.17,
            -11.28,
        ],
        'format': 'netcdf',
    },
    'scratch/obs/era5/hourly_100m_wind_u.nc')

# daily precipitation from MESCAN 5.5 km
# https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-uerra-europe-single-levels?tab=form

c.retrieve(
    'reanalysis-uerra-europe-single-levels',
    {
        'format': 'netcdf',
        'origin': 'mescan_surfex',
        'variable': 'total_precipitation',
        'year': [
            '2006', '2007', '2008',
            '2009', '2010', '2011',
            '2012', '2013', '2014',
            '2015',
        ],
        'month': [
            '01', '02', '03',
            '04', '05', '06',
            '07', '08', '09',
            '10', '11', '12',
        ],
        'day': [
            '01', '02', '03',
            '04', '05', '06',
            '07', '08', '09',
            '10', '11', '12',
            '13', '14', '15',
            '16', '17', '18',
            '19', '20', '21',
            '22', '23', '24',
            '25', '26', '27',
            '28', '29', '30',
            '31',
        ],
        'time': '06:00',
    },
    'scratch/obs/mescan/mescan_daily_pre_europe_2006_15.nc')

c.retrieve(
    'reanalysis-era5-single-levels',
    {
        'product_type': 'reanalysis',
        'format': 'netcdf',
        'variable': [
            '100m_u_component_of_wind', '100m_v_component_of_wind', 'mean_sea_level_pressure',
        ],
        'month': '02',
        'year': '2010',
        'day': [
            '01', '02', '03',
            '04', '05', '06',
            '07', '08', '09',
            '10', '11', '12',
            '13', '14', '15',
            '16', '17', '18',
            '19', '20', '21',
            '22', '23', '24',
            '25', '26', '27',
            '28',
        ],
        'time': [
            '00:00', '01:00', '02:00',
            '03:00', '04:00', '05:00',
            '06:00', '07:00', '08:00',
            '09:00', '10:00', '11:00',
            '12:00', '13:00', '14:00',
            '15:00', '16:00', '17:00',
            '18:00', '19:00', '20:00',
            '21:00', '22:00', '23:00',
        ],
        'area': [
            60, -60, 0,
            0,
        ],
    },
    'scratch/obs/era5/siglelev_uvp_201002_60_60.nc')

# https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels?tab=form
c.retrieve(
    'reanalysis-era5-single-levels',
    {
        'product_type': 'reanalysis',
        'format': 'netcdf',
        'variable': [
            '100m_u_component_of_wind', '100m_v_component_of_wind',
        ],
        'year': '2010',
        'month': [
            '01', '02', '03',
            '04', '05', '06',
            '07', '08', '09',
            '10', '11', '12',
        ],
        'day': [
            '01', '02', '03',
            '04', '05', '06',
            '07', '08', '09',
            '10', '11', '12',
            '13', '14', '15',
            '16', '17', '18',
            '19', '20', '21',
            '22', '23', '24',
            '25', '26', '27',
            '28', '29', '30',
            '31',
        ],
        'time': [
            '00:00', '01:00', '02:00',
            '03:00', '04:00', '05:00',
            '06:00', '07:00', '08:00',
            '09:00', '10:00', '11:00',
            '12:00', '13:00', '14:00',
            '15:00', '16:00', '17:00',
            '18:00', '19:00', '20:00',
            '21:00', '22:00', '23:00',
        ],
        'area': [
            34.88, -23.42, 24.17,
            -11.28,
        ],
    },
    'scratch/obs/era5/siglelev_uv_2010.nc')


# https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels?tab=form
c.retrieve(
    'reanalysis-era5-single-levels',
    {
        'product_type': 'reanalysis',
        'format': 'netcdf',
        'variable': [
            '100m_u_component_of_wind', '100m_v_component_of_wind', 'mean_sea_level_pressure',
        ],
        'year': '2010',
        'month': '08',
        'day': [
            '03', '04', '05',
            '06', '07', '08',
            '09',
        ],
        'time': [
            '00:00', '01:00', '02:00',
            '03:00', '04:00', '05:00',
            '06:00', '07:00', '08:00',
            '09:00', '10:00', '11:00',
            '12:00', '13:00', '14:00',
            '15:00', '16:00', '17:00',
            '18:00', '19:00', '20:00',
            '21:00', '22:00', '23:00',
        ],
        'area': [
            60, -60, 0,
            0,
        ],
    },
    'scratch/obs/era5/siglelev_100uvp_20100803_09_60_60.nc')

c.retrieve(
    'reanalysis-era5-single-levels',
    {
        'product_type': 'reanalysis',
        'format': 'netcdf',
        'variable': [
            '10m_u_component_of_wind', '10m_v_component_of_wind', 'mean_sea_level_pressure',
        ],
        'year': '2010',
        'month': '08',
        'day': [
            '03', '04', '05',
            '06', '07', '08',
            '09',
        ],
        'time': [
            '00:00', '01:00', '02:00',
            '03:00', '04:00', '05:00',
            '06:00', '07:00', '08:00',
            '09:00', '10:00', '11:00',
            '12:00', '13:00', '14:00',
            '15:00', '16:00', '17:00',
            '18:00', '19:00', '20:00',
            '21:00', '22:00', '23:00',
        ],
        'area': [
            60, -60, 0,
            0,
        ],
    },
    'scratch/obs/era5/siglelev_uvp_20100803_09_60_60.nc')

c.retrieve(
    'reanalysis-era5-single-levels',
    {
        'product_type': 'reanalysis',
        'variable': [
            '10m_u_component_of_wind', '10m_v_component_of_wind', 'mean_sea_level_pressure',
        ],
        'year': '2015',
        'month': '03',
        'day': [
            '01', '02', '03',
            '04', '05', '06',
            '07', '08', '09',
            '10', '11', '12',
            '13', '14', '15',
            '16', '17', '18',
            '19', '20', '21',
            '22', '23', '24',
            '25', '26', '27',
            '28', '29', '30',
            '31',
        ],
        'time': [
            '00:00', '01:00', '02:00',
            '03:00', '04:00', '05:00',
            '06:00', '07:00', '08:00',
            '09:00', '10:00', '11:00',
            '12:00', '13:00', '14:00',
            '15:00', '16:00', '17:00',
            '18:00', '19:00', '20:00',
            '21:00', '22:00', '23:00',
        ],
        'format': 'netcdf',
        'area': [
            60, -60, -30,
            30,
        ],
    },
    'scratch/obs/era5/siglelev_uvp_201503_90_90.nc')


c.retrieve(
    'reanalysis-era5-pressure-levels',
    {
        'product_type': 'reanalysis',
        'format': 'netcdf',
        'pressure_level': [
            '500', '800', '850',
            '900', '925', '950',
            '975', '1000',
        ],
        'variable': [
            'geopotential', 'u_component_of_wind', 'v_component_of_wind',
        ],
        'year': '2010',
        'month': '08',
        'day': [
            '03', '04', '05',
            '06', '07', '08',
            '09',
        ],
        'time': [
            '00:00', '01:00', '02:00',
            '03:00', '04:00', '05:00',
            '06:00', '07:00', '08:00',
            '09:00', '10:00', '11:00',
            '12:00', '13:00', '14:00',
            '15:00', '16:00', '17:00',
            '18:00', '19:00', '20:00',
            '21:00', '22:00', '23:00',
        ],
        'area': [
            60, -60, 0,
            0,
        ],
    },
    'scratch/obs/era5/plev_zuv_20100803_09_60_60.nc')


# https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels?tab=form
# ERA5 single level variables from 20100801 to 20100809
c.retrieve(
    'reanalysis-era5-single-levels',
    {
        'product_type': 'reanalysis',
        'variable': [
            'boundary_layer_height', 'orography',
        ],
        'year': '2010',
        'month': '08',
        'day': [
            '01', '02', '03',
            '04', '05', '06',
            '07', '08', '09',
        ],
        'time': [
            '00:00', '01:00', '02:00',
            '03:00', '04:00', '05:00',
            '06:00', '07:00', '08:00',
            '09:00', '10:00', '11:00',
            '12:00', '13:00', '14:00',
            '15:00', '16:00', '17:00',
            '18:00', '19:00', '20:00',
            '21:00', '22:00', '23:00',
        ],
        'format': 'netcdf',
        'area': [
            34.88, -23.42, 24.17,
            -11.28,
        ],
    },
    'scratch/obs/era5/single_level_BLheight_20100801_09.nc')
c.retrieve(
    'reanalysis-era5-single-levels',
    {
        'product_type': 'reanalysis',
        'variable': [
            'sea_surface_temperature',
        ],
        'year': '2010',
        'month': '08',
        'day': [
            '01', '02', '03',
            '04', '05', '06',
            '07', '08', '09',
        ],
        'time': [
            '00:00', '01:00', '02:00',
            '03:00', '04:00', '05:00',
            '06:00', '07:00', '08:00',
            '09:00', '10:00', '11:00',
            '12:00', '13:00', '14:00',
            '15:00', '16:00', '17:00',
            '18:00', '19:00', '20:00',
            '21:00', '22:00', '23:00',
        ],
        'format': 'netcdf',
        'area': [
            34.88, -23.42, 24.17,
            -11.28,
        ],
    },
    'scratch/obs/era5/single_level_sst_20100801_09.nc')

# https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-pressure-levels?tab=form
# ERA5 pressure level variables from 20100801 to 20100809
c.retrieve(
    'reanalysis-era5-pressure-levels',
    {
        'product_type': 'reanalysis',
        'variable': [
            'divergence', 'fraction_of_cloud_cover', 'geopotential',
            'potential_vorticity', 'relative_humidity', 'specific_cloud_ice_water_content',
            'specific_cloud_liquid_water_content', 'specific_humidity', 'specific_rain_water_content',
            'temperature', 'u_component_of_wind', 'v_component_of_wind',
            'vertical_velocity', 'vorticity',
        ],
        'pressure_level': [
            '1', '2', '3',
            '5', '7', '10',
            '20', '30', '50',
            '70', '100', '125',
            '150', '175', '200',
            '225', '250', '300',
            '350', '400', '450',
            '500', '550', '600',
            '650', '700', '750',
            '775', '800', '825',
            '850', '875', '900',
            '925', '950', '975',
            '1000',
        ],
        'year': '2010',
        'month': '08',
        'day': [
            '01', '02', '03',
            '04', '05', '06',
            '07', '08', '09',
        ],
        'time': [
            '00:00', '01:00', '02:00',
            '03:00', '04:00', '05:00',
            '06:00', '07:00', '08:00',
            '09:00', '10:00', '11:00',
            '12:00', '13:00', '14:00',
            '15:00', '16:00', '17:00',
            '18:00', '19:00', '20:00',
            '21:00', '22:00', '23:00',
        ],
        'format': 'netcdf',
        'area': [
            34.88, -23.42, 24.17,
            -11.28,
        ],
    },
    'scratch/obs/era5/pressure_level_variables_20100801_09.nc')

# https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-pressure-levels?tab=form
c.retrieve(
    'reanalysis-era5-pressure-levels',
    {
        'product_type': 'reanalysis',
        'variable': [
            'geopotential', 'potential_vorticity', 'temperature',
            'u_component_of_wind', 'v_component_of_wind', 'vertical_velocity',
        ],
        'pressure_level': [
            '200', '500', '800', '900', '1000',
        ],
        'year': '2010',
        'month': '08',
        'day': [
            '03', '04', '05', '06', '07', '08', '09',
        ],
        'time': [
            '00:00', '01:00', '02:00',
            '03:00', '04:00', '05:00',
            '06:00', '07:00', '08:00',
            '09:00', '10:00', '11:00',
            '12:00', '13:00', '14:00',
            '15:00', '16:00', '17:00',
            '18:00', '19:00', '20:00',
            '21:00', '22:00', '23:00',
        ],
        'format': 'netcdf',
    },
    'scratch/obs/era5/pressure_level_variables_20100801_09_global.nc')


# 2m temperature https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels-monthly-means?tab=form

c.retrieve(
    'reanalysis-era5-single-levels-monthly-means',
    {
        'format': 'netcdf',
        'product_type': 'monthly_averaged_reanalysis',
        'variable': '2m_temperature',
        'year': [
            '1979', '1980', '1981',
            '1982', '1983', '1984',
            '1985', '1986', '1987',
            '1988', '1989', '1990',
            '1991', '1992', '1993',
            '1994', '1995', '1996',
            '1997', '1998', '1999',
            '2000', '2001', '2002',
            '2003', '2004', '2005',
            '2006', '2007', '2008',
            '2009', '2010', '2011',
            '2012', '2013', '2014',
            '2015', '2016', '2017',
            '2018', '2019', '2020',
        ],
        'month': [
            '01', '02', '03',
            '04', '05', '06',
            '07', '08', '09',
            '10', '11', '12',
        ],
        'time': '00:00',
    },
    'scratch/obs/era5/monthly_2m_tem_1979_2020_global.nc')

c.retrieve(
    'reanalysis-era5-single-levels-monthly-means',
    {
        'format': 'netcdf',
        'product_type': 'monthly_averaged_reanalysis',
        'variable': '2m_temperature',
        'year': [
            '1979', '1980', '1981',
            '1982', '1983', '1984',
            '1985', '1986', '1987',
            '1988', '1989', '1990',
            '1991', '1992', '1993',
            '1994', '1995', '1996',
            '1997', '1998', '1999',
            '2000', '2001', '2002',
            '2003', '2004', '2005',
            '2006', '2007', '2008',
            '2009', '2010', '2011',
            '2012', '2013', '2014',
            '2015', '2016', '2017',
            '2018', '2019', '2020',
        ],
        'month': [
            '01', '02', '03',
            '04', '05', '06',
            '07', '08', '09',
            '10', '11', '12',
        ],
        'time': '00:00',
        'area': [
            34.88, -23.42, 24.17,
            -11.28,
        ],
    },
    'scratch/obs/era5/monthly_2m_tem_1979_2020.nc')

# r vorticity
c.retrieve(
    'reanalysis-era5-pressure-levels',
    {
        'product_type': 'reanalysis',
        'variable': [
            'geopotential', 'vorticity',
        ],
        'pressure_level': '1000',
        'year': '2010',
        'month': '08',
        'day': [
            '03', '04', '05',
            '06', '07', '08',
            '09',
        ],
        'time': [
            '00:00', '01:00', '02:00',
            '03:00', '04:00', '05:00',
            '06:00', '07:00', '08:00',
            '09:00', '10:00', '11:00',
            '12:00', '13:00', '14:00',
            '15:00', '16:00', '17:00',
            '18:00', '19:00', '20:00',
            '21:00', '22:00', '23:00',
        ],
        'area': [
            45, -35, 10,
            0,
        ],
        'format': 'netcdf',
    },
    'scratch/era5/rvorticity_1000hPa_20100803_09.nc')


# cloud
c.retrieve(
    'reanalysis-era5-pressure-levels',
    {
        'product_type': 'reanalysis',
        'variable': [
            'specific_cloud_ice_water_content', 'specific_cloud_liquid_water_content',
        ],
        'pressure_level': [
            '1', '2', '3',
            '5', '7', '10',
            '20', '30', '50',
            '70', '100', '125',
            '150', '175', '200',
            '225', '250', '300',
            '350', '400', '450',
            '500', '550', '600',
            '650', '700', '750',
            '775', '800', '825',
            '850', '875', '900',
            '925', '950', '975',
            '1000',
        ],
        'year': '2010',
        'month': '08',
        'day': [
            '03', '04', '05',
            '06', '07', '08',
            '09',
        ],
        'time': [
            '00:00', '03:00', '06:00',
            '09:00', '12:00', '15:00',
            '18:00', '21:00',
        ],
        'area': [
            45, -35, 10,
            0,
        ],
        'format': 'netcdf',
    },
    'scratch/era5/clouds_3h_20100803_09.nc')


# hourly sst https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels?tab=overview
c.retrieve(
    'reanalysis-era5-single-levels',
    {
        'product_type': 'reanalysis',
        'format': 'netcdf',
        'variable': 'sea_surface_temperature',
        'year': '2010',
        'month': [
            '01', '02', '03',
            '04', '05', '06',
            '07', '08', '09',
            '10', '11', '12',
        ],
        'day': [
            '01', '02', '03',
            '04', '05', '06',
            '07', '08', '09',
            '10', '11', '12',
            '13', '14', '15',
            '16', '17', '18',
            '19', '20', '21',
            '22', '23', '24',
            '25', '26', '27',
            '28', '29', '30',
            '31',
        ],
        'time': [
            '00:00', '01:00', '02:00',
            '03:00', '04:00', '05:00',
            '06:00', '07:00', '08:00',
            '09:00', '10:00', '11:00',
            '12:00', '13:00', '14:00',
            '15:00', '16:00', '17:00',
            '18:00', '19:00', '20:00',
            '21:00', '22:00', '23:00',
        ],
        'area': [
            34.86, -23.41, 24.18,
            -11.29,
        ],
    },
    'scratch/era5/hourly_sst_2010.nc')


# daily sst https://cds.climate.copernicus.eu/cdsapp#!/dataset/satellite-sea-surface-temperature?tab=form

c.retrieve(
    'satellite-sea-surface-temperature',
    {
        'variable': 'all',
        'format': 'zip',
        'processinglevel': 'level_4',
        'sensor_on_satellite': 'combined_product',
        'version': '2_0',
        'year': '2010',
        'month': [
            '01', '02', '03',
            '04', '05', '06',
        ],
        'day': [
            '01', '02', '03',
            '04', '05', '06',
            '07', '08', '09',
            '10', '11', '12',
            '13', '14', '15',
            '16', '17', '18',
            '19', '20', '21',
            '22', '23', '24',
            '25', '26', '27',
            '28', '29', '30',
            '31',
        ],
    },
    'scratch/era5/daily_sst_201001_6.zip')

c.retrieve(
    'satellite-sea-surface-temperature',
    {
        'variable': 'all',
        'format': 'zip',
        'processinglevel': 'level_4',
        'sensor_on_satellite': 'combined_product',
        'version': '2_0',
        'year': '2010',
        'month': [
            '07', '08', '09',
            '10', '11', '12',
        ],
        'day': [
            '01', '02', '03',
            '04', '05', '06',
            '07', '08', '09',
            '10', '11', '12',
            '13', '14', '15',
            '16', '17', '18',
            '19', '20', '21',
            '22', '23', '24',
            '25', '26', '27',
            '28', '29', '30',
            '31',
        ],
    },
    'scratch/era5/daily_sst_201007_12.zip')

# monthly sst https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels-monthly-means?tab=form
c.retrieve(
    'reanalysis-era5-single-levels-monthly-means',
    {
        'format': 'netcdf',
        'product_type': 'monthly_averaged_reanalysis',
        'variable': 'sea_surface_temperature',
        'year': [
            '1979', '1980', '1981',
            '1982', '1983', '1984',
            '1985', '1986', '1987',
            '1988', '1989', '1990',
            '1991', '1992', '1993',
            '1994', '1995', '1996',
            '1997', '1998', '1999',
            '2000', '2001', '2002',
            '2003', '2004', '2005',
            '2006', '2007', '2008',
            '2009', '2010', '2011',
            '2012', '2013', '2014',
            '2015', '2016', '2017',
            '2018', '2019', '2020',
        ],
        'month': [
            '01', '02', '03',
            '04', '05', '06',
            '07', '08', '09',
            '10', '11', '12',
        ],
        'time': '00:00',
        'area': [
            34.88, -23.42, 24.17,
            -11.28,
        ],
    },
    'scratch/obs/era5/monthly_sst_1979_2020.nc')


# monthly pre https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels-monthly-means?tab=form

c.retrieve(
    'reanalysis-era5-single-levels-monthly-means',
    {
        'format': 'netcdf',
        'variable': 'total_precipitation',
        'product_type': 'monthly_averaged_reanalysis',
        'year': [
            '1979', '1980', '1981',
            '1982', '1983', '1984',
            '1985', '1986', '1987',
            '1988', '1989', '1990',
            '1991', '1992', '1993',
            '1994', '1995', '1996',
            '1997', '1998', '1999',
            '2000', '2001', '2002',
            '2003', '2004', '2005',
            '2006', '2007', '2008',
            '2009', '2010', '2011',
            '2012', '2013', '2014',
            '2015', '2016', '2017',
            '2018', '2019', '2020',
        ],
        'month': [
            '01', '02', '03',
            '04', '05', '06',
            '07', '08', '09',
            '10', '11', '12',
        ],
        'time': '00:00',
        'area': [
            34.88, -23.42, 24.17,
            -11.28,
        ],
    },
    'scratch/obs/era5/monthly_pre_1979_2020.nc')

c.retrieve(
    'reanalysis-era5-single-levels-monthly-means',
    {
        'format': 'grib',
        'product_type': 'monthly_averaged_reanalysis',
        'variable': 'total_precipitation',
        'year': [
            '1979', '1980', '1981',
            '1982', '1983', '1984',
            '1985', '1986', '1987',
            '1988', '1989', '1990',
            '1991', '1992', '1993',
            '1994', '1995', '1996',
            '1997', '1998', '1999',
            '2000', '2001', '2002',
            '2003', '2004', '2005',
            '2006', '2007', '2008',
            '2009', '2010', '2011',
            '2012', '2013', '2014',
            '2015', '2016', '2017',
            '2018', '2019', '2020',
        ],
        'month': [
            '01', '02', '03',
            '04', '05', '06',
            '07', '08', '09',
            '10', '11', '12',
        ],
        'time': '00:00',
        'area': [
            34.88, -23.42, 24.17,
            -11.28,
        ],
    },
    'scratch/obs/era5/monthly_pre_1979_2020_check.grib')

c.retrieve(
    'reanalysis-era5-single-levels-monthly-means',
    {
        'format': 'netcdf',
        'product_type': 'monthly_averaged_reanalysis',
        'variable': 'total_precipitation',
        'year': [
            '1979', '1980', '1981',
            '1982', '1983', '1984',
            '1985', '1986', '1987',
            '1988', '1989', '1990',
            '1991', '1992', '1993',
            '1994', '1995', '1996',
            '1997', '1998', '1999',
            '2000', '2001', '2002',
            '2003', '2004', '2005',
            '2006', '2007', '2008',
            '2009', '2010', '2011',
            '2012', '2013', '2014',
            '2015', '2016', '2017',
            '2018', '2019', '2020',
        ],
        'month': [
            '01', '02', '03',
            '04', '05', '06',
            '07', '08', '09',
            '10', '11', '12',
        ],
        'time': '00:00',
    },
    'scratch/obs/era5/monthly_pre_1979_2020_global.nc')


# ecv pre https://cds.climate.copernicus.eu/cdsapp#!/dataset/ecv-for-climate-change?tab=form

c.retrieve(
    'ecv-for-climate-change',
    {
        'format': 'zip',
        'variable': 'precipitation',
        'product_type': [
            'anomaly', 'monthly_mean',
        ],
        'time_aggregation': '1_month',
        'year': [
            '1979', '1980', '1981',
            '1982', '1983', '1984',
            '1985', '1986', '1987',
            '1988', '1989', '1990',
            '1991', '1992', '1993',
            '1994', '1995', '1996',
            '1997', '1998', '1999',
            '2000', '2001', '2002',
            '2003', '2004', '2005',
            '2006', '2007', '2008',
            '2009', '2010', '2011',
            '2012', '2013', '2014',
            '2015', '2016', '2017',
            '2018', '2019', '2020',
        ],
        'origin': 'era5',
        'month': [
            '01', '02', '03',
            '04', '05', '06',
            '07', '08', '09',
            '10', '11', '12',
        ],
    },
    'scratch/obs/era5/ecv_monthly_pre_1979_2020.zip')

# mescan_surfex,wind direction and speed 2010,
# https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-uerra-europe-single-levels?tab=form
c.retrieve(
    'reanalysis-uerra-europe-single-levels',
    {
        'format': 'netcdf',
        'origin': 'mescan_surfex',
        'variable': '10m_wind_direction',
        'year': '2010',
        'month': [
            '01', '02', '03',
            '04', '05', '06',
            '07', '08', '09',
            '10', '11', '12',
        ],
        'day': [
            '01', '02', '03',
            '04', '05', '06',
            '07', '08', '09',
            '10', '11', '12',
            '13', '14', '15',
            '16', '17', '18',
            '19', '20', '21',
            '22', '23', '24',
            '25', '26', '27',
            '28', '29', '30',
            '31',
        ],
        'time': [
            '00:00', '06:00', '12:00',
            '18:00',
        ],
    },
    'scratch/obs/mescan/wind_direction_10m_2010.nc')


c.retrieve(
    'reanalysis-uerra-europe-single-levels',
    {
        'format': 'netcdf',
        'origin': 'mescan_surfex',
        'variable': '10m_wind_speed',
        'year': '2010',
        'month': [
            '01', '02', '03',
            '04', '05', '06',
            '07', '08', '09',
            '10', '11', '12',
        ],
        'day': [
            '01', '02', '03',
            '04', '05', '06',
            '07', '08', '09',
            '10', '11', '12',
            '13', '14', '15',
            '16', '17', '18',
            '19', '20', '21',
            '22', '23', '24',
            '25', '26', '27',
            '28', '29', '30',
            '31',
        ],
        'time': [
            '00:00', '06:00', '12:00',
            '18:00',
        ],
    },
    'scratch/obs/mescan/wind_speed_10m_2010.nc')


c.retrieve(
    'reanalysis-uerra-europe-single-levels',
    {
        'format': 'netcdf',
        'origin': 'mescan_surfex',
        'variable': '10m_wind_direction',
        'year': '2010',
        'month': '08',
        'day': [
            '01', '02', '03',
            '04', '05', '06',
            '07', '08', '09',
            '10', '11', '12',
            '13', '14', '15',
            '16', '17', '18',
            '19', '20', '21',
            '22', '23', '24',
            '25', '26', '27',
            '28', '29', '30',
            '31',
        ],
        'time': [
            '00:00', '06:00', '12:00',
            '18:00',
        ],
    },
    'scratch/obs/mescan/wind_direction_10m_201008.nc')


# endregion
# =============================================================================


# =============================================================================
# region plot relative vorticity from era5 from 20100803 - 09

rvorticity_1000hPa_20100803_09 = xr.open_dataset(
    'scratch/era5/rvorticity_1000hPa_20100803_09.nc')
# rvorticity_1000hPa_20100803_09.vo

lon = rvorticity_1000hPa_20100803_09.longitude.values
lat = rvorticity_1000hPa_20100803_09.latitude.values
time = rvorticity_1000hPa_20100803_09.time.values

# set colormap level and ticks
vorlevel = np.arange(-6, 6.1, 0.1)
ticks = np.arange(-6, 6.1, 2)

# create a color map
top = cm.get_cmap('Blues_r', int(np.floor(len(vorlevel) / 2)))
bottom = cm.get_cmap('Reds', int(np.floor(len(vorlevel) / 2)))
newcolors = np.vstack(
    (top(np.linspace(0, 1, int(np.floor(len(vorlevel) / 2)))),
     [1, 1, 1, 1],
     bottom(np.linspace(0, 1, int(np.floor(len(vorlevel) / 2))))))
newcmp = ListedColormap(newcolors, name='RedsBlues_r')


fig, ax = plt.subplots(1, 1, figsize=np.array([8.8, 8.8]) / 2.54,
                       subplot_kw={'projection': transform}, dpi=600)
ims = []
istart = 20
ifinal = 152  # 152

for i in np.arange(istart, ifinal):
    rvor = rvorticity_1000hPa_20100803_09.vo[
        i, :, :].values * 10**4
    plt_rvor = ax.pcolormesh(
        lon, lat, rvor, cmap=newcmp, rasterized=True, transform=transform,
        norm=BoundaryNorm(vorlevel, ncolors=newcmp.N, clip=False), zorder=-2,)
    
    rvor_time = ax.text(
        -23, 34, str(time[i])[0:10] + ' ' + str(time[i])[11:13] + ':00 UTC')
    
    ims.append([plt_rvor, rvor_time])
    print(str(i) + '/' + str(ifinal - 1))

gl = ax.gridlines(crs=transform, linewidth=0.25, color='gray',
                  alpha=0.5, linestyle='--')
gl.xlocator = mticker.FixedLocator(ticklabel1km_lb[0])
gl.ylocator = mticker.FixedLocator(ticklabel1km_lb[2])

ax.add_feature(borders, lw = 0.25); ax.add_feature(coastline, lw = 0.25)
ax.set_xticks(ticklabel1km_lb[0]); ax.set_xticklabels(ticklabel1km_lb[1])
ax.set_yticks(ticklabel1km_lb[2]); ax.set_yticklabels(ticklabel1km_lb[3])

cbar = fig.colorbar(
    plt_rvor, ax=ax, orientation="horizontal",  pad=0.1, fraction=0.09,
    shrink=1, aspect=25, ticks=ticks, extend='both',
    anchor=(0.5, 1), panchor=(0.5, 0))
cbar.ax.set_xlabel("Relative vorticity [$10^{-4}\;s^{-1}$]")

ax.set_extent(extent1km_lb, crs=transform)
scale_bar(ax, bars=2, length=200, location=(0.08, 0.06),
          barheight=20, linewidth=0.15, col='black', middle_label=False)
ax.set_rasterization_zorder(-1)
fig.subplots_adjust(left=0.12, right=0.99, bottom=0.08, top=0.99)
ani = animation.ArtistAnimation(fig, ims, interval=100, blit=True)

ani.save('figures/02_vorticity/2.5.0_Relative_vorticity_in_era5_2010080320_2010080907.mp4')

# endregion
# =============================================================================


# =============================================================================
# region check era5 vertical temperature profiles

era5_example = xr.open_dataset(
    'data_source/era5/adaptor.mars.internal-1603200664.7498538-23233-12-0f7403c4-7ee1-44c2-8d5f-ab5a6d67e189.nc')

height = mpcalc.geopotential_to_height(
    era5_example.z.squeeze().values * units('meter ** 2 / second ** 2'))

era5_example.level


# plot T and theta

fig, ax = plt.subplots(1, 1, figsize=np.array([8.8, 8]) / 2.54)

plt_T, = ax.plot(
    era5_example.t[0, -11:, np.where(era5_example.latitude == 33.0)[0][0],
                   np.where(era5_example.longitude == -16.75)[0][0]],
    height[0, -11:, np.where(era5_example.latitude == 33.0)[0][0],
           np.where(era5_example.longitude == -16.75)[0][0]],
    linewidth=0.25, color='blue'
)

ax.set_xticks(np.arange(285, 316, 5))
ax.set_yticks(np.arange(0, 3001, 500))
ax.set_xticklabels(np.arange(285, 316, 5), size=8)
ax.set_yticklabels(np.arange(0, 3.1, 0.5), size=8)
ax.set_xlabel("Temperature in ERA5 [K]", size=10)
ax.set_ylabel("Height [km]", size=10)

fig.subplots_adjust(left=0.14, right=0.95, bottom=0.2, top=0.99)
fig.savefig(
    'figures/06_inversion_layer/6.0.1 inversion layer in 201008 in ERA5.png', dpi=1200)
plt.close('all')

# endregion
# =============================================================================


# =============================================================================
# region check vorticity calculation

era5_example = xr.open_dataset(
    'data_source/era5/adaptor.mars.internal-1603200664.7498538-23233-12-0f7403c4-7ee1-44c2-8d5f-ab5a6d67e189.nc')

lon = era5_example.longitude.values
lat = era5_example.latitude.values
dx, dy = mpcalc.lat_lon_grid_deltas(lon, lat)

u = era5_example.u[0, -1, :, :].values * units('m/s')
v = era5_example.v[0, -1, :, :].values * units('m/s')
vo = era5_example.vo[0, -1, :, :].values

vo_calc = mpcalc.vorticity(
    u, v, dx, dy, dim_order='yx')

np.quantile(abs(vo - vo_calc.magnitude), 0.95)


# endregion
# =============================================================================


# =============================================================================
# region check era5 clouds during 201008 03 - 09
clouds_3h_20100803_09 = xr.open_dataset('scratch/era5/clouds_3h_20100803_09.nc')



'''
filelist_1h = sorted(glob.glob(folder_1km + '1h/lffd2010080*[0-9].nc'))

clouds_1h = xr.open_mfdataset(
    filelist_1h, concat_dim="time", data_vars='minimal',
    coords='minimal', compat='override')

lon = clouds_1h.lon[80:920, 80:920].values
lat = clouds_1h.lat[80:920, 80:920].values
time = clouds_1h.time.values

# Transparent colormap
colors = [(1, 1, 1, c) for c in np.linspace(0, 1, 100)]
cmapwhite = mpcolors.LinearSegmentedColormap.from_list('mycmap', colors, N=100)
color_bg = plt.cm.Blues(0.5)
vmax = 1


fig, ax = plt.subplots(1, 1, figsize=np.array([8.8, 8.2]) / 2.54,
                       subplot_kw={'projection': transform}, dpi=600)
ims = []
istart = 68
ifinal = 200  # 200

for i in np.arange(istart, ifinal):

    tqc = clouds_1h.TQC[i, 80:920, 80:920].values

    plt_tqc = ax.pcolormesh(
        lon, lat, tqc, cmap=cmapwhite, vmin=0.0, vmax=vmax,
        transform=transform, rasterized=True,
        # zorder=-2,
    )

    tqc_time = ax.text(
        -23, 34, str(time[i])[0:10] + ' ' + str(time[i])[11:13] + ':00 UTC',
        color='white',
    )

    ims.append([plt_tqc, tqc_time])
    print(str(i) + '/' + str(ifinal - 1))

scale_bar(ax, bars=2, length=200, location=(0.08, 0.06),
          barheight=20, linewidth=0.15, col='black', middle_label=False,
          fontcolor='white')
# ax.stock_img()
# ax.background_img(name='blue_marble_jun', resolution='high')
# ax.background_img(name='natural_earth', resolution='high')
ax.set_xticks(ticklabel1km_lb[0])
ax.set_xticklabels(ticklabel1km_lb[1])
ax.set_yticks(ticklabel1km_lb[2])
ax.set_yticklabels(ticklabel1km_lb[3])
ax.set_xlabel('Atmospheric cloud liquid water content [mm]')

ax.add_feature(coastline)
ax.add_feature(borders)
ax.set_facecolor(color_bg)

gl = ax.gridlines(crs=transform, linewidth=0.5, color='gray',
                  alpha=0.5, linestyle='--')
gl.xlocator = mticker.FixedLocator(ticklabel1km_lb[0])
gl.ylocator = mticker.FixedLocator(ticklabel1km_lb[2])

ax.set_extent(extent1km_lb, crs=transform)
fig.subplots_adjust(left=0.12, right=0.99, bottom=0.12, top=0.99)

ani = animation.ArtistAnimation(fig, ims, interval=100, blit=True)
ani.save('figures/05_clouds/5.0.1_Clouds_in_2010080320_2010080907.mp4')



'''

# endregion
# =============================================================================




