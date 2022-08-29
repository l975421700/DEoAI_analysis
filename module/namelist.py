# -*- coding: utf-8 -*-
"""
File defining the global variables used in the main program
and all subfunctions.
"""

# --------------------------------------------------------
# --------------------- USER NAMELIST --------------------
# --------------------------------------------------------

import numpy as np
import cartopy as ctp
from DEoAI_analysis.module.mapplot import (
    ticks_labels,
)


# Seasons
# -------------------------------------------------

month = np.array(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])

seasons = np.array(['Spring', 'Summer', 'Autumn', 'Winter'])

hours = ['00', '01', '02', '03', '04', '05',
         '06', '07', '08', '09', '10', '11',
         '12', '13', '14', '15', '16', '17',
         '18', '19', '20', '21', '22', '23',
         ]

months = ['01', '02', '03', '04', '05', '06',
          '07', '08', '09', '10', '11', '12']

years = ['06', '07', '08', '09', '10', '11', '12', '13', '14', '15']

years_months = [i + j for i, j in
                zip(np.repeat(years, 12), np.tile(months, 10))]

timing = np.concatenate((np.array(('Annual'), ndmin=1), month, seasons))

month_days = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])

# computation
# -------------------------------------------------

quantiles = np.array(
    ([0, 5, 10, 25, 50, 75, 90, 95, 100],
     ['0%', '5%', '10%', '25%', '50%', '75%', '90%', '95%', '100%',
      'ptp100_0', 'ptp95_5', 'ptp90_10', 'ptp75_25',
      'mean', 'std']), dtype=object)


# folder
# -------------------------------------------------

folder_1km = '/store/c2sm/pr04/jvergara/RUNS_IN_SCRATCH/MAC1/lm_f/'
folder_12km = '/store/c2sm/pr04/jvergara/RUNS_IN_SCRATCH/MAC12/lm_c/'
era5_folder = '/net/thermo/atmosdyn2/era5/cdf/'

# Physical constants
# -------------------------------------------------

g = 9.80665  # gravity [m/s^2]
m = 0.0289644  # Molar mass of dry air [kg/mol]
r0 = 8.314462618  # Universal gas constant [J/(mol·K)]
cp = 1004.68506  # specific heat of air at constant pressure [J/(kg·K)]
r = 287.0  # gas constant of air [J/kgK]
r_v = 461.0  # gas cons tant of vapor [J/kgK]
p0sl = 100000.0  # pressure at see level
t0sl = 288.1499938964844   # temperature at see level

# plot
# -------------------------------------------------
extent1km = [-24.642454, -10.228505, 23.151627, 35.85266]
extent3d_m = [-17.319347, -16.590143, 32.50762, 33.0472]
extent3d_g = [-16.004251, -15.163912, 27.55076, 28.373308]
extent3d_t = [-17.07781, -15.909306, 27.865873, 28.743168]
extentm = [-17.32, -16.25, 32.35, 33.15]
extentc = [-18.2, -13.2, 27.5, 29.5]
extent12km = [-30.504213, -4.761099, 17.60372, 40.405067]
extent1km_lb = [-23.401758, -11.290954, 24.182158, 34.85296]
extent_global = [-180, 180, -90, 90]
extent12km_out = [-35, 0, 10, 45]

ticklabel1km = ticks_labels(-24, -12, 25, 35, 3, 2)
ticklabelm = ticks_labels(-17.3, -16.3, 32.4, 32.9, 0.2, 0.2)
ticklabelc = ticks_labels(-18, -14, 27.5, 29.5, 1, 0.5)
ticklabel12km = ticks_labels(-30, -5, 20, 40, 5, 5)
ticklabel1km_lb = ticks_labels(-22, -12, 26, 34, 2, 2)
ticklabel_global = ticks_labels(-180, 180, -90, 90, 60, 30)
ticklabel12km_out = ticks_labels(-30, 0, 10, 40, 10, 10)

transform = ctp.crs.PlateCarree()
coastline = ctp.feature.NaturalEarthFeature(
    'physical', 'coastline', '10m', edgecolor='black',
    facecolor='none', lw = 0.5)
borders = ctp.feature.NaturalEarthFeature(
    'cultural', 'admin_0_boundary_lines_land', '10m', edgecolor='black',
    facecolor='none', lw = 0.5)

center_madeira = np.array([-16.99095096,  32.74360856])
angle_deg_madeira = 78.84727915314083
radius_madeira = np.array([0.13104647 * 0.7, 0.38622103 * 0.7])
hm_m_model = 1546.0217

# set colormap level and ticks
from matplotlib import cm
from matplotlib.colors import ListedColormap

rvor_level = np.arange(-12, 12.1, 0.1)
rvor_ticks = np.arange(-12, 12.1, 3)
# create a color map
rvor_top = cm.get_cmap('Blues_r', int(np.floor(len(rvor_level) / 2)))
rvor_bottom = cm.get_cmap('Reds', int(np.floor(len(rvor_level) / 2)))
rvor_colors = np.vstack(
    (rvor_top(np.linspace(0, 1, int(np.floor(len(rvor_level) / 2)))),
     [1, 1, 1, 1],
     rvor_bottom(np.linspace(0, 1, int(np.floor(len(rvor_level) / 2))))))
rvor_cmp = ListedColormap(rvor_colors, name='RedsBlues_r')

# END OF NAMELIST.PY



