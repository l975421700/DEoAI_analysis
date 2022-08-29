

# =============================================================================
# region import packages


# basic library
from matplotlib.path import Path
import pywt.data
import pywt
import datetime
import numpy as np
import xarray as xr
import os
import glob
import pickle
import gc
from scipy import stats
import tables as tb

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
# Background image
os.environ["CARTOPY_USER_BACKGROUNDS"] = "data_source/share/bg_cartopy"

from DEoAI_analysis.module.mapplot import (
    ticks_labels,
    scale_bar,
    framework_plot,
    framework_plot1,
)

from DEoAI_analysis.module.vortex_namelist import (
    correctly_reidentified,
    track_p1, track_p2, track_p3, track_p4,
    track_n1, track_n2, track_n3, track_n4,
)

# data analysis
import pandas as pd
import xesmf as xe
import metpy.calc as mpcalc
from metpy.interpolate import cross_section
from metpy.units import units
from haversine import haversine
from scipy import interpolate
from scipy import ndimage
import dask
dask.config.set({"array.slicing.split_large_chunks": False})
import h5py
from scipy.ndimage.filters import median_filter, maximum_filter

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
    extent_global,
    ticklabel1km,
    ticklabelm,
    ticklabelc,
    ticklabel12km,
    ticklabel1km_lb,
    ticklabel_global,
    transform,
    coastline,
    borders,
    rvor_level,
    rvor_ticks,
    rvor_cmp,
    center_madeira,
)

from DEoAI_analysis.module.statistics_calculate import(
    get_statistics
)

from DEoAI_analysis.module.spatial_analysis import(
    rotate_wind,
    sig_coeffs,
    vortex_identification,
    vortex_identification1,
)


# endregion
# =============================================================================


# =============================================================================
# region import data
iyear = 4
# years[iyear]

################################ 100m vorticity
rvor_100m_f = np.array(sorted(glob.glob(
    'scratch/rvorticity/relative_vorticity_1km_1h_100m/relative_vorticity_1km_1h_100m20' + years[iyear] + '*.nc')))
rvor_100m = xr.open_mfdataset(
    rvor_100m_f, concat_dim="time", data_vars='minimal',
    coords='minimal', compat='override', chunks={'time': 1})
time = rvor_100m.time.values
lon = rvor_100m.lon[80:920, 80:920].values
lat = rvor_100m.lat[80:920, 80:920].values

################################ model topography
dsconst = xr.open_dataset(
    'scratch/simulation/20100801_09_3d/02_lm/lfsd20051101000000c.nc')
model_topo = dsconst.HSURF[0].values

################################ parameters setting
grid_size = 1.2  # in km^2
median_filter_size = 3
maximum_filter_size = 50

min_rvor = 3.
min_max_rvor = 4.
min_size = 100.
min_size_theta = 450.
min_size_dir = 450
min_size_dir1 = 900
# max_dir = 30
# max_dir1 = 60
# max_dir2 = 90
max_dir = 30
max_dir1 = 40
max_dir2 = 50
max_distance2radius = 5


################################ original simulation to calculate surface theta
orig_simulation_f = np.array(sorted(
    glob.glob('/store/c2sm/pr04/jvergara/RUNS_IN_SCRATCH/MAC1/lm_f/1h/lffd20' +
              years[iyear] + '*[0-9].nc')
))

################################ create a mask for Madeira
polygon = [
    (300, 0), (390, 320), (260, 580),
    (380, 839), (839, 839), (839, 0),
]
poly_path = Path(polygon)
x, y = np.mgrid[:840, :840]
coors = np.hstack((x.reshape(-1, 1), y.reshape(-1, 1)))
madeira_mask = poly_path.contains_points(coors).reshape(840, 840)

################################ import wind
wind_100m_f = np.array(sorted(glob.glob(
    'scratch/wind_earth/wind_earth_1h_100m/wind_earth_1h_100m20' + years[iyear] + '*.nc')))
wind_100m = xr.open_mfdataset(
    wind_100m_f, concat_dim="time", data_vars='minimal',
    coords='minimal', compat='override', chunks={'time': 1})

# endregion
# =============================================================================


# =============================================================================
# region extract info of tracked reidentified rvor


################################ define time period and I/O files
initial = np.where(time == np.datetime64('2010-08-03T20:00:00.000000000'))[0][0]

inputfile = 'scratch/rvorticity/rvor_identify/re_identify/identified_transformed_rvor_20100803_09_stricter_dir.h5'
outputfile = 'scratch/rvorticity/rvor_track/identified_transformed_rvor_20100803_09_track_4p_4n.h5'


################################ import identified vortices
identified_rvor = tb.open_file(inputfile, mode="r")
experiment = identified_rvor.root.exp1
ilat = lon.shape[0]
ilon = lon.shape[1]


################################ create a new file to store tracked vortices

class ExtractVortexInfo(tb.IsDescription):
    is_vortex = tb.Int8Col(shape=(ilat, ilon))
    time_index = tb.Int64Col()
    vortex_index = tb.Int8Col()
    center_lat = tb.Float32Col()
    center_lon = tb.Float32Col()
    size = tb.Float64Col()
    radius = tb.Float64Col()
    distance2radius = tb.Float64Col()
    peak_magnitude = tb.Float64Col()
    mean_magnitude = tb.Float64Col()
    mean_wind_u = tb.Float64Col()
    mean_wind_v = tb.Float64Col()
    angle = tb.Float32Col()

tracked_rvor = tb.open_file(outputfile, mode="w")

tracked_rvor.create_array('/', 'lon', lon, 'longitude')
tracked_rvor.create_array('/', 'lat', lat, 'latitude')

################################ manual extracted vortices
extracted_rvor_dic = [
    track_p1, track_p2, track_p3, track_p4,
    track_n1, track_n2, track_n3, track_n4,
]
################################ group name
group_name = ['P1', 'P2', 'P3', 'P4', 'N1', 'N2', 'N3', 'N4']


for j in range(len(extracted_rvor_dic)):
    ################################ create group
    iexp = tracked_rvor.create_group('/', group_name[j])
    iexp._v_attrs.manual_track = extracted_rvor_dic[j]
    iexp._v_attrs.time = np.array(
        list(iexp._v_attrs.manual_track.keys()), dtype='datetime64')
    
    ################################ create table
    vortex_info = tracked_rvor.create_table(
        iexp, 'extracted_vortex_info', ExtractVortexInfo)
    vortex_info_r = vortex_info.row
    
    ################################ extract vortex info at each time point
    istart = np.where(time == iexp._v_attrs.time[0])[0][0]
    ifinal = np.where(time == iexp._v_attrs.time[-1])[0][0] + 1
    # time[[istart, ifinal - 1]]
    
    for i in np.arange(istart, ifinal):
        if (len(extracted_rvor_dic[j][str(time[i])[0:13]]) == 0):
            continue
        
        ################################ extract vortices information
        is_vortex = experiment.vortex_info.cols.is_vortex[i - initial]
        vortex_indices = experiment.vortex_info.cols.vortex_indices[i - initial]
        
        ivortex = extracted_rvor_dic[j][str(time[i])[0:13]][0]
        is_vortex[vortex_indices != ivortex] = 0
        vortex_de_r = [
            row[:] for row in experiment.vortex_de_info.where(
                '(time_index == ' + str(i) +
                ') & (vortex_index == ' + str(ivortex) + ')')][0]
        vortex_de_cols = experiment.vortex_de_info.colnames
        
        ################################ store vortex info
        vortex_info_r['is_vortex'] = is_vortex
        vortex_info_r['time_index'] = i
        vortex_info_r['vortex_index'] = ivortex
        vortex_info_r['center_lat'] = \
            vortex_de_r[vortex_de_cols.index('center_lat')]
        vortex_info_r['center_lon'] = \
            vortex_de_r[vortex_de_cols.index('center_lon')]
        vortex_info_r['size'] = \
            vortex_de_r[vortex_de_cols.index('size')]
        vortex_info_r['radius'] = \
            vortex_de_r[vortex_de_cols.index('radius')]
        vortex_info_r['distance2radius'] = \
            vortex_de_r[vortex_de_cols.index('distance2radius')]
        vortex_info_r['peak_magnitude'] = \
            vortex_de_r[vortex_de_cols.index('peak_magnitude')]
        vortex_info_r['mean_magnitude'] = \
            vortex_de_r[vortex_de_cols.index('mean_magnitude')]
        vortex_info_r['mean_wind_u'] = \
            vortex_de_r[vortex_de_cols.index('mean_wind_u')]
        vortex_info_r['mean_wind_v'] = \
            vortex_de_r[vortex_de_cols.index('mean_wind_v')]
        vortex_info_r['angle'] = \
            vortex_de_r[vortex_de_cols.index('angle')]
        
        vortex_info_r.append()
    
    vortex_info.flush()


identified_rvor.close()
tracked_rvor.close()


'''
# plot as check in next region
tracked_rvor = tb.open_file(
    'scratch/rvorticity/rvor_track/identified_transformed_rvor_20100803_09_track_4p_4n.h5',
    mode="r")

'''

# endregion
# =============================================================================


# =============================================================================
# region plot tracked vortices

######## mask topography
dsconst = xr.open_dataset(
    'scratch/simulation/20100801_09_3d/02_lm/lfsd20051101000000c.nc')
model_topo = dsconst.HSURF[0].values
model_topo_mask = np.ones_like(model_topo)
model_topo_mask[model_topo == 0] = np.nan

initial = np.where(time == np.datetime64('2010-08-03T20:00:00.000000000'))[0][0]
inputfile = 'scratch/rvorticity/rvor_track/identified_transformed_rvor_20100803_09_track_4p_4n.h5'
outputfile_prefix = 'figures/09_decades_vortex/09_05_vortex_track/9_5_0.'

################################ import track vortices
tracked_rvor = tb.open_file(inputfile, mode="r")
group_name = ['P1', 'P2', 'P3', 'P4', 'N1', 'N2', 'N3', 'N4']

################################ extract information and visualization
# index of group
for j in range(len(group_name)):
    # j = 0
    iexp = tracked_rvor.root[group_name[j]]
    istart = np.where(time == iexp._v_attrs.time[0])[0][0]
    ifinal = np.where(time == iexp._v_attrs.time[-1])[0][0] + 1
    # time[[istart, ifinal - 1]]
    
    ################################ plot the vortex information
    rvor = rvor_100m.relative_vorticity[istart, 80:920, 80:920].values * 10**4
    ######## plot rvor
    fig, ax = framework_plot1(
        "1km_lb",
        plot_vorticity=True,
        xlabel="100-meter relative vorticity [$10^{-4}\;s^{-1}$]",
        vorticity_elements={
            'rvor': rvor,
            'lon': lon,
            'lat': lat,
            'vorlevel': np.arange(-12, 12.1, 0.1), 'ticks': np.arange(-12, 12.1, 3),
            'time_point': time[istart], 'time_location': [-23, 34], },
    )
    
    ################################ plot every time points
    # for i in range(iexp.extracted_vortex_info.nrows):
        # vortex_info_row = iexp.extracted_vortex_info[i, ]
        ######## get center latitude and longitude
        # center_lat = iexp.extracted_vortex_info.cols.center_lat[i]
        # center_lon = iexp.extracted_vortex_info.cols.center_lon[i]
        # vortex_size = iexp.extracted_vortex_info.cols.size[i]
        # vortex_angle = iexp.extracted_vortex_info.cols.angle[i]
        # distance2radius = iexp.extracted_vortex_info.cols.distance2radius[i]
        # vortex_index = iexp.extracted_vortex_info.cols.vortex_index[i]
        
        ######## plot contour
        # ax.contour(
        #     lon, lat, iexp.extracted_vortex_info.cols.is_vortex[i],
        #     colors='lime', levels=np.array([-0.5, 0.5]),
        #     linewidths=0.3, linestyles='solid', zorder=2
        # )
        ######## plot wind
        # ax.quiver(
        #     center_lon, center_lat,
        #     iexp.extracted_vortex_info.cols.mean_wind_u[i],
        #     iexp.extracted_vortex_info.cols.mean_wind_v[i],
        #     rasterized=True,
        # )
        ######## plot text
        # ax.text(
        #     center_lon+0.5, center_lat,
        #     str(vortex_index) + ':' + str(int(vortex_size)) +
        #     ':' + str(int(vortex_angle)) + ':' +
        #     str(round(distance2radius, 1)),
        #     color='m', size=4, fontweight='bold'
        # )
    
    ################################ plot important information
    # contour
    ax.contour(
        lon, lat, iexp.extracted_vortex_info.cols.is_vortex[0],
        colors='lime', levels=np.array([-0.5, 0.5]),
        linewidths=0.3, linestyles='solid', zorder=2
    )
    # track
    ax.plot(iexp.extracted_vortex_info.cols.center_lon[:],
            iexp.extracted_vortex_info.cols.center_lat[:],
            '.-', color = 'black',
            linestyle='dashed', linewidth=0.5, markersize = 2.5,
            )
    
    # topography
    ax.contourf(
        lon, lat, model_topo_mask,
        colors='white', levels=np.array([0.5, 1.5]))
    
    # label
    ax.text(
        iexp.extracted_vortex_info.cols.center_lon[0] + 0.4,
        iexp.extracted_vortex_info.cols.center_lat[1] - 0.1, group_name[j],
        color='k', size=10, fontweight='normal')
    
    fig.savefig(
        outputfile_prefix + str(j) + '.1 vortex track of ' + group_name[j] +
        # '_detail' +
        '.png',
        dpi=600)
    plt.close('all')
    print(j)


tracked_rvor.close()

# endregion
# =============================================================================






