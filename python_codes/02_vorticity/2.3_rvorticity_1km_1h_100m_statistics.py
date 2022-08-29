

# region import packages ----

import numpy as np
import xarray as xr
import datetime
import glob

import sys
sys.path.append('/Users/gao/OneDrive - whu.edu.cn/ETH/Courses/4. Semester/DEoAI')
sys.path.append('/project/pr94/qgao/DEoAI')
sys.path.append('/scratch/snx3000/qgao')

from DEoAI_analysis.module.mapplot import (
    ticks_labels,
    scale_bar,
)

from DEoAI_analysis.module.namelist import (
    month,
    seasons,
    quantiles,
    timing,
)

from DEoAI_analysis.module.statistics_calculate import(
    get_statistics
)

# endregion

folder_rvorticity_1km_1h_100m = \
    'scratch/relative_vorticity_1km_1h_100m'

filelist_rvorticity_1km_1h_100m = \
    sorted(glob.glob(folder_rvorticity_1km_1h_100m + '/*100m2010*'))


begin_time = datetime.datetime.now()
rvorticity_1km_1h_100m = xr.open_mfdataset(
    filelist_rvorticity_1km_1h_100m, concat_dim="time",
    data_vars='minimal', coords='minimal', compat='override'
)

rlon = rvorticity_1km_1h_100m.rlon.data
rlat = rvorticity_1km_1h_100m.rlat.data
lon = rvorticity_1km_1h_100m.lon.data
lat = rvorticity_1km_1h_100m.lat.data


rvorticity_1km_1h_100m_statistics = xr.Dataset(
    {"rvorticity_quantiles": (
        ("timing", "quantiles", "rlat", "rlon"),
        np.zeros((len(timing),
                  len(quantiles[1]),
                  len(rlat) - 160,
                  len(rlon) - 160,
                  ))),
     "lat": (("rlat", "rlon"), lat[80:920, 80:920]),
     "lon": (("rlat", "rlon"), lon[80:920, 80:920]),
     },
    coords={
        "timing": timing,
        "quantiles": quantiles[1],
        "rlat": rlat[80:920],
        "rlon": rlon[80:920],
        }
)

q = quantiles[0]
axis = 0

# range(21)
for i in range(21):
    arr = np.array(rvorticity_1km_1h_100m.relative_vorticity[
        :, np.arange(80, 120) + i * 40, 80:920])
    rvorticity_1km_1h_100m_statistics.rvorticity_quantiles[
        0, :, np.arange(0, 40) + i * 40, :] = get_statistics(
        arr=arr, q=q, axis=axis
    )

rvorticity_1km_1h_100m_statistics.to_netcdf(
    "scratch/relative_vorticity_1km_1h_100m2010_statistics.nc"
)


# region check ----
# ddd = xr.open_dataset(
#     '/project/pr94/qgao/DEoAI/scratch/relative_vorticity_1km_1h_100m2010_statistics.nc')

# np.array(ddd.rvorticity_quantiles[0, 0, 839, 839])
# np.array(np.min(rvorticity_1km_1h_100m.relative_vorticity[:, 919, 919]))
# np.array(ddd.rvorticity_quantiles[0, 8, 839, 839])
# np.array(np.max(rvorticity_1km_1h_100m.relative_vorticity[:, 919, 919]))

# np.array(ddd.rvorticity_quantiles[0, 11, 839, 839])
# np.array(ddd.rvorticity_quantiles[0, 6, 839, 839]) - \
#     np.array(ddd.rvorticity_quantiles[0, 2, 839, 839])

# np.array(ddd.rvorticity_quantiles[0, 13, 839, 839])
# np.array(np.mean(rvorticity_1km_1h_100m.relative_vorticity[:, 919, 919]))


# ticklabel = ticks_labels(-30, 0, 10, 40, 10, 10)
# extent = [-35, 0, 10, 45]
# transform = ctp.crs.PlateCarree()

# fig, ax = plt.subplots(
#     1, 1, figsize = np.array([8.8, 9.6]) / 2.54,
#     subplot_kw={'projection': transform})
# ax.set_extent(extent, crs = transform)
# ax.set_xticks(ticklabel[0])
# ax.set_xticklabels(ticklabel[1])
# ax.set_yticks(ticklabel[2])
# ax.set_yticklabels(ticklabel[3])

# gl = ax.gridlines(crs = transform, linewidth = 0.5,
#                   color = 'gray', alpha = 0.5, linestyle='--')
# gl.xlocator = mticker.FixedLocator(ticklabel[0])
# gl.ylocator = mticker.FixedLocator(ticklabel[2])

# coastline = ctp.feature.NaturalEarthFeature(
#     'physical', 'coastline', '10m', edgecolor='black',
#     facecolor='none', lw = 0.5)
# ax.add_feature(coastline)
# borders = ctp.feature.NaturalEarthFeature(
#     'cultural', 'admin_0_boundary_lines_land', '10m', edgecolor='black',
#     facecolor='none', lw = 0.5)
# ax.add_feature(borders)

# scale_bar(ax, bars = 2, length = 1000, location = (0.1, 0.05),
#           barheight = 60, linewidth = 0.2, col = 'black')

# plt_12km = ax.contourf(
#     ddd.lon, ddd.lat, np.ones(ddd.lon.shape),
#     transform = transform, colors = 'lightgrey')

# fig.subplots_adjust(left=0.12, right = 0.96, bottom = 0.2, top = 0.99)
# fig.savefig('figures/test/test.png', dpi = 300)


# endregion


