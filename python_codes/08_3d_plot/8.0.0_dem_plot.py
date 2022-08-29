

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

import sys  # print(sys.path)
sys.path.append(
    '/Users/gao/OneDrive - whu.edu.cn/ETH/Courses/4. Semester/DEoAI')
sys.path.append('/project/pr94/qgao/DEoAI')
sys.path.append('/scratch/snx3000/qgao')
sys.path.append('/home/qigao/DEoAI')

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
import rasterio as rio
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
    era5_folder,
    
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
# region plot 2d dem

srtm_files = sorted(
    glob.glob('data_source/topograph/srtm_1arc_second_global/*.tif'))
# srtm_files[-9]
dem = []
dem_bounds = []
dem_lon = []
dem_lat = []
for i in np.arange(0, len(srtm_files)):
    
    idem = rio.open(srtm_files[i], masked=True)
    idem_bounds = idem.bounds
    
    idem_data = idem.read(1)
    
    idem_data = np.ma.array(idem_data)
    idem_data[idem_data == 0] = np.ma.masked
    
    # if len(np.where(idem_data == 0)[0]) > 0:
    #     idem_data[idem_data == 0] = np.nan
    # idem_data = np.ma.array(idem_data)
    # idem_data[np.isnan(idem_data)] = np.ma.masked
    
    ilon, ilat = np.meshgrid(
        np.linspace(
            idem_bounds.left, idem_bounds.right,
            int(idem_data.shape[1])),
        np.linspace(
            idem_bounds.bottom, idem_bounds.top,
            int(idem_data.shape[1])),
        sparse=True)
    
    dem.append([idem_data])
    dem_bounds.append([idem_bounds])
    dem_lon.append([ilon])
    dem_lat.append([ilat])
    del idem, idem_bounds, ilon, ilat
    
    print(str(i) + '   ' + str(np.max(dem[i][0])))


begin_time = datetime.datetime.now()
fig, ax = plt.subplots(
    1, 1, figsize=np.array([8.8, 8.8]) / 2.54,
    subplot_kw={'projection': transform}, dpi=300)


demlevel = np.arange(0, 3500.1, 25)
ticks = np.arange(0, 3500.1, 500)

for i in np.arange(0, len(srtm_files)):  # len(srtm_files)
    
    plt_dem = ax.pcolormesh(
        dem_lon[i][0], dem_lat[i][0], dem[i][0][::-1, :],
        norm=BoundaryNorm(demlevel, ncolors=len(demlevel), clip=False),
        cmap=cm.get_cmap('terrain', len(demlevel)), rasterized=True,
        transform=transform,
    )
    
    if i == 0:
        cbar = fig.colorbar(
            plt_dem, orientation="horizontal",  pad=0.1, fraction=0.09,
            shrink=1, aspect=25, ticks=ticks, extend='both')
        cbar.ax.set_xlabel("Topography [m]")
    
    print(str(i) + '/' + str(len(srtm_files)))


ax.add_feature(borders, lw = 0.25); ax.add_feature(coastline, lw = 0.25)
ax.set_xticks(ticklabel1km_lb[0]); ax.set_xticklabels(ticklabel1km_lb[1])
ax.set_yticks(ticklabel1km_lb[2]); ax.set_yticklabels(ticklabel1km_lb[3])

gl = ax.gridlines(crs=transform, linewidth=0.25, color='gray',
                  alpha=0.5, linestyle='--')
gl.xlocator = mticker.FixedLocator(ticklabel1km_lb[0])
gl.ylocator = mticker.FixedLocator(ticklabel1km_lb[2])

ax.set_extent(extent1km_lb, crs=transform)

scale_bar(ax, bars=2, length=200, location=(0.04, 0.03),
          barheight=20, linewidth=0.15, col='black', middle_label=False)

fig.subplots_adjust(left=0.155, right=0.975, bottom=0.08, top=0.99)
fig.savefig('figures/08_3d_plot/8.0.0 domain dem.png')

print(str(datetime.datetime.now() - begin_time))



'''

'''

# endregion
# =============================================================================


# =============================================================================
# region plot 3d dem mplot3d

from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt

def f(x, y):
    return np.sin(np.sqrt(x ** 2 + y ** 2))

x = np.linspace(-6, 6, 30)
y = np.linspace(-6, 6, 30)

X, Y = np.meshgrid(x, y)
Z = f(X, Y)

fig = plt.figure(dpi = 600)
ax = plt.axes(projection='3d')
ax.contour3D(X, Y, Z, 50, cmap='binary')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z');

ax.view_init(60, 35)
fig.savefig('figures/08_3d_plot/8.0.1 domain 3d dem.png')






import itertools

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PolyCollection
import numpy as np

import cartopy.feature
from cartopy.mpl.patch import geos_to_path
import cartopy.crs as ccrs


fig = plt.figure()
ax = Axes3D(fig, xlim=[-180, 180], ylim=[-90, 90])
ax.set_zlim(bottom=0)

concat = lambda iterable: list(itertools.chain.from_iterable(iterable))

target_projection = ccrs.PlateCarree()

feature = cartopy.feature.NaturalEarthFeature('physical', 'land', '110m')
geoms = feature.geometries()

geoms = [target_projection.project_geometry(geom, feature.crs)
         for geom in geoms]

paths = concat(geos_to_path(geom) for geom in geoms)

polys = concat(path.to_polygons() for path in paths)

lc = PolyCollection(polys, edgecolor='black',
                    facecolor='green', closed=False)

ax.add_collection3d(lc)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Height')

fig.savefig('figures/08_3d_plot/8.0.1 domain 3d dem.png')





import cartopy.crs as ccrs
import cartopy.feature
from cartopy.mpl.patch import geos_to_path

import itertools
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.collections import PolyCollection
import numpy as np


def f(x,y):
    x, y = np.meshgrid(x, y)
    return (1 - x / 2 + x**5 + y**3 + x*y**2) * np.exp(-x**2 -y**2)

nx, ny = 256, 512
X = np.linspace(-180, 180, nx)
Y = np.linspace(-90, 90, ny)
Z = f(np.linspace(-3, 3, nx), np.linspace(-3, 3, ny))


fig = plt.figure()
ax3d = fig.add_axes([0, 0, 1, 1], projection='3d')

# Make an axes that we can use for mapping the data in 2d.
proj_ax = plt.figure().add_axes([0, 0, 1, 1], projection=ccrs.Mercator())
cs = proj_ax.contourf(X, Y, Z, transform=ccrs.PlateCarree(), alpha=0.4)


for zlev, collection in zip(cs.levels, cs.collections):
    paths = collection.get_paths()
    # Figure out the matplotlib transform to take us from the X, Y coordinates
    # to the projection coordinates.
    trans_to_proj = collection.get_transform() - proj_ax.transData

    paths = [trans_to_proj.transform_path(path) for path in paths]
    verts3d = [np.concatenate([path.vertices,
                               np.tile(zlev, [path.vertices.shape[0], 1])],
                              axis=1)
               for path in paths]
    codes = [path.codes for path in paths]
    pc = Poly3DCollection([])
    pc.set_verts_and_codes(verts3d, codes)

    # Copy all of the parameters from the contour (like colors) manually.
    # Ideally we would use update_from, but that also copies things like
    # the transform, and messes up the 3d plot.
    pc.set_facecolor(collection.get_facecolor())
    pc.set_edgecolor(collection.get_edgecolor())
    pc.set_alpha(collection.get_alpha())

    ax3d.add_collection3d(pc)

proj_ax.autoscale_view()

ax3d.set_xlim(*proj_ax.get_xlim())
ax3d.set_ylim(*proj_ax.get_ylim())
ax3d.set_zlim(Z.min(), Z.max())


# Now add coastlines.
concat = lambda iterable: list(itertools.chain.from_iterable(iterable))

target_projection = proj_ax.projection

feature = cartopy.feature.NaturalEarthFeature('physical', 'land', '110m')
geoms = feature.geometries()

# Use the convenience (private) method to get the extent as a shapely geometry.
boundary = proj_ax._get_extent_geom()

# Transform the geometries from PlateCarree into the desired projection.
geoms = [target_projection.project_geometry(geom, feature.crs)
         for geom in geoms]
# Clip the geometries based on the extent of the map (because mpl3d can't do it for us)
# geoms = [boundary.intersection(geom) for geom in geoms]

# Convert the geometries to paths so we can use them in matplotlib.
paths = concat(geos_to_path(geom) for geom in geoms)
polys = concat(path.to_polygons() for path in paths)
lc = PolyCollection(polys, edgecolor='black',
                    facecolor='green', closed=False)
ax3d.add_collection3d(lc, zs=ax3d.get_zlim()[0])

plt.close(proj_ax.figure)
plt.savefig('figures/08_3d_plot/8.0.1 domain 3d dem.png')
plt.close('all')
# endregion
# =============================================================================


# =============================================================================
# region plot 3d dem pyvista

import pyvista as pv
from pyvista import examples

mesh = pv.read(examples.planefile)

plotter = pv.Plotter(off_screen=True, interactive=False)
plotter.add_mesh(mesh)
plotter.show(screenshot="myscreenshot.png")



from pyvista import set_plot_theme
set_plot_theme('document')

import pyvista as pv
from pyvista import examples

earth_alt = examples.download_topo_global()

pl = pv.Plotter(off_screen=True)
actor = pl.add_mesh(examples.load_airplane(), smooth_shading=True)
# pl.add_background_image(examples.mapfile)
pl.save('figures/08_3d_plot/8.0.1 domain 3d dem.vtk')

# endregion
# =============================================================================

