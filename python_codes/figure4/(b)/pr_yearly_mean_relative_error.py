import Ngl,Nio     #-- import PyNGL and PyNIO modules
import netCDF4 as nc
import numpy as np
import sys
import os
import cartopy
import cartopy.crs as ccrs
from cartopy.feature import ShapelyFeature
from cartopy.io.shapereader import Reader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm


# dirData = '/mnt/f/ETH/Madeira/Validação/Resultados/pr_MAD-12km_17Stat_2006-2015/'
# dirDataObs = '/mnt/f/ETH/Madeira/Validação/Resultados/pr_MAD-1km_17Stat_2006-2015/'
# dirFig = '/mnt/f/ETH/Madeira/Validação/Figuras/Mapas/'
dirData = 'DEoAI_analysis/python_codes/figure4/(b)/12km/'
dirDataObs = 'DEoAI_analysis/python_codes/figure4/(b)/1km/'
dirFig = 'figures/11_figure4/'

fileout = dirFig + 'pr_yearly_mean_relative_error.png'

ncols=3
nlon=17
nlat=1
nrows=nlon*nlat

extent = [-17.3, -16.6, 32.6, 32.9]
fig = plt.figure(figsize=(6.14,1.5))

isolinhas = np.arange(-90,100,10)
N = len(isolinhas)
mapacor = plt.cm.RdBu
mapacorlist = [mapacor(i) for i in range(mapacor.N)]
mapacor = mapacor.from_list('Custom cmap',mapacorlist,mapacor.N)
norm = BoundaryNorm(isolinhas,mapacor.N)
cbartitle = '%'

#ax = fig.add_subplot(1,3,1,projection=ccrs.PlateCarree())
#ax.set_extent(extent)
#ax.set_title('MAD-Stat',fontsize=9)

shpfilename = 'DEoAI_analysis/python_codes/figure4/shapefile/gadm36_PRT_0.shp'
shape_feature = ShapelyFeature(Reader(shpfilename).geometries(),ccrs.PlateCarree(),facecolor='gainsboro',edgecolor='k',linewidth=0.5) 
#ax.add_feature(shape_feature)

#ax.background_patch.set_visible(False)
#ax.outline_patch.set_visible(False)

#Make 1km plot
fname = dirDataObs + 'pr_MAD_Madeira_v_Stat_2006-2015_yearly_accumulated_relative_error_MAD.out'
data=Ngl.asciiread(fname,(nrows,ncols),"float")
lat=data[:,1]
lon=data[:,0]
pr=data[:,2]

ax = fig.add_subplot(1,3,2,projection=ccrs.PlateCarree())
ax.set_extent(extent)
ax.set_title('MAD-1km',fontsize=9)

ax.add_feature(shape_feature)

ax.background_patch.set_visible(False)
ax.outline_patch.set_visible(False)

cp=plt.scatter(lon,lat,c=pr,s=8,cmap=mapacor,transform=ccrs.PlateCarree(),norm=norm,zorder=2)

#Make 12km plot
fname = dirData + 'pr_MAD_Madeira_v_Stat_2006-2015_yearly_accumulated_relative_error_MAD.out'
data=Ngl.asciiread(fname,(nrows,ncols),"float")
lat=data[:,1]
lon=data[:,0]
pr=data[:,2]

ax = fig.add_subplot(1,3,3,projection=ccrs.PlateCarree())
ax.set_extent(extent)
ax.set_title('MAD-12km',fontsize=9)

ax.add_feature(shape_feature)

ax.background_patch.set_visible(False)
ax.outline_patch.set_visible(False)

cp=plt.scatter(lon,lat,c=pr,s=8,cmap=mapacor,transform=ccrs.PlateCarree(),norm=norm,zorder=2)

fig.subplots_adjust(left=0,right=0.93,bottom=0,top=0.98,wspace=0,hspace=0)
cbar_ax = fig.add_axes([0.93,0.1,0.02,0.7])     
cbar=fig.colorbar(cp,spacing='proportional',cax=cbar_ax,shrink=0.6)
cbar.ax.set_title(cbartitle,fontsize=6)
cbar.ax.tick_params(labelsize=6)

fig.savefig(fileout,dpi=600)
    
plt.close('all')