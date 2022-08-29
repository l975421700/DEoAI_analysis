

# =============================================================================
# region Function to generate ticks and labels
import cartopy.crs as ccrs
import numpy as np

def ticks_labels(
    xmin, xmax, ymin, ymax, xspacing, yspacing
    ):
    '''
    # input ----
    xmin, xmax, ymin, ymax: range of labels
    xspacing: spacing of x ticks
    yspacing: spacing of y ticks
    
    # output ----
    xticks_pos, xticks_label, yticks_pos, yticks_label
    '''
    
    # get the x ticks
    xticks_pos = np.arange(xmin, xmax + xspacing, xspacing)
    if not isinstance(xspacing, int):
        xticks_pos = np.around(xticks_pos, 1)
    
    # Associate with '° W', '°', and '° E'
    xticks_label = [''] * len(xticks_pos)
    for i in np.arange(len(xticks_pos)):
        if xticks_pos[i] < 0:
            xticks_label[i] = str(abs(xticks_pos[i])) + '° W'
        if xticks_pos[i] == 0:
            xticks_label[i] = str(xticks_pos[i]) + '°'
        if xticks_pos[i] > 0:
            xticks_label[i] = str(xticks_pos[i]) + '° E'
    
    # get the y ticks
    yticks_pos = np.arange(ymin, ymax + yspacing, yspacing)
    if not isinstance(yspacing, int):
        yticks_pos = np.around(yticks_pos, 1)
    
    # Associate with '° N', '°', and '° S'
    yticks_label = [''] * len(yticks_pos)
    for i in np.arange(len(yticks_pos)):
        if yticks_pos[i] < 0:
            yticks_label[i] = str(abs(yticks_pos[i])) + '° S'
        if yticks_pos[i] == 0:
            yticks_label[i] = str(yticks_pos[i]) + '°'
        if yticks_pos[i] > 0:
            yticks_label[i] = str(yticks_pos[i]) + '° N'
    
    return xticks_pos, xticks_label, yticks_pos, yticks_label

'''
xmin = -180; xmax = 180; ymin = -90; ymax = 90; xspacing = 60; yspacing = 30
xmin = -24; xmax = -12; ymin = 24; ymax = 33; xspacing = 3; yspacing = 3
xmin = -17.3; xmax = -16.6; ymin = 32.6; ymax = 33.0; xspacing = 0.1; yspacing = 0.1
xticks_pos = np.arange(xmin, xmax + xspacing, xspacing)
ddd = ticks_labels(xmin, xmax, ymin, ymax, xspacing, yspacing)
'''
# endregion
# =============================================================================


# =============================================================================
# region Function to plot scale bar
from matplotlib.patches import Rectangle

# plot a scale bar with 4 subdivisions on the left side of the map
# https://github.com/SciTools/cartopy/issues/490
def scale_bar(ax, bars = 2, length = None, location = (0.1, 0.05),
              barheight = 5, linewidth = 3, col = 'black',
              middle_label=True, fontcolor='black', vline = 1400):
    """
    ax: the axes to draw the scalebar on.
    bars: the number of subdivisions of the bar (black and white chunks)
    length: the length of the scalebar in km.
    location: left side of the scalebar in axis coordinates.
    (ie. 0 is the left side of the plot)
    barheight: height of bar in [km]
    linewidth: the thickness of the scalebar.
    color: the color of the scale bar
    middle_label: whether to plot the middle label
    """
    
    # Get the limits of the axis in lat lon
    llx0, llx1, lly0, lly1 = ax.get_extent(ccrs.PlateCarree())
    
    # Make tmc aligned to the left of the map, vertically at scale bar location
    sbllx = llx0 + (llx1 - llx0) * location[0]
    sblly = lly0 + (lly1 - lly0) * location[1]
    tmc = ccrs.TransverseMercator(sbllx, sblly, approx = False)
    
    # Get the extent of the plotted area in coordinates in metres
    x0, x1, y0, y1 = ax.get_extent(tmc)
    
    # Turn the specified scalebar location into coordinates in metres
    sbx = x0 + (x1 - x0) * location[0]
    sby = y0 + (y1 - y0) * location[1]
    
    # Generate the x coordinate for the ends of the scalebar
    bar_xs = [sbx, sbx + length * 1000 / bars]
    # Plot the scalebar chunks
    barcol = 'white'
    
    for i in range(0, bars):
        # Generate the x coordinate for the left of ith bar
        barleft_x = sbx + i * length * 1000 / bars
        
        # plot the chunk
        ax.add_patch(
            Rectangle((barleft_x, sby),
                      length * 1000 / bars, barheight * 1000, ec = 'black',
                      color = barcol, lw = linewidth, transform = tmc))
        
        # ax.plot(bar_xs, [sby, sby], transform=tmc, color=barcol, linewidth=linewidth)
        
        # alternate the colour
        if barcol == 'white':
            barcol = col
        else:
            barcol = 'white'
        # Plot the scalebar label for that chunk
        if i == 0 or middle_label:
            ax.text(barleft_x, sby + barheight * 1200,
                    str(round(i * length / bars)), transform=tmc,
                    horizontalalignment='center', verticalalignment='bottom',
                    color=fontcolor)
        
    # Generate the x coordinate for the last number
    bar_xt = sbx + length * 1000 * 1.1
    # Plot the last scalebar label
    ax.text(bar_xt, sby + barheight * vline, str(round(length)) + ' km',
            transform=tmc, horizontalalignment='center',
            verticalalignment='bottom', color=fontcolor)

'''
# bars = 2; length = 1000; location=(0.1, 0.05); linewidth=3; col='black'
'''
# endregion
# =============================================================================


# =============================================================================
# region Function to plot advance framework

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib as mpl
import matplotlib.ticker as mticker
mpl.rc('font', family='Times New Roman', size=10)
import cartopy as ctp
from matplotlib import cm
from matplotlib.colors import ListedColormap
from matplotlib.colors import BoundaryNorm

# set of namelist----
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
    facecolor='none', lw=0.5)
borders = ctp.feature.NaturalEarthFeature(
    'cultural', 'admin_0_boundary_lines_land', '10m', edgecolor='black',
    facecolor='none', lw=0.5)

'''
'''

def framework_plot1(
    which_area,
    xlabel = None,
    output_png=None,
    dpi=600,
    figsize=None,
    country_boundaries=True, border_color = 'black',
    gridlines=True, grid_color = 'gray',
    ticks_and_labels = True,
    lw=0.25, labelsize=10, extent=None, ticklabel = None,
    plot_scalebar = True,
    scalebar_elements={'bars': 2, 'length': 200, 'location': (0.02, 0.015),
                       'barheight': 20, 'linewidth': 0.15, 'col': 'black',
                       'middle_label': False},
    set_figure_margin = True,
    figure_margin=None,
    plot_vorticity = False,
    vorticity_elements = {
        'rvor': None, 'lon': None, 'lat': None, 'vorlevel': None,
        'ticks': None, 'time_point': None, 'time_location': None,
        },
    ):
    '''
    ----Input
    which_area: indicate which area to plot, {"1km", "12km", "madeira", "canary"}
    
    figsize: figure size
    
    
    ----output
    
    
    '''
    
    if which_area == "1km_lb":
        extent = extent1km_lb
        ticklabel = ticklabel1km_lb
        if plot_scalebar:
            scalebar_elements = {
                'bars': 2, 'length': 200, 'location': (0.02, 0.015),
                'barheight': 20, 'linewidth': 0.15, 'col': 'black',
                'middle_label': False}
        if (figure_margin is None) & (set_figure_margin):
            figure_margin = {
                'left': 0.12, 'right': 0.99, 'bottom': 0.08, 'top': 0.99
            }
        if (figsize is None):
            figsize = np.array([8.8, 8.8]) / 2.54
    
    if which_area == "1km":
        extent = extent1km
        ticklabel = ticklabel1km
    
    if which_area == "12km":
        extent = extent12km
        ticklabel = ticklabel12km
    
    if which_area == "12km_out":
        extent = extent12km_out
        ticklabel = ticklabel12km_out
        plot_scalebar = False
        if (figsize is None):
            figsize = np.array([8.8, 8]) / 2.54
        if (figure_margin is None) & (set_figure_margin):
            figure_margin = {
                'left': 0.12, 'right': 0.96, 'bottom': 0.05, 'top': 0.995
            }
    
    if which_area == "madeira":
        extent = extentm
        ticklabel = ticklabelm
    
    if which_area == "canary":
        extent = extentc
        ticklabel = ticklabelc
    
    if which_area == 'global':
        extent = extent_global
        ticklabel = ticklabel_global
        plot_scalebar = False
        lw = 0.1
        if (figure_margin is None) & (set_figure_margin):
            figure_margin = {
                'left': 0.06, 'right': 0.97, 'bottom': 0.05, 'top': 0.995
            }
        if (figsize is None):
            figsize = np.array([8.8*2, 8.8]) / 2.54
    
    if which_area == 'self_defined':
        extent = extent
        ticklabel = ticklabel
        plot_scalebar = False
        if (figsize is None):
            figsize = np.array([8.8, 8.8]) / 2.54
    
    fig, ax = plt.subplots(
        1, 1, figsize=figsize, subplot_kw={'projection': transform}, dpi=dpi)
    
    ax.set_extent(extent, crs = transform)
    if ticks_and_labels:
        ax.set_xticks(ticklabel[0])
        ax.set_xticklabels(ticklabel[1])
        ax.set_yticks(ticklabel[2])
        ax.set_yticklabels(ticklabel[3])
        ax.tick_params(length=2, labelsize=labelsize)
    
    if country_boundaries:
        coastline = cfeature.NaturalEarthFeature(
            'physical', 'coastline', '10m', edgecolor=border_color,
            facecolor='none', lw=lw)
        ax.add_feature(coastline, zorder=2)
        borders = cfeature.NaturalEarthFeature(
            'cultural', 'admin_0_boundary_lines_land', '10m',
            edgecolor=border_color,
            facecolor='none', lw=lw)
        ax.add_feature(borders, zorder=2)
    
    if gridlines:
        gl = ax.gridlines(crs=transform, linewidth=lw, zorder=2,
                          color=grid_color, alpha=0.5, linestyle='--')
        gl.xlocator = mticker.FixedLocator(ticklabel[0])
        gl.ylocator = mticker.FixedLocator(ticklabel[2])
    
    if plot_vorticity:
        top = cm.get_cmap('Blues_r', int(np.floor(len(
            vorticity_elements['vorlevel']) / 2)))
        bottom = cm.get_cmap('Reds', int(np.floor(len(
            vorticity_elements['vorlevel']) / 2)))
        newcolors = np.vstack(
            (top(np.linspace(0, 1, int(np.floor(len(
                vorticity_elements['vorlevel']) / 2)))),
             [1, 1, 1, 1],
             bottom(np.linspace(0, 1, int(np.floor(len(
                 vorticity_elements['vorlevel']) / 2))))))
        newcmp = ListedColormap(newcolors, name='RedsBlues_r')
        
        plt_rvor = ax.pcolormesh(
            vorticity_elements['lon'],
            vorticity_elements['lat'],
            vorticity_elements['rvor'],
            cmap=newcmp,
            norm=BoundaryNorm(
                vorticity_elements['vorlevel'], ncolors=newcmp.N, clip=False),
            rasterized=True, transform=transform, zorder=-2,)
        rvor_time = ax.text(
            vorticity_elements['time_location'][0],
            vorticity_elements['time_location'][1],
            str(vorticity_elements['time_point'])[0:10] + ' ' +
            str(vorticity_elements['time_point'])[11:13] + ':00 UTC')
        cbar = fig.colorbar(
            plt_rvor, ax=ax, orientation="horizontal",  pad=0.1, fraction=0.09,
            shrink=1, aspect=25, ticks=vorticity_elements['ticks'],
            extend='both', anchor=(0.5, 1), panchor=(0.5, 0))
        if xlabel is None:
            xlabel = "Relative vorticity [$10^{-4}\;s^{-1}$]"
        cbar.ax.set_xlabel(xlabel)
    
    if plot_scalebar:
        scale_bar(ax, bars=scalebar_elements['bars'],
                  length=scalebar_elements['length'],
                  location=scalebar_elements['location'],
                  barheight=scalebar_elements['barheight'],
                  linewidth=scalebar_elements['linewidth'],
                  col=scalebar_elements['col'],
                  middle_label=scalebar_elements['middle_label'])
    
    if set_figure_margin & (not(figure_margin is None)):
        fig.subplots_adjust(
            left=figure_margin['left'], right=figure_margin['right'],
            bottom=figure_margin['bottom'], top=figure_margin['top'])
    else:
        fig.tight_layout()
    
    if (not(output_png is None)):
        fig.savefig(output_png, dpi=dpi)
    else:
        return fig, ax


'''
framework_plot1("1km_lb", output_png = 'figures/00_test/trial.png')


# plot natural earth
fig, ax = framework_plot1("global")
ax.background_img(name='natural_earth', resolution='high')
fig.savefig('figures/00_test/natural_earth.png', dpi=1200)

fig, ax = framework_plot1("madeira", plot_scalebar = False)
fig.savefig('figures/00_test/trial.png')

'''

# endregion
# =============================================================================


# =============================================================================
# region Function to plot a framework
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 1200
import matplotlib.font_manager
import matplotlib.ticker as mticker
mpl.rc('font', size = 10)
from DEoAI_analysis.module.namelist import (
    extent1km,
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
)


def framework_plot(
    which_area, figsize = np.array([8.8, 9.6]) / 2.54,
    country_boundaries=True, border_color = 'black',
    gridlines=True, grid_color = 'gray',
    ticks_and_labels = True,
    lw=0.25, labelsize=10, extent=None, ticklabel = None,
    plot_vorticity = False,
    vorticity_elements = {}
    ):
    '''
    ----Input
    which_area: indicate which area to plot, {"1km", "12km", "madeira", "canary"}
    
    figsize: figure size
    
    
    ----output
    
    
    '''
    
    if which_area == "1km_lb":
        extent = extent1km_lb
        ticklabel = ticklabel1km_lb
    
    if which_area == "1km":
        extent = extent1km
        ticklabel = ticklabel1km
    
    if which_area == "12km":
        extent = extent12km
        ticklabel = ticklabel12km
    
    if which_area == "madeira":
        extent = extentm
        ticklabel = ticklabelm
    
    if which_area == "canary":
        extent = extentc
        ticklabel = ticklabelc
    
    if which_area == 'global':
        extent = extent_global
        ticklabel = ticklabel_global
    
    if which_area == 'self_defined':
        extent = extent
        ticklabel = ticklabel
    
    fig, ax = plt.subplots(
        1, 1, figsize = figsize, subplot_kw={'projection': transform})
    
    ax.set_extent(extent, crs = transform)
    if ticks_and_labels:
        ax.set_xticks(ticklabel[0])
        ax.set_xticklabels(ticklabel[1])
        ax.set_yticks(ticklabel[2])
        ax.set_yticklabels(ticklabel[3])
        ax.tick_params(length=2, labelsize=labelsize)
    
    if country_boundaries:
        coastline = cfeature.NaturalEarthFeature(
            'physical', 'coastline', '10m', edgecolor=border_color,
            facecolor='none', lw=lw)
        ax.add_feature(coastline, zorder=2)
        borders = cfeature.NaturalEarthFeature(
            'cultural', 'admin_0_boundary_lines_land', '10m',
            edgecolor=border_color,
            facecolor='none', lw=lw)
        ax.add_feature(borders, zorder=2)
    
    if gridlines:
        gl = ax.gridlines(crs=transform, linewidth=lw, zorder=2,
                          color=grid_color, alpha=0.5, linestyle='--')
        gl.xlocator = mticker.FixedLocator(ticklabel[0])
        gl.ylocator = mticker.FixedLocator(ticklabel[2])
    
    return fig, ax

'''
fig, ax = framework_plot(
    "global", figsize = np.array([8.8, 4.4]) / 2.54, lw=0.1, labelsize = 8
    )
fig.subplots_adjust(left=0.09, right = 0.96, bottom = 0.08, top = 0.99)
fig.savefig('figures/00_test/trial.png', dpi = 600)

'''
# endregion
# =============================================================================


# =============================================================================
# region Function to plot confidence ellipse
# https://matplotlib.org/3.3.2/gallery/statistics/confidence_ellipse.html#sphx-glr-gallery-statistics-confidence-ellipse-py
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.
    
    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.
    
    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.
    
    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.
    
    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`
    
    Returns
    -------
    matplotlib.patches.Ellipse
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")
    
    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)
    
    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)
    
    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)
    
    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)
    
    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

# endregion
# =============================================================================




