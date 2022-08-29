


# =============================================================================
# region pycwt tutorials
# http://regeirk.github.io/pycwt/tutorial.html
"""
In this example we will load the NINO3 sea surface temperature anomaly dataset
between 1871 and 1996. This and other sample data files are kindly provided by
C. Torrence and G. Compo at
<http://paos.colorado.edu/research/wavelets/software.html>.

"""
# We begin by importing the relevant libraries. Please make sure that PyCWT is
# properly installed in your system.
from __future__ import division
import numpy
from matplotlib import pyplot

import pycwt as wavelet
from pycwt.helpers import find

# Then, we load the dataset and define some data related parameters. In this
# case, the first 19 lines of the data file contain meta-data, that we ignore,
# since we set them manually (*i.e.* title, units).
url = 'http://paos.colorado.edu/research/wavelets/wave_idl/nino3sst.txt'
dat = numpy.genfromtxt(url, skip_header=19)
title = 'NINO3 Sea Surface Temperature'
label = 'NINO3 SST'
units = 'degC'
t0 = 1871.0
dt = 0.25  # In years

# We also create a time array in years.
N = dat.size
t = numpy.arange(0, N) * dt + t0

# We write the following code to detrend and normalize the input data by its
# standard deviation. Sometimes detrending is not necessary and simply
# removing the mean value is good enough. However, if your dataset has a well
# defined trend, such as the Mauna Loa CO\ :sub:`2` dataset available in the
# above mentioned website, it is strongly advised to perform detrending.
# Here, we fit a one-degree polynomial function and then subtract it from the
# original data.
p = numpy.polyfit(t - t0, dat, 1)
dat_notrend = dat - numpy.polyval(p, t - t0)
std = dat_notrend.std()  # Standard deviation
var = std ** 2  # Variance
dat_norm = dat_notrend / std  # Normalized dataset

# The next step is to define some parameters of our wavelet analysis. We
# select the mother wavelet, in this case the Morlet wavelet with
# :math:`\omega_0=6`.
mother = wavelet.Morlet(6)
s0 = 2 * dt  # Starting scale, in this case 2 * 0.25 years = 6 months
dj = 1 / 12  # Twelve sub-octaves per octaves
J = 7 / dj  # Seven powers of two with dj sub-octaves
alpha, _, _ = wavelet.ar1(dat)  # Lag-1 autocorrelation for red noise

# The following routines perform the wavelet transform and inverse wavelet
# transform using the parameters defined above. Since we have normalized our
# input time-series, we multiply the inverse transform by the standard
# deviation.
wave, scales, freqs, coi, fft, fftfreqs = wavelet.cwt(dat_norm, dt, dj, s0, J,
                                                      mother)
iwave = wavelet.icwt(wave, scales, dt, dj, mother) * std

# We calculate the normalized wavelet and Fourier power spectra, as well as
# the Fourier equivalent periods for each wavelet scale.
power = (numpy.abs(wave)) ** 2
fft_power = numpy.abs(fft) ** 2
period = 1 / freqs

# We could stop at this point and plot our results. However we are also
# interested in the power spectra significance test. The power is significant
# where the ratio ``power / sig95 > 1``.
signif, fft_theor = wavelet.significance(1.0, dt, scales, 0, alpha,
                                         significance_level=0.95,
                                         wavelet=mother)
sig95 = numpy.ones([1, N]) * signif[:, None]
sig95 = power / sig95

# Then, we calculate the global wavelet spectrum and determine its
# significance level.
glbl_power = power.mean(axis=1)
dof = N - scales  # Correction for padding at edges
glbl_signif, tmp = wavelet.significance(var, dt, scales, 1, alpha,
                                        significance_level=0.95, dof=dof,
                                        wavelet=mother)

# We also calculate the scale average between 2 years and 8 years, and its
# significance level.
sel = find((period >= 2) & (period < 8))
Cdelta = mother.cdelta
scale_avg = (scales * numpy.ones((N, 1))).transpose()
scale_avg = power / scale_avg  # As in Torrence and Compo (1998) equation 24
scale_avg = var * dj * dt / Cdelta * scale_avg[sel, :].sum(axis=0)
scale_avg_signif, tmp = wavelet.significance(var, dt, scales, 2, alpha,
                                             significance_level=0.95,
                                             dof=[scales[sel[0]],
                                                  scales[sel[-1]]],
                                             wavelet=mother)

# Finally, we plot our results in four different subplots containing the
# (i) original series anomaly and the inverse wavelet transform; (ii) the
# wavelet power spectrum (iii) the global wavelet and Fourier spectra ; and
# (iv) the range averaged wavelet spectrum. In all sub-plots the significance
# levels are either included as dotted lines or as filled contour lines.

# Prepare the figure
pyplot.close('all')
pyplot.ioff()
figprops = dict(figsize=(11, 8), dpi=72)
fig = pyplot.figure(**figprops)

# First sub-plot, the original time series anomaly and inverse wavelet
# transform.
ax = pyplot.axes([0.1, 0.75, 0.65, 0.2])
ax.plot(t, iwave, '-', linewidth=1, color=[0.5, 0.5, 0.5])
ax.plot(t, dat, 'k', linewidth=1.5)
ax.set_title('a) {}'.format(title))
ax.set_ylabel(r'{} [{}]'.format(label, units))

# Second sub-plot, the normalized wavelet power spectrum and significance
# level contour lines and cone of influece hatched area. Note that period
# scale is logarithmic.
bx = pyplot.axes([0.1, 0.37, 0.65, 0.28], sharex=ax)
levels = [0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16]
bx.contourf(t, numpy.log2(period), numpy.log2(power), numpy.log2(levels),
            extend='both', cmap=pyplot.cm.viridis)
extent = [t.min(), t.max(), 0, max(period)]
bx.contour(t, numpy.log2(period), sig95, [-99, 1], colors='k', linewidths=2,
           extent=extent)
bx.fill(numpy.concatenate([t, t[-1:] + dt, t[-1:] + dt,
                           t[:1] - dt, t[:1] - dt]),
        numpy.concatenate([numpy.log2(coi), [1e-9], numpy.log2(period[-1:]),
                           numpy.log2(period[-1:]), [1e-9]]),
        'k', alpha=0.3, hatch='x')
bx.set_title('b) {} Wavelet Power Spectrum ({})'.format(label, mother.name))
bx.set_ylabel('Period (years)')
#
Yticks = 2 ** numpy.arange(numpy.ceil(numpy.log2(period.min())),
                           numpy.ceil(numpy.log2(period.max())))
bx.set_yticks(numpy.log2(Yticks))
bx.set_yticklabels(Yticks)

# Third sub-plot, the global wavelet and Fourier power spectra and theoretical
# noise spectra. Note that period scale is logarithmic.
cx = pyplot.axes([0.77, 0.37, 0.2, 0.28], sharey=bx)
cx.plot(glbl_signif, numpy.log2(period), 'k--')
cx.plot(var * fft_theor, numpy.log2(period), '--', color='#cccccc')
cx.plot(var * fft_power, numpy.log2(1./fftfreqs), '-', color='#cccccc',
        linewidth=1.)
cx.plot(var * glbl_power, numpy.log2(period), 'k-', linewidth=1.5)
cx.set_title('c) Global Wavelet Spectrum')
cx.set_xlabel(r'Power [({})^2]'.format(units))
cx.set_xlim([0, glbl_power.max() + var])
cx.set_ylim(numpy.log2([period.min(), period.max()]))
cx.set_yticks(numpy.log2(Yticks))
cx.set_yticklabels(Yticks)
pyplot.setp(cx.get_yticklabels(), visible=False)

# Fourth sub-plot, the scale averaged wavelet spectrum.
dx = pyplot.axes([0.1, 0.07, 0.65, 0.2], sharex=ax)
dx.axhline(scale_avg_signif, color='k', linestyle='--', linewidth=1.)
dx.plot(t, scale_avg, 'k-', linewidth=1.5)
dx.set_title('d) {}--{} year scale-averaged power'.format(2, 8))
dx.set_xlabel('Time (year)')
dx.set_ylabel(r'Average variance [{}]'.format(units))
ax.set_xlim([t.min(), t.max()])

pyplot.show()

# endregion
# =============================================================================


# =============================================================================
# region PyTable tutorial
import tables
import numpy as np

class Particle(tables.IsDescription):
    name = tables.StringCol(16)     # 16-character String
    idnumber = tables.Int64Col()    # Signed 64-bit integer
    ADCcount = tables.UInt16Col()   # Unsigned short integer
    TDCcount = tables.UInt8Col()    # unsigned byte
    grid_i = tables.Int32Col()      # integer
    grid_j = tables.Int32Col()      # integer
    pressure = tables.Float32Col()  # float  (single-precision)
    energy = tables.Float64Col()    # double (double-precision)

h5file = tables.open_file(
    "scratch/rvorticity/rvor_identify/identified_rvor_20100803_09_001.h5",
    mode="w", title="Test file")
group = h5file.create_group("/", 'detector', 'Detector information')
table = h5file.create_table(group, 'readout', Particle, "Readout example")

# Get a shortcut to the record object in table
particle = table.row

# Fill the table with 10 particles
for i in range(10):
    particle['name'] = 'Particle: %6d' % (i)
    particle['TDCcount'] = i % 256
    particle['ADCcount'] = (i * 256) % (1 << 16)
    particle['grid_i'] = i
    particle['grid_j'] = 10 - i
    particle['pressure'] = float(i * i)
    particle['energy'] = float(particle['pressure'] ** 4)
    particle['idnumber'] = i * (2 ** 34)
    # Insert a new particle record
    particle.append()

# Flush the buffers for table
table.flush()

table = h5file.root.detector.readout
pressure = [x['pressure'] for x in table.iterrows()
            if x['TDCcount'] > 3 and 20 <= x['pressure'] < 50]

names = [x['name'] for x in table.where(
    """(TDCcount > 3) & (20 <= pressure) & (pressure < 50)""")]

condition = '(name == b"Particle:      5") | (name == b"Particle:      7")'
for record in table.where(condition):
    print(record)

gcolumns = h5file.create_group(h5file.root, "columns", "Pressure and Name")
h5file.create_array(gcolumns, 'pressure', np.array(
    pressure), "Pressure column selection")
h5file.create_array(gcolumns, 'name', names, "Name column selection")

h5file.close()

h5file = tables.open_file(
    "scratch/rvorticity/rvor_identify/identified_rvor_20100803_09_001.h5", "a")
table = h5file.root.detector.readout
table.attrs.gath_date = "Wed, 06/12/2003 18:33"
table.attrs.temperature = 18.4
table.attrs.temp_scale = "Celsius"
detector = h5file.root.detector
detector._v_attrs.stuff = [5, (2.3, 4.5), "Integer and tuple"]
table.attrs._f_rename("temp_scale", "tempScale")

for name in table.colnames:
    print(name, ':= %s, %s' % (table.coldtypes[name], table.coldtypes[name].shape))

pressureObject = h5file.get_node("/columns", "pressure")
pressureArray = pressureObject.read()

particle = table.row
for i in range(10, 15):
    particle['name'] = 'Particle: %6d' % (i)
    particle['TDCcount'] = i % 256
    particle['ADCcount'] = (i * 256) % (1 << 16)
    particle['grid_i'] = i
    particle['grid_j'] = 10 - i
    particle['pressure'] = float(i*i)
    particle['energy'] = float(particle['pressure'] ** 4)
    particle['idnumber'] = i * (2 ** 34)
    particle.append()
table.flush()

for r in table.iterrows():
    print("%-16s | %11.1f | %11.4g | %6d | %6d | %8d \|" %
          (r['name'], r['pressure'], r['energy'], r['grid_i'], r['grid_j'],
           r['TDCcount']))


from tables import *
from numpy import *

# Describe a particle record
class Particle(IsDescription):
    name        = StringCol(itemsize=16)  # 16-character string
    lati        = Int32Col()              # integer
    longi       = Int32Col()              # integer
    pressure    = Float32Col(shape=(2,3)) # array of floats (single-precision)
    temperature = Float64Col(shape=(2,3)) # array of doubles (double-precision)

# Native NumPy dtype instances are also accepted
Event = dtype([
    ("name"     , "S16"),
    ("TDCcount" , uint8),
    ("ADCcount" , uint16),
    ("xcoord"   , float32),
    ("ycoord"   , float32)
    ])

fileh = open_file(
    "scratch/rvorticity/rvor_identify/identified_rvor_20100803_09_001.h5",
    mode="w")
root = fileh.root

for groupname in ("Particles", "Events"):
    group = fileh.create_group(root, groupname)

gparticles = root.Particles

# fileh.root.Particles.TParticle1._f_remove()


for tablename in ("TParticle1", "TParticle2", "TParticle3"):
    table = fileh.create_table(
        "/Particles", tablename, Particle, "Particles: "+tablename)
    particle = table.row
    for i in range(257):
        particle['name'] = 'Particle: %6d' % (i)
        particle['lati'] = i
        particle['longi'] = 10 - i
        # particle['pressure'] = array(i*arange(2*3)).reshape((2,4))  # Incorrect
        particle['pressure'] = array(i*arange(2*3)).reshape((2,3)) # Correct
        particle['temperature'] = (i**2)     # Broadcasting
        particle.append()
    table.flush()


for tablename in ("TEvent1", "TEvent2", "TEvent3"):
    table = fileh.create_table(
        root.Events, tablename, Event, "Events: "+tablename)
    event = table.row
    for i in range(257):
        event['name']  = 'Event: %6d' % (i)
        event['TDCcount'] = i % (1<<8)   # Correct range
        event['xcoord'] = float(i**2)   # Correct spelling
        event['ADCcount'] = i * 2       # Correct type
        event['ycoord'] = float(i)**4
        event.append()
    table.flush()


table = root.Events.TEvent3
e = [p['TDCcount']
     for p in table if p['ADCcount'] < 20 and 4 <= p['TDCcount'] < 15]
fileh.close()



'''
experiment = identified_rvor.create_table(
    '/', 'exp1', VortexInfo, 'Vortex Information in Experiment 1')

# class VortexPoints(tb.IsDescription):
#     lat = tb.Float32Col()
#     lon = tb.Float32Col()
#     magnitude = tb.Float64Col()

class Vortices(tb.IsDescription):
    # vortex_points = VortexPoints()
    # index = tb.Int8Col()
    center_lat = tb.Float32Col()
    center_lon = tb.Float32Col()
    size = tb.Float64Col()
    radius = tb.Float64Col()
    distance2radius = tb.Float64Col()
    peak_magnitude = tb.Float64Col()
    mean_magnitude = tb.Float64Col()
    ellipse_eccentricity = tb.Float64Col()

class VortexInfo(tb.IsDescription):
    vortex_count = tb.Int8Col()
    is_vortex = tb.Int8Col(shape=(ilat, ilon))
    vortex_indices = tb.Int8Col(shape=(ilat, ilon))
    # vortices = Vortices()

    vortex_info_vortices = vortex_info['vortices'].row
    for j in range(vortices_count):
        # vortex_info_vortices['index'] = vortices[j]['index']
        vortex_info_vortices['center_lat'] = vortices[j]['center_lat']
        vortex_info_vortices['center_lon'] = vortices[j]['center_lon']
        vortex_info_vortices['size'] = vortices[j]['size']
        vortex_info_vortices['radius'] = vortices[j]['radius']
        vortex_info_vortices['distance2radius'] = vortices[j][
            'distance2radius']
        vortex_info_vortices['peak_magnitude'] = vortices[j][
            'peak_magnitude']
        vortex_info_vortices['mean_magnitude'] = vortices[j]['mean_magnitude']
        vortex_info_vortices['ellipse_eccentricity'] = vortices[j][
            'ellipse_eccentricity']
        
        # vortex_info_points = vortex_info_vortices['vortex_points']
        # for k in range(vortices[j]['lon'].shape[0]):
        #     vortex_info_points['lat'] = vortices[j]['lat'][k]
        #     vortex_info_points['lon'] = vortices[j]['lon'][k]
        #     vortex_info_points['magnitude'] = vortices[j]['magnitude'][k]
        #     vortex_info_points.append()

'''
# endregion
# =============================================================================


# =============================================================================
# region interpolate example from metpy
# https://unidata.github.io/MetPy/latest/examples/cross_section.html#sphx-glr-examples-cross-section-py

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

import metpy.calc as mpcalc
from metpy.cbook import get_test_data
from metpy.interpolate import cross_section

data = xr.open_dataset(get_test_data('narr_example.nc', False))
data = data.metpy.parse_cf().squeeze()
print(data)

start = (37.0, -105.0)
end = (35.5, -65.0)

cross = cross_section(data, start, end).set_coords(('lat', 'lon'))
print(cross)

temperature, pressure, specific_humidity = xr.broadcast(
    cross['Temperature'],
    cross['isobaric'],
    cross['Specific_humidity'])

theta = mpcalc.potential_temperature(pressure, temperature)
rh = mpcalc.relative_humidity_from_specific_humidity(specific_humidity, temperature, pressure)

# These calculations return unit arrays, so put those back into DataArrays in our Dataset
cross['Potential_temperature'] = xr.DataArray(theta,
                                              coords=temperature.coords,
                                              dims=temperature.dims,
                                              attrs={'units': theta.units})
cross['Relative_humidity'] = xr.DataArray(rh,
                                          coords=specific_humidity.coords,
                                          dims=specific_humidity.dims,
                                          attrs={'units': rh.units})

cross['u_wind'].metpy.convert_units('knots')
cross['v_wind'].metpy.convert_units('knots')
cross['t_wind'], cross['n_wind'] = mpcalc.cross_section_components(
    cross['u_wind'],
    cross['v_wind'])

print(cross)

fig = plt.figure(1, figsize=(16., 9.))
ax = plt.axes()

# Plot RH using contourf
rh_contour = ax.contourf(
    cross['lon'], cross['isobaric'], cross['Relative_humidity'],
    levels=np.arange(0, 1.05, .05), cmap='YlGnBu')
rh_colorbar = fig.colorbar(rh_contour)

# Plot potential temperature using contour, with some custom labeling
theta_contour = ax.contour(
    cross['lon'], cross['isobaric'], cross['Potential_temperature'],
    levels=np.arange(250, 450, 5), colors='k', linewidths=2)
theta_contour.clabel(
    theta_contour.levels[1::2], fontsize=8, colors='k', inline=1,
    inline_spacing=8, fmt='%i', rightside_up=True, use_clabeltext=True)

# Plot winds using the axes interface directly, with some custom indexing to make the barbs
# less crowded
wind_slc_vert = list(range(0, 19, 2)) + list(range(19, 29))
wind_slc_horz = slice(5, 100, 5)
ax.barbs(cross['lon'][wind_slc_horz], cross['isobaric'][wind_slc_vert],
         cross['t_wind'][wind_slc_vert, wind_slc_horz],
         cross['n_wind'][wind_slc_vert, wind_slc_horz], color='k')

# Adjust the y-axis to be logarithmic
ax.set_yscale('symlog')
ax.set_yticklabels(np.arange(1000, 50, -100))
ax.set_ylim(cross['isobaric'].max(), cross['isobaric'].min())
ax.set_yticks(np.arange(1000, 50, -100))

# Define the CRS and inset axes
data_crs = data['Geopotential_height'].metpy.cartopy_crs
ax_inset = fig.add_axes([0.125, 0.665, 0.25, 0.25], projection=data_crs)

# Plot geopotential height at 500 hPa using xarray's contour wrapper
ax_inset.contour(
    data['x'], data['y'], data['Geopotential_height'].sel(isobaric=500.),
    levels=np.arange(5100, 6000, 60), cmap='inferno')

# Plot the path of the cross section
endpoints = data_crs.transform_points(
    ccrs.Geodetic(), *np.vstack([start, end]).transpose()[::-1])
ax_inset.scatter(endpoints[:, 0], endpoints[:, 1], c='k', zorder=2)
ax_inset.plot(cross['x'], cross['y'], c='k', zorder=2)

# Add geographic features
ax_inset.coastlines()
ax_inset.add_feature(cfeature.STATES.with_scale('50m'), edgecolor='k', alpha=0.2, zorder=0)

# Set the titles and axes labels
ax_inset.set_title('')
ax.set_title('NARR Cross-Section \u2013 {} to {} \u2013 Valid: {}\n'
             'Potential Temperature (K), Tangential/Normal Winds (knots), '
             'Relative Humidity (dimensionless)\n'
             'Inset: Cross-Section Path and 500 hPa Geopotential Height'.format(
                 start, end, cross['time'].dt.strftime('%Y-%m-%d %H:%MZ').item()))
ax.set_ylabel('Pressure (hPa)')
ax.set_xlabel('Longitude (degrees east)')
rh_colorbar.set_label('Relative Humidity (dimensionless)')

fig.savefig('figures/00_test/trial.png')



# endregion
# =============================================================================


# =============================================================================
# region examples mapping_GOES16_TrueColor
# https://unidata.github.io/python-training/gallery/mapping_goes16_truecolor/

FILE = ('https://ramadda.scigw.unidata.ucar.edu/repository/opendap'
        '/4ef52e10-a7da-4405-bff4-e48f68bb6ba2/entry.das#fillmismatch')
C = xr.open_dataset(FILE)

scan_start = datetime.strptime(C.time_coverage_start, '%Y-%m-%dT%H:%M:%S.%fZ')

# Scan's end time, converted to datetime object
scan_end = datetime.strptime(C.time_coverage_end, '%Y-%m-%dT%H:%M:%S.%fZ')

# File creation time, convert to datetime object
file_created = datetime.strptime(C.date_created, '%Y-%m-%dT%H:%M:%S.%fZ')

# The 't' variable is the scan's midpoint time
midpoint = str(C['t'].data)[:-8]
scan_mid = datetime.strptime(midpoint, '%Y-%m-%dT%H:%M:%S.%f')

print('Scan Start    : {}'.format(scan_start))
print('Scan midpoint : {}'.format(scan_mid))
print('Scan End      : {}'.format(scan_end))
print('File Created  : {}'.format(file_created))
print('Scan Duration : {:.2f} minutes'.format(
    (scan_end-scan_start).seconds/60))
for band in [2, 3, 1]:
    print('{} is {:.2f} {}'.format(
        C['band_wavelength_C{:02d}'.format(band)].long_name,
        float(C['band_wavelength_C{:02d}'.format(band)][0]),
        C['band_wavelength_C{:02d}'.format(band)].units))
# RED: 0.64, GREEN 0.86, BLUE: 0.47 10^-6m

# Load the three channels into appropriate R, G, and B variables
R = C['CMI_C02'].data
G = C['CMI_C03'].data
B = C['CMI_C01'].data

# Apply range limits for each channel. RGB values must be between 0 and 1
R = np.clip(R, 0, 1)
G = np.clip(G, 0, 1)
B = np.clip(B, 0, 1)

# Apply a gamma correction to the image to correct ABI detector brightness
gamma = 2.2
R = np.power(R, 1/gamma)
G = np.power(G, 1/gamma)
B = np.power(B, 1/gamma)

# Calculate the "True" Green
G_true = 0.45 * R + 0.1 * G + 0.45 * B
G_true = np.clip(G_true, 0, 1)  # apply limits again, just in case.

# =============================================================================
fig, ([ax1, ax2, ax3, ax4]) = plt.subplots(1, 4, figsize=(16, 3))

ax1.imshow(R, cmap='Reds', vmax=1, vmin=0)
ax1.set_title('Red', fontweight='bold')
ax1.axis('off')

ax2.imshow(G, cmap='Greens', vmax=1, vmin=0)
ax2.set_title('Veggie', fontweight='bold')
ax2.axis('off')

ax3.imshow(G_true, cmap='Greens', vmax=1, vmin=0)
ax3.set_title('"True" Green', fontweight='bold')
ax3.axis('off')

ax4.imshow(B, cmap='Blues', vmax=1, vmin=0)
ax4.set_title('Blue', fontweight='bold')
ax4.axis('off')

plt.subplots_adjust(wspace=.02)
plt.savefig('figures/00_test/sat.png')


# =============================================================================
# The RGB array with the raw veggie band
RGB_veggie = np.dstack([R, G, B])

# The RGB array for the true color image
RGB = np.dstack([R, G_true, B])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# The RGB using the raw veggie band
ax1.imshow(RGB_veggie)
ax1.set_title('GOES-16 RGB Raw Veggie', fontweight='bold', loc='left',
              fontsize=12)
ax1.set_title('{}'.format(scan_start.strftime('%d %B %Y %H:%M UTC ')),
              loc='right')
ax1.axis('off')

# The RGB for the true color image
ax2.imshow(RGB)
ax2.set_title('GOES-16 RGB True Color', fontweight='bold', loc='left',
              fontsize=12)
ax2.set_title('{}'.format(scan_start.strftime('%d %B %Y %H:%M UTC ')),
              loc='right')
ax2.axis('off')
fig.savefig('figures/00_test/sat.png')


# =============================================================================

# We'll use the `CMI_C02` variable as a 'hook' to get the CF metadata.
dat = C.metpy.parse_cf('CMI_C02')

geos = dat.metpy.cartopy_crs

# We also need the x (north/south) and y (east/west) axis sweep of the ABI data
x = dat.x
y = dat.y

fig = plt.figure(figsize=(15, 12))

# Create axis with Geostationary projection
ax = fig.add_subplot(1, 1, 1, projection=geos)

# Add the RGB image to the figure. The data is in the same projection as the
# axis we just created.
ax.imshow(RGB, origin='upper',
          extent=(x.min(), x.max(), y.min(), y.max()), transform=geos)

# Add Coastlines and States
ax.coastlines(resolution='50m', color='black', linewidth=0.25)
ax.add_feature(ccrs.cartopy.feature.STATES, linewidth=0.25)

plt.title('GOES-16 True Color', loc='left', fontweight='bold', fontsize=15)
plt.title('{}'.format(scan_start.strftime('%d %B %Y %H:%M UTC ')), loc='right')
fig.savefig('figures/00_test/sat.png')


# =============================================================================

fig = plt.figure(figsize=(15, 12))

# Generate an Cartopy projection
lc = ccrs.LambertConformal(central_longitude=-97.5,
                           standard_parallels=(38.5, 38.5))

ax = fig.add_subplot(1, 1, 1, projection=lc)
ax.set_extent([-135, -60, 10, 65], crs=ccrs.PlateCarree())

ax.imshow(RGB, origin='upper',
          extent=(x.min(), x.max(), y.min(), y.max()),
          transform=geos,
          interpolation='none')
ax.coastlines(resolution='50m', color='black', linewidth=0.5)
ax.add_feature(ccrs.cartopy.feature.STATES, linewidth=0.5)

plt.title('GOES-16 True Color', loc='left', fontweight='bold', fontsize=15)
plt.title('{}'.format(scan_start.strftime('%d %B %Y %H:%M UTC ')), loc='right')
fig.savefig('figures/00_test/sat.png')


# =============================================================================
fig = plt.figure(figsize=(8, 8))

pc = ccrs.PlateCarree()

ax = fig.add_subplot(1, 1, 1, projection=pc)
ax.set_extent([-114.75, -108.25, 36, 43], crs=pc)

ax.imshow(RGB, origin='upper',
          extent=(x.min(), x.max(), y.min(), y.max()),
          transform=geos,
          interpolation='none')

ax.coastlines(resolution='50m', color='black', linewidth=1)
ax.add_feature(ccrs.cartopy.feature.STATES)

plt.title('GOES-16 True Color', loc='left', fontweight='bold', fontsize=15)
plt.title('{}'.format(scan_start.strftime('%d %B %Y %H:%M UTC ')), loc='right')
fig.savefig('figures/00_test/sat.png')


# endregion
# =============================================================================


# =============================================================================
# region Overlay Nighttime IR when dark

# A GOES-16 file with half day and half night
FILE = ('https://ramadda.scigw.unidata.ucar.edu/repository/opendap'
        '/85da3304-b910-472b-aedf-a6d8c1148131/entry.das#fillmismatch')
C = xr.open_dataset(FILE)

# Scan's start time, converted to datetime object
scan_start = datetime.strptime(C.time_coverage_start, '%Y-%m-%dT%H:%M:%S.%fZ')

# Create the RGB like we did before

# Load the three channels into appropriate R, G, and B
R = C['CMI_C02'].data
G = C['CMI_C03'].data
B = C['CMI_C01'].data

# Apply range limits for each channel. RGB values must be between 0 and 1
R = np.clip(R, 0, 1)
G = np.clip(G, 0, 1)
B = np.clip(B, 0, 1)

# Apply the gamma correction
gamma = 2.2
R = np.power(R, 1/gamma)
G = np.power(G, 1/gamma)
B = np.power(B, 1/gamma)

# Calculate the "True" Green
G_true = 0.45 * R + 0.1 * G + 0.45 * B
G_true = np.clip(G_true, 0, 1)

# The final RGB array :)
RGB = np.dstack([R, G_true, B])

# Apply the normalization...
cleanIR = C['CMI_C13'].data

# Normalize the channel between a range.
#       cleanIR = (cleanIR-minimumValue)/(maximumValue-minimumValue)
cleanIR = (cleanIR-90)/(313-90)

# Apply range limits to make sure values are between 0 and 1
cleanIR = np.clip(cleanIR, 0, 1)

# Invert colors so that cold clouds are white
cleanIR = 1 - cleanIR

# Lessen the brightness of the coldest clouds so they don't appear so bright
# when we overlay it on the true color image.
cleanIR = cleanIR/1.4

# Yes, we still need 3 channels as RGB values. This will be a grey image.
RGB_cleanIR = np.dstack([cleanIR, cleanIR, cleanIR])

# =============================================================================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

ax1.set_title('True Color', fontweight='bold')
ax1.imshow(RGB)
ax1.axis('off')

ax2.set_title('Clean IR', fontweight='bold')
ax2.imshow(RGB_cleanIR)
ax2.axis('off')
fig.savefig('figures/00_test/sat.png', dpi = 600)


# =============================================================================
# Maximize the RGB values between the True Color Image and Clean IR image
RGB_ColorIR = np.dstack([np.maximum(R, cleanIR), np.maximum(G_true, cleanIR),
                         np.maximum(B, cleanIR)])

fig = plt.figure(figsize=(15, 12))

ax = fig.add_subplot(1, 1, 1, projection=geos)

ax.imshow(RGB_ColorIR, origin='upper',
          extent=(x.min(), x.max(), y.min(), y.max()),
          transform=geos)

ax.coastlines(resolution='50m', color='black', linewidth=2)
ax.add_feature(ccrs.cartopy.feature.STATES)

plt.title('GOES-16 True Color and Night IR', loc='left', fontweight='bold',
          fontsize=15)
plt.title('{}'.format(scan_start.strftime('%H:%M UTC %d %B %Y'), loc='right'),
          loc='right')
fig.savefig('figures/00_test/sat.png', dpi=600)


# =============================================================================
def contrast_correction(color, contrast):
    """Modify the contrast of an RGB.
    See:
    https://www.dfstudios.co.uk/articles/programming/image-programming-algorithms/image-processing-algorithms-part-5-contrast-adjustment/

    Input:
        color    - an array representing the R, G, and/or B channel
        contrast - contrast correction level
    """
    F = (259*(contrast + 255))/(255.*259-contrast)
    COLOR = F*(color-.5)+.5
    COLOR = np.clip(COLOR, 0, 1)  # Force value limits 0 through 1.
    return COLOR


# Amount of contrast
contrast_amount = 105

# Apply contrast correction
RGB_contrast = contrast_correction(RGB, contrast_amount)

# Add in clean IR to the contrast-corrected True Color image
RGB_contrast_IR = np.dstack([np.maximum(RGB_contrast[:, :, 0], cleanIR),
                             np.maximum(RGB_contrast[:, :, 1], cleanIR),
                             np.maximum(RGB_contrast[:, :, 2], cleanIR)])

fig = plt.figure(figsize=(15, 12))

ax1 = fig.add_subplot(1, 2, 1, projection=geos)
ax2 = fig.add_subplot(1, 2, 2, projection=geos)

ax1.imshow(RGB_ColorIR, origin='upper',
           extent=(x.min(), x.max(), y.min(), y.max()),
           transform=geos)
ax1.coastlines(resolution='50m', color='black', linewidth=2)
ax1.add_feature(ccrs.cartopy.feature.BORDERS)
ax1.set_title('True Color and Night IR')

ax2.imshow(RGB_contrast_IR, origin='upper',
           extent=(x.min(), x.max(), y.min(), y.max()),
           transform=geos)
ax2.coastlines(resolution='50m', color='black', linewidth=2)
ax2.add_feature(ccrs.cartopy.feature.BORDERS)
ax2.set_title('Contrast Correction = {}'.format(contrast_amount))

plt.subplots_adjust(wspace=.02)
fig.savefig('figures/00_test/sat.png', dpi=600)


# endregion
# =============================================================================

# =============================================================================
# region make plots for a Mesoscale scan
# M1 is for the Mesoscale1 NetCDF file
FILE = ('https://ramadda.scigw.unidata.ucar.edu/repository/opendap'
        '/5e02eafa-5cee-4d00-9f58-6e201e69b014/entry.das#fillmismatch')
M1 = xr.open_dataset(FILE)

# Load the RGB arrays
R = M1['CMI_C02'][:].data
G = M1['CMI_C03'][:].data
B = M1['CMI_C01'][:].data

# Apply range limits for each channel. RGB values must be between 0 and 1
R = np.clip(R, 0, 1)
G = np.clip(G, 0, 1)
B = np.clip(B, 0, 1)

# Apply the gamma correction
gamma = 2.2
R = np.power(R, 1/gamma)
G = np.power(G, 1/gamma)
B = np.power(B, 1/gamma)

# Calculate the "True" Green
G_true = 0.45 * R + 0.1 * G + 0.45 * B
G_true = np.clip(G_true, 0, 1)

# The final RGB array :)
RGB = np.dstack([R, G_true, B])

# Scan's start time, converted to datetime object
scan_start = datetime.strptime(M1.time_coverage_start, '%Y-%m-%dT%H:%M:%S.%fZ')

# We'll use the `CMI_C02` variable as a 'hook' to get the CF metadata.
dat = M1.metpy.parse_cf('CMI_C02')

# Need the satellite sweep x and y values, too.
x = dat.x
y = dat.y

# =============================================================================
fig = plt.figure(figsize=(10, 8))

ax = fig.add_subplot(1, 1, 1, projection=lc)
ax.set_extent([-125, -70, 25, 50], crs=ccrs.PlateCarree())

ax.imshow(RGB, origin='upper',
          extent=(x.min(), x.max(), y.min(), y.max()),
          transform=geos)

ax.coastlines(resolution='50m', color='black', linewidth=0.5)
ax.add_feature(ccrs.cartopy.feature.STATES, linewidth=0.5)
ax.add_feature(ccrs.cartopy.feature.BORDERS, linewidth=0.5)

plt.title('GOES-16 True Color', fontweight='bold', fontsize=15, loc='left')
plt.title('Mesoscale Section 1')
plt.title('{}'.format(scan_start.strftime('%H:%M UTC %d %B %Y')), loc='right')
fig.savefig('figures/00_test/sat.png', dpi=600)

# =============================================================================
fig = plt.figure(figsize=(15, 12))

ax = fig.add_subplot(1, 1, 1, projection=geos)

ax.imshow(RGB, origin='upper',
          extent=(x.min(), x.max(), y.min(), y.max()),
          transform=geos)

ax.coastlines(resolution='50m', color='black', linewidth=0.25)
ax.add_feature(ccrs.cartopy.feature.STATES, linewidth=0.25)

plt.title('GOES-16 True Color', fontweight='bold', fontsize=15, loc='left')
plt.title('Mesoscale Section 1')
plt.title('{}'.format(scan_start.strftime('%H:%M UTC %d %B %Y')), loc='right')
fig.savefig('figures/00_test/sat.png', dpi=600)
# endregion
# =============================================================================


# region PP_to_P
# https://github.com/broglir/pgw-python/blob/master/Preprocess_CCLM/PP_to_P.py
import numpy as np
import xarray as xr

"""
Preprocessing script to convert PP from COSMO-CLM to absolute pressure. This is only necessary if one subsequently wants to compute the mearelative Humidity from Specific humidity. 
"""

#enter paths
superpath = '/project/pr04/robro/inputfiles_for_surrogate_hadgem/inputfiles/'

#enter path to pp
ppname = 'HadGEM2-ES_Hist_RCP85_cosmo4-8_17_new_PP_2070-2099_ydaymean.nc'
savename = 'HadGEM2-ES_Hist_RCP85_cosmo4-8_17_new_P_2070-2099_ydaymean.nc'

const = xr.open_dataset(
	'/store/c2sm/ch4/robro/surrogate_input/lffd1969120100c.nc')
hsurf = const['HSURF'].squeeze()


# altitude where coordinate system becomes flat (you can find that in runscripts)
vcflat = 11430.

#CCLM uses a referece pressure system, to compute the actual pressure it PP needs to be added to the reference. These are the height levels for the 50km simulations!

height_flat = np.asanyarray([22700.0, 20800.0000, 19100.0, 17550.0, 16150.0, 14900.0, 13800.0, 12785.0, 11875.0, 11020.0, 10205.0, 		9440.0, 8710.0, 8015.0, 7355.0, 6725.0, 6130.0,
                             5565.0, 5035.0, 4530.0, 4060.0, 3615.0, 3200.0, 2815.0, 2455.0, 2125.0, 1820.0, 1545.0, 1295.0, 1070.0, 870.0, 695.0, 542.0, 412.0, 303.0, 214.0, 143.0, 89.0, 49.0, 20.0])

smoothing = (vcflat - height_flat) / vcflat
smoothing = np.where(smoothing > 0, smoothing, 0)


#the height at which the reference pressure needs to be computed needs to be derived form the terrain 	following coordinates:
newheights = np.zeros((len(height_flat), hsurf.shape[0], hsurf.shape[1]))

#add the surface height but respect the terrain following coordinates
for x in range(hsurf.shape[0]):
	for y in range(hsurf.shape[1]):
		newheights[:, x, y] = height_flat + hsurf[x, y].values * smoothing

pref = 100000*np.exp(-(9.80665*0.0289644*newheights/(8.31447*288.15)))

PP = xr.open_dataset(superpath+ppname)['PP']

p = PP + pref

p = p.astype('float32')
pds = p.to_dataset(name='P')
pds.to_netcdf(superpath+savename)
# endregion


# region examples from jesus
import numpy as np
def uvrot2uv_vec(u, v, rlat, rlon, pollat, pollon, idim, jdim):
    
#------------------------------------------------------------------------------
#
# Description:
#   This routine converts the wind components u and v from the rotated
#   system to the real geographical system. This is the vectorized form
#   of the routine above, i.e. the computation is for a whole 2D field.
#
# Method:
#   Transformation formulas for converting between these two systems.
#
#------------------------------------------------------------------------------
    
#------------------------------------------------------------------------------
# Parameter list:
#  idim, jdim        # dimensions of the field
    
#  u  (idim,jdim), & # wind components in the true geographical system
#  v  (idim,jdim)    #
    
#  rlat(idim,jdim),& # coordinates in the true geographical system
#  rlon(idim,jdim),& #
#  pollat, pollon    # latitude and longitude of the north pole of the
#                    # rotated grid
    
# Local variables
#  zsinpol, zcospol, zlonp, zlat, zarg1, zarg2, znorm, zugeo, zvgeo
    
    
    zrpi18 = 57.2957795
    zpir18 = 0.0174532925
    
#------------------------------------------------------------------------------
# Begin Subroutine uvrot2uv_vec
#------------------------------------------------------------------------------
    unrot_v=np.zeros_like(v)
    unrot_u=np.zeros_like(u)
# Converting from degree to radians
    zsinpol = np.sin(pollat * zpir18)
    zcospol = np.cos(pollat * zpir18)
    
    zlonp   = (pollon-rlon[:,:]) * zpir18
    zlat    =         rlat[:,:]  * zpir18
    zarg1   = zcospol*np.sin(zlonp)
    zarg2   = zsinpol*np.cos(zlat) - zcospol*np.sin(zlat)*np.sin(zlonp)
    znorm   = 1.0/np.sqrt(zarg1**2 + zarg2**2)
    # Convert the u- and v-components
    unrot_u   =  u[:,:]*zarg2*znorm + v[:,:]*zarg1*znorm
    unrot_v   = -u[:,:]*zarg1*znorm + v[:,:]*zarg2*znorm
    return unrot_u,unrot_v


if __name__=='__main__':
    from netCDF4 import Dataset
    file='/scratch/snx3000/jvergara/lffd20061031210000.nc'
    ds=Dataset(file)
    U=ds.variables['U_10M'][0,]
    V=ds.variables['V_10M'][0,]
    rlat=ds.variables['lat']
    rlon=ds.variables['lon']
    pollon=ds.variables['rotated_pole'].grid_north_pole_longitude
    pollat=ds.variables['rotated_pole'].grid_north_pole_latitude
    idim=U.shape[0]
    jdim=U.shape[1]
    unr_U,unr_V=uvrot2uv_vec(U,V, rlat, rlon, pollat, pollon, idim, jdim)
#    jle.Quick_plot(unr_U**2+unr_V**2-U**2-V**2,'V_10M_unrot',metadata_dataset=ds)
#    jle.Quick_plot(unr_V,'V_10M_unrot',metadata_dataset=ds)
#    jle.Quick_plot(ds,'V_10M')


# endregion


# region examples from Christian ----



#!/usr/bin/python

import timeit
import numpy as np
import xarray as xr
from scipy.interpolate import interp1d

start = timeit.default_timer()

infile = 'lffd20180530081500.nc'    # small
outfile = 'w_new_interp1d.nc'
variable = 'W'
height_variable = 'HHL'
heights = np.arange(0, 16100, 100)

# Open file and extract variable of interest
ds_orig = xr.open_dataset(infile)

# Get height and w
hhl = np.array(ds_orig['HHL'])
w = np.array(ds_orig['W'])

# Target grid for w
nz = len(heights)
nt, _, ny, nx = hhl.shape
w_intp = np.zeros((nt, nz, ny, nx))

# Interpolate it on new height levels
for t in range(0, nt):
    for j in range(0, ny):
        for i in range(0, nx):
            interpolator = interp1d(hhl[t,:,j,i], w[t,:,j,i],
                                    bounds_error=False,
                                    fill_value=np.nan)
            w_intp[t,:,j,i] = interpolator(heights)

ds_w_new = xr.Dataset({'W': (['time','lev','rlat','rlon'], w_intp)},
                             coords={'time': ('time', ds_orig['time']),
                                     'lev': ('lev',
                                             np.arange(1,162,dtype=np.int32)),
                                     'rlat': ('rlat', ds_orig['rlat']),
                                     'rlon': ('rlon', ds_orig['rlon'])})
ds_w_new.to_netcdf(outfile)

stop = timeit.default_timer()
print("Converted in: " + str(stop - start) + " s")



#!/usr/bin/python

import timeit
import os
import numpy as np
import xarray as xr
from cdo import Cdo

start = timeit.default_timer()

# infile = 'lfsd20100801220000.nc'
infile = 'lfsd20100802070000.nc'
constfile = 'lfsd20051101000000c.nc'
# outfile = 'lfsd20100801220000z.nc'
outfile = 'lfsd20100802070000z.nc'
variables_fl = 'QC'
height_variable = 'HHL'
heights = np.arange(0, 13100, 100)
cdo = Cdo()

# Open file and extract variable of interest
ds_orig = xr.open_dataset(infile)
cdo.selvar(variables_fl, input=infile, output='orig_fl.nc')

# Write old height file for full level
ds_const = xr.open_dataset(constfile)
heights_fl = (ds_const.HHL[:,1:,:,:] + ds_const.HHL[:,:-1,:,:]) / 2.
ds_height_old_fl = xr.Dataset({'height': (['time','level1','rlat','rlon'],
                                          heights_fl.values)},
                               coords={'time': ('time', ds_const['time']),
                                       'rlat': ('rlat', ds_const['rlat']),
                                       'rlon': ('rlon', ds_const['rlon'])})
ds_height_old_fl.to_netcdf('heights_old_fl.nc')

# new height file
nz = len(heights)
nt, nmlp1, ny, nx = ds_const['HHL'].shape
height_new = np.zeros((nt, nz, ny, nx))
for i in range(nz):
    height_new[:,i,:,:] = heights[i]
ds_height_new = xr.Dataset({'height': (['time','level1','rlat','rlon'], height_new)},
                           coords={'time': ('time', ds_const['time']),
                                   'rlat': ('rlat', ds_const['rlat']),
                                   'rlon': ('rlon', ds_const['rlon'])})
ds_height_new.to_netcdf('heights_new.nc')

# Interpolate to new height levels
cdo.intlevel3d('heights_old_fl.nc', input='orig_fl.nc heights_new.nc',
               output=outfile)

# Clean up
os.remove('heights_old_fl.nc')
os.remove('orig_fl.nc')
os.remove('heights_new.nc')

stop = timeit.default_timer()
print("Converted in: " + str(stop - start) + " s")



dset = xr.open_dataset(
    '/store/c2sm/pr04/jvergara/RUNS_IN_SCRATCH/MAC1/lm_f/1h/lffd20100210080000.nc'
)

# Transparent colormap
colors = [(1,1,1,c) for c in np.linspace(0,1,100)]
cmapwhite = mpcolors.LinearSegmentedColormap.from_list('mycmap', colors, N=100)

# Background image
os.environ["CARTOPY_USER_BACKGROUNDS"] = "data_source/share/bg_cartopy"

# Figure
transform = ctp.crs.PlateCarree()
projection = transform
fig, ax = plt.subplots(1, 1, subplot_kw={'projection': projection})

# ax.background_img(name='blue_marble_jun', resolution='high')
ax.background_img(name='natural_earth', resolution='high')

# Plot precipitation
plt_clouds = ax.pcolormesh(dset.lon[80:920, 80:920], dset.lat[80:920, 80:920],
                           dset.TQC[0, 80:920, 80:920],
                           cmap=cmapwhite, vmin=0.0, vmax=1.0,
                           transform=transform, zorder=10)
ax.set_extent(extent1km_lb, crs=transform)
fig.subplots_adjust(left=0.155, right=0.975, bottom=0.08, top=0.99)
fig.savefig('figures/00_test/clouds.png', dpi=1200)


# cd /project/pr94/qgao/DEoAI
# cd $SCRATCH 
# conda activate deoai
# /users/qgao/miniconda3/envs/deoai/bin/python

import numpy as np
import xarray as xr
import cartopy as ctp
import cartopy.crs as ccrs
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mpcolors
import xesmf as xe
import metpy.calc as mpcalc
from metpy.interpolate import cross_section
from metpy.units import units
from matplotlib.legend import Legend
from netCDF4 import Dataset
from haversine import haversine
from scipy import interpolate
mpl.rcParams['figure.dpi'] = 600

# plot scope of the study area



# 2  Plot u on 850 hPa ---- 

path = '/store/c2sm/pr04/jvergara/RUNS_IN_SCRATCH/MAC1/lm_f/3h_CORDEX/lffd20051101000000p.nc'
dset = xr.open_dataset(path)
level = 6   # 850 hPa


transform = ctp.crs.PlateCarree()
fig, ax = plt.subplots(1, 1, subplot_kw={'projection': transform})
coastline = ctp.feature.NaturalEarthFeature('physical', 'coastline', '10m',
                edgecolor='black', facecolor='none')
ax.add_feature(coastline)
borders = ctp.feature.NaturalEarthFeature('cultural', 
              'admin_0_boundary_lines_land', '10m', edgecolor='grey',
              facecolor='none')
ax.add_feature(borders)
plt_u = ax.contourf(dset.lon, dset.lat, dset.U[0, level], transform = transform)
cbar = fig.colorbar(plt_u)
cbar.ax.set_ylabel("u [m/s]")
fig.savefig('DEoAI_analysis/figures/trial.png')


# Figure
transform = ctp.crs.PlateCarree()
projection = transform
fig, ax = plt.subplots(1, 1, subplot_kw={'projection': projection})

# Add coastline and boarders
coastline = ctp.feature.NaturalEarthFeature('physical', 'coastline', '10m',
                edgecolor='black', facecolor='none')
borders = ctp.feature.NaturalEarthFeature('cultural', 
              'admin_0_boundary_lines_land', '10m', edgecolor='grey',
              facecolor='none')
ax.add_feature(coastline)
ax.add_feature(borders)

# Plot u
plt_u = ax.contourf(dset.lon, dset.lat, dset.U[0, level], transform=transform)
cbar = fig.colorbar(plt_u)
cbar.ax.set_ylabel("u [m/s]")
fig.savefig('DEoAI_analysis/figures/trial.png')

# use this for next plot
extent = ax.get_extent()


# 3  Plot precipitation ----
# Read file
path = '/store/c2sm/pr04/jvergara/RUNS_IN_SCRATCH/MAC1/lm_f/1h_TOT_PREC/lffd20100220100000.nc'
dset = xr.open_dataset(path)

# Range for contourf plot
tprange = [0.1,0.5,1,2,5,10,20,50]
tplabels = ['0.1','0.5','1','2','5','10','20','50']
norm = mpcolors.LogNorm(vmin=0.1, vmax=100.)
cmap = plt.cm.viridis_r

# Figure
transform = ctp.crs.PlateCarree()
projection = transform
fig, ax = plt.subplots(1, 1, subplot_kw={'projection': projection})

color_bg = plt.cm.Greys(0.25)
ax.set_extent(extent, projection)
ax.background_patch.set_facecolor(color_bg)

# Add coastline
coastline = ctp.feature.NaturalEarthFeature('physical', 'coastline', '10m',
                edgecolor='black', facecolor='none')
borders = ctp.feature.NaturalEarthFeature('cultural', 
              'admin_0_boundary_lines_land', '10m', edgecolor='grey',
              facecolor='none')
ax.add_feature(coastline)
ax.add_feature(borders)

# Plot precipitation
domain = ax.fill(dset.lon, dset.lat, "white", transform=transform, zorder=0)
plt_precip = ax.contourf(dset.lon, dset.lat, dset.TOT_PREC[0], tprange,
                         cmap=cmap, norm=norm, transform=transform, extend='max', zorder=10)
cbar = fig.colorbar(plt_precip, extend='max', ticks=tprange)
cbar.ax.set_yticklabels(tplabels)
cbar.ax.set_ylabel("Precipitation [mm/h]")
fig.savefig('DEoAI_analysis/figures/trial.png')


# region 4  Cross section ----

# 4.1  Dataset
path_ml = '/store/c2sm/pr04/jvergara/RUNS_IN_SCRATCH/MAC1/lm_f/1h/lffd20051101000000.nc'
path_sl = '/store/c2sm/pr04/jvergara/RUNS_IN_SCRATCH/MAC1/lm_f/1h_second/lffd20051101000000c.nc'
ds_ml = xr.open_dataset(path_ml)
ds_sl = xr.open_dataset(path_sl)

# create regional latlon grid and regridder
target_grid = xe.util.grid_2d(lon0_b=ds_sl.lon.min(), lon1_b=ds_sl.lon.max(), d_lon=0.01,
                              lat0_b=ds_sl.lat.min(), lat1_b=ds_sl.lat.max(), d_lat=0.01)
delta_x = 1100.
delta_y = 1100.
regridder_ml = xe.Regridder(ds_ml, target_grid, 'bilinear', reuse_weights=True)
regridder_sl = xe.Regridder(ds_sl, target_grid, 'bilinear', reuse_weights=True)

# regrid fields
u_st = regridder_ml(ds_ml.U_10M)
qc = regridder_ml(ds_ml.TQC)
qi = regridder_ml(ds_ml.TQI)
t_s = regridder_ml(ds_ml.T_2M)
hsurf_c = regridder_ml(ds_sl.HSURF)
hsurf = t_s.copy()
hsurf.name = 'HSURF'
hsurf.values = hsurf_c.values
u = qc.copy()
u.name = 'U'
u.values = np.zeros_like(u.values)
u.values[:, :, :-1] = 0.5 * (u_st[:, :, 1:].values + u_st[:,:,:-1].values)
# u.values[:,:,:,:-1] = 0.5 * (u_st[:,:,:,1:].values + u_st[:,:,:,:-1].values)
hhl = regridder_ml(ds_sl.HHL)
height = qc.copy()
height.name = 'H'
# height.values = 0.5 * (hhl[:,1:,:,:].values + hhl[:,:-1,:,:].values)
height.values = 0.5 * (hhl[:,1,:,:].values + hhl[:,-1,:,:].values)

# create new dataset
ds = xr.merge([hsurf, height, u, qc, qi,])
ds = ds.squeeze('time')


# 4.2  Create cross section
dset_cross = ds.metpy.parse_cf()

start_lat = 32.7
start_lon = -17.2
end_lat = 32.7
end_lon = -16.7
distance = haversine((start_lat, start_lon), (end_lat, end_lon))

start = (start_lat, start_lon)
end = (end_lat, end_lon)
dset_cross['y'] = dset_cross['lat'].values[:,0]
dset_cross['x'] = dset_cross['lon'].values[0,:]

cross = cross_section(dset_cross, start, end, steps=int(distance/1.1)+1).set_coords(('lat', 'lon'))
nl, nx = cross.U.shape
z_min = 0
z_max = 7000
z_diff = 25
z_levs = np.arange(z_min, z_max+z_diff, z_diff)
nz = len(z_levs)
u_interp = np.empty((nz, nx))
qc_interp = np.empty((nz, nx))
qi_interp = np.empty((nz, nx))
qc_interp.fill(np.nan)
qi_interp.fill(np.nan)

delta_t = 1100.
pos_t = np.arange(0, nx) * delta_t
nk, nx = cross.H.shape
pos = np.zeros((nk, nx))
for k in range(0, nk):
    pos[k, :] = pos_t
    
positions = np.array([pos.flatten(), cross.H.values.flatten()]).transpose()
grid_t, grid_z = np.meshgrid(pos_t, z_levs)

u_interp = interpolate.griddata(positions, cross.U.values.flatten(), (grid_t, grid_z), method='linear')
qc_interp = interpolate.griddata(positions, cross.QC.values.flatten(), (grid_t, grid_z), method='linear')
qi_interp = interpolate.griddata(positions, cross.QI.values.flatten(), (grid_t, grid_z), method='linear')

for i in range(0, nx):
    h = cross.H[:,i].values
    valid_z_levs = z_levs[z_levs >= cross.H.values[-1,i]]
    max_k = len(z_levs) - len(valid_z_levs)
    u_interp[0:max_k, i] = np.nan
    qc_interp[0:max_k, i] = np.nan
    qi_interp[0:max_k, i] = np.nan

textsize = 14.
labelsize = 16.
ticksize = 14.
titlesize = 18.

# Define the figure and axes
fig, ax = plt.subplots(1, 1, figsize=(16, 6.))
spec = mpl.gridspec.GridSpec(ncols=2, nrows=2, height_ratios=[5, 1], width_ratios=[5, 1])


# x-axis values in km
x_km = np.linspace(0, distance, len(cross.lon))

# For mountains (nans)
ax.set_facecolor('xkcd:charcoal')

# u
umax = 11.0  # 11 for 4.4
umin = -umax
udiff = 1.
u_contourf = ax.contourf(x_km, z_levs, u_interp,
                         cmap='RdBu_r', levels=np.arange(umin, umax+udiff, udiff), extend='both')

# Clouds
cloud_contour = ax.contour(x_km , z_levs, qc_interp + qi_interp,
                           levels=np.arange(0.00001, 0.00002, 0.00001), colors='xkcd:blue',
                           linewidths=2.)

# Pressure ticks
prs_ticks = np.arange(100, 1100, 100)
height_ticks = mpcalc.pressure_to_height_std(prs_ticks * units.hectopascal).to(units.meter)
ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
ax2.set_ylabel('Pressure [hPa]', color='xkcd:black', size=labelsize)
ax2.set_ylim(z_min, z_max)
ax2.set_yticks(height_ticks.magnitude)
ax2.set_yticklabels(prs_ticks, size=ticksize)

# Colorbar and labels
u_colorbar = fig.colorbar(u_contourf, pad=0.09)
u_colorbar.set_label('u [m/s]', size=labelsize)
u_colorbar.ax.tick_params(labelsize=ticksize)
ax.set_xlabel('x [km]', size=labelsize)
ax.set_ylabel('Height [km]', size=labelsize)

# title
ax.tick_params(axis='x', which='major', labelsize=labelsize)

# endregion

# endregion


