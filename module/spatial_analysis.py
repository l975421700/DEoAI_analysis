
# =============================================================================
# region Find the indices of the nearest grid to a point

def find_nearest_grid(
    point_lat, point_lon, grid_lat, grid_lon):
    '''
    Input --------
    point_lat, point_lon: latitude and longitude of a point, single values
    grid_lat, grid_lon: latitude and longitude of model grid, 2d array
    
    Output --------
    indices: indices of that grid
    
    Example --------
    point_lat = 32.6333
    point_lon = -16.9000
    grid_lat = inversion_height_madeira3d.lat.values
    grid_lon = inversion_height_madeira3d.lon.values
    '''
    
    import numpy as np
    from geopy import distance
    
    point2gird_dist = np.zeros((grid_lon.shape[0], grid_lon.shape[1]))
    
    for i in range(grid_lon.shape[0]):
        # i = 0
        for j in range(grid_lon.shape[1]):
            # j = 0
            point2gird_dist[i, j] = distance.distance(
                [point_lat, point_lon],
                [grid_lat[i, j], grid_lon[i, j]]).km
    
    
    indices = np.unravel_index(
        np.argmin(point2gird_dist, axis=None), point2gird_dist.shape)
    
    '''
    np.amin(point2gird_dist)
    distance.distance(
        [point_lat, point_lon],
        [grid_lat[indices[0], indices[1]],
        grid_lon[indices[0], indices[1]]]).km
    '''
    return(indices)


# endregion
# =============================================================================


# =============================================================================
# region function to calculate the number of significant coefficients

import pywt
import numpy as np

def sig_coeffs(coeffs, signal, wavelet, n_0ratio = 1,
               remove_coeffs=None, ):
    '''
    ----Input
    coeffs: coefficients output by pywt.wavedec2
    signal: the set of discrete values of a signal
    
    ----output
    
    '''
    
    # flatten the coeffients
    coeffs_nd, coeff_slices = pywt.coeffs_to_array(
        coeffs, padding=0, axes=None)
    
    # normalized square modulus of the ith element of the signal
    nsm_rvor = signal ** 2 / np.sum(signal**2)
    
    # The entropy
    entropy = -np.sum(nsm_rvor * np.log(nsm_rvor))
    
    # the number of significant coefficient
    n_0 = np.int(np.e ** entropy * n_0ratio)
    
    # np.sum(abs(coeffs_nd) > np.sort(abs(coeffs_nd.flatten()))[-n_0-1])
    coeffs_nd[abs(coeffs_nd) <= np.sort(abs(coeffs_nd.flatten()))[-n_0-1]] = 0
    
    rec_coeffs_nd = pywt.array_to_coeffs(
        coeffs_nd, coeff_slices, output_format='wavedec2')
    if(not remove_coeffs is None):
        for i in remove_coeffs:
            rec_coeffs_nd[i] = tuple([np.zeros_like(v)
                                      for v in rec_coeffs_nd[i]])
    
    rec_rvor = pywt.waverec2(
        rec_coeffs_nd, wavelet, mode='periodic', axes=(-2, -1))
    
    return(n_0, rec_rvor)


'''
# n_0, rec_rvor = sig_coeffs(coeffs, rvor, wavelets[i_wavelet])
coeffs = coeffs
signal = rvor
wavelet = wavelets[i_wavelet]
n_0ratio = 1
# remove_coeffs=None
remove_coeffs=[1, 2, 3]
'''

# endregion
# =============================================================================


# =============================================================================
# region rotate the wind

import numpy as np
def rotate_wind(u, v, lat, lon, pollat, pollon):
    '''
    Input--------
    u, v: wind vector related to rotated pole, 2D.
    lat, lon: latitude and longitude, 2D.
    pollat, pollon: latitude and longitude of rotated pole, 0D.
    
    Output--------
    u_earth, v_earth: wind vector related to earth, 2D.
    '''
    
    pollat_sin = np.sin(np.deg2rad(pollat))
    pollat_cos = np.cos(np.deg2rad(pollat))
    
    lon_rad = np.deg2rad(pollon - lon)
    lat_rad = np.deg2rad(lat)
    
    arg1 = pollat_cos * np.sin(lon_rad)
    arg2 = pollat_sin * np.cos(lat_rad) - pollat_cos * \
        np.sin(lat_rad)*np.sin(lon_rad)
    
    norm = 1.0/np.sqrt(arg1**2 + arg2**2)
    
    u_earth = u * arg2 * norm + v * arg1 * norm
    v_earth = -u * arg1 * norm + v * arg2 * norm
    
    return u_earth, v_earth


'''
import xarray as xr
import numpy as np

# functions from Jesus
def uvrot2uv_vec(u, v, rlat, rlon, pollat, pollon, idim, jdim):
    zrpi18 = 57.2957795
    zpir18 = 0.0174532925
    unrot_v=np.zeros_like(v)
    unrot_u=np.zeros_like(u)
    zsinpol = np.sin(pollat * zpir18)
    zcospol = np.cos(pollat * zpir18)
    zlonp   = (pollon-rlon[:,:]) * zpir18
    zlat    =         rlat[:,:]  * zpir18
    zarg1   = zcospol*np.sin(zlonp)
    zarg2   = zsinpol*np.cos(zlat) - zcospol*np.sin(zlat)*np.sin(zlonp)
    znorm   = 1.0/np.sqrt(zarg1**2 + zarg2**2)
    unrot_u   =  u[:,:]*zarg2*znorm + v[:,:]*zarg1*znorm
    unrot_v   = -u[:,:]*zarg1*znorm + v[:,:]*zarg2*znorm
    return unrot_u,unrot_v

nc1h_100m = xr.open_dataset(
    '/store/c2sm/pr04/jvergara/RUNS_IN_SCRATCH/MAC1/lm_f/1h_100m/lffd20051101000000z.nc')
u = nc1h_100m.U.squeeze().data
v = nc1h_100m.V.squeeze().data
lat = nc1h_100m.lat.data
lon = nc1h_100m.lon.data
pollat = nc1h_100m.rotated_pole.grid_north_pole_latitude
pollon = nc1h_100m.rotated_pole.grid_north_pole_longitude

u_earth, v_earth = rotate_wind(u, v, lat, lon, pollat, pollon)
unrot_u, unrot_v = uvrot2uv_vec(
    u, v, lat, lon, pollat, pollon, u.shape[0], u.shape[1])

print(np.max(abs(u_earth - unrot_u)))
print(np.max(abs(v_earth - unrot_v)))
print(np.max(abs(u_earth**2 + v_earth**2 - u**2 - v**2)))

i = 200
[u[i, i], v[i, i], u_earth[i, i], v_earth[i, i]]

from line_profiler import LineProfiler
lp = LineProfiler()
lp_wrapper = lp(rotate_wind)
lp_wrapper(u, v, lat, lon, pollat, pollon)
lp.print_stats()

'''

# endregion
# =============================================================================


# =============================================================================
# region calculate dividing streamline

from metpy.units import units
from metpy.calc.thermo import brunt_vaisala_frequency_squared
from scipy import interpolate
from scipy.integrate import quad
from scipy.optimize import fsolve

def dividing_streamline(ds_height, ds_theta, ds_u, hm):
    '''
    Input --------
    ds_height: height levels
    ds_theta: potential temperature at height levels
    ds_u: velocity at height levels
    hm: maximum height of mountain
    
    Output --------
    ds: dividing stream line
    
    '''
    
    brunt_v_sqr = brunt_vaisala_frequency_squared(
        ds_height * units.meter, ds_theta * units.kelvin)
    brunt_v_sqr_f = interpolate.interp1d(
        ds_height, brunt_v_sqr.magnitude, fill_value="extrapolate")
    
    velocity_sqr_f = interpolate.interp1d(
        ds_height, ds_u**2, fill_value="extrapolate")
    
    def integrand(z):
        return brunt_v_sqr_f(z)*(hm - z)
    
    def func(hc):
        y, err = quad(
            integrand,
            hc, hm, epsabs=10**-4, epsrel=10**-4, limit=50)
        y = y - velocity_sqr_f(hc)/2
        return y
    
    hc = fsolve(func, hm*0.6, col_deriv=True)
    
    return hc[0]

# endregion
# =============================================================================


# =============================================================================
# region calculate inversion layer
import numpy as np
def inversion_layer(temperature, altitude, topo = 0):
    '''
    Input --------
    temperature: vertical temperature profiles
    altitude: height of tem
    
    Output --------
    dinv: inversion_layer height
    '''
    
    temperature = temperature[altitude > topo].copy()
    altitude = altitude[altitude > topo].copy()
    
    try:
        level = np.where(temperature[1:] - temperature[:-1] > 0)[0][0]
        dinv = altitude[level]
        # teminv = temperature[level]
    except:
        dinv = np.nan
    
    return(dinv)


'''
# igrids = 400
# temperature = ml_3d_sim.T[i_hours, :, igrids, igrids].values
# altitude = ml_3d_sim.altitude.values
# temperature = np.linspace(30, 0, 30)
# altitude = np.arange(0, 30, 1)

temperature = temperature_sim[::-1]
altitude = altitude_sim[::-1]
topo = 158

dinv = inversion_layer(temperature, altitude, topo = 158)
teminv = temperature[np.where(altitude == dinv)]
fig, ax = plt.subplots(1, 1, figsize=np.array([8.8, 8]) / 2.54)
ax.plot(temperature, altitude, lw=0.5)
ax.scatter(teminv, dinv, s = 5)

ax.set_yticks(np.arange(0, 5.1, 0.5) * 1000)
ax.set_yticklabels((
    '0', '0.5', '1', '1.5', '2', '2.5', '3', '3.5', '4', '4.5', '5'))
ax.set_xticks(np.arange(245, 305.1, 10))
ax.set_ylim(0, 5000)
ax.set_xlim(245, 305)
ax.set_ylabel('Height [km]')
ax.set_xlabel('Temperature [K]')
ax.grid(True, linewidth=0.25, color='gray', alpha=0.5, linestyle='--')
fig.subplots_adjust(left=0.16, right=0.96, bottom=0.15, top=0.96)
fig.savefig('figures/00_test/trial.png')

'''
# endregion
# =============================================================================


# =============================================================================
# region calculate geometry


from haversine import haversine_vector, Unit, haversine

def calculate_geometry(vortex_lat, vortex_lon):
    '''
    # Input --------
    vortex_lat, vortex_lon: 1D array of latitude and longitude
    
    # Output --------
    (center_lat, center_lon, vortex_max_distance)
    center_lat, center_lon: center latitude and longitude
    vortex_max_distance: max distance between any pair of lon-lat
    '''
    
    center_lat = np.mean(vortex_lat)
    center_lon = np.mean(vortex_lon)
    center_lat_lon = (center_lat, center_lon)
    
    lat_lons = []
    for i in range(len(vortex_lat)):
        lat_lons.append((vortex_lat[i], vortex_lon[i]))
    
    distance2center = haversine_vector(
        center_lat_lon, lat_lons, Unit.KILOMETERS, comb=True)
    endpoint1 = np.where(distance2center==np.max(distance2center))[0][0]
    endpoint1_lon_lat = lat_lons[endpoint1]
    
    distance2endpoint1 = haversine_vector(
        endpoint1_lon_lat, lat_lons, Unit.KILOMETERS, comb=True)
    endpoint2 = np.where(distance2endpoint1 ==
                         np.max(distance2endpoint1))[0][0]
    endpoint2_lon_lat = lat_lons[endpoint2]
    
    vortex_max_distance = haversine(endpoint1_lon_lat, endpoint2_lon_lat)
    
    return(center_lat, center_lon, vortex_max_distance)

'''
# check
lyon = (45.7597, 4.8422) # (lat, lon)
paris = (48.8567, 2.3508)
new_york = (40.7033962, -74.2351462)
haversine_vector([lyon, lyon, paris], [paris, new_york, new_york],
                 Unit.KILOMETERS)
calculate_geometry(np.array([45.7597, 48.8567, 40.7033962]),
                   np.array([4.8422, 2.3508, -74.2351462]))

'''

# endregion
# =============================================================================


# =============================================================================
# region initial functions for vortex identification

import sys
sys.path.insert(0, '/project/pr133/qgao/DEoAI/git_repo/identify_cells')
import identify
import numpy as np

def vortex_identification(
        rvorticity, lat, lon, model_topo,
        min_rvor, min_max_rvor, min_size,
        max_distance2radius, max_ellipse_eccentricity, small_size_cell,
        grid_size=1.2, original_rvorticity = None, reject_info = False):
    '''
    Input --------
    rvorticity: 2d array, relative vorticity, unit 10^(-4)/s.
    lat, lon: corresponding latitude and longitude
    model_topo: model topography
    min_rvor: Minimum value for the boundaries
    min_max_rvor: Minimum maximum value of a cell. If no value within this cell
                  is larger than this, it will not be counted as a cell.
    min_size: Minimum size for the cell, in km^2
    max_distance2radius: max ratio between max_distance to radius
    grid_size: area of one grid, in km^2
    max_ellipse_eccentricity: max ratio between major-minor axes of fitted ellipse
    small_size_cell: criteria for small cell
    grid_size: area of a grid
    original_rvorticity: untransformed vorticity field
    reject_info: whether to track information of rejected vortices
    
    Output --------
    
    '''
    
    ################################ pre processing
    # <identify> is designed for 3D arrays, so firstly broadcast
    ny, nx = rvorticity.shape
    nz = 3
    rvor = np.zeros((nz, ny, nx))
    rvor[1] = rvorticity
    
    # <identify> is designed to use height criteria, but we don't need them.
    height = np.ones_like(rvor) * 100.
    min_height = 0.
    max_height = 200.
    min_topheight = 0.
    
    ################################ vortex identification
    # Get clusters (positive and negative vorticity)
    clusters_pos = identify.get_clusters(
        rvor, height, min_rvor, min_max_rvor,
        min_height, max_height, min_topheight)
    clusters_neg = identify.get_clusters(
        -rvor, height, min_rvor, min_max_rvor,
        min_height, max_height, min_topheight)
    # Aggregate clusters
    clusters = np.zeros((nz, ny, nx), dtype=np.float64)
    clusters += clusters_pos
    clusters -= clusters_neg
    
    ################################ vortex information extraction
    ######## 1st: Extract cells identifyier and count number of grid points
    # c_val_pos: Unique identifyier for cell
    # c_count_pos: How many grid points are part of this cell
    c_val_pos, c_count_pos = np.unique(clusters_pos, return_counts=True)
    c_count_pos = c_count_pos[c_val_pos != 0]
    c_val_pos = c_val_pos[c_val_pos != 0]
    c_val_neg, c_count_neg = np.unique(clusters_neg, return_counts=True)
    c_count_neg = c_count_neg[c_val_neg != 0]
    c_val_neg = c_val_neg[c_val_neg != 0]
    # Aggregate
    c_val = np.concatenate((c_val_pos, -c_val_neg.astype(np.int64)))
    c_count = np.concatenate((c_count_pos, c_count_neg))
    
    ################################ post processing
    # information extraction and noise filter
    
    # accepted vortices
    vortices = []
    is_vortex = np.zeros_like((clusters[1]), dtype='int8')
    vortex_indices = np.ma.array(
        np.zeros_like((clusters[1]), dtype='int8'), mask=True)
    # clusters_masked = np.ma.array(clusters[1])
    vortex_index = 0
    
    # rejected vortices
    rejected_vortices = []
    rejected_is_vortex = np.zeros_like((clusters[1]), dtype='int8')
    rejected_vortex_indices = np.ma.array(
        np.zeros_like((clusters[1]), dtype='int8'), mask=True)
    rejected_vortex_index = 0
    
    for i in range(len(c_val)):
        # i=63
        
        accepted = False
        
        vortex_size = c_count[i] * grid_size
        # 1_1st criterion: tatal area >= min_size in km^2
        criterion1_1 = (vortex_size >= min_size)
        
        if criterion1_1:
            vortex_cluster = np.where(clusters[1] == c_val[i])
            # 1_2nd criterion: no contact with model topograph
            criterion1_2 = (np.sum(model_topo[vortex_cluster]) == 0)
        else:
            criterion1_2 = False
        
        if (criterion1_1 & criterion1_2):
            vortex_lon = lon[vortex_cluster]
            vortex_lat = lat[vortex_cluster]
            cov = np.cov(vortex_lon.T, vortex_lat.T)
            pearson = abs(cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1]))
            radius_x = np.sqrt(1 + pearson)
            radius_y = np.sqrt(1 - pearson)
            ellipse_eccentricity = radius_x/radius_y
            # ellipse_eccentricity = np.sqrt(1 - radius_y**2 / radius_x**2)
            # 2_1st criterion: upper bound of ellipse eccentricity
            criterion2_1 = (ellipse_eccentricity <= max_ellipse_eccentricity)
            
            vortex_radius = (vortex_size / np.pi) ** 0.5
            center_lat, center_lon, vortex_max_distance = \
                calculate_geometry(vortex_lat, vortex_lon)
            # 2_2nd criterion: the ratio between max_distance to radius
            # >=max_distance2radius
            criterion2_2 = (vortex_max_distance /
                            vortex_radius <= max_distance2radius)
            
            if (criterion2_1 & criterion2_2):
                if original_rvorticity is None:
                    magnitude = rvorticity[vortex_cluster]
                else:
                    magnitude = original_rvorticity[vortex_cluster]
                mean_magnitude = np.mean(magnitude)
                if mean_magnitude < 0:
                    peak_magnitude = np.min(magnitude)
                else:
                    peak_magnitude = np.max(magnitude)
                
                # 3_1st criterion: special constraint on small cells
                criterion3_1 = (
                    (vortex_size > small_size_cell['size']) |
                    ((abs(peak_magnitude) >=
                      small_size_cell['peak_magnitude']) &
                     (ellipse_eccentricity <=
                      small_size_cell['max_ellipse_eccentricity']) &
                     (vortex_max_distance/vortex_radius <=
                      small_size_cell['max_distance2radius'])))
                
                if (criterion3_1):
                    vortices.append({
                        'index': i,
                        'lon': vortex_lon,
                        'lat': vortex_lat,
                        'center_lat': center_lat,
                        'center_lon': center_lon,
                        'size': vortex_size,
                        'radius': vortex_radius,
                        'distance2radius': vortex_max_distance/vortex_radius,
                        'magnitude': magnitude,
                        'peak_magnitude': peak_magnitude,
                        'mean_magnitude': mean_magnitude,
                        'ellipse_eccentricity': ellipse_eccentricity,
                    })
                    is_vortex[vortex_cluster] = 1
                    vortex_indices.mask[vortex_cluster] = False
                    vortex_indices.data[vortex_cluster] = vortex_index
                    vortex_index += 1
                    accepted = True
        
        
        
        if ( (not accepted) & reject_info):
            vortex_cluster = np.where(clusters[1] == c_val[i])
            vortex_lon = lon[vortex_cluster]
            vortex_lat = lat[vortex_cluster]
            cov = np.cov(vortex_lon.T, vortex_lat.T)
            pearson = abs(cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1]))
            radius_x = np.sqrt(1 + pearson)
            radius_y = np.sqrt(1 - pearson)
            ellipse_eccentricity = radius_x/radius_y
            vortex_radius = (vortex_size / np.pi) ** 0.5
            center_lat, center_lon, vortex_max_distance = \
                calculate_geometry(vortex_lat, vortex_lon)
            if original_rvorticity is None:
                magnitude = rvorticity[vortex_cluster]
            else:
                magnitude = original_rvorticity[vortex_cluster]
            mean_magnitude = np.mean(magnitude)
            if mean_magnitude < 0:
                peak_magnitude = np.min(magnitude)
            else:
                peak_magnitude = np.max(magnitude)
            
            rejected_vortices.append({
                'index': i,
                'lon': vortex_lon,
                'lat': vortex_lat,
                'center_lat': center_lat,
                'center_lon': center_lon,
                'size': vortex_size,
                'radius': vortex_radius,
                'distance2radius': vortex_max_distance/vortex_radius,
                'magnitude': magnitude,
                'peak_magnitude': peak_magnitude,
                'mean_magnitude': mean_magnitude,
                'ellipse_eccentricity': ellipse_eccentricity,
            })
            rejected_is_vortex[vortex_cluster] = 1
            rejected_vortex_indices.mask[vortex_cluster] = False
            rejected_vortex_indices.data[vortex_cluster] = vortex_index
            rejected_vortex_index += 1
    
    vortices_count = len(vortices)
    if (not reject_info):
        return([vortices, is_vortex, vortices_count, vortex_indices])
    else:
        rejected_vortices_count = len(rejected_vortices)
        return([vortices, is_vortex, vortices_count, vortex_indices,
                rejected_vortices, rejected_is_vortex,
                rejected_vortices_count, rejected_vortex_indices
                ])


'''
# test
import xarray as xr
import matplotlib.pyplot as plt
sys.path.append('/project/pr94/qgao/DEoAI')
from DEoAI_analysis.module.mapplot import framework_plot1

dsconst = xr.open_dataset(
    'scratch/simulation/20100801_09_3d/02_lm/lfsd20051101000000c.nc')
model_topo = dsconst.HSURF[0].values
dset = xr.open_dataset(
    'scratch/rvorticity/relative_vorticity_1km_1h_100m/relative_vorticity_1km_1h_100m201008.nc'
    )
time = dset.time.values
lon = dset.lon[80:920, 80:920].values
lat = dset.lat[80:920, 80:920].values
grid_size = 1.2 # in km^2
# import metpy.calc as mpcalc
# dx, dy = mpcalc.lat_lon_grid_deltas(lon.data, lat.data)
# from scipy import stats
# stats.describe((dx * dy.T).flatten())

ihours = 120 # 141 # 200
min_rvor=3.
min_max_rvor=6.
min_size=100
max_distance2radius = 4.5
max_ellipse_eccentricity = 3
small_size_cell = {
    'size': min_size * 1.4,
    'max_ellipse_eccentricity': max_ellipse_eccentricity / 2,
    'peak_magnitude': min_max_rvor * 1.5,
    'max_distance2radius': max_distance2radius - 1}
reject_info = False

time[ihours]
rvor = dset.relative_vorticity[ihours, 80:920, 80:920].values * 10**4

coeffs = pywt.wavedec2(rvor, 'haar', mode='periodic')
n_0, rec_rvor = sig_coeffs(coeffs, rvor, 'haar',)
rvorticity = rec_rvor
original_rvorticity = rvor


vortices, is_vortex, vortices_count, vortex_indices = vortex_identification(
    rvorticity, lat, lon, model_topo,
    min_rvor, min_max_rvor, min_size,
    max_distance2radius, max_ellipse_eccentricity, small_size_cell,
    original_rvorticity = original_rvorticity, reject_info = reject_info,
    )

(vortices1, is_vortex1, vortices_count1, vortex_indices1, rejected_vortices1,
 rejected_is_vortex1, rejected_vortices_count1, rejected_vortex_indices1) = vortex_identification(
    rvorticity, lat, lon, model_topo,
    min_rvor, min_max_rvor, min_size,
    max_distance2radius, max_ellipse_eccentricity, small_size_cell,
    original_rvorticity = original_rvorticity, reject_info = True,
    )

# np.sum(vortex_indices == 0)
# len(vortices[0]['lat'])
# is_vortex[np.where(vortex_indices.mask == False)]
# np.where(is_vortex[np.where(vortex_indices.mask == False)] == 0)
# np.where(vortex_indices.mask[np.where(is_vortex == 1)] == True)

for j in range(len(vortices)):
    print(
        str(j),
        ' index: ' + str(vortices[j]['index']),
        ' size: ' + str(round(vortices[j]['size'], 0)),
        ' dis2rad:'+str(round(vortices[j]['distance2radius'], 1)),
        ' peak:'+str(round(vortices[j]['peak_magnitude'], 1)),
        ' mean:'+str(round(vortices[j]['mean_magnitude'], 1)),
        ' ecc:'+str(round(vortices[j]['ellipse_eccentricity'], 1)),
        )

fig, ax = framework_plot1(
    "1km_lb",
    plot_vorticity=True,
    xlabel="Relative vorticity [$10^{-4}\;s^{-1}$]",
    vorticity_elements={
        'rvor': rvor,
        'lon': lon,
        'lat': lat,
        'vorlevel': np.arange(-12, 12.1, 0.1), 'ticks': np.arange(-12, 12.1, 3),
        'time_point': '2010-08-06 2100', 'time_location': [-23, 34],},
    )
for j in range(len(vortices)):
    ax.text(vortices[j]['center_lon'], vortices[j]['center_lat'],
            str(j), color = 'red', size = 8)
ax.contour(lon, lat, is_vortex,
           colors='lime', levels=np.array([-0.5, 0.5]),
           linewidths=0.2, linestyles='solid'
           )
fig.savefig('figures/00_test/trial.png', dpi=600)
plt.close('all')

for j in range(len(vortices1)):
    print(
        str(j),
        ' index: ' + str(vortices1[j]['index']),
        ' size: ' + str(round(vortices1[j]['size'], 0)),
        ' dis2rad:'+str(round(vortices1[j]['distance2radius'], 1)),
        ' peak:'+str(round(vortices1[j]['peak_magnitude'], 1)),
        ' mean:'+str(round(vortices1[j]['mean_magnitude'], 1)),
        ' ecc:'+str(round(vortices1[j]['ellipse_eccentricity'], 1)),
        )

for j in range(len(rejected_vortices1)):
    print(
        str(j),
        ' index: ' + str(rejected_vortices1[j]['index']),
        ' size: ' + str(round(rejected_vortices1[j]['size'], 0)),
        ' dis2rad:'+str(round(rejected_vortices1[j]['distance2radius'], 1)),
        ' peak:'+str(round(rejected_vortices1[j]['peak_magnitude'], 1)),
        ' mean:'+str(round(rejected_vortices1[j]['mean_magnitude'], 1)),
        ' ecc:'+str(round(rejected_vortices1[j]['ellipse_eccentricity'], 1)),
        )

fig, ax = framework_plot1(
    "1km_lb",
    plot_vorticity=True,
    xlabel="Relative vorticity [$10^{-4}\;s^{-1}$]",
    vorticity_elements={
        'rvor': rvor,
        'lon': lon,
        'lat': lat,
        'vorlevel': np.arange(-12, 12.1, 0.1), 'ticks': np.arange(-12, 12.1, 3),
        'time_point': '2010-08-00 0000', 'time_location': [-23, 34],},
    )
for j in range(len(vortices1)):
    ax.text(vortices1[j]['center_lon'], vortices1[j]['center_lat'],
            str(j), color = 'm', size = 8)
# for j in range(len(rejected_vortices1)):
#     ax.text(rejected_vortices1[j]['center_lon'],
#             rejected_vortices1[j]['center_lat'],
#             str(j), color = 'red', size = 8)
ax.contour(lon, lat, is_vortex1,
           colors='lime', levels=np.array([-0.5, 0.5]),
           linewidths=0.2, linestyles='solid'
           )
ax.contour(lon, lat, rejected_is_vortex1,
           colors='red', levels=np.array([-0.5, 0.5]),
           linewidths=0.2, linestyles='solid'
           )
fig.savefig('figures/00_test/trial.png', dpi=600)
plt.close('all')

'''
# endregion
# =============================================================================


# =============================================================================
# region functions for Madeira vortex identification extension

def contains_point(vortex_cluster, theta_anomalies):
    '''
    Input --------
    vortex_cluster:  a tuple containing two arrays with same length,
                     it contains the points where there is a vortex
    theta_anomalies: a tuple containing two arrays with same length,
                     it contains the points where there is local maximum theta.
    Output --------
    '''
    # check if each theta is inside the vortex
    contains = np.zeros_like(theta_anomalies[0])
    for i in range(len(contains)):
        # i = 0
        contains[i] = np.vstack((
            theta_anomalies[0][i] == vortex_cluster[0],
            theta_anomalies[1][i] == vortex_cluster[1])).all(axis = 0).any()
        if (contains.sum() > 0):
            return(True)
        else:
            return(False)



def vortex_identification1(
    rvorticity, lat, lon, model_topo, theta, wind_u, wind_v,
    center_madeira, poly_path, madeira_mask,
    min_rvor, min_max_rvor, min_size, min_size_theta,
    min_size_dir, min_size_dir1, max_dir, max_dir1, max_dir2,
    max_distance2radius,
    original_rvorticity = None, reject_info = False,
    grid_size=1.2, median_filter_size = 3, maximum_filter_size = 50, ):
    '''
    Input --------
    #### 2d spatial variables
    rvorticity: 2d array, relative vorticity, unit 10^(-4)/s.
    lat, lon: corresponding latitude and longitude
    model_topo: model topography
    theta: potential temperature
    wind_u, wind_v: 100m u and v component wind
    
    #### Madeira related
    center_madeira: center longitude and latitude
    poly_path: path contain madeira_mask
    madeira_mask: area where vortex from Madeira can appear
    
    #### criteria related
    min_rvor: Minimum value for the boundaries, unit [10^-4 s^-1] (e.g. 3.)
    min_max_rvor: Minimum peak vorticity magnitude of a cell. (e.g. 4.)
    min_size: Minimum size of a cell, in km^2. (e.g. 100.)
    min_size_theta: minimum size of a cell to check theta. (e.g. 450.)
    min_size_dir: minimum size of a cell to check direction. (e.g. 450.)
    min_size_dir1: minimum size of a cell to check direction. (e.g. 900.)
    max_dir: maximum direction deviation for min_size_dir. (e.g. 30.)
    max_dir1: maximum direction deviation for min_size_dir1. (e.g. 60.)
    
    #### others
    original_rvorticity: untransformed vorticity field
    reject_info: whether to track information of rejected vortices
    grid_size: area of one grid, in km^2
    
    Output --------
    
    '''
    
    ################################ load packages
    import sys
    sys.path.insert(0, '/project/pr94/qgao/DEoAI/git_repo/identify_cells')
    import identify
    import numpy as np
    from scipy.ndimage.filters import median_filter, maximum_filter
    from matplotlib.path import Path
    from haversine import haversine_vector, Unit
    
    ################################ pre processing
    # <identify> is designed for 3D arrays, so firstly broadcast
    ny, nx = rvorticity.shape
    nz = 3
    rvor = np.zeros((nz, ny, nx))
    rvor[1] = rvorticity
    
    # <identify> is designed to use height criteria, but we don't need them.
    height = np.ones_like(rvor) * 100.
    min_height = 0.
    max_height = 200.
    min_topheight = 0.
    
    ################################ vortex identification
    # Get clusters (positive and negative vorticity)
    clusters_pos = identify.get_clusters(
        rvor, height, min_rvor, min_max_rvor,
        min_height, max_height, min_topheight)
    clusters_neg = identify.get_clusters(
        -rvor, height, min_rvor, min_max_rvor,
        min_height, max_height, min_topheight)
    # Aggregate clusters
    clusters = np.zeros((nz, ny, nx), dtype=np.float64)
    clusters += clusters_pos
    clusters -= clusters_neg
    
    ################################ vortex information extraction
    ######## 1st: Extract cells identifyier and count number of grid points
    # c_val_pos: Unique identifyier for cell
    # c_count_pos: How many grid points are part of this cell
    c_val_pos, c_count_pos = np.unique(clusters_pos, return_counts=True)
    c_count_pos = c_count_pos[c_val_pos != 0]
    c_val_pos = c_val_pos[c_val_pos != 0]
    c_val_neg, c_count_neg = np.unique(clusters_neg, return_counts=True)
    c_count_neg = c_count_neg[c_val_neg != 0]
    c_val_neg = c_val_neg[c_val_neg != 0]
    # Aggregate
    c_val = np.concatenate((c_val_pos, -c_val_neg.astype(np.int64)))
    c_count = np.concatenate((c_count_pos, c_count_neg))
    
    ################################ post processing
    # information extraction and noise filter
    
    #### accepted vortices
    vortices = []
    is_vortex = np.zeros_like((clusters[1]), dtype='int8')
    vortex_indices = np.zeros_like((clusters[1]), dtype='int8')
    vortex_index = 0
    
    #### rejected vortices
    rejected_vortices = []
    rejected_is_vortex = np.zeros_like((clusters[1]), dtype='int8')
    rejected_vortex_indices = np.zeros_like((clusters[1]), dtype='int64')
    rejected_vortex_index = 0
    
    ################################ local theta maximum detection
    theta_filtered = median_filter(theta, size=median_filter_size)
    # add dummy variables to avoid identical neighbor values
    theta_dummy = theta_filtered # + np.random.normal(0, 0.000001, theta.shape)
    theta_ext = maximum_filter(
        theta_dummy, maximum_filter_size, mode='nearest')
    theta_anomalies = np.where(theta_ext == theta_dummy)
    
    ################################ iterate for each extracted vortices
    for i in range(len(c_val)):
        # i=63
        ################ 1st criterion: tatal area >= min_size in km^2
        vortex_size = c_count[i] * grid_size
        criterion1 = (vortex_size >= min_size)
        
        if (criterion1 == False) & (reject_info == False):
            continue
        
        ################ 2nd criterion: no contact with model topograph
        vortex_cluster = np.where(clusters[1] == c_val[i])
        criterion2 = (np.sum(model_topo[vortex_cluster]) == 0)
        
        if (criterion2 == False) & (reject_info == False):
            continue
        
        ################ 3rd criterion: within Madeira mask
        criterion3 = (poly_path.contains_point((
            vortex_cluster[0].mean(), vortex_cluster[1].mean()
        )))
        
        if (criterion3 == False) & (reject_info == False):
            continue
        
        ################ 4th criterion: If the vortex size is less than
        # min_size_theta, there must be a local potential temperature maximum
        # at the surface within the vortex or within its nominal radius.
        vortex_radius = (vortex_size / np.pi) ** 0.5
        vortex_lat = lat[vortex_cluster]
        center_lat = vortex_lat.mean()
        vortex_lon = lon[vortex_cluster]
        center_lon = vortex_lon.mean()
        if (vortex_size < min_size_theta):
            # within the vortex
            criterion4_1 = contains_point(vortex_cluster, theta_anomalies)
            
            # within its nominal radius
            distance2theta_anomaly = haversine_vector(
                [(center_lat, center_lon)],
                [tuple(i)
                 for i in zip(lat[theta_anomalies], lon[theta_anomalies])],
                Unit.KILOMETERS, comb=True)
            criterion4_2 = (distance2theta_anomaly[:, 0] <= vortex_radius).any()
            
            criterion4 = np.array((criterion4_1, criterion4_2)).any()
        else:
            criterion4 = True
        
        if (criterion4 == False) & (reject_info == False):
            continue
        
        ################ 5th criterion: wind angle and vortex direction
        vortex_wind_u = wind_u[vortex_cluster].mean()
        vortex_wind_v = wind_v[vortex_cluster].mean()
        vector_1 = [vortex_wind_u, vortex_wind_v]
        vector_2 = [center_lon - center_madeira[0],
                    center_lat - center_madeira[1]]
        unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
        unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
        dot_product = np.dot(unit_vector_1, unit_vector_2)
        angle = np.rad2deg(np.arccos(dot_product))
        
        if (vortex_size >= min_size_dir1):
            criterion5 = (angle < max_dir2)
        elif (vortex_size >= min_size_dir):
            criterion5 = (angle < max_dir1)
        else:
            criterion5 = (angle < max_dir)
        
        if (criterion5 == False) & (reject_info == False):
            continue
        
        ################ 6th criterion: maximum distance to radius
        center_lat1, center_lon1, vortex_max_distance = \
            calculate_geometry(vortex_lat, vortex_lon)
        criterion6 = (vortex_max_distance / vortex_radius <= \
            max_distance2radius)
        
        if (criterion6 == False) & (reject_info == False):
            continue
        
        ################################ get info
        if original_rvorticity is None:
            original_rvorticity = rvorticity
        
        magnitude = original_rvorticity[vortex_cluster]
        mean_magnitude = np.mean(magnitude)
        if mean_magnitude < 0:
            peak_magnitude = np.min(magnitude)
        else:
            peak_magnitude = np.max(magnitude)
        
        if(criterion1 & criterion2 & criterion3 & criterion4 & criterion5 & criterion6):
            vortices.append({
                'index': i,
                'center_lat': center_lat,
                'center_lon': center_lon,
                'size': vortex_size,
                'radius': vortex_radius,
                'peak_magnitude': peak_magnitude,
                'mean_magnitude': mean_magnitude,
                'mean_wind_u': vortex_wind_u,
                'mean_wind_v': vortex_wind_v,
                'angle': angle,
                'distance2radius': vortex_max_distance / vortex_radius,
            })
            is_vortex[vortex_cluster] = 1
            vortex_indices[vortex_cluster] = vortex_index
            vortex_index += 1
        else:
            rejected_vortices.append({
                'index': i,
                'center_lat': center_lat,
                'center_lon': center_lon,
                'size': vortex_size,
                'radius': vortex_radius,
                'peak_magnitude': peak_magnitude,
                'mean_magnitude': mean_magnitude,
                'mean_wind_u': vortex_wind_u,
                'mean_wind_v': vortex_wind_v,
                'angle': angle,
                'distance2radius': vortex_max_distance / vortex_radius,
            })
            rejected_is_vortex[vortex_cluster] = 1
            rejected_vortex_indices[vortex_cluster] = rejected_vortex_index
            rejected_vortex_index += 1
    
    vortices_count = len(vortices)
    if (not reject_info):
        return([
            vortices, is_vortex, vortices_count, vortex_indices,
            theta_anomalies])
    else:
        rejected_vortices_count = len(rejected_vortices)
        return([vortices, is_vortex, vortices_count, vortex_indices,
                theta_anomalies,
                rejected_vortices, rejected_is_vortex,
                rejected_vortices_count, rejected_vortex_indices
                ])


'''
################################################################ check
################################ package import
import xarray as xr
import matplotlib.pyplot as plt
import sys
sys.path.append('/project/pr94/qgao/DEoAI')
from DEoAI_analysis.module.mapplot import framework_plot1
import numpy as np
import glob
from DEoAI_analysis.module.namelist import (
    p0sl, r, cp, years, center_madeira)
from scipy import stats

################################ data import
######## model topograph
dsconst = xr.open_dataset(
    'scratch/simulation/20100801_09_3d/02_lm/lfsd20051101000000c.nc')
model_topo = dsconst.HSURF[0].values

######## 100m vorticity
iyear = 4
rvor_100m_f = np.array(sorted(glob.glob(
    'scratch/rvorticity/relative_vorticity_1km_1h_100m/relative_vorticity_1km_1h_100m20' + years[iyear] + '*.nc')))
rvor_100m = xr.open_mfdataset(
    rvor_100m_f, concat_dim="time", data_vars='minimal',
    coords='minimal', compat='override', chunks={'time': 1})
time = rvor_100m.time.values
lon = rvor_100m.lon[80:920, 80:920].values
lat = rvor_100m.lat[80:920, 80:920].values

######## parameter settings
grid_size = 1.2 # in km^2
median_filter_size = 3
maximum_filter_size = 50

min_rvor=3.
min_max_rvor=4.
min_size=100.
min_size_theta = 450.
min_size_dir = 450
min_size_dir1 = 900
max_dir = 30
max_dir1 = 40
max_dir2 = 50
max_distance2radius = 5
reject_info = True

######## original simulation to calculate surface theta
orig_simulation_f = np.array(sorted(
    glob.glob('/store/c2sm/pr04/jvergara/RUNS_IN_SCRATCH/MAC1/lm_f/1h/lffd20' +
              years[iyear] + '*[0-9].nc')
))

######## create a mask for Madeira
from matplotlib.path import Path
polygon=[
    (300, 0), (390, 320), (260, 580),
    (380, 839), (839, 839), (839, 0),
    ]
poly_path=Path(polygon)
x, y = np.mgrid[:840, :840]
coors = np.hstack((x.reshape(-1, 1), y.reshape(-1, 1)))
madeira_mask = poly_path.contains_points(coors).reshape(840, 840)

######## import wind
wind_100m_f = np.array(sorted(glob.glob(
    'scratch/wind_earth/wind_earth_1h_100m/wind_earth_1h_100m20' + years[iyear] + '*.nc')))
wind_100m = xr.open_mfdataset(
    wind_100m_f, concat_dim="time", data_vars='minimal',
    coords='minimal', compat='override', chunks={'time': 1})

################################ data read
# istart = np.where(time==np.datetime64('2010-08-03T20:00:00.000000000'))[0][0]
# istart = np.where(time==np.datetime64('2010-02-14T00:00:00.000000000'))[0][0]
# i = np.where(time==np.datetime64('2010-08-06T21:00:00.000000000'))[0][0]
# i = np.where(time==np.datetime64('2010-02-17T14:00:00.000000000'))[0][0]
# i = np.where(time==np.datetime64('2010-08-08T11:00:00.000000000'))[0][0]
i = 5221

######## relative vorticity and wind
rvor = rvor_100m.relative_vorticity[i, 80:920, 80:920].values * 10**4
wind_u = wind_100m.u_earth[i, 80:920, 80:920].values
wind_v = wind_100m.v_earth[i, 80:920, 80:920].values

######## theta
orig_simulation = xr.open_dataset(orig_simulation_f[i])
pres = orig_simulation.PS[0, 80:920, 80:920].values
tem2m = orig_simulation.T_2M[0, 80:920, 80:920].values
theta = tem2m * (p0sl/pres)**(r/cp)

################################ wavelet transform
coeffs = pywt.wavedec2(rvor, 'haar', mode='periodic')
n_0, rec_rvor = sig_coeffs(coeffs, rvor, 'haar',)
rvorticity = rec_rvor
original_rvorticity = rvor


vortices, is_vortex, vortices_count, vortex_indices, theta_anomalies = \
    vortex_identification1(
    rvorticity, lat, lon, model_topo, theta, wind_u, wind_v,
    center_madeira, poly_path, madeira_mask,
    min_rvor, min_max_rvor, min_size, min_size_theta,
    min_size_dir, min_size_dir1, max_dir, max_dir1, max_dir2,
    max_distance2radius,
    original_rvorticity = original_rvorticity, reject_info = False,
    grid_size=1.2, median_filter_size = 3, maximum_filter_size = 50,
    )

(vortices1, is_vortex1, vortices_count1, vortex_indices1, theta_anomalies1,
 rejected_vortices1, rejected_is_vortex1, rejected_vortices_count1,
 rejected_vortex_indices1) = vortex_identification1(
    rvorticity, lat, lon, model_topo, theta, wind_u, wind_v,
    center_madeira, poly_path, madeira_mask,
    min_rvor, min_max_rvor, min_size, min_size_theta,
    min_size_dir, min_size_dir1, max_dir, max_dir1, max_dir2,
    max_distance2radius,
    original_rvorticity = original_rvorticity, reject_info = True,
    grid_size=1.2, median_filter_size = 3, maximum_filter_size = 50,
    )

for i in range(len(rejected_vortices1)):
    if rejected_vortices1[i]['mean_magnitude'] == 2.9653055391263625:
        print(str(i) + ':' + str(rejected_vortices1[i]['size']))
np.mean(rvor[rejected_vortex_indices1 == 386])
np.mean(rec_rvor[rejected_vortex_indices1 == 386])


################################ check results
(is_vortex == is_vortex1).all()
vortices_count1 == vortices_count
(vortex_indices == vortex_indices1).all()
(theta_anomalies[0] == theta_anomalies1[0]).all()
(theta_anomalies[1] == theta_anomalies1[1]).all()

for j in range(len(vortices)):
    print(
        str(j),
        ' index:' + str(vortices[j]['index']),
        ' size:' + str(round(vortices[j]['size'], 0)),
        ' peak:'+str(round(vortices[j]['peak_magnitude'], 1)),
        ' mean:'+str(round(vortices[j]['mean_magnitude'], 1)),
        ' wind_u:'+str(round(vortices[j]['mean_wind_u'], 1)),
        ' wind_v:'+str(round(vortices[j]['mean_wind_v'], 1)),
        ' angle:'+str(round(vortices[j]['angle'], 1)),
        )

################################ plot rvor and results
fig, ax = framework_plot1(
    "1km_lb",
    plot_vorticity=True,
    xlabel="Relative vorticity [$10^{-4}\;s^{-1}$]",
    vorticity_elements={
        'rvor': rvor,
        'lon': lon,
        'lat': lat,
        'vorlevel': np.arange(-12, 12.1, 0.1), 'ticks': np.arange(-12, 12.1, 3),
        'time_point': str(time[i])[0:16], 'time_location': [-23, 34],},
    )
for j in range(len(vortices)):
    ax.text(vortices[j]['center_lon'], vortices[j]['center_lat'],
            str(j) + ':' + str(round(vortices[j]['size'], 0)) + \
                    ':' + str(round(vortices[j]['angle'], 1)),
            color = 'm', size=6, fontweight='normal')
    ax.quiver(vortices[j]['center_lon'], vortices[j]['center_lat'],
              vortices[j]['mean_wind_u'], vortices[j]['mean_wind_v'],
              rasterized=True)
    vortex_circle = plt.Circle(
        (vortices[j]['center_lon'], vortices[j]['center_lat']),
        vortices[j]['radius']/1.1*0.01,
        edgecolor='lime', facecolor = 'None', lw = 0.3, zorder = 2)
    ax.add_artist(vortex_circle)
ax.contour(lon, lat, is_vortex,
           colors='lime', levels=np.array([-0.5, 0.5]),
           linewidths=0.2, linestyles='solid'
           )
ax.contour(lon, lat, rejected_is_vortex1,
           colors='red', levels=np.array([-0.5, 0.5]),
           linewidths=0.1, linestyles='solid'
           )
ax.scatter(lon[theta_anomalies], lat[theta_anomalies],s=1, c='b',)
fig.savefig('figures/00_test/trial.png', dpi=600)
plt.close('all')

################################ plot theta and detected anomalies
stats.describe(theta.flatten())
theta_mid = 293
theta_min = theta_mid - 3
theta_max = theta_mid + 3

theta_ticks = np.arange(theta_min, theta_max + 0.01, 1)
theta_level = np.arange(theta_min, theta_max + 0.01, 0.025)
from DEoAI_analysis.module.namelist import rvor_cmp, transform
from matplotlib.colors import BoundaryNorm

fig, ax = framework_plot1("1km_lb",)
theta_time = ax.text(
    -23, 34, str(time[i])[0:10] + ' ' + str(time[i])[11:13] + ':00 UTC')
plt_theta = ax.pcolormesh(
    lon, lat, theta, cmap=rvor_cmp, rasterized=True, transform=transform,
    norm=BoundaryNorm(theta_level, ncolors=rvor_cmp.N, clip=False), )
cbar = fig.colorbar(
    plt_theta, ax=ax, orientation="horizontal",  pad=0.1, fraction=0.09,
    shrink=1, aspect=25, ticks=theta_ticks, extend='both',
    anchor=(0.5, 1), panchor=(0.5, 0))
cbar.ax.set_xlabel("Surface potential temperature [K]")
ax.contour(lon, lat, is_vortex,
           colors='lime', levels=np.array([-0.5, 0.5]),
           linewidths=0.2, linestyles='solid'
           )
ax.scatter(lon[theta_anomalies], lat[theta_anomalies],s=1, c='b',)
fig.savefig('figures/00_test/trial.png', dpi=600)


'''
# endregion
# =============================================================================






