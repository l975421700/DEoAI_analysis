

# =============================================================================
# region import packages

import datetime
import numpy as np
import pandas as pd
import xarray as xr
import metpy.calc as mpcalc
from metpy.units import units
import glob

print(datetime.datetime.now())

# endregion

# =============================================================================
# region calculate vorticity

dir_1km = '/store/c2sm/pr04/jvergara/RUNS_IN_SCRATCH/MAC1/lm_f'

# folder = '/1h/'
folder = '/1h_100m/'


years = ['06', '07', '08', '09', '10', '11', '12', '13', '14', '15']
months = ['01', '02', '03', '04', '05', '06',
          '07', '08', '09', '10', '11', '12']
years_months = [i + j for i,
    j in zip(np.repeat(years, 12), np.tile(months, 10))]


with xr.open_dataset(dir_1km + folder + 'lffd20060101000000z.nc') as ds:
    rlon = ds.rlon.data
    rlat = ds.rlat.data
    lon = ds.lon.data
    lat = ds.lat.data
    dx, dy = mpcalc.lat_lon_grid_deltas(lon, lat)

# range(len(years_months))
# np.arange(49, 60)
for k in np.arange(49, 60):
    begin_time = datetime.datetime.now()
    
    print(begin_time)
    
    filelist = np.array(sorted(
        glob.glob(dir_1km + folder + 'lffd20' + years_months[k] + '*[0-9z].nc')
    ))
    
    # create a file to store the results
    time = pd.date_range(
        "20" + years_months[k][0:2] + "-" + years_months[k][2:4] +
        "-01-00",
        "20" + years_months[k][0:2] + "-" + years_months[k][2:4] +
        "-" + filelist[-1][-12:-10] + "-23",
        freq="60min")
    relative_vorticity_1km_1h_100m = xr.Dataset(
        {"relative_vorticity": (
            ("time", "rlat", "rlon"),
            np.zeros((len(time), len(rlat), len(rlon)))),
         "lat": (("rlat", "rlon"), lat),
         "lon": (("rlat", "rlon"), lon),
         },
        coords={
            "time": time,
            "rlat": rlat,
            "rlon": rlon,
        }
    )
    
    # calculate vorticity
    ncfiles = xr.open_mfdataset(
        filelist, concat_dim="time",
        data_vars='minimal', coords='minimal', compat='override'
    ).metpy.parse_cf()
    
    print(datetime.datetime.now())
    
    u_100m = ncfiles.U.data.squeeze() * units('m/s')
    v_100m = ncfiles.V.data.squeeze() * units('m/s')
    relative_vorticity_1km_1h_100m.relative_vorticity[:, :, :] = mpcalc.vorticity(
        u_100m, v_100m, dx[None, :], dy[None, :], dim_order='yx')
    
    # write out files
    relative_vorticity_1km_1h_100m.to_netcdf(
        "/project/pr94/qgao/DEoAI/scratch/relative_vorticity_1km_" + folder[1:] + "relative_vorticity_1km_" + folder[1:-1] + "20" + years_months[k] + ".nc"
    )
    print(str(k) + "/" + str(len(years_months)) + "   " +
          str(datetime.datetime.now() - begin_time))


# endregion

# /project/pr94/qgao/miniconda3/envs/deoai/bin/python -c "from IPython import start_ipython; start_ipython()" --no-autoindent /project/pr94/qgao/DEoAI/DEoAI_analysis/2.2_vorticity_calculation1km_100m.py


