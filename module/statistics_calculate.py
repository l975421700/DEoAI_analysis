

# Function to calculate quantiles, mean and standard deviation
import numpy as np


def get_statistics(arr, q, axis=0):
    '''
    # input ----
    arr: an array
    q: the quantile to be calculated
    axis: along which the quantile of arr is calculated
    
    # output ----
    arrays: numpy arrays: first dimension corresponds to [quantiles, mean, std]
    '''
    
    # if not isinstance(arr, np.ndarray):
    #     arr = np.array(arr)
    
    # calculate percentile
    qtl = np.percentile(a=arr, q=q, axis=axis)
    
    ptp100_0 = qtl[8, :] - qtl[0, :]
    ptp95_5 = qtl[7, :] - qtl[1, :]
    ptp90_10 = qtl[6, :] - qtl[2, :]
    ptp75_25 = qtl[5, :] - qtl[3, :]
    
    m = np.mean(a=arr, axis=axis, dtype=np.float64)
    std = np.std(a=arr, axis=axis, dtype=np.float64)
    
    res = np.concatenate((
        qtl,
        ptp100_0[None, :],
        ptp95_5[None, :],
        ptp90_10[None, :],
        ptp75_25[None, :],
        m[None, :],
        std[None, :],
        ), axis=0)
    
    return res

# line profiles and check ----
# arr = np.array(rvorticity_1km_1h_100m.relative_vorticity[:, 80:100, 80:100])
# q = quantiles[0]
# axis = 0

# from line_profiler import LineProfiler
# lp = LineProfiler()
# lp_wrapper = lp(get_statistics)
# lp_wrapper(
#     arr=arr,
#     q=q,
#     axis = axis)
# lp.print_stats()


# res = get_statistics(
#     arr=np.array(rvorticity_1km_1h_100m.relative_vorticity[:, 1:5, 1:5]),
#     q=quantiles[0],
#     axis=0
# )

# res[2, 1, 1]
# np.quantile(
#     a=np.array(rvorticity_1km_1h_100m.relative_vorticity[:, 2, 2]),
#     q=0.1
# )

# res[11, 1, 1]
# res[6, 1, 1] - res[2, 1, 1]

# res[13, 1, 1]
# np.mean(
#     a=np.array(rvorticity_1km_1h_100m.relative_vorticity[:, 2, 2]),
#     axis=0
# )


# check ----
# a = np.array([[[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]],
#               [[10, 20, 30, 40, 50], [60, 70, 80, 90, 100],
#                 [110, 120, 130, 140, 150]]],
#              )
# ddd = get_statistics(
#     arr=a,
#     q=np.array([0, 5, 10, 25, 50, 75, 90, 95, 100]),
#     axis=0
# )
# ddd[1, 1, 1]
# np.quantile(a[:, 1, 1], 0.05)
# ddd[11, 1, 1]
# ddd[6, 1, 1] - ddd[2, 1, 1]
# ddd[13, 1, 1]
# np.mean(a[:, 1, 1])







