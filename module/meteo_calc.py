

# =============================================================================
# region function to calculate lifting condensation level

# reference
# https://romps.berkeley.edu/papers/pubs-2016-lcl.html
# https://romps.berkeley.edu/papers/pubdata/2016/lcl/lcl.py
# Version 1.0 released by David Romps on September 12, 2017.


import math
import scipy.special
import numpy as np
def lifting_condensation_level(pres, tem, rh):
    '''
    ----Input
    pres: in Pascals
    tem: in Kelvins
    rh: relative humidity with respect to liquid water if T >= 273.15 K
                          with respect to ice if T < 273.15 K.
    
    ----output
    lcl: the height of the lifting condensation level (LCL) in meters.
    '''
    
    # Parameters
    Ttrip = 273.16     # K
    ptrip = 611.65     # Pa
    E0v = 2.3740e6   # J/kg
    E0s = 0.3337e6   # J/kg
    ggr = 9.81       # m/s^2
    rgasa = 287.04     # J/kg/K
    rgasv = 461        # J/kg/K
    cva = 719        # J/kg/K
    cvv = 1418       # J/kg/K
    cvl = 4119       # J/kg/K
    cvs = 1861       # J/kg/K
    cpa = cva + rgasa
    cpv = cvv + rgasv
    
    # The saturation vapor pressure over liquid water
    def pvstarl(T):
        return ptrip * (T/Ttrip)**((cpv-cvl)/rgasv) * \
        math.exp((E0v - (cvv-cvl)*Ttrip) / rgasv * (1/Ttrip - 1/T))
    
    # The saturation vapor pressure over solid ice
    def pvstars(T):
        return ptrip * (T/Ttrip)**((cpv-cvs)/rgasv) * \
        math.exp((E0v + E0s - (cvv-cvs)*Ttrip) / rgasv * (1/Ttrip - 1/T))
    
    # Calculate pv from rh
    # The variable rh is assumed to be
    # with respect to liquid if T > Ttrip and
    # with respect to solid if T < Ttrip
    if tem > Ttrip:
        pv = rh * pvstarl(tem)
    else:
        pv = rh * pvstars(tem)
    
    rhl = pv / pvstarl(tem)
    rhs = pv / pvstars(tem)
    
    if pv > pres:
        return np.nan
    
    # Calculate lcl_liquid and lcl_solid
    qv = rgasa*pv / (rgasv*pres + (rgasa-rgasv)*pv)
    rgasm = (1-qv)*rgasa + qv*rgasv
    cpm = (1-qv)*cpa + qv*cpv
    if rh == 0:
        return cpm*tem/ggr
    
    aL = -(cpv-cvl)/rgasv + cpm/rgasm
    bL = -(E0v-(cvv-cvl)*Ttrip)/(rgasv*tem)
    cL = pv/pvstarl(tem)*math.exp(-(E0v-(cvv-cvl)*Ttrip)/(rgasv*tem))
    lcl = cpm*tem/ggr*(
        1 - bL/(aL*scipy.special.lambertw(bL/aL*cL**(1/aL), -1).real))
    
    return lcl


'''
from metpy.units import units
import metpy.calc as mpcalc
pres = 101320.75
tem = 296.5619
rh = 0.90995485
lifting_condensation_level(pres, tem, rh)

tem_dew = mpcalc.dewpoint_from_relative_humidity(
    tem * units('K'), rh
)

# mpcalc.lcl(pres * units('Pa'), tem * units('K'),
#            tem_dew, max_iters=50, eps=1e-05)


def lcl_david(p,T,rh=None,rhl=None,rhs=None,return_ldl=False,return_min_lcl_ldl=False):

   import math
   import scipy.special

   # Parameters
   Ttrip = 273.16     # K
   ptrip = 611.65     # Pa
   E0v   = 2.3740e6   # J/kg
   E0s   = 0.3337e6   # J/kg
   ggr   = 9.81       # m/s^2
   rgasa = 287.04     # J/kg/K 
   rgasv = 461        # J/kg/K 
   cva   = 719        # J/kg/K
   cvv   = 1418       # J/kg/K 
   cvl   = 4119       # J/kg/K 
   cvs   = 1861       # J/kg/K 
   cpa   = cva + rgasa
   cpv   = cvv + rgasv

   # The saturation vapor pressure over liquid water
   def pvstarl(T):
      return ptrip * (T/Ttrip)**((cpv-cvl)/rgasv) * \
         math.exp( (E0v - (cvv-cvl)*Ttrip) / rgasv * (1/Ttrip - 1/T) )
   
   # The saturation vapor pressure over solid ice
   def pvstars(T):
      return ptrip * (T/Ttrip)**((cpv-cvs)/rgasv) * \
         math.exp( (E0v + E0s - (cvv-cvs)*Ttrip) / rgasv * (1/Ttrip - 1/T) )

   # Calculate pv from rh, rhl, or rhs
   rh_counter = 0
   if rh  is not None:
      rh_counter = rh_counter + 1
   if rhl is not None:
      rh_counter = rh_counter + 1
   if rhs is not None:
      rh_counter = rh_counter + 1
   if rh_counter != 1:
      print(rh_counter)
      exit('Error in lcl: Exactly one of rh, rhl, and rhs must be specified')
   if rh is not None:
      # The variable rh is assumed to be 
      # with respect to liquid if T > Ttrip and 
      # with respect to solid if T < Ttrip
      if T > Ttrip:
         pv = rh * pvstarl(T)
      else:
         pv = rh * pvstars(T)
      rhl = pv / pvstarl(T)
      rhs = pv / pvstars(T)
   elif rhl is not None:
      pv = rhl * pvstarl(T)
      rhs = pv / pvstars(T)
      if T > Ttrip:
         rh = rhl
      else:
         rh = rhs
   elif rhs is not None:
      pv = rhs * pvstars(T)
      rhl = pv / pvstarl(T)
      if T > Ttrip:
         rh = rhl
      else:
         rh = rhs
   if pv > p:
      return NA

   # Calculate lcl_liquid and lcl_solid
   qv = rgasa*pv / (rgasv*p + (rgasa-rgasv)*pv)
   rgasm = (1-qv)*rgasa + qv*rgasv
   cpm = (1-qv)*cpa + qv*cpv
   if rh == 0:
      return cpm*T/ggr
   aL = -(cpv-cvl)/rgasv + cpm/rgasm
   bL = -(E0v-(cvv-cvl)*Ttrip)/(rgasv*T)
   cL = pv/pvstarl(T)*math.exp(-(E0v-(cvv-cvl)*Ttrip)/(rgasv*T))
   aS = -(cpv-cvs)/rgasv + cpm/rgasm
   bS = -(E0v+E0s-(cvv-cvs)*Ttrip)/(rgasv*T)
   cS = pv/pvstars(T)*math.exp(-(E0v+E0s-(cvv-cvs)*Ttrip)/(rgasv*T))
   lcl = cpm*T/ggr*( 1 - \
      bL/(aL*scipy.special.lambertw(bL/aL*cL**(1/aL),-1).real) )
   ldl = cpm*T/ggr*( 1 - \
      bS/(aS*scipy.special.lambertw(bS/aS*cS**(1/aS),-1).real) )

   # Return either lcl or ldl
   if return_ldl and return_min_lcl_ldl:
      exit('return_ldl and return_min_lcl_ldl cannot both be true')
   elif return_ldl:
      return ldl
   elif return_min_lcl_ldl:
      return min(lcl,ldl)
   else:
      return lcl

lcl(pres, tem, rh = rh)

'''
# endregion
# =============================================================================




