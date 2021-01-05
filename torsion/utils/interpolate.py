"""Utilities for interpolating torsion profiles.
"""
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from scipy.signal import argrelextrema


def get_global_min_interp1d(x, y, kind='cubic'):
    xmin_global, ymin_global = get_global_minimum(x, y)
    y = y - ymin_global
    x = np.concatenate((x[:-1]-360.0, x, x[1:]+360))
    y = np.concatenate((y[:-1], y, y[1:]))
    f = interp1d(x, y, kind=kind, bounds_error=False)
    return xmin_global, f


def get_global_minimum(xp, yp):
    '''
    @type xp: np.array
    @type yp: np.array
    @return: tuple(float, float)
    Algorithm:
    1. Generate cubic interpolation function
    2. Identify data points corresponding to the local minimum
    3. Identify local minimum using interpolation function starting from
       each local minimum location
    4. Local minimum with the lowest function value is taken as
       the global minimum
    '''
    xm, ym = get_minima(xp, yp)

    m = np.argmin(ym)

    if ym[m] > 9999:
        raise Exception('Unable to find global minimum')

    return xm[m], ym[m]


def get_minima(xp, yp, offsetGlobalMin = False):
    """
    Return the angles corresponding to the location of minima in the energy profile
    For each angle, it also returns the corresponding relative energies
    @type xp: np.array
    @type yp: np.array
    @param xp: angles
    @param yp: relative energies
    @return: np.array, np.array
    """
    f = interp1d(xp, yp, kind='cubic', bounds_error=False)
    x1 = argrelextrema(yp, np.less_equal)[0]
    # add the first and the last points to the set of starting points
    # from where the minimization will be done
    x1 = np.insert(x1, 0, 0)
    x1 = np.insert(x1, x1.size, yp.size-1)
    x0 = xp[x1]

    ym = np.ones(x0.shape)*10000
    xm = np.zeros(x0.shape)

    for i in range(0, x0.size):
        res = minimize(f, x0[i], bounds=((min(xp), max(xp)),))
        if res.success:
            ym[i] = f(res.x)
            xm[i] = res.x

    ia = np.argsort(ym)
    xm = xm[ia]
    ym = ym[ia]

    m = np.argmin(ym)

    if ym[m] > 9999:
        raise Exception('Unable to find global minimum')

    if offsetGlobalMin:
        ym = ym - ym[m]

    return xm, ym
