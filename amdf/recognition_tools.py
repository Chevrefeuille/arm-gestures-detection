#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 17 22:48:19 2018

@author: zeynep
"""

import numpy as np
import scipy.optimize
# from sensei.constants import FPS, MEDFILT, MEDFILT_SIZE
from constants import FPS, MEDFILT, MEDFILT_SIZE

import sys
import warnings

# supress warnings. this may be bad...
if not sys.warnoptions:
    warnings.simplefilter("ignore")
    

    
def amdf(x):
    """
    Average magnitude difference function
    
    D_m = 1/L * sum_{n = 1}^{n = L} |x[n] - x[n-m]|
    where 0 < m < M
    
    Here I limit m by M = X*FPS (X sec)
    """
        
    t_max = int(2*FPS) # 2 or 3 sec?
    amd = []
        
    for i in range(1,t_max):
        temp = np.abs(x[0:-i] - x[i:])
        temp = temp[np.where(np.invert( np.isnan(temp)))]
        amd.append(np.mean(np.abs(temp)))
        
    return amd   

def sin_fct(t, A, w, p, c):  
    """
    y = A sin(wt+p) + c
    """
    return A * np.sin(w*t + p) + c


def fit_sin(tt, yy):
    """
    Fit a sine to the input yy with time sequence tt, 
    and return fitting parameters :
        A amp
        w frequency of oscillation omega
        p phase
        c offset (AC)
        freq 
        fitfunc
    """
    tt = np.array(tt)
    yy = np.array(yy)
    ff = np.fft.fftfreq(len(tt), (tt[1]-tt[0]))   # assume uniform spacing
    Fyy = abs(np.fft.fft(yy))
    guess_freq = abs(ff[np.argmax(Fyy[1:])+1])   # excluding the zero frequency "peak", which is related to offset
    guess_amp = np.std(yy) * 2.**0.5
    guess_offset = np.mean(yy)
    guess = np.array([guess_amp, 2.*np.pi*guess_freq, 0., guess_offset])

    try:
        popt, pcov = scipy.optimize.curve_fit(sin_fct, tt, yy, p0=guess)
        A, w, p, c = popt
        f = w/(2.*np.pi)
        fitfunc = lambda t: A * np.sin(w*t + p) + c
        #print('Fit ')
        return {"amp": A, "omega": w, "phase": p, "offset": c, "freq": f,"fitfunc": fitfunc}

    
    except RuntimeError:
        # if i cannot find fit, probably there is a gesture...
        #print('No fit')
        return False

def auto_corr_valid(angles):
    """
    worth trying once replacing amdf with this
    """
    
    mu = np.mean(angles[np.logical_not(np.isnan(angles))])
    angles_n = angles - mu
    
    r = []
    for a in range(1,len(angles_n)):
        temp = np.multiply(angles_n[a:], angles_n[:-a])
        temp = temp[np.logical_not(np.isnan(temp))]
        r.append(np.sum(temp))
        
    return r


def medfilt (x, k):
    """
    Apply a length-k median filter to a 1D array x.
    Boundaries are extended by repeating endpoints.
    """
    assert k % 2 == 1, "Median filter length must be odd."
    assert x.ndim == 1, "Input must be one-dimensional."
    k2 = (k - 1) // 2
    y = np.zeros ((len (x), k), dtype=x.dtype)
    y[:,k2] = x
    for i in range (k2):
        j = k2 - i
        y[j:,i] = x[:-j]
        y[:j,i] = x[0]
        y[:-j,-(i+1)] = x[j:]
        y[-j:,-(i+1)] = x[-1]
    return np.median (y, axis=1)