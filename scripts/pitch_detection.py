import numpy as np
from pathlib import Path
import pickle
from tools import utils as ut
import matplotlib.pyplot as plt
import scipy.signal as sg
import os

# present some pitch detections methods


def cepstrum(x, n=None):
    """
    Return the real cepstrum of the signal
    """
    spectrum = np.fft.fft(x, n=n)
    ceps = np.fft.ifft(np.log(np.abs(spectrum))).real

    return ceps

def power_cepstrum(x, n=None):
    """
    Return the power cepstrum of the signal
    """
    spectrum = np.fft.fft(x, n=n)
    ceps = np.fft.ifft(np.log(np.power(np.abs(spectrum), 2))).real

    return ceps

def autocorrelation(x):
    """
    Return the autocorrelation of the signal
    """
    return np.correlate(x, x, mode='same')

def nan_helper(y):
    """
    Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indicesx
    """

    return np.isnan(y), lambda z: z.nonzero()[0]



pathlist = Path('data/angles/samples').glob('*Elbow*')
for path in pathlist:
    path_in_str = str(path)
    file_name = os.path.splitext(os.path.basename(path_in_str))[0]
    angles = np.array(pickle.load(open(path_in_str, 'rb')))

    nan_ratio = len(angles[[np.isnan(angles)]]) / len(angles)
    if nan_ratio < 0.5:
        nans, x = nan_helper(angles)
        interp_angles = angles
        interp_angles[nans] = np.interp(x(nans), x(~nans), interp_angles[~nans])
        filtered_angles = sg.medfilt(interp_angles, 9)
        filtered_angles = filtered_angles - np.mean(filtered_angles)

        
        # pceps = power_cepstrum(interp_angles)
        auto_correl = autocorrelation(interp_angles)

        dt = 1 / ut.FPS
        t = [i * dt for i in range(len(angles))]

        n = len(auto_correl)
        m = [i - n//2 for i in range(n)]
    
        plt.subplot('211')
        plt.title('{}'.format(file_name))
        # plt.plot(t, angles)
        plt.plot(t, interp_angles)
        plt.plot(t, filtered_angles)
        # plt.plot(t, pceps)
        plt.xlabel('t (in s)')
        plt.ylabel('angle (in deg)')
        # plt.ylim(0, 200)

        plt.subplot('212')
        plt.title('Autocorrelation')
        plt.plot(m, auto_correl)
        plt.xlabel('lag')
        plt.ylabel('Correlation')
        plt.show()
