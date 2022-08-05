from scipy.signal import detrend
import numpy as np

_units = {'Hz': lambda freq: freq,
          'mHz': lambda freq: freq * 1000.,
          'min': lambda freq: (1.0 / freq) / 60.0,
          'sec': lambda freq: (1.0 / freq),
          'msec': lambda freq: (1.0 / freq * 1000),
          'kHz': lambda freq: freq/1000.
          }


def power_spectrum_1d(array, cadence, units='Hz',
                      tp='constant'):
    """
    The function calculates the power spectrum.

    Parameters
    ----------
    array : ndarray
        The input 1-d array.
    cadence : int
        The cadence of the data in seconds.
    units : str
        The type of units for output [Hz or mHz or min or sec](default='Hz').

    Returns
     -------
    return_value : [float, float, float]
            power values,
            frequency values in Hz corresponded to power values,
            dispersion of the original data


    """
    if not units in _units.keys():
        print("Wrong units. Units must be {0}.".format(list(_units.keys())))
        return 0
    array = detrend(array, type=tp)
    sigma = np.std(array, ddof=1)
    array = np.hstack((array, np.zeros(array.shape[0], dtype=array.dtype)))
    cValue = np.fft.fft(array)
    disp = sigma ** 2
    amplitude = abs(cValue) / array.shape[0]
    power = (amplitude ** 2)
    freq = np.fft.fftfreq(array.shape[0], d=cadence)
    aa = np.where(freq < 0)
    index = aa[0][0]
    freq = freq[1:index]
    power = power[1:index] * 2
    return power, _units[units](freq), disp, units