import scipy
import scipy.signal as signal
import numpy as np


def interpol(dset, i_factor, out=None):
    if out is None:
        ret_array = np.zeros((dset.shape[0] * i_factor), dtype=np.float64)
        x = np.arange(dset.shape[0])
    else:
        ret_array = out
        x = np.arange(dset.shape[1])


    row_i_f = scipy.interpolate.interp1d(x, dset)
    ret_array[:] = row_i_f(np.linspace(0, dset.shape[0] - 1, dset.shape[0]*i_factor))

    return ret_array

def interpol_1d(dset, i_factor):
    """
    Interpolate
    """
    ret_array = np.zeros((dset.shape[0], dset.shape[1]*i_factor), dtype=np.float64)

    x = np.arange(dset.shape[1])

    for i in range(dset.shape[0]):
        row_i_f = scipy.interpolate.interp1d(x, dset[i])
        row_i = row_i_f(np.linspace(0, dset.shape[1] - 1, dset.shape[1]*i_factor))
        ret_array[i,:] = row_i


    return ret_array


def interpol_2d_by_1d(dset, i_factor):
    return interpol_1d(interpol_1d(dset, i_factor).T, i_factor).T


def interpol_1d_rot(dset, i_factor):
    """
    Interpolate
    """
    return interpol_1d(dset.T, i_factor).T

def interpol_1d_rot90(dset, i_factor):
    """
    Interpolate
    """
    return np.rot90(interpol_1d(np.rot90(dset,3), i_factor),1)