import numpy as np


def cross_correlation(array1, array2, max_shift=None):
    """
    The function returns the shift of the axis x,y and coefficient of
    correlation.

    Parameters
    ----------
    array1 : ndarray
        The input array.
    array2 : ndarray
        The input array
    max_shift: int
        maximal shift for cross-correlation maximum
        (only for 1d arrays)

    Returns
    -------
    return_value : tuple

        result 1d : correlation coefficient, x-axis shift
        result 2d : correlation coefficient, y-axis shift, x-axis shift

    Examples
    --------
    >>> import numpy as np
    >>> array1 = np.sin(np.arange(1024, dtype = 'f4'))
    >>> array2 = np.roll(array1,10,0))
    >>> result = cross_correlation(array1, array2)
    >>> print(result)
    """

    if array1.shape != array2.shape:
        print ('array1.shape and array2.shape mast be equal!')
        return
    fft = np.fft.fft
    ifft = np.fft.ifft
    alen = array1.shape[0]
    if array1.ndim == 2:
        fft = np.fft.fft2
        ifft = np.fft.ifft2
        blen = array1.shape[1]

    f1 = np.ma.conjugate(fft(array1))
    f2 = fft(array2)
    corr = abs(ifft(f1 * f2))
    c = corr / np.sqrt(((array1 ** 2).sum()) * ((array2 ** 2).sum()))

    if array1.ndim == 2:
        yp, xp = np.unravel_index(c.argmax(), c.shape)
        yshift = yp - alen
        xshift = xp - blen
        xshift = xshift + blen if abs(xshift) > blen / 2 else xshift
        yshift = yshift + alen if abs(yshift) > alen / 2 else yshift
        return c.max(), yshift, xshift

    elif array1.ndim == 1:
        if max_shift is not None:
            max_shift = int(max_shift)
            mid = int(alen/2)
            c = np.roll(c, mid, axis=0)
            ii = mid - max_shift
            fi = mid + max_shift
            c = c[ii:fi]
            xshift = c.argmax() - max_shift
            return c.max(), xshift

        else:
            xp = c.argmax()
            xshift = xp - alen
            xshift = xshift + alen if abs(xshift) > alen / 2 else xshift
            return c.max(), xshift, c


def correlation(a1, a2, ddof=0):
    """

    Parameters
    ----------

    Returns
    -------

    Examples
    --------

    """
    return np.mean((a1 - np.mean(a1)) * 
                   (a2 - np.mean(a2))) / (np.std(a1, ddof=ddof) * 
                                          np.std(a2, ddof=ddof))


def auto_correlation(x, lag=1, squezee=False):
    """

    Parameters
    ----------

    Returns
    -------

    Examples
    --------

    """
    if squezee:
        x = np.squeeze(x)
    ln = x.shape[0] - lag
    mu = x.mean()
    s = x.std()
    return ((x[:-lag] - mu) * (x[lag:] - mu)).sum() / (s ** 2) / ln


if __name__ == '__main__':
    x = np.arange(2 * 5).reshape(2, 5)
    y = np.roll(np.arange(2 * 5).reshape(2, 5), 3)

    print(cross_correlation(x, y))