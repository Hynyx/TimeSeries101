import numpy as np

from sdppy.wavelet import wavelet, wave_signif


from cycler import cycler
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from scipy.signal import detrend

from math import floor, ceil
import time
import datetime

from .correlation import cross_correlation, auto_correlation
from .spectrum import power_spectrum_1d

class TimeSeries:
    def __init__(self, data, cadence=1.0, name='', signif_lvl=0.95,
                 timing=None):
        self.data = np.array(data)
        self.cadence = cadence
        self.name = name
        self.sig_lvl = signif_lvl
        self.fspectrum = {}
        self.waveletfiltered = dict()
        self.time_axis = np.arange(self.data.shape[0])*self.cadence/60.
        self.local_wvlt = None
        self.timing = timing

    def get_data(self, normalize=None):
        d = self.data
        if normalize is not None:
            m = np.mean(d, axis=0)
            d = d - m
            if normalize == 'max':
                d = d / d.max()
            elif normalize == 'std':
                d = d / np.std(d, ddof=1)
        return d

    def get_time(self):
        return self.timing

    def set_name(self, name):
        self.name = name

    def set_sig_lvl(self, signif):
        self.sig_lvl = signif

    def get_fspectrum(self, normalize=False, flimit=None):
        fs = {}
        fs.update(self.fspectrum)
        variance = np.std(self.data, ddof=1) ** 2
        lens = fs['power'].shape[0]
        freq = fs['freq']
        # norm_factor  = 1
        if normalize is True:
            fs['power'] = fs['power'] * lens / variance
        if flimit is not None:
            fmin = flimit[0]
            fmax = flimit[1]
            ind_min = np.where(freq < fmin)[0].tolist()
            if not ind_min:
                ind_min = 0
            else:
                ind_min = ind_min[-1]
            ind_max = np.where(freq > fmax)[0].tolist()
            if not ind_max:
                ind_max = -1
            else:
                ind_max = ind_max[0]
            fs['power'] = fs['power'][ind_min:ind_max]
            fs['freq'] = fs['freq'][ind_min:ind_max]
        return fs

    def cros_corr(self, data, max_shift=None):
        return cross_correlation(self.data, data, max_shift=max_shift)

    def calc_fspectrum(self, units='mHz', trend='constant'):
        print(self.data.shape)
        fps = power_spectrum_1d(self.data, self.cadence,
                                units=units, tp=trend)
        self.fspectrum['power'] = fps[0]
        self.fspectrum['freq'] = fps[1]
        self.fspectrum['variance'] = fps[2]
        self.fspectrum['units'] = fps[3]

    def get_cadence(self):
        return self.cadence

    def filter_by_wavelet(self, s0, j, dj=0.05, save=True):
        wv = wavelet(self.data, dt=self.cadence*1000., s0=s0, dj=dj, j=j,
                     core='morlet')
        # print (wv.period)
        # period = wv['period']
        band_label = format((wv.period[0]), '0.2f') + '-' +\
                format((wv.period[-1]), '0.2f') + 'msec'
        print('label: ', band_label, 'period:', wv.period)
        if save is True:
            self.waveletfiltered[band_label] = np.copy(wv.y1)
#            print id(self.waveletfiltered[label])
        return band_label

    def calc_local_wavelet(self, s0, dj, j, save=True,
                           background='white', core='morlet'):
        siglvl = self.sig_lvl
        lag1 = [0.0]
        gws = None
        if background == 'red':
            lag1 = self.calc_lag1()
        if background == 'gws':
            gws = self.calc_global_wavelet()

        dt = self.cadence/60.
        wv = wavelet(self.data, dt=dt, s0=s0, dj=dj, j=j,
                     core=core, recon=False, siglvl=siglvl)
        scal = wv['scale']
        signif = wave_signif(self.data, dt=dt, scale=scal,
                             siglvl=siglvl, gws=gws,
                             core=core, lag1=lag1)
        wv['signif'] = signif[0]
        wv['power'] = abs(wv['wave']) ** 2
        wv['time'] = self.time_axis

        self.local_wvlt = wv

    def calc_lag1(self):
        lag1 = auto_correlation(self.data[0])
        return lag1

    def calc_global_wavelet(self):
        pass

    def get_local_wavelet(self):
        return self.local_wvlt

    def get_filtered(self, band_label=None, normalize=False):
        wave_max = 1
        if normalize is True:
            wave_max = np.max(self.waveletfiltered[band_label])
        return self.waveletfiltered[band_label] / wave_max

    def get_filtered_list(self):
        keys = list(self.waveletfiltered)
        return keys

    def get_duration(self, units='min'):
        if units == 'min':
            return (self.data.shape[0] * self.cadence) / 60.0
        elif units == 'sec':
            return (self.data.shape[0] * self.cadence)
        elif units == 'hr':
            return (self.data.shape[0] * self.cadence) / 3600.0
        elif units == 'msec':
            return (self.data.shape[0] * self.cadence) * 1000

    def max_power(self, freq_range=[0, None], normalize=True):
        power = self.fspectrum['power']
        freq = self.fspectrum['freq']
        fmin = freq_range[0]
        fmax = freq_range[1]
        ind_min = np.where(freq < fmin)[0].tolist()
        if not ind_min:
            ind_min = 0
        else:
            ind_min = ind_min[-1]
        ind_max = np.where(freq > fmax)[0].tolist()
        if not ind_max:
            ind_max = -1
        else:
            ind_max = ind_max[0]
        p = power[ind_min:ind_max]
        f = freq[ind_min:ind_max]
        if normalize is True:
            variance = np.std(self.data, ddof=1) ** 2
            p = p * self.data.shape[0] / variance
        pmax = p.max()
        fmax = f[p.argmax()]

        print('max power: ', pmax, ' freq: ', fmax)
        return pmax, fmax

    def plot_fspectrum(self, subplot=None, signif_lvl=3.0,
                       flimit=None, plimit=None, **kwargs):

        power = self.fspectrum['power']
        freq = self.fspectrum['freq']
        units = self.fspectrum['units']

        x_label = kwargs.pop('xlabel', 'Frequency')
        y_label = kwargs.pop('ylabel', 'Power')
        x_units = kwargs.pop('xunits', units)
        y_units = kwargs.pop('yunits', 'Arbitrary units')
        no_labels = kwargs.pop('no_label', False)
        extra_title = kwargs.pop('extra_title', '')
        title = kwargs.pop('title', 'Fourier')
        normalization = kwargs.pop('normalization', 'white')
        signif_lvl = kwargs.pop('signif_lvl', 3.0)



        # var = self.fspectrum['variance']
        variance = np.std(self.data, ddof=1) ** 2
        lens = power.shape[0]
        # norm_factor  = 1
        if normalization == 'white':
            power = power * lens / variance
        print('name: ' + 'Fourier, ' + self.name + '  is plotting')
        print('plimit: ', plimit, 'flimit: ', flimit)
        ax = None
        if subplot is None:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
        else:
            ax = subplot
        xmin = None
        xmax = None
        ymin = None
        ymax = None
        if flimit is not None:
            xmin = flimit[0]
            xmax = flimit[1]
        else:
            xmin = freq[1]
            xmax = freq[-1]
        if plimit is not None:
            # ymin = plimit[0] # always zero-level
            ymax = plimit[1]
        else:
            ind_min = np.where(freq < xmin)[0].tolist()
            if not ind_min:
                ind_min = 0
            else:
                ind_min = ind_min[-1]
            ind_max = np.where(freq > xmax)[0].tolist()
            if not ind_max:
                ind_max = -1
            else:
                ind_max = ind_max[0]
            # ymin = np.min(power[ind_min:ind_max])
            print('ind', ind_min, ind_max)
            ymax = np.max(power[ind_min:ind_max])
        ax.plot(freq, power, **kwargs)
        ax._get_lines.prop_cycler.__next__()
        ax.plot([xmin, xmax], [signif_lvl, signif_lvl])
        ax.axis('tight')
        if no_labels is False:
            ax.set_xlabel(x_label+', '+x_units)
            ax.set_ylabel(y_label+', '+y_units)
            ax.set_title(title + ', ' + self.name + ', ' + extra_title)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        if subplot is None:
            plt.show()

    def plot_filtered(self, band_label, subplot=None, units='min',
                      **kwargs):
        x_label = kwargs.pop('xlabel', 'Time')
        y_label = kwargs.pop('ylabel', '')
        x_units = kwargs.pop('xunits', 'min')
        y_units = kwargs.pop('yunits', '')
        time_range = kwargs.pop('time_range', None)
        no_labels = kwargs.pop('no_label', False)
        bl = band_label
        if units =='mHz':
            a = band_label.split('-')
            i = format(16.6/float(a[0]), '0.2f')
            b = a[1].split('m')
            f = format(16.6/float(b[0]), '0.2f')
            bl = f + '-' + i +'mHz'
        extra_title = kwargs.pop('extra_title', '')
        title = kwargs.pop('title', 'Filtered signal'+', ' + self.name)
        normalize = kwargs.pop('normalize', False)
        min_step = 60.0 / self.cadence
        if units == 'msec':
            min_step *= 1000
        ax = None
        if subplot is None:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
        else:
            ax = subplot
        y = self.waveletfiltered[band_label]
        if normalize == 'max':
            y = y / y.max()
        elif normalize == 'std':
            y = y / np.std(y, ddof=1)
        x = np.linspace(0, self.get_duration(x_units), y.shape[0])
        if time_range is not None:
            itime = time_range[0]
            ftime = time_range[1]
            ii = int(itime * min_step)
            fi = int(ftime * min_step) + 1
            x = x[ii:fi]
            y = y[ii:fi]

        g1,  = ax.plot(x, y, **kwargs)
        if no_labels is False:
            ax.set_xlabel(x_label+', '+x_units)
            ax.set_ylabel(y_label+', '+y_units)
            ax.set_title(title + ', ' + band_label + extra_title)
        if subplot is None:
            plt.show()
        return g1

    def plot_data(self, subplot=None, **kwargs):
        x_label = kwargs.pop('xlabel', 'Time')
        y_label = kwargs.pop('ylabel', '')
        x_units = kwargs.pop('xunits', 'min')
        y_units = kwargs.pop('yunits', '')
        no_labels = kwargs.pop('no_label', False)
        extra_title = kwargs.pop('extra_title', '')
        title = kwargs.pop('title', 'Uniltered signal')
        normalize = kwargs.pop('normalize', 'std')
        smooth = kwargs.pop('smooth', None)
        d = self.get_data(normalize=normalize)
        y = d
        if smooth is not None:
            y = signal_smooth(y, window_len=smooth)
        ax = None
        if subplot is None:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
        else:
            ax = subplot
        x = np.linspace(0, self.get_duration(x_units), y.shape[0])
        g1,  = ax.plot(x, y, **kwargs)
        if no_labels is False:
            ax.set_xlabel(x_label+', '+x_units)
            ax.set_ylabel(y_label+', '+y_units)
            ax.set_title(title + ', ' + self.name + ', ' + extra_title)
        if subplot is None:
            plt.show()
        return g1

    def plot_wavelet(self, subplot=None, freq=[1, 3, 5, 10, 15],
                     levels=[], **kwargs):
        x_label = kwargs.pop('xlabel', 'Time')
        y_label1 = kwargs.pop('ylabel1', 'Period')
        y_label2 = kwargs.pop('ylabel2', 'Frequency')
        x_units = kwargs.pop('xunits', 'min')
        y_units1 = kwargs.pop('yunits1', 'min')
        y_units2 = kwargs.pop('yunits2', 'mhz')
        extra_title = kwargs.pop('extra_title', '')
        title = kwargs.pop('title', 'local wavelet')
        signif = kwargs.pop('signif', None)

        w_power = self.local_wvlt['power']
        per = self.local_wvlt['period']
        scal = self.local_wvlt['scale']
        signif = self.local_wvlt['signif']
        variance = np.std(self.data, ddof=1) ** 2
        signif = signif / variance
        coi = self.local_wvlt['coi']
        tim = self.time_axis
        ax = None
        if subplot is None:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
        else:
            ax = subplot
        ax.imshow(w_power, extent=[tim[0], tim[-1], per[1], per[-1]], **kwargs)
        formatter = ticker.FormatStrFormatter('%1.2f')
        formatter2 = ticker.FormatStrFormatter('%1.2f')
        ax.yaxis.set_major_formatter(formatter)
        tk = np.arange(1, 22, 1).tolist()
        # tk2 = [1.0/(l * 60) * 1000 for l in tk]
        ax.set_yscale('log')
        ax.set_yticks(tk)
        tkl = []
        # tkl2 = []
        for j in range(1, len(tk) + 1, 1):
            if j > 10 and j % 4 != 0:
                tkl.append('')
                continue
            tkl.append(str(tk[j-1]))
            # tkl2.append('{:1.2f}'.format(tk2[j-1]))
        labels = ax.set_yticklabels((tkl))
        tk2 = 1.0/(np.arange(1, 21, 1)/1000) / 60
        bx = ax.twinx()
        bx.set_aspect(ax.get_aspect())
        bx.set_ylim(ax.get_ylim())
        bx.yaxis.set_major_formatter(formatter2)
        bx.set_yscale('log')
        bx.set_yticks(tk2)
        tkl2 = []
        f = np.arange(1, 21, 1).tolist()
        for j in range(0, len(tk2), 1):
            if f[j] in freq:
                tkl2.append(f[j])
            else:
                tkl2.append('')
        labels2 = bx.set_yticklabels((tkl2))

        ax.title(self.name + ', ' + title + ', ' + extra_title)
        ax.set_xlabel(x_label + ', ' + x_units)
        ax.set_ylabel(y_label1 + ', ' + y_units1)
        bx.set_ylabel(y_label2 + ', ' + y_units2, rotation=270)

        index = np.where(coi > per[-1])
        index = index[0]
        for i in range(index.shape[0]):
            coi[index[i]] = per[-1]
        index = np.where(coi < per[0])
        index = index[0]
        for i in range(index.shape[0]):
            coi[index[i]] = per[0]

        # c1 = w_power/signif.reshape(signif.shape[0], 1)
        c1 = w_power / variance
        levels = np.arange(1, 10., 1)
        # levels[0] = 0.5
        # levels=[5e-1,1e0, 2e0, 3e0]
        # c2 = c1 / signif.reshape(signif.shape[0], 1)


        CS = ax.contour(tim, per, c1, levels, extent=[tim[0], tim[-1], per[1], per[-1]],
                        **kwargs)

        # CS1 = ax.contour(tim, per, c2, levels, extent=[tim[0], tim[-1], per[1], per[-1]], 
        #                colors='green', linewidth=1)

        # CS = plt.contour(X, Y, Z)
        # plt.clabel(CS, inline=1, fontsize=10)
        ax.clabel(CS, levels[:2],  # label every second level
                    inline=1,
                fmt='%1i$\sigma^2$',
                fontsize=10)



def min_to_mhz(label):
    temp = label.split('-')
    i = 16.6/float(temp[0])
    btemp = temp[1].split('m')
    f = 16.6/float(btemp[0])
    return format(f, '0.2f') + '-' + format(i, '0.2f')+'mHz'


def signal_smooth(x, window_len=11, window='hanning'):

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len<3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s = np.r_[x[window_len-1:0:-1], x, x[-1:-window_len:-1]]
    #print(len(s))
    if window == 'flat':   #moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('numpy.' + window + '(window_len)')

    y = np.convolve(w / w.sum(), s, mode='valid')
    return y
