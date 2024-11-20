# import from plugins/cluster_metrics.py
"""Show how to add a custom cluster metrics."""

import numpy as np
from phy import IPlugin
from scipy.optimize import curve_fit
from scipy.special import ndtr


class qMetricsPlugin(IPlugin):
    def attach_to_controller(self, controller):
        """Note that this function is called at initialization time, *before* the supervisor is
        created. The `controller.cluster_metrics` items are then passed to the supervisor when
        constructing it."""
        # fraction refractory period violations
        # amplitude clipped off percent
        # waveform somatic/non somatic
        # waveform peaks
        # waveform decay
        # waveform baseline
        # waveform troughs
        # waveform amplitude

        def fractionRPV(cluster_id): #add this written in the view
            t = controller.get_spike_times(cluster_id).data
            tauR = 0.0020
            tauC = 0.0010
            a = 2 * (tauR - tauC) * len(t) / np.abs(np.max(t) - np.min(t))
            r = np.sum(np.diff(t) <= tauR)
            if r ==0:
                fp = 0
            else:
                rts = np.roots([-1, 1, -r/a])
                fp = np.min(rts)
                if np.iscomplex(fp):
                    if r < len(t):
                        fp = r / (2 * (tauR - tauC) * (len(t) - r))
                    else:
                        fp = 1
            return fp

        def percentSpikesMissing(cluster_id):
           amp = controller.get_amplitudes(cluster_id)
           num, bins = np.histogram(amp, bins=50)

           def gaussian(x, a, x0, sigma):
               return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))

           def gaussian_cut(x, a, x0, sigma, xcut):
               g = a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))
               g[x < xcut] = 0
               return g

           mean_seed = bins[np.argmax(num)]  # mode of mean_seed
           bin_steps = np.diff(bins[:2])[0]
           x = bins[:-1] + bin_steps / 2
           next_low_bin = x[0] - bin_steps
           add_points = np.flipud(np.arange(next_low_bin,
                                            0, -bin_steps))
           x = np.append(add_points, x)
           num = np.append(np.zeros(len(add_points)), num)

           p0 = (num.max(), mean_seed, 2 * amp.std(),
                 np.percentile(amp, 1))
           try:
               popt, pcov = curve_fit(gaussian_cut, x, num, p0=p0,
                                          maxfev=10000)
               was_fit = True
           except:
               try:
                    popt, pcov = curve_fit(gaussian_cut, x, num, p0=p0,
                                          maxfev=50000)
                    was_fit = True
               except:
                    was_fit = False
                    percent_missing_ndtr = float("NaN")
                    n_fit = float("NaN")
           if was_fit:
               #n_fit = gaussian_cut(x, popt[0], popt[1], popt[2], popt[3])


               min_amplitude = popt[3]

               norm_area_ndtr = ndtr((popt[1] - min_amplitude) /
                                        popt[2])
               percent_missing_ndtr = 100 * (1 - norm_area_ndtr)
           return percent_missing_ndtr#add this written in the view + curve_fit



        # def numTroughs(cluster_id):
        #     waveforms = controller._get_template_waveforms(cluster_id).data
        #     return waveforms[0]# add locations + number in the view

        # def numPeaks(cluster_id):
        #     return # add locations + number in the view
        #
        # def baselineFlatness(cluster_id):
        #     return # add lines in the view
        #
        # def spatialDecay(cluster_id):
        #     return


        # Use this dictionary to define custom cluster metrics.
        # We memcache the function so that cluster metrics are only computed once and saved
        # within the session, and also between sessions (the memcached values are also saved
        # on disk).
        controller.cluster_metrics['fractionRPV'] = controller.context.memcache(fractionRPV)

        controller.cluster_metrics['percentSpikesMissing'] = controller.context.memcache(percentSpikesMissing)
        controller.context.save_memcache()
        #controller.cluster_metrics['numTroughs'] = controller.context.memcache(numTroughs)
