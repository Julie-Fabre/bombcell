
% from pykilosort 
% def adc_shifts(version=1):
%     """
%     The sampling is serial within the same ADC, but it happens at the same time in all ADCs.
%     The ADC to channel mapping is done per odd and even channels:
%     ADC1: ch1, ch3, ch5, ch7...
%     ADC2: ch2, ch4, ch6....
%     ADC3: ch33, ch35, ch37...
%     ADC4: ch34, ch36, ch38...
%     Therefore, channels 1, 2, 33, 34 get sample at the same time. I hope this is more or
%     less clear. In 1.0, it is similar, but there we have 32 ADC that sample each 12 channels."
%     - Nick on Slack after talking to Carolina - ;-)
%     """
%     if version == 1:
%         adc_channels = 12
%         # version 1 uses 32 ADC that sample 12 channels each
%     elif np.floor(version) == 2:
%         # version 2 uses 24 ADC that sample 16 channels each
%         adc_channels = 16
%     adc = np.floor(np.arange(NC) / (adc_channels * 2)) * 2 + np.mod(np.arange(NC), 2)
%     sample_shift = np.zeros_like(adc)
%     for a in adc:
%         sample_shift[adc == a] = np.arange(adc_channels) / adc_channels
%     return sample_shift, adc
%     
%     def trace_header(version=1):
%     """
%     Returns the channel map for the dense layout used at IBL
%     :param version: major version number: 1 or 2
%     :return: , returns a dictionary with keys
%     x, y, row, col, ind, adc and sampleshift vectors corresponding to each site
%     """
%     h = dense_layout(version=version)
%     h['sample_shift'], h['adc'] = adc_shifts(version=version)
%     return h
%     
%     def dense_layout(version=1):
%     """
%     Returns a dense layout indices map for neuropixel, as used at IBL
%     :param version: major version number: 1 or 2 or 2.4
%     :return: dictionary with keys 'ind', 'col', 'row', 'x', 'y'
%     """
%     ch = {'ind': np.arange(NC),
%           'row': np.floor(np.arange(NC) / 2),
%           'shank': np.zeros(NC)}
% 
%     if version == 2:
%         ch.update({'col': np.tile(np.array([0, 1]), int(NC / 2))})
%     elif version == 2.4:
%         # the 4 shank version default is rather complicated
%         shank_row = np.tile(np.arange(NC / 16), (2, 1)).T[:, np.newaxis].flatten()
%         shank_row = np.tile(shank_row, 8)
%         shank_row += np.tile(np.array([0, 0, 1, 1, 0, 0, 1, 1])[:, np.newaxis], (1, int(NC / 8))).flatten() * 24
%         ch.update({
%             'col': np.tile(np.array([0, 1]), int(NC / 2)),
%             'shank': np.tile(np.array([0, 1, 0, 1, 2, 3, 2, 3])[:, np.newaxis], (1, int(NC / 8))).flatten(),
%             'row': shank_row})
%     elif version == 1:
%         ch.update({'col': np.tile(np.array([2, 0, 3, 1]), int(NC / 4))})
%     # for all, get coordinates
%     ch.update(rc2xy(ch['row'], ch['col'], version=version))
%     return ch
%     
%     
%     x = fshift(x, h['sample_shift'], axis=1)
%     
%     def fshift(w, s, axis=-1, ns=None):
%     """
%     Shifts a 1D or 2D signal in frequency domain, to allow for accurate non-integer shifts
%     :param w: input signal (if complex, need to provide ns too)
%     :param s: shift in samples, positive shifts forward
%     :param axis: axis along which to shift (last axis by default)
%     :param axis: axis along which to shift (last axis by default)
%     :param ns: if a rfft frequency domain array is provided, give a number of samples as there
%      is an ambiguity
%     :return: w
%     """
%     # create a vector that contains a 1 sample shift on the axis
%     ns = ns or w.shape[axis]
%     shape = np.array(w.shape) * 0 + 1
%     shape[axis] = ns
%     dephas = np.zeros(shape)
%     np.put(dephas, 1, 1)
%     dephas = scipy.fft.rfft(dephas, axis=axis)
%     # fft the data along the axis and the dephas
%     do_fft = np.invert(np.iscomplexobj(w))
%     if do_fft:
%         W = scipy.fft.rfft(w, axis=axis)
%     else:
%         W = w
%     # if multiple shifts, broadcast along the other dimensions, otherwise keep a single vector
%     if not np.isscalar(s):
%         s_shape = np.array(w.shape)
%         s_shape[axis] = 1
%         s = s.reshape(s_shape)
%     # apply the shift (s) to the fft angle to get the phase shift and broadcast
%     W *= np.exp(1j * np.angle(dephas) * s)
%     if do_fft:
%         W = np.real(scipy.fft.irfft(W, ns, axis=axis))
%         W = W.astype(w.dtype)
%     return W
%     