%% based on Olivier Winter's python function 
function bc_fshift() 
% def fshift(w, s, axis=-1, ns=None):
%     """
%     Shifts a 1D or 2D signal in frequency domain, to allow for accurate non-integer shifts
%     :param w: input signal (if complex, need to provide ns too)
%     :param s: shift in samples, positive shifts forward
%     :param axis: axis along which to shift (last axis by default)
%     :param ns: if a rfft frequency domain array is provided, give a number of samples as there
%      is an ambiguity
%     :return: w
%     """

% create a vector that contains a 1 sample shift on the axis
ns = size(w,2);
    %ns = ns or w.shape[axis]
    shape = ones(size(w));
    %shape = np.array(w.shape) * 0 + 1
    shape(:,1) = ns;
    %shape[axis] = ns
    %dephas = np.zeros(shape)
    dephas = zeros(shape);
    %np.put(dephas, 1, 1)
    dephas(2) = 1;
    %dephas = scipy.fft.rfft(dephas, axis=axis)
    # fft the data along the axis and the dephas
    do_fft = np.invert(np.iscomplexobj(w))
    if do_fft:
        W = scipy.fft.rfft(w, axis=axis)
    else:
        W = w
    # if multiple shifts, broadcast along the other dimensions, otherwise keep a single vector
    if not np.isscalar(s):
        s_shape = np.array(w.shape)
        s_shape[axis] = 1
        s = s.reshape(s_shape)
    # apply the shift (s) to the fft angle to get the phase shift and broadcast
    W *= np.exp(1j * np.angle(dephas) * s)
    if do_fft:
        W = np.real(scipy.fft.irfft(W, ns, axis=axis))
        W = W.astype(w.dtype)
    return W