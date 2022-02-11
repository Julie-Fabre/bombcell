function [p,mu,stdev,n,x] = gaussian_mssing(amplitudes,threshes)
% UltraMegaSort2000 by Hill DN, Mehta SB, & Kleinfeld D  - 07/12/201

%
%
% Input:
%   waveforms  - [Events X Samples X Channels] the waveforms of the cluster
%   threshes   - [1 X Channels] the threshold for each channel
%   criteria_func - Used to determine what the detection metric is on each
%                   waveform.  If this is the string "auto" or "manual" then
%                   it is assumed that a simple voltage threshold was used. 
%                   The detection criterion then is to divide each channel
%                   by its threhsold and use the maximum value.  Otherwise
%                   the criteria_func is assumed to be a function handle that
%                   takes in waveforms and threshes and returns the detection
%                   metric for each event [Events x 1].  The function will
%                   be called as
%                      criteria = criteria_func( waveforms, threshes)
%                   It is assumed that the values of criteria are normalized 
%                   to use a threshold value of + 1.
%
% Output:
%  p            - estimate of probability that a spike is missing because it didn't reach threshhold
%  mu           - mean estimated for gaussian fit
%  stdev        - standard deviation estimated for gaussian fit
%  n            - bin counts for histogram used to fit Gaussian
%  x            - bin centers for histogram used to fit Gaussian
%

bins = 75;
   
    % create the histogram values
    global_max = max(amplitudes);
    mylims = linspace( 1,global_max,bins+1);
    x = mylims +  (mylims(2) - mylims(1))/2;
    n = histc( amplitudes,mylims );
    
    % fit the histogram with a cutoff gaussian
    m = mode_guesser(amplitudes, .05);    % use mode instead of mean, since tail might be cut off
    [stdev,mu] = stdev_guesser(amplitudes, n, x, m); % fit the standard deviation as well

    % Now make an estimate of how many spikes are missing, given the Gaussian and the cutoff
    p = normcdf( 1,mu,stdev);
    

end

% fit the standard deviation to the histogram by looking for an accurate
% match over a range of possible values
function [stdev,m] = stdev_guesser( thresh_val, n, x, m)

    % initial guess is juts the RMS of just the values below the mean
    init = sqrt( mean( (m-thresh_val(thresh_val>=m)).^2  ) );

    % try 20 values, within a factor of 2 of the initial guess
    num = 20;
    st_guesses = linspace( init/2, init*2, num );
    m_guesses  = linspace( m-init,max(m+init,1),num);
    for j = 1:length(m_guesses)
        for k = 1:length(st_guesses)
              b = normpdf(x,m_guesses(j),st_guesses(k));
              b = b *sum(n) / sum(b);
              error(j,k) = sum(abs(b(:)-n(:)));
        end        
    end
    
    % which one has the least error?
    [val,pos] = min(error(:));
    jpos = mod( pos, num ); if jpos == 0, jpos = num; end
    kpos = ceil(pos/num);
    stdev = st_guesses(kpos);
    
    % refine mode estimate
    m     = m_guesses(jpos);

end







