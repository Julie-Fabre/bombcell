# bombcell

Quality metrics for neuropixels data, used in Peters et al., 2021 to get some bombshell cells! 


### quality metrics 

####  % spikes missing 

estimate the percent of spikes missing (false nagatives) by fitting a gaussian the distribution of amplitudes, with a cutoff parameter. This assumes the spike amplitudes follow a gaussian distribution, which is not strictly true for bursty cells, like MSNs. This can then define which epochs of the recording to keep for a unit, if it has for example drifted relative to the recording sites and in only some recording epochs a substantial amount of spikes are missing.

![alt text](https://github.com/Julie-Fabre/bombcell/blob/master/images/percSpikesMissingDrift.png?raw=true)

#### number of spikes 

Number of spikes over the recording. Below a certain amount of spikes, ephys properties like ACGs will not be reliable. A good minimum to use is 300 empirically, because Kilosort2 does not attempt to split any clusters that have less than 300 spikes in the post-processing phase.

#### refractory period violations

Estimate fraction of refractory period violations (false positives) using  r = 2*(tauR - tauC) * N^2 * (1-Fp) * Fp / T , solving for Fp, with tauR the refractory period, tauC the censored period, T the total experiment duration, r the number of refractory period violations, Fp the fraction of contamination. method from Hill et al., 2011. 

#### axonal

Axonal waveforms are defined as waveforms where the largest peak precedes the largest trough (Deligkaris,Bullmann& Frey, 2016).

#### number of peaks and troughs

Count the number of peaks and troughs to eliminate non-cell-like waveforms due to noise.

#### amplitude 

Amplitude of the mean raw waveformelfor a unit, to eliminate noisy, further away units, that are more likely to be MUA. 

#### distance metrics  

Compute measure of unit isolation quality: the isolation distance (see Harris et al., 2001), l-ratio (see Schmitzer-Torbert and Redish, 2004) and silhouette-score (see Rousseeuw, 1987). 

### ephys properties 

#### post spike suppresssion 

#### waveform duration

#### proportion long ISIs 

### Used in Peters et al., 2021


## dependancies:

- https://github.com/kwikteam/npy-matlab (to load data in)

- https://github.com/buzsakilab/buzcode, modified from http://www.fieldtriptoolbox.org/ (to compute fast ACG/CCGs. download the repo and run compileBuzcode.m)

- https://uk.mathworks.com/matlabcentral/fileexchange/181-keep 

- https://uk.mathworks.com/matlabcentral/fileexchange/1805-rgb-m
