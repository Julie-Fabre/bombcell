# bombcell

Quality metrics for neuropixels data, used in Peters et al., 2021 to get some bombshell cells! 

## Used in Peters et al., 2021

### quality metrics 

#### number of spikes 

minimum number of spikes over recording. The minimum was set to 300 empirically, and because Kilosort2 does not attempt to split any clusters that have less than 300 spikes in the post-processing phase. 

#### refractory period violations

method from Hill et al., 2011. Estimate fraction of refractory period violations using  r = 2*(tauR - tauC) * N^2 * (1-Fp) * Fp / T , solving for Fp, with tauR the refractory period, tauC the censored period, T the total experiment duration, r the number of refractory period violations, Fp the fraction of contamination

#### % spikes missing 

estimate the percent of spikes missing by fitting a gaussian the distribution of amplitudes, with a cutoff parameter. This assumes the spike amplitudes follow a gaussian distribution, which is not strictly true for bursty cells, like MSNs.

#### somatic

eliminate axonal spikes by keeping only cells where the waveform trough precedes the waveform peak (Deligkaris,Bullmann& Frey, 2016)

#### amplitude 

eliminate waveforms of a low amplitude to remove noisy, further away units, that are more likely to be MUA. 


### ephys properties 

#### post spike suppresssion 

#### waveform duration

#### proportion long ISIs 


## Additional

### quality metrics 

### ephys properties 


## dependancies:

- https://github.com/kwikteam/npy-matlab (to load data in)

- https://github.com/petersaj/AP_scripts_cortexlab (to load data in and most other things)

- https://github.com/buzsakilab/buzcode, modified from http://www.fieldtriptoolbox.org/ (to compute fast ACG/CCGs. download the repo and run compileBuzcode.m)

- https://uk.mathworks.com/matlabcentral/fileexchange/181-keep 

- https://uk.mathworks.com/matlabcentral/fileexchange/1805-rgb-m
