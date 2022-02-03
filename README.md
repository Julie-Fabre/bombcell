# bombcell: evaluate and plot the quality of neuropixel-recorded units & compute and plot electrophysiological characteristics

Used in Peters et al., 2021 to classify striatal units .

<img src="https://github.com/Julie-Fabre/bombcell/blob/master/images/bombcell2.png" width=20% height=20%>

### Quality metrics 

Run all quality metrics with the function bc_runAllQualityMetrics. Eg:

    [qMetric, goodUnits] = bc_runAllQualityMetrics(param, spikeTimes, spikeTemplates, ...
      templateWaveforms, templateAmplitudes,pcFeatures,pcFeatureIdx,usedChannels, savePath);
    					
Then, plot a GUI to flip through the quality metrics for each cell with the function bc_unitQualityGUI Eg:

    bc_unitQualityGUI(memMapData, ephysData, qMetrics, param, probeLocation)

<img src="https://github.com/Julie-Fabre/bombcell/blob/master/images/GUI.png" width=40% height=40%>


						
####  % spikes missing 

estimate the percent of spikes missing (false nagatives) by fitting a gaussian the distribution of amplitudes, with a cutoff parameter. This assumes the spike amplitudes follow a gaussian distribution, which is not strictly true for bursty cells, like MSNs. This can then define which epochs of the recording to keep for a unit, if it has for example drifted relative to the recording sites and in only some recording epochs a substantial amount of spikes are missing.

Below: example of unit with many spikes below the detection threshold in the first two time chunks of the recording. 

![alt text](https://github.com/Julie-Fabre/bombcell/blob/master/images/percSpikesMissingDrift.png?raw=true)

#### number of spikes 

Number of spikes over the recording. Below a certain amount of spikes, ephys properties like ACGs will not be reliable. A good minimum to use is 300 empirically, because Kilosort2 does not attempt to split any clusters that have less than 300 spikes in the post-processing phase.


#### refractory period violations

Estimate fraction of refractory period violations (false positives) using  r = 2*(tauR - tauC) * N^2 * (1-Fp) * Fp / T , solving for Fp, with tauR the refractory period, tauC the censored period, T the total experiment duration, r the number of refractory period violations, Fp the fraction of contamination. method from Hill et al., 2011. 

Below: examples of a unit with a small fraction of refractory period violations (left) and one with a large fraction (right).

![alt text](https://github.com/Julie-Fabre/bombcell/blob/master/images/rpv.png?raw=true)

#### axonal

Axonal waveforms are defined as waveforms where the largest peak precedes the largest trough (Deligkaris,Bullmann& Frey, 2016).

#### number of peaks and troughs

Count the number of peaks and troughs to eliminate non-cell-like waveforms due to noise.

Below: examples of a unit with a a cell-like waveform (left) and a unit with a noise-like waveform (right).

![alt text](https://github.com/Julie-Fabre/bombcell/blob/master/images/numberTroughsPeaks.png?raw=true)

#### amplitude 

Amplitude of the mean raw waveformelfor a unit, to eliminate noisy, further away units, that are more likely to be MUA. 

Below: examples of a unit with high amplitude (blue) and one with low amplitude (red).

![alt text](https://github.com/Julie-Fabre/bombcell/blob/master/images/amplitude.png?raw=true)

#### distance metrics  

Compute measure of unit isolation quality: the isolation distance (see Harris et al., 2001), l-ratio (see Schmitzer-Torbert and Redish, 2004) and silhouette-score (see Rousseeuw, 1987). 

Below: examples of a unit with high isolation distance (left) and one with low isolation distance (right).

![alt text](https://github.com/Julie-Fabre/bombcell/blob/master/images/isolationDistance.png?raw=true)

### Ephys properties - work in progress 

#### post spike suppression 

#### waveform duration

#### proportion long ISIs 

### Used in Peters et al., 2021

work in progress

## Dependancies:

- https://github.com/kwikteam/npy-matlab (to load data in)

- https://github.com/buzsakilab/buzcode, modified from http://www.fieldtriptoolbox.org/ (to compute fast ACG/CCGs. download the repo and run compileBuzcode.m)

- https://uk.mathworks.com/matlabcentral/fileexchange/181-keep 

- https://uk.mathworks.com/matlabcentral/fileexchange/1805-rgb-m

- https://github.com/tuckermcclure/matlab-plot-big
