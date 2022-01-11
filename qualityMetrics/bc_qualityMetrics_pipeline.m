%% loadData 
load('/home/netshare/zserver-lab/Share/Celian/dataForJulie.mat')
d1_folder = strrep(dat.d1.ephysSortingFolder, '\', filesep); % windows -> linux
d2_folder = strrep(dat.d2.ephysSortingFolder, '\', filesep); % windows -> linux
d1_folder = strrep(d1_folder, '/128.40.224.65/Subjects/', 'home/netshare/tempserver/'); % folder location
d2_folder = strrep(d2_folder, '/128.40.224.65/Subjects/', 'home/netshare/tempserver/'); % folder location
d1_rawfolder = strrep(dat.d1.ephysFolder, '\', filesep); % windows -> linux
d2_rawfolder = strrep(dat.d2.ephysFolder, '\', filesep); % windows -> linux
d1_rawfolder = strrep(d1_rawfolder, '/128.40.224.65/Subjects/', 'home/netshare/tempserver/'); % folder location
d2_rawfolder = strrep(d2_rawfolder, '/128.40.224.65/Subjects/', 'home/netshare/tempserver/'); % folder location
dat.d1.clu.ID = dat.d1.clu.ID + 1; % 0-idx -> 1-idx
dat.d2.clu.ID = dat.d2.clu.ID + 1; % 0-idx -> 1-idx
foldersAll = {d1_folder; d2_folder};

for iDay = 1:size(foldersAll, 1)
    
    templates{iDay} = readNPY([foldersAll{iDay, 1}, filesep, 'templates.npy']);
    channel_positions{iDay} = readNPY([foldersAll{iDay, 1}, filesep, 'channel_positions.npy']);
    spike_times_seconds{iDay} = double(readNPY([foldersAll{iDay, 1}, filesep, 'spike_times.npy']))./30000;
    spike_times{iDay} = double(readNPY([foldersAll{iDay, 1}, filesep, 'spike_times.npy'])); % sample rate hard-coded as 30000 - should load this in from params 
    spike_templates{iDay} = readNPY([foldersAll{iDay, 1}, filesep, 'spike_templates.npy']) + 1; % 0-idx -> 1-idx
    template_amplitude{iDay} = readNPY([foldersAll{iDay, 1}, filesep, 'amplitudes.npy']);
    spike_clusters{iDay} = readNPY([foldersAll{iDay, 1}, filesep, 'spike_clusters.npy']) + 1;
    %mean_template_ampltiude
end

%% params
param = struct;
param.tauR = 0.0010; %refractory period time (s)
param.tauC = 0.0002; %censored period time (s)
param.maxPercSpikesMissing = 30;
param.minNumSpikes = 300;
param.maxNtroughsPeaks = 3;
param.axonal = 0; 
param.maxRPVviolations = 0.2;
param.minAmplitude = 70; 
param.plotThis = 1;
param.rawFolder = d1_rawfolder;
param.deltaTimeChunk = 600; % 10 min time chunk
param.ephys_sample_rate = 30000;
param.nChannels = 385;
param.nRawSpikesToExtract = 100; 
%% run and save qualityMetrics 
iDay = 1;
qMetrics = bc_runAllQualityMetrics(param, spike_times{iDay}, spike_templates{iDay}, ...
    templates{iDay}, template_amplitude{iDay});
