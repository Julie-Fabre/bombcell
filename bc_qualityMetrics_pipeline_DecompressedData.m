% Example calling QMatrix (Bombcell) - Enny
%% Quality matrix - Bombcell (https://github.com/Julie-Fabre/bombcell)
ephysap_tmp = [];
ephysap_path = fullfile(lfpD.folder,lfpD.name);

DecompressLocal = 1; %When you have compressed data there's two options: 1) decompress locally and use the decompressed data 2) decompress on the fly
qMetricsExist = dir(fullfile(myClusFile(1).folder, 'qMetric*.mat'));
if isempty(qMetricsExist) || RedoQM
    clear param
    bc_qualityParamValues;
    param.plotThis = 0;
    param.plotGlobal=1;

    % First check if we want to use python for compressed data. If not, uncompress data first
    if any(strfind(lfpD.name,'cbin')) && DecompressLocal
        if ~exist(fullfile(tmpdatafolder,strrep(lfpD.name,'cbin','bin')))
            disp('This is compressed data and we do not want to use Python integration... uncompress temporarily')
            % Decompression
            success = pyrunfile("MTSComp_From_Matlab.py","success",datapath = strrep(fullfile(lfpD.folder,lfpD.name),'\','/'),...
                JsonPath =  strrep(fullfile(lfpD.folder,strrep(lfpD.name,'cbin','ch')),'\','/'), savepath = strrep(fullfile(tmpdatafolder,strrep(lfpD.name,'cbin','bin')),'\','/'))
            % Also copy metafile
            copyfile(strrep(fullfile(lfpD.folder,lfpD.name),'cbin','meta'),strrep(fullfile(tmpdatafolder,lfpD.name),'cbin','meta'))
        end
        ephysap_tmp = fullfile(tmpdatafolder,strrep(lfpD.name,'cbin','bin'));
        DecompressionFlag = 1;

    end

    %             idx = ismember(sp{countid}.spikeTemplates,clusidtmp(Good_IDtmp)); %Only include good units
    idx = true(1,length(sp{countid}.spikeTemplates));
    %careful; spikeSites zero indexed
    [qMetric, unitType] = bc_runAllQualityMetrics(param, round(sp{countid}.st(idx).*sp{countid}.sample_rate), sp{countid}.spikeTemplates(idx)+1, ...
        sp{countid}.temps, sp{countid}.tempScalingAmps(idx),sp{countid}.pcFeat(idx,:,:),sp{countid}.pcFeatInd,channelpostmp, myClusFile(1).folder);
else
    load(fullfile(myClusFile(1).folder, 'qMetric.mat'))
    load(fullfile(myClusFile(1).folder, 'param.mat'))
    bc_getQualityUnitType;
end
AllQMsPaths{countid} = fullfile(myClusFile(1).folder, 'qMetric.mat');

if InspectQualityMatrix
    bc_getRawMemMap;

    %% view units + quality metrics in GUI
    % put ephys data into structure
    ephysData = struct;
    ephysData.spike_times = sp{countid}.st.*round(sp{countid}.sample_rate);
    ephysData.ephys_sample_rate = sp{countid}.sample_rate;
    ephysData.spike_times_timeline = sp{countid}.st;
    ephysData.spike_templates = sp{countid}.spikeTemplates+1; %0-indexed
    ephysData.templates = sp{countid}.temps;
    ephysData.template_amplitudes = sp{countid}.tempScalingAmps;
    ephysData.channel_positions = channelpostmp;
    ephysData.waveform_t = 1e3*((0:size(sp{countid}.temps, 2) - 1) / ephysData.ephys_sample_rate);
    ephysParams = struct;
    plotRaw = 1;
    probeLocation=[];
    %% Inspect the results? - Work in progress: Decompressing on the fly still needs to be implemented here
    unitQualityGuiHandle = bc_unitQualityGUI(memMapData,ephysData,qMetric, param, probeLocation, unitType, plotRaw);
    disp('Not continuing until GUI closed')
    while isvalid(unitQualityGuiHandle) && InspectQualityMatrix
        pause(0.01)
    end
    disp('GUI closed')
end